import Compression
import CryptoKit
import Foundation

/// The pre-shared key and everything derived from it.
///
/// The key never appears in a frame. It is used exactly twice: once to prove
/// possession of it during the handshake, and once as the input key material for
/// the session keys that seal the connection afterwards.
enum RemoteCrypto {
    static let keyByteCount = 32
    static let nonceByteCount = 32
    static let proofByteCount = 32

    // MARK: Key material

    /// 32 bytes from the system CSPRNG. `SystemRandomNumberGenerator` is
    /// `SecRandomCopyBytes` underneath and is the same generator the app already
    /// relies on implicitly everywhere else.
    static func generateKey() -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: keyByteCount)
        var generator = SystemRandomNumberGenerator()
        for index in bytes.indices { bytes[index] = generator.next() }
        return bytes
    }

    /// The public, non-secret identifier a beacon can carry: the first eight bytes
    /// of a hash of the key.
    ///
    /// A hash prefix of a 256-bit random secret discloses nothing, and it lets a
    /// client say "Ready" or "Key needed" for an autodiscovered server without
    /// opening a socket — and notice a rotation when the value changes.
    static func keyID(for key: [UInt8]) -> Data {
        Data(SHA256.hash(data: Data("WF1/keyid".utf8) + Data(key)).prefix(8))
    }

    // MARK: Display format

    /// `WF1-` plus eight groups of eight uppercase hex characters, then a
    /// four-character check group.
    ///
    /// Hex rather than base32 because it is unambiguous when read aloud and when
    /// compared by eye, and this is a human-facing rendering of the key — it is
    /// never what travels on the socket.
    static func displayString(for key: [UInt8]) -> String {
        precondition(key.count == keyByteCount, "a remote key is 32 bytes")
        var groups: [String] = []
        for offset in stride(from: 0, to: key.count, by: 4) {
            groups.append(key[offset..<(offset + 4)].map { String(format: "%02X", $0) }.joined())
        }
        groups.append(checkCharacters(for: key))
        return "WF1-" + groups.joined(separator: "-")
    }

    /// Parses the display format, rejecting a bad check group before the key is
    /// used for anything. A mistyped key fails here with a clear message instead
    /// of becoming a confusing authentication error.
    static func parseDisplayString(_ text: String) throws -> [UInt8] {
        let cleaned = text.uppercased().filter { $0 != "-" && $0 != " " }
        guard cleaned.hasPrefix("WF1") else {
            throw RemoteError.badKey
        }
        let body = String(cleaned.dropFirst(3))
        guard body.count == keyByteCount * 2 + 4 else {
            throw RemoteError.badKey
        }
        let data = Array(body.utf8)
        func nibble(_ byte: UInt8) throws -> UInt8 {
            switch byte {
            case 0x30...0x39: byte - 0x30
            case 0x41...0x46: byte - 0x41 + 10
            default: throw RemoteError.badKey
            }
        }
        var key = [UInt8]()
        key.reserveCapacity(keyByteCount)
        var index = 0
        while index < keyByteCount * 2 {
            let high = try nibble(data[index])
            let low = try nibble(data[index + 1])
            key.append(high << 4 | low)
            index += 2
        }
        guard checkCharacters(for: key) == String(body.suffix(4)) else {
            throw RemoteError.badKey
        }
        return key
    }

    /// Four hex characters: the first two bytes of a hash over the key.
    private static func checkCharacters(for key: [UInt8]) -> String {
        let digest = SHA256.hash(data: Data("WF1/check".utf8) + Data(key))
        return digest.prefix(2).map { String(format: "%02X", $0) }.joined()
    }

    // MARK: Handshake proofs

    static func clientProof(key: [UInt8], clientNonce: Data, serverNonce: Data) -> Data {
        proof(key: key, label: "WF1/auth", clientNonce: clientNonce, serverNonce: serverNonce)
    }

    static func serverProof(key: [UInt8], sessionID: String, clientNonce: Data, serverNonce: Data) -> Data {
        proof(key: key, label: "WF1/server", sessionID: sessionID,
              clientNonce: clientNonce, serverNonce: serverNonce)
    }

    private static func proof(key: [UInt8], label: String, sessionID: String = "",
                              clientNonce: Data, serverNonce: Data) -> Data {
        var message = Data(label.utf8)
        message.append(Data(sessionID.utf8))
        message.append(clientNonce)
        message.append(serverNonce)
        let mac = HMAC<SHA256>.authenticationCode(for: message, using: SymmetricKey(data: Data(key)))
        return Data(mac)
    }

    /// Compares in time.
    ///
    /// `==` on `[UInt8]` short-circuits at the first differing byte, which turns
    /// the verification into an oracle for how many leading bytes were correct.
    /// Every comparison of a secret-derived value in this file goes through here.
    static func constantTimeEquals(_ lhs: Data, _ rhs: Data) -> Bool {
        guard lhs.count == rhs.count else { return false }
        var difference: UInt8 = 0
        for (left, right) in zip(lhs, rhs) { difference |= left ^ right }
        return difference == 0
    }

    // MARK: Session keys

    /// The two direction-separated session keys.
    ///
    /// Named by direction rather than by "send" and "receive", because which one
    /// is the send key depends on which end you are. The client seals with
    /// `clientToServer` and opens with `serverToClient`; the server does the
    /// reverse. Deriving both in one place is what stops the two ends from
    /// disagreeing about it.
    ///
    /// The nonces are fresh per connection, so the derived keys are too: a
    /// recorded session cannot be replayed against a later one even when the
    /// shared key is unchanged.
    static func sessionKeys(key: [UInt8], clientNonce: Data, serverNonce: Data)
        -> (clientToServer: SymmetricKey, serverToClient: SymmetricKey) {
        let salt = clientNonce + serverNonce
        func material(_ direction: RemoteDirection) -> SymmetricKey {
            let label = direction == .clientToServer ? "c2s" : "s2c"
            return HKDF<SHA256>.deriveKey(inputKeyMaterial: SymmetricKey(data: Data(key)),
                                          salt: salt, info: Data("waifufinder-remote/v1 \(label)".utf8),
                                          outputByteCount: 32)
        }
        return (clientToServer: material(.clientToServer), serverToClient: material(.serverToClient))
    }

    // MARK: Frame sealing

    /// One direction's sealing state.
    ///
    /// The 24-byte header is authenticated as associated data rather than
    /// encrypted, so a reader can find the payload boundary without a key while
    /// an attacker still cannot rewrite an opcode, a length or a stream id.
    struct Sealer {
        private let key: SymmetricKey
        private let directionConstant: UInt32
        private var counter: UInt64 = 0

        init(key: SymmetricKey, direction: RemoteDirection) {
            self.key = key
            directionConstant = direction == .clientToServer ? 1 : 2
        }

        mutating func seal(_ header: RemoteHeader, payload: [UInt8]) throws -> [UInt8] {
            guard counter < UInt64.max else { throw RemoteError.framing("frame counter exhausted") }
            counter += 1
            let nonce = try RemoteCrypto.nonce(directionConstant: directionConstant, counter: counter)
            let box = try ChaChaPoly.seal(payload, using: key, nonce: nonce,
                                          authenticating: Data(header.encoded()))
            return [UInt8](box.ciphertext) + [UInt8](box.tag)
        }
    }

    /// One direction's opening state.
    struct Opener {
        private let key: SymmetricKey
        private let directionConstant: UInt32
        private var highestCounter: UInt64 = 0

        init(key: SymmetricKey, direction: RemoteDirection) {
            self.key = key
            // A server opens what the client sends and vice versa, so the
            // expected counter constant is the peer's.
            directionConstant = direction == .clientToServer ? 2 : 1
        }

        /// Returns the plaintext, or throws. A frame whose nonce is not exactly
        /// one past the highest seen is a replay and closes the connection.
        mutating func open(_ header: RemoteHeader, sealed: [UInt8]) throws -> [UInt8] {
            guard sealed.count >= 16 else { throw RemoteError.framing("sealed frame is shorter than its tag") }
            let tag = Array(sealed.suffix(16))
            let ciphertext = Array(sealed.dropLast(16))
            let nonce = try RemoteCrypto.nonce(directionConstant: directionConstant, counter: highestCounter + 1)
            do {
                let box = try ChaChaPoly.SealedBox(nonce: nonce, ciphertext: ciphertext, tag: tag)
                let plaintext = try ChaChaPoly.open(box, using: key,
                                                   authenticating: Data(header.encoded()))
                highestCounter += 1
                return [UInt8](plaintext)
            } catch {
                throw RemoteError.framing("frame failed authentication")
            }
        }
    }

    /// 4-byte direction constant, then an 8-byte big-endian frame counter.
    ///
    /// A counter nonce is correct here precisely because there is one connection:
    /// the same (key, nonce) pair cannot occur twice, and a replay is caught by
    /// requiring the counter to advance by exactly one.
    private static func nonce(directionConstant: UInt32, counter: UInt64) throws -> ChaChaPoly.Nonce {
        guard counter > 0 else { throw RemoteError.framing("frame counter underflow") }
        var bytes = [UInt8](repeating: 0, count: 12)
        bytes[0] = UInt8(truncatingIfNeeded: directionConstant >> 24)
        bytes[1] = UInt8(truncatingIfNeeded: directionConstant >> 16)
        bytes[2] = UInt8(truncatingIfNeeded: directionConstant >> 8)
        bytes[3] = UInt8(truncatingIfNeeded: directionConstant)
        for index in 0..<8 { bytes[4 + index] = UInt8(truncatingIfNeeded: counter >> UInt64(56 - 8 * index)) }
        return try ChaChaPoly.Nonce(data: Data(bytes))
    }

    /// Removes a frame's zlib wrapper when the flag says it is there.
    ///
    /// Compression is a transport optimisation for control payloads and the
    /// vocabulary. It never stands in for the raw-bytes rule: what comes out is
    /// byte-identical to what went in.
    static func decompress(_ payload: [UInt8], expected: Int) throws -> [UInt8] {
        guard expected >= 0, expected <= RemoteLimits.maximumMessagePayload else { throw RemoteError.tooLarge }
        if expected == 0, payload.isEmpty { return [] }
        guard !payload.isEmpty else { throw RemoteError.framing("empty compressed payload") }
        var output = [UInt8](repeating: 0, count: expected + 1)
        let written = output.withUnsafeMutableBytes { destination in
            payload.withUnsafeBytes { source in
                compression_decode_buffer(destination.bindMemory(to: UInt8.self).baseAddress!,
                                          destination.count,
                                          source.bindMemory(to: UInt8.self).baseAddress!,
                                          source.count, nil, COMPRESSION_ZLIB)
            }
        }
        guard written == expected else {
            throw RemoteError.framing("decompressed payload is \(written) bytes, expected \(expected)")
        }
        output = Array(output.prefix(written))
        return output
    }

    static func compress(_ payload: [UInt8]) -> [UInt8] {
        guard !payload.isEmpty else { return [] }
        // Worst case for incompressible data is slightly larger than the input.
        var output = [UInt8](repeating: 0, count: payload.count + 64)
        let written = output.withUnsafeMutableBytes { destination in
            payload.withUnsafeBytes { source in
                compression_encode_buffer(destination.bindMemory(to: UInt8.self).baseAddress!,
                                          destination.count,
                                          source.bindMemory(to: UInt8.self).baseAddress!,
                                          source.count, nil, COMPRESSION_ZLIB)
            }
        }
        guard written > 0 else {
            // Retry with room for the per-block overhead of incompressible data.
            output = [UInt8](repeating: 0, count: payload.count * 2 + 1024)
            let size = output.withUnsafeMutableBufferPointer { destination in
                payload.withUnsafeBufferPointer { source in
                    compression_encode_buffer(destination.baseAddress!, destination.count,
                                              source.baseAddress!, source.count, nil, COMPRESSION_ZLIB)
                }
            }
            return Array(output.prefix(size))
        }
        return Array(output.prefix(written))
    }
}

/// Which way round a key pair is. Named so neither end has to reason about
/// "c2s" being the send key for exactly one of them.
enum RemoteDirection: Sendable {
    case clientToServer, serverToClient
}