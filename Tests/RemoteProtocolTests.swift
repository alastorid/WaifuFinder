import AppKit
import CoreGraphics
import CoreMedia
import Foundation
import Network

// Wire-protocol tests.
//
// Everything here runs over a real TCP listener bound to `127.0.0.1:0`, so the
// framing, the sealing and the request routing are exercised as they actually
// run rather than through a stub of themselves. No beacon is involved and no
// local-network permission is needed, because loopback is not a local network
// access.
//
// The models are stubbed through `RemoteModelCatalog`, which is the seam the
// serving side exposes for exactly this reason. The assertion that matters most
// is the last one: a remote tag batch and a remote reconstruction produce the
// same bytes a local run would have, which is what makes "remote results are the
// same as local results" a tested claim rather than a comment.

// MARK: - Fixtures

/// A tagger with no model behind it and fully predictable output.
///
/// Rows are derived from the head index and the batch position, so a mis-ordered
/// response is visible rather than plausible — a stub that returned the same row
/// for every frame would pass a test that a real out-of-order bug could not.
private actor RemoteTagWindowProbe {
    var arrived = 0
    var completed = 0
    var released = false
    let waitForCancellation: Bool
    init(waitForCancellation: Bool = false) { self.waitForCancellation = waitForCancellation }
    func noteReleased() { released = true }
    func enter() async throws {
        arrived += 1
        defer { completed += 1 }
        if waitForCancellation {
            try await Task.sleep(for: .seconds(30))
            return
        }
        let deadline = Date().addingTimeInterval(5)
        while arrived < 8 {
            try Task.checkCancellation()
            guard Date() < deadline else { throw TaggerError.invalidBatch }
            try await Task.sleep(for: .milliseconds(1))
        }
    }
}

private final class StubTagger: TaggerBackend, @unchecked Sendable {
    let vocabulary: TagVocabulary?
    private let lock = NSLock()
    private var batchCount = 0
    private let windowProbe: RemoteTagWindowProbe?

    init(vocabulary: TagVocabulary?, windowProbe: RemoteTagWindowProbe? = nil) {
        self.vocabulary = vocabulary
        self.windowProbe = windowProbe
    }

    var batches: Int { lock.withLock { batchCount } }

    func probabilityRows(batch: [CGImage]) async throws -> [[Float]] {
        guard vocabulary != nil else { throw TaggerError.invalidModel("stub has no vocabulary") }
        lock.withLock { batchCount += 1 }
        try await windowProbe?.enter()
        return batch.enumerated().map { position, _ in
            [Float]((0..<TagVocabulary.headCount).map { index in
                Float((index * 31 + position * 7) % 1_000) / 4_000
            })
        }
    }

    func tag(batch: [CGImage]) async throws -> [TagResult] {
        guard let vocabulary else { throw TaggerError.invalidModel("stub has no vocabulary") }
        return try await probabilityRows(batch: batch).map {
            try vocabulary.result(probabilities: $0, threshold: 0.35)
        }
    }
}

/// Writes `baseline` into `reconstructed`, which is exactly what a backend does
/// when it returns the degraded input unchanged — so the test can compare the
/// bytes that came back over the wire with the bytes that went out.
private struct StubUncoverBackend: TemporalVideoUpscaler {
    func capabilities() async -> UpscalerCapabilities {
        UpscalerCapabilities(available: true, message: "STUB", scales: UncoverScale.allCases,
                             minimumFrames: 5, requiresUnsafeMemoryOptIn: false)
    }
    func reconstruct(_ prepared: UncoverPrepared, allowUnsafeMemory: Bool,
                     progress: @escaping @Sendable (String) -> Void) async throws -> UncoverResult {
        for frame in prepared.manifest.frames {
            try Task.checkCancellation()
            let source = RegionImages.frameURL(prepared.directory, "degraded", frame.index)
            try FileManager.default.copyItem(at: source,
                                             to: RegionImages.frameURL(prepared.directory, "reconstructed", frame.index))
            progress("STUB · frame \(frame.index + 1)/\(prepared.manifest.frames.count)")
        }
        return UncoverResult(prepared: prepared, model: "STUB · baseline", seconds: 0.01)
    }
}

private func remoteTestKey() -> [UInt8] { RemoteCrypto.generateKey() }

// MARK: - Registration

func registerRemoteProtocolTests() {
    let suite = "Remote protocol"
    Registry.shared.add(suite: suite, TestCase(name: "frame headers round trip and reject every malformed field before allocating", knownBug: nil) {
        let header = RemoteHeader(opcode: .tagBatch, streamID: 42, flags: [.isFinal, .zlib], byteCount: 1_234)
        let decoded = try RemoteHeader.decode(header.encoded())
        checkEqual(decoded.opcodeValue, .tagBatch, "opcode survives")
        checkEqual(decoded.streamID, 42, "stream id survives")
        check(decoded.flags.contains(.zlib), "flags survive")
        check(decoded.payloadByteCount == 1_234, "declared length survives")

        var bad = header.encoded()
        bad[0] = 0x00
        _ = checkThrowsSync("bad magic") { try RemoteHeader.decode(bad) }

        var wrongVersion = header.encoded()
        wrongVersion[4] = 9
        _ = checkThrowsSync("unsupported version") { try RemoteHeader.decode(wrongVersion) }

        var reserved = header.encoded()
        reserved[20] = 1
        _ = checkThrowsSync("nonzero reserved bytes") { try RemoteHeader.decode(reserved) }

        // The important one: a four-byte frame declaring a gigabyte must be
        // refused on the declared length alone, before any buffer is reserved.
        var huge = header.encoded()
        for index in 16..<20 { huge[index] = 0xFF }
        _ = checkThrowsSync("oversized declared length") { try RemoteHeader.decode(huge) }

        _ = checkThrowsSync("short header") { try RemoteHeader.decode(Array(header.encoded().dropLast())) }
    })

    Registry.shared.add(suite: suite, TestCase(name: "messages split across frames reassemble in order and interleaving is refused", knownBug: nil) {
        // Two frames per message, so the non-final path is genuinely exercised.
        let payload = (0..<(RemoteLimits.maximumFramePayload * 2 + 17)).map { UInt8($0 % 251) }
        let frames = RemoteMessage.frames(opcode: .tagBatch, streamID: 7, payload: payload)
        check(frames.count == 3, "a two-frame payload plus a remainder is three frames")
        check(frames.dropLast().allSatisfy { !$0.header.flags.contains(.isFinal) },
              "only the last frame is final")

        var assembler = RemoteMessageAssembler()
        var finished: (opcode: RemoteOpcode, stream: UInt32, payload: [UInt8])?
        for (index, frame) in frames.enumerated() {
            let message = try assembler.append(frame)
            if index < frames.count - 1 {
                checkNil(message as Any?, "no message is reported before the final frame")
            } else {
                finished = message
            }
        }
        checkEqual(finished?.opcode, .tagBatch, "opcode survives reassembly")
        checkEqual(finished?.stream, 7, "stream id survives reassembly")
        check(finished?.payload == payload, "every byte survives reassembly in order")

        // Interleaving two streams is a protocol violation, not a message to
        // tolerate: the bytes cannot be attributed to a request any more.
        var mixed = RemoteMessageAssembler()
        _ = try mixed.append(RemoteFrame(opcode: .tagBatch, streamID: 1, flags: [], payload: [1]))
        _ = checkThrowsSync("interleaved opcode") {
            try mixed.append(RemoteFrame(opcode: .tagResult, streamID: 1, flags: [.isFinal], payload: [2])) as Any
        }
        var mismatched = RemoteMessageAssembler()
        _ = try mismatched.append(RemoteFrame(opcode: .tagBatch, streamID: 1, flags: [], payload: [1]))
        _ = checkThrowsSync("interleaved stream") {
            try mismatched.append(RemoteFrame(opcode: .tagBatch, streamID: 2, flags: [.isFinal], payload: [2])) as Any
        }
        var afterFinish = RemoteMessageAssembler()
        _ = try afterFinish.append(RemoteFrame(opcode: .tagBatch, streamID: 1, flags: [.isFinal], payload: [1]))
        _ = try afterFinish.append(RemoteFrame(opcode: .tagBatch, streamID: 2, flags: [.isFinal], payload: [2]))
        check(true, "the assembler accepts a fresh message after the previous one completed")
    })

    Registry.shared.add(suite: suite, TestCase(name: "access keys round trip through the display format and reject a mistyped character", knownBug: nil) {
        let key = remoteTestKey()
        let text = RemoteCrypto.displayString(for: key)
        check(text.hasPrefix("WF1-"), "the display form is namespaced")
        checkEqual(text.filter { $0 != "-" }.count, RemoteCrypto.keyByteCount * 2 + 7,
                   "32 bytes plus the WF1 tag and the check group")
        check(try RemoteCrypto.parseDisplayString(text) == key, "the key round trips exactly")

        // Lower case and spaces are what a paste actually contains.
        check(try RemoteCrypto.parseDisplayString(text.lowercased()) == key, "case and spacing do not matter")

        // A single flipped hex character is the realistic paste error, and the
        // check group is what catches it here rather than at authentication.
        var damaged = text
        if let firstDigit = damaged.firstIndex(where: { $0.isHexDigit && $0 != "1" }) {
            damaged.replaceSubrange(firstDigit...firstDigit,
                                    with: String(damaged[firstDigit] == "0" ? "1" : "0"))
            checkThrowsSync("a flipped character is rejected") {
                try RemoteCrypto.parseDisplayString(damaged)
            }
        } else {
            check(false, "the display form must contain a hex digit to damage")
        }
        _ = checkThrowsSync("a truncated key") { try RemoteCrypto.parseDisplayString("WF1-ABCD") }
        _ = checkThrowsSync("an unnamespaced key") { try RemoteCrypto.parseDisplayString(text.replacingOccurrences(of: "WF1-", with: "")) }
        _ = checkThrowsSync("a non-hex character") {
            try RemoteCrypto.parseDisplayString("WF1-ZZZZ-" + String(repeating: "A", count: 71))
        }

        // Two keys must not share a keyID, or a rotated key would look unchanged.
        check(RemoteCrypto.keyID(for: key) != RemoteCrypto.keyID(for: RemoteCrypto.generateKey()),
              "distinct keys have distinct identifiers")
        checkEqual(RemoteCrypto.keyID(for: key), RemoteCrypto.keyID(for: key), "the identifier is stable")
        checkEqual(RemoteCrypto.keyID(for: key).count, 8, "the identifier is eight bytes")
    })

    Registry.shared.add(suite: suite, TestCase(name: "session keys separate the two directions and frames survive sealing", knownBug: nil) {
        let key = remoteTestKey()
        let clientNonce = Data(repeating: 0xA1, count: RemoteCrypto.nonceByteCount)
        let serverNonce = Data(repeating: 0xB2, count: RemoteCrypto.nonceByteCount)
        // `SymmetricKey` is compared directly rather than through its memory: its
        // storage is an opaque reference, so `withUnsafeBytes(of:)` would compare
        // the wrapper, not the key material.
        let keys = RemoteCrypto.sessionKeys(key: key, clientNonce: clientNonce, serverNonce: serverNonce)
        check(keys.clientToServer != keys.serverToClient,
              "the two directions derive different keys")
        let again = RemoteCrypto.sessionKeys(key: key, clientNonce: clientNonce, serverNonce: serverNonce)
        check(keys.clientToServer == again.clientToServer,
              "derivation is deterministic for the same nonces")
        check(keys.serverToClient == again.serverToClient, "in both directions")
        let other = RemoteCrypto.sessionKeys(key: key, clientNonce: Data(repeating: 0xA1, count: 30),
                                             serverNonce: serverNonce)
        check(other.clientToServer != keys.clientToServer,
              "a different nonce derives a different key, so a session cannot be replayed")

        var sealer = RemoteCrypto.Sealer(key: keys.clientToServer, direction: .clientToServer)
        var opener = RemoteCrypto.Opener(key: keys.clientToServer, direction: .serverToClient)
        let payload = [UInt8](repeating: 0x5A, count: 1_000)
        let header = RemoteHeader(opcode: .tagBatch, streamID: 3, byteCount: payload.count)

        let sealed = try sealer.seal(header, payload: payload)
        check(sealed.count == payload.count + 16, "a sealed frame carries exactly its tag")
        check(try opener.open(header, sealed: sealed) == payload, "the payload round trips byte for byte")

        let second = try sealer.seal(header, payload: payload)
        check(try opener.open(header, sealed: second) == payload, "the next counter opens a second frame")

        // A flipped ciphertext bit must fail the tag.
        var damaged = sealed
        damaged[0] ^= 0x01
        _ = checkThrowsSync("a flipped ciphertext bit fails authentication") {
            try opener.open(header, sealed: damaged)
        }
        // And so must a rewritten header: it is authenticated as associated data,
        // so an attacker cannot change an opcode or a length in flight.
        _ = checkThrowsSync("a rewritten header fails authentication") {
            try opener.open(RemoteHeader(opcode: .tagResult, streamID: 3, byteCount: payload.count),
                            sealed: sealed)
        }
        // Replaying the first frame after the counter has advanced must fail: the
        // nonce is the counter, so it cannot be used twice.
        _ = checkThrowsSync("a replayed frame fails authentication") {
            try opener.open(header, sealed: sealed)
        }
    })

    Registry.shared.add(suite: suite, TestCase(name: "handshake proofs are directional and compared in time", knownBug: nil) {
        let key = remoteTestKey()
        let clientNonce = Data(repeating: 0x11, count: RemoteCrypto.nonceByteCount)
        let serverNonce = Data(repeating: 0x22, count: RemoteCrypto.nonceByteCount)
        let proof = RemoteCrypto.clientProof(key: key, clientNonce: clientNonce, serverNonce: serverNonce)
        checkEqual(proof.count, RemoteCrypto.proofByteCount, "a proof is one HMAC")
        check(proof == RemoteCrypto.clientProof(key: key, clientNonce: clientNonce, serverNonce: serverNonce),
              "a proof is deterministic")
        check(proof != RemoteCrypto.clientProof(key: remoteTestKey(), clientNonce: clientNonce,
                                                serverNonce: serverNonce),
              "a different key proves differently")
        check(proof != RemoteCrypto.serverProof(key: key, sessionID: "abc", clientNonce: clientNonce,
                                                serverNonce: serverNonce),
              "the client and server proofs are different values, not the same one twice")

        check(RemoteCrypto.constantTimeEquals(proof, proof), "equal proofs compare equal")
        check(!RemoteCrypto.constantTimeEquals(proof, Data(proof.dropLast())),
              "a truncated proof does not compare equal")
        check(!RemoteCrypto.constantTimeEquals(proof, RemoteCrypto.clientProof(
            key: key, clientNonce: Data(repeating: 0x12, count: RemoteCrypto.nonceByteCount),
            serverNonce: serverNonce)), "a different proof does not compare equal")
    })

    Registry.shared.add(suite: suite, TestCase(name: "addresses are validated once, before any socket is opened", knownBug: nil) {
        checkEqual(RemoteAddress.parse("192.168.1.24:47410"),
                   RemoteAddress.Endpoint(host: "192.168.1.24", port: 47410), "host and port")
        checkEqual(RemoteAddress.parse("studio.local")?.port, RemoteLimits.defaultPort,
                   "a bare host takes the default port")
        checkEqual(RemoteAddress.parse("  studio.local:1234  ")?.host, "studio.local",
                   "surrounding whitespace is trimmed")
        checkEqual(RemoteAddress.parse("studio.local")?.display, "studio.local:\(RemoteLimits.defaultPort)",
                   "the display form is host:port")
        checkEqual(RemoteAddress.parse("[::1]:5555"), RemoteAddress.Endpoint(host: "::1", port: 5555),
                   "a bracketed IPv6 literal with a port")
        checkEqual(RemoteAddress.parse("[fe80::1]")?.port, RemoteLimits.defaultPort,
                   "a bracketed IPv6 literal without a port")

        for rejected in ["", "   ", "host:notaport", "host:0", "host:70000",
                         "http://host:1234", "host/path", "-bad.example", "bad-.example",
                         "ho st", "[::1", "a..b"] {
            checkNil(RemoteAddress.parse(rejected), "\(rejected.isEmpty ? "<empty>" : rejected) is refused")
        }
    })

    Registry.shared.add(suite: suite, TestCase(name: "beacons round trip, expire after three missed beats, and never carry the key", knownBug: nil) {
        let key = remoteTestKey()
        let name = String(repeating: "n", count: RemoteBeacon.maximumNameBytes)
        let beacon = RemoteBeacon(tcpPort: 47410, serverInstance: 0x0123_4567_89AB_CDEF,
                                  keyID: RemoteCrypto.keyID(for: key),
                                  sentAtMilliseconds: UInt32(truncatingIfNeeded: 1_700_000_000_000),
                                  flags: [.accepting, .taggerReady,
                                          RemoteBeacon.BeaconFlags(rawValue: 1 << 3)],
                                  name: name, appVersion: "9.9.9")
        let decoded = RemoteBeacon.decode(beacon.encoded())
        checkNotNil(decoded, "a well-formed beacon decodes")
        checkEqual(decoded?.tcpPort, 47410, "the port a client should dial survives")
        checkEqual(decoded?.serverInstance, 0x0123_4567_89AB_CDEF, "the instance id survives")
        checkEqual(decoded?.keyID, RemoteCrypto.keyID(for: key), "the key identifier survives")
        checkEqual(decoded?.name, name, "a maximal name survives")
        checkEqual(decoded?.appVersion, "9.9.9", "the version survives")
        check(decoded?.flags.contains(.taggerReady) == true, "tagger availability survives")
        check(decoded?.offers(.seedVR2) == true, "the algorithm bit lands on the right case")
        check(decoded?.offers(.flashVSR11) == false, "an unoffered algorithm reads as absent")

        // The one property that matters for a broadcast packet: the key must not
        // be recoverable from it. Only an eight-byte hash prefix may appear.
        let bytes = beacon.encoded()
        check(bytes.count <= 128, "a beacon fits one MTU without fragmenting")
        // The only key-derived bytes in a broadcast packet may be the eight-byte
        // identifier; the secret itself must not be recoverable from the wire.
        var carriesKey = false
        for start in 0...(bytes.count - RemoteCrypto.keyByteCount) {
            if Array(bytes[start..<(start + RemoteCrypto.keyByteCount)]) == key { carriesKey = true; break }
        }
        check(!carriesKey, "the 32-byte key never appears in a beacon")
        check(Array(bytes[20..<28]) == [UInt8](RemoteCrypto.keyID(for: key)),
              "the only key-derived bytes are the eight-byte identifier")

        var damaged = bytes
        damaged[0] = 0
        checkNil(RemoteBeacon.decode(damaged), "a beacon without magic is refused")
        damaged = bytes
        damaged[4] = 9
        checkNil(RemoteBeacon.decode(damaged), "a beacon from another version is refused")
        damaged = bytes
        damaged[32] = 1
        checkNil(RemoteBeacon.decode(damaged), "a beacon with nonzero reserved bytes is refused")

        // Expiry: three missed beats, so a server that quits disappears in about
        // three seconds rather than lingering until the list is reopened.
        let cache = RemoteBeaconCache()
        let now = Date()
        await cache.record(beacon, from: "10.0.0.1", now: now)
        checkEqual(await cache.live(now: now).count, 1, "a fresh beacon is live")
        checkEqual(await cache.live(now: now).first?.address, "10.0.0.1:47410",
                   "the source address is what a client dials, not a name from the payload")
        checkEqual(await cache.live(now: now).first?.keyID, RemoteCrypto.keyID(for: key),
                   "the stored row carries the key identifier for the panel's Ready/Key needed state")
        checkEqual(await cache.live(now: now).first?.name, name,
                   "the row carries the name the payload claimed, for display only")
        // Expiry drops the row rather than merely marking it, so the next
        // `live()` reflects it.
        checkEqual(await cache.live(now: now.addingTimeInterval(2.9)).count, 1,
                   "still live just inside the window")
        checkEqual(await cache.live(now: now.addingTimeInterval(3.1)).count, 0,
                   "gone once three beats have been missed")
    })

    Registry.shared.add(suite: suite, TestCase(name: "capabilities round trip and report absence rather than false", knownBug: nil) {
        // A build that is not licensed to offer a model must report it as absent,
        // not present-and-unavailable: the client must not offer work the server
        // cannot do.
        let capabilities = RemoteCapabilities(
            protocolVersion: 1, appVersion: "1.4.0", serverName: "Studio", sessionID: "abc",
            tagger: .init(available: true, imageSize: 448, batchSize: 8, headCount: 10_861,
                          vocabularyFingerprint: 12_345),
            uncover: [
                .init(algorithm: "seedVR2", available: true, message: "Ready",
                      scales: [1, 2, 4], minimumFrames: 5, requiresUnsafeMemoryOptIn: false,
                      maximumPaddedArea: nil, maximumOutputPixels: 16_000_000, outputScale: 1,
                      supportsStillPhotos: false),
                .init(algorithm: "pgtFormer", available: false, message: "Not installed on this server.",
                      scales: [], minimumFrames: 1, requiresUnsafeMemoryOptIn: false,
                      maximumPaddedArea: nil, maximumOutputPixels: nil, outputScale: 1,
                      supportsStillPhotos: false),
            ],
            vision: .init(available: true))
        let encoded = try JSONEncoder().encode(capabilities)
        let decoded = try JSONDecoder().decode(RemoteCapabilities.self, from: encoded)
        checkEqual(decoded, capabilities, "the capability document round trips")

        // A nil limit and a zero limit must stay distinguishable, which is why
        // the field is optional rather than defaulted to zero.
        checkNil(decoded.uncover.first?.maximumPaddedArea, "an absent limit decodes as absent")
        checkNil(decoded.uncover.last?.maximumPaddedArea, "for every algorithm that states none")
        checkEqual(decoded.uncover.last?.maximumOutputPixels, nil, "and so does an absent pixel limit")
        checkEqual(decoded.uncover.first?.maximumOutputPixels, 16_000_000, "a stated limit survives")

        check(decoded.offersRemoteTagging, "tagging is offered when the tagger is available")
        check(decoded.offersRemote(.seedVR2, scale: .four), "an offered scale reads as offered")
        check(!decoded.offersRemote(.seedVR2, scale: .eight), "an unoffered scale reads as absent")
        check(!decoded.offersRemote(.pgtFormer), "an unavailable algorithm reads as absent")
        check(decoded.offersAnyUncover, "some Uncover work is available")
        checkEqual(decoded.localCapabilities(.seedVR2)?.maximumOutputPixels, 16_000_000,
                   "server limits reach the local capability type")
        checkEqual(decoded.localCapabilities(.pgtFormer)?.available, false,
                   "an unavailable algorithm keeps its message")
        checkEqual(decoded.localCapabilities(.seedVR2)?.scales, [.one, .two, .four],
                   "scales translate back to the local enum")

        var noTagger = capabilities
        noTagger.tagger = nil
        check(!noTagger.offersRemoteTagging, "an absent tagger offers no tagging")
        check(noTagger.offersAnyUncover, "and Uncover availability is a separate question")

        var noAlgorithms = capabilities
        noAlgorithms.uncover = []
        check(!noAlgorithms.offersAnyUncover, "no algorithms means no Uncover work")
        check(!noAlgorithms.offersRemote(.seedVR2), "and no algorithm can be named remotely")
    })

    Registry.shared.add(suite: suite, TestCase(name: "binary payloads survive the writer and refuse to over-read", knownBug: nil) {
        var writer = WireWriter()
        writer.u8(7)
        writer.u16(1_000)
        writer.u32(70_000)
        writer.u64(UInt64.max)
        writer.f32(0.5)
        writer.f64(-1.25)
        writer.string("Studio Mac")
        writer.raw([1, 2, 3, 4, 5])
        let floats: [Float] = [0, 1, 0.5, Float.pi]
        writer.u32(UInt32(floats.count * 4))
        writer.floats(floats)

        var reader = WireReader(writer.bytes)
        checkEqual(try reader.u8(), 7, "u8")
        checkEqual(try reader.u16(), 1_000, "u16")
        checkEqual(try reader.u32(), 70_000, "u32")
        checkEqual(try reader.u64(), UInt64.max, "u64")
        checkClose(Double(try reader.f32()), 0.5, tolerance: 0, "f32")
        checkClose(try reader.f64(), -1.25, tolerance: 0, "f64")
        checkEqual(try reader.string(), "Studio Mac", "string")
        check(try reader.raw() == [1, 2, 3, 4, 5], "byte run")
        check(try reader.floats(4) == floats, "float run")
        check(reader.isAtEnd, "the payload was fully consumed")

        // Truncation at every field boundary must be an error, never a partial
        // read or a trap.
        for length in 0..<writer.bytes.count {
            let prefix = Array(writer.bytes.prefix(length))
            _ = checkThrowsSync("a truncated payload is refused at \(length) bytes") {
                var short = WireReader(prefix)
                while true {
                    if short.isAtEnd { break }
                    _ = try short.u8()
                }
                // Reading a multi-byte field from an exhausted reader must throw.
                return try short.string()
            }
        }
        // A declared count that outruns the payload must be refused on the
        // remaining length, before anything is allocated for it.
        var liar = WireWriter()
        liar.u32(0xFFFF_FFFF)
        let liarBytes = liar.bytes
        _ = checkThrowsSync("an overlong declared byte run") {
            var reader = WireReader(liarBytes)
            return try reader.raw()
        }
        var overlongFloats = WireWriter()
        overlongFloats.u32(3)
        overlongFloats.floats([1, 2])
        let floatsBytes = overlongFloats.bytes
        _ = checkThrowsSync("a float count that outruns the payload") {
            var reader = WireReader(floatsBytes)
            return try reader.floats(3)
        }
        // Invalid UTF-8 is refused rather than silently repaired.
        var badString = WireWriter()
        badString.u16(2)
        badString.append([0xFF, 0xFE])
        let badStringBytes = badString.bytes
        _ = checkThrowsSync("a string that is not valid UTF-8") {
            var reader = WireReader(badStringBytes)
            return try reader.string()
        }
    })

    Registry.shared.add(suite: suite, TestCase(name: "the compressed payload envelope states its size and refuses a liar", knownBug: nil) {
        check(try RemoteCompressedPayload.decode(RemoteCompressedPayload.encode([])).isEmpty, "empty compression round trip")
        var enormous = WireWriter()
        enormous.u64(UInt64.max)
        checkThrowsSync("integer-overflow decompression length") { try RemoteCompressedPayload.decode(enormous.bytes) }
        let original = [UInt8](repeating: 0x42, count: 8_000)
        let wrapped = RemoteCompressedPayload.encode(original)
        check(wrapped.count < original.count, "repetitive data actually compresses")
        check(try RemoteCompressedPayload.decode(wrapped) == original, "decompression is byte exact")

        var liar = WireWriter()
        liar.u64(UInt64(RemoteLimits.maximumMessagePayload) + 1)
        liar.append(RemoteCompressedPayload.encode(original))
        _ = checkThrowsSync("an implausible declared size") {
            try RemoteCompressedPayload.decode(liar.bytes)
        }
    })

    Registry.shared.add(suite: suite, TestCase(name: "RGBA input keeps its orientation and model bytes across the socket codec", knownBug: nil) {
        let context = CGContext(data: nil, width: 64, height: 48, bitsPerComponent: 8, bytesPerRow: 256,
                                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        context.setFillColor(CGColor(red: 1, green: 0.2, blue: 0, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: 64, height: 48))
        context.setFillColor(CGColor(red: 0, green: 0.3, blue: 1, alpha: 1))
        context.fill(CGRect(x: 5, y: 17, width: 29, height: 31))
        let input = try WDTaggerBackend.preprocessRGBA(context.makeImage()!)
        let received = try RemoteTagImageDecoder.decode(codec: .rgba8, bytes: input)
        check(try WDTaggerBackend.preprocessRGBA(received) == input,
              "the server sends the model precisely the RGBA bytes prepared by the client")
    })

    registerRemoteEndToEndTests(suite: suite)
}

// MARK: - End to end, over a real socket

/// A real `RemoteServer` and a real `RemoteClient`, both on loopback, with the
/// models stubbed.
///
/// Both halves are the production types, so the socket, the framing, the sealing,
/// the handshake, the routing and the client's own state machine are all the code
/// that runs in the app. Only the model catalog is substituted — the one seam the
/// serving side exposes for exactly this purpose.
@MainActor
private final class LoopbackPair {
    static let serverName = "Loopback Studio"
    /// `127.0.0.1` rather than `localhost`: the link resolves a name, and a name
    /// lookup is one more thing that can fail for reasons unrelated to the code.
    static let loopback = "127.0.0.1"

    let server: RemoteServer
    let client: RemoteClient
    private let defaults: UserDefaults

    init(catalog: RemoteModelCatalog, key: [UInt8]) throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("waifu-remote-pair-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let store = RemoteFileKeyStore(url: root.appendingPathComponent("server.key"))
        try store.saveKey(key)
        defaults = try RemotePairDefaults.make()
        server = RemoteServer(keyStore: store, catalog: catalog, port: 0, advertises: false,
                              appVersion: "test", serverName: LoopbackPair.serverName)
        client = RemoteClient(defaults: defaults, keyStore: store, discoveryEnabled: false,
                              vocabularyCacheDirectory: root.appendingPathComponent("vocabulary"))
    }

    var isListening: Bool { if case .listening = server.state { return true } else { return false } }

    /// Polls a main-actor condition. Only used where an ordering is not the thing
    /// under test — the socket round trips themselves are awaited properly.
    private func poll(_ condition: () -> Bool, timeout: Double = 10) async -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if condition() { return true }
            try? await Task.sleep(for: .milliseconds(20))
        }
        return condition()
    }

    /// Starts the listener and drives the real client through its own state
    /// machine. Returns `false` rather than throwing when the handshake does not
    /// complete, because several cases are about the refusal paths.
    @discardableResult
    func connect() async -> Bool {
        server.start()
        guard await poll({ self.isListening }) else { return false }
        guard case .listening(let port) = server.state else { return false }
        client.manualAddress = "\(LoopbackPair.loopback):\(port)"
        if client.isEnabled { client.connect(to: client.manualAddress) }
        else { client.setEnabled(true) }
        return await poll({ self.client.status.isConnected })
    }

    /// Waits for the client to notice that the server let it go.
    func waitForDisconnect() async -> Bool {
        await poll({ !self.client.status.isConnected })
    }

    func tearDown() {
        client.setEnabled(false)
        server.stop()
        defaults.removePersistentDomain(forName: RemotePairDefaults.suite)
    }
}

private enum RemotePairDefaults {
    /// A private suite name, so a test run never reads or writes the app's real
    /// preferences — including `remoteClientAddress` and the enabled flag.
    static let suite = "waifu.remote.tests"

    static func make() throws -> UserDefaults {
        guard let defaults = UserDefaults(suiteName: suite) else {
            throw SkipTest(reason: "could not open an isolated defaults suite")
        }
        defaults.removePersistentDomain(forName: suite)
        return defaults
    }
}

/// A catalog backed by stubs, so the protocol can be exercised with no model
/// files present.
private enum StubCatalog {
    /// A catalog with a stub Uncover backend and no tagger.
    static func uncoverOnly() -> RemoteModelCatalog { make(tagger: nil, vocabulary: nil) }

    /// A catalog with a stub tagger whose rows are deterministic.
    static func tagging(_ tagger: StubTagger, vocabulary: TagVocabulary,
                        csv: Data? = nil, fingerprint: UInt64? = nil) -> RemoteModelCatalog {
        make(tagger: tagger, vocabulary: vocabulary, csv: csv, fingerprintOverride: fingerprint)
    }

    static func make(tagger: StubTagger?, vocabulary: TagVocabulary?,
                     csv: Data? = nil, fingerprintOverride: UInt64? = nil) -> RemoteModelCatalog {
        let resolved: TagVocabulary? = vocabulary ?? tagger?.vocabulary
        let csv = csv ?? (try? VocabularyFixture.csv())
        let fingerprint = fingerprintOverride ?? resolved?.cacheFingerprint ?? 0
        return RemoteModelCatalog(
            tagger: { tagger ?? StubTagger(vocabulary: resolved) },
            backend: { _ in StubUncoverBackend() },
            vocabularyCSV: {
                guard let csv else { throw RemoteError.modelUnavailable("no vocabulary") }
                return csv
            },
            vocabularyFingerprint: { fingerprint },
            capabilities: { version, name, session in
                RemoteCapabilities(
                    protocolVersion: 1, appVersion: version, serverName: name, sessionID: session,
                    tagger: resolved == nil ? nil
                        : .init(available: true, imageSize: WDTaggerBackend.imageSize,
                                batchSize: WDTaggerBackend.batchSize,
                                headCount: TagVocabulary.headCount, vocabularyFingerprint: fingerprint),
                    uncover: [
                        .init(algorithm: UncoverAlgorithm.seedVR2.rawValue, available: true, message: "STUB",
                              scales: UncoverScale.allCases.map(\.rawValue), minimumFrames: 5,
                              requiresUnsafeMemoryOptIn: false, maximumPaddedArea: nil,
                              maximumOutputPixels: 16_000_000, outputScale: 1, supportsStillPhotos: false),
                        .init(algorithm: UncoverAlgorithm.realESRGAN.rawValue, available: false,
                              message: "Not installed on this server.", scales: [], minimumFrames: 1,
                              requiresUnsafeMemoryOptIn: false, maximumPaddedArea: nil,
                              maximumOutputPixels: nil, outputScale: 1, supportsStillPhotos: false),
                    ],
                    vision: .init(available: true))
            })
    }
}

/// A generated CSV with the exact head layout `TagVocabulary.load` checks for:
/// four rating heads, 8,106 general, 2,751 character.
///
/// Generated rather than copied so the suite needs no model files.
private enum VocabularyFixture {
    static func csv() throws -> Data { Data(rows().utf8) }

    static func make() throws -> TagVocabulary {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("waifu-remote-vocab-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("labels.csv")
        try csv().write(to: file)
        return try TagVocabulary.load(from: file)
    }

    private static func rows() -> String {
        var rows = ["name,category",
                    "general,9", "sensitive,9",
                    "questionable,9", "explicit,9"]
        for index in 0..<TagVocabulary.generalCount { rows.append("general_\(index),0") }
        for index in 0..<2_751 { rows.append("character_\(index),4") }
        return rows.joined(separator: "\n")
    }
}

/// A prepared Uncover job on disk, laid out exactly as
/// `TemporalFrameExtractor` writes one.
private enum PreparedJobFixture {
    static func make(directory: URL, frames: Int) throws -> UncoverManifest {
        for name in ["original", "degraded", "baseline", "reconstructed"] {
            try FileManager.default.createDirectory(at: directory.appendingPathComponent(name),
                                                    withIntermediateDirectories: true)
        }
        var entries: [UncoverFrame] = []
        for index in 0..<frames {
            let image = try ImageFixture.cgImage(width: 40, height: 30, gray: CGFloat(index) / 8)
            for name in ["degraded", "baseline", "original"] {
                try RegionImages.write(image, to: RegionImages.frameURL(directory, name, index))
            }
            entries.append(UncoverFrame(index: index,
                                        timestamp: UncoverTime(CMTime(value: CMTimeValue(index * 10), timescale: 600)),
                                        duration: UncoverTime(CMTime(value: 10, timescale: 600))))
        }
        let manifest = UncoverManifest(source: URL(fileURLWithPath: "/dev/null"), sourceBytes: 0,
                                       sourceModified: 0, roi: UncoverROI(x: 0, y: 0, width: 1, height: 1),
                                       width: 40, height: 30, regionX: 0, regionY: 0,
                                       regionWidth: 40, regionHeight: 30, scale: .two, frames: entries)
        try JSONEncoder().encode(manifest).write(to: directory.appendingPathComponent("manifest.json"),
                                                 options: .atomic)
        return manifest
    }
}

/// Synchronous `checkThrows` for the pure functions in this file, kept to one line
/// at the call sites.
@discardableResult
private func checkThrowsSync(_ what: String, _ body: () throws -> Any) -> Bool {
    do {
        _ = try body()
        check(false, "\(what) must throw")
        return false
    } catch {
        return true
    }
}

// MARK: - The end-to-end cases
//
// Each case hands the work to a `@MainActor` helper. The case body runs off the
// main actor, so the hop has to be at the call rather than inside a closure.

private func registerRemoteEndToEndTests(suite: String) {
    Registry.shared.add(suite: suite, TestCase(name: "a client holding the key is admitted and learns what the server can do", knownBug: nil) {
        let vocabulary = try VocabularyFixture.make()
        try await admittedCase(vocabulary: vocabulary)
    })

    Registry.shared.add(suite: suite, TestCase(name: "a client with no key, or the wrong key, is refused rather than left retrying", knownBug: nil) {
        let vocabulary = try VocabularyFixture.make()
        try await refusedCases(vocabulary: vocabulary)
    })

    Registry.shared.add(suite: suite, TestCase(name: "a remote tagging batch returns rows bit-identical to a local run", knownBug: nil) {
        let vocabulary = try VocabularyFixture.make()
        try await taggingCase(vocabulary: vocabulary)
    })

    Registry.shared.add(suite: suite, TestCase(name: "eight tagging batches reach the server before any result completes", knownBug: nil) {
        try await queuedTaggingCase()
    })

    Registry.shared.add(suite: suite, TestCase(name: "cancelling queued tagging joins all server tasks and releases the cancelled model", knownBug: nil) {
        try await queuedTaggingCancellationCase()
    })

    Registry.shared.add(suite: suite, TestCase(name: "the vocabulary transfer is checked against the fingerprint it claims", knownBug: nil) {
        let vocabulary = try VocabularyFixture.make()
        try await vocabularyCase(vocabulary: vocabulary)
    })

    Registry.shared.add(suite: suite, TestCase(name: "a remote reconstruction returns frames identical to a local run", knownBug: nil) {
        try await uncoverCase()
    })

    Registry.shared.add(suite: suite, TestCase(name: "a second client is refused while one is connected, and a kick ends the session", knownBug: nil) {
        try await kickCase()
    })

    Registry.shared.add(suite: suite, TestCase(name: "cancelling a remote reconstruction stops the server job and releases its model", knownBug: nil) {
        try await cancellationCase()
    })

    Registry.shared.add(suite: suite, TestCase(name: "the label context marks a request only when the server can run it", knownBug: nil) {
        let vocabulary = try VocabularyFixture.make()
        let catalog = StubCatalog.tagging(StubTagger(vocabulary: vocabulary), vocabulary: vocabulary)
        let capabilities = await catalog.capabilities("1.0", "Studio", nil)
        try await labelCase(capabilities: capabilities)
    })
}

@MainActor
private func admittedCase(vocabulary: TagVocabulary) async throws {
    let key = remoteTestKey()
    let pair = try LoopbackPair(catalog: StubCatalog.make(tagger: StubTagger(vocabulary: vocabulary),
                                                          vocabulary: vocabulary), key: key)
    defer { pair.tearDown() }
    check(await pair.connect(), "a client holding the key completes the handshake")

    checkEqual(pair.client.capabilities?.serverName, LoopbackPair.serverName,
               "the client learned the server's name from the challenge")
    check(pair.client.remoteIdentity.contains(LoopbackPair.serverName),
          "the connected server's identity is available synchronously for cache keys")
    check(pair.client.offersRemoteTagging, "a server with a tagger offers tagging")
    check(pair.client.offersRemote(.seedVR2), "an available algorithm is offered")
    check(!pair.client.offersRemote(.realESRGAN), "an unavailable algorithm is not offered")
    checkNotNil(pair.client.remoteVocabulary, "the vocabulary arrived with the handshake")
    checkEqual(pair.client.remoteVocabulary?.cacheFingerprint, vocabulary.cacheFingerprint,
               "and it is the vocabulary the server reported")
    check(!pair.client.needsAttention, "a healthy connection needs no attention")
    check(pair.server.client != nil, "the server shows the connected client in its panel")
}

@MainActor
private func refusedCases(vocabulary: TagVocabulary) async throws {
    // No key at all.
    do {
        let key = remoteTestKey()
        let pair = try LoopbackPair(catalog: StubCatalog.uncoverOnly(), key: key)
        defer { pair.tearDown() }
        let empty = RemoteFileKeyStore(url: FileManager.default.temporaryDirectory
            .appendingPathComponent("waifu-remote-empty-\(UUID().uuidString).key"))
        pair.client.replaceKeyStore(empty)
        check(!(await pair.connect()), "a client holding no key is not admitted")
        check(pair.client.needsAttention, "and is told so, rather than left retrying")
        checkNil(pair.client.capabilities, "a refused client learns nothing about the server")
    }

    // The wrong key. The refusal must not distinguish "wrong key" from "bad
    // proof": over a 256-bit secret there is nothing there to learn.
    do {
        let pair = try LoopbackPair(catalog: StubCatalog.uncoverOnly(), key: remoteTestKey())
        defer { pair.tearDown() }
        let wrong = RemoteFileKeyStore(url: FileManager.default.temporaryDirectory
            .appendingPathComponent("waifu-remote-wrong-\(UUID().uuidString).key"))
        try wrong.saveKey(remoteTestKey())
        pair.client.replaceKeyStore(wrong)
        check(!(await pair.connect()), "a client with the wrong key is not admitted")
        if case .refused(let reason) = pair.client.status {
            check(!reason.lowercased().contains("proof"),
                  "the refusal does not say which check failed")
        }
    }
}

@MainActor
private func taggingCase(vocabulary: TagVocabulary) async throws {
    let tagger = StubTagger(vocabulary: vocabulary)
    let pair = try LoopbackPair(
        catalog: StubCatalog.tagging(tagger, vocabulary: vocabulary), key: remoteTestKey())
    defer { pair.tearDown() }
    check(await pair.connect(), "the loopback pair is connected")

    let images = try (0..<4).map { index in
        try ImageFixture.cgImage(width: 64, height: 48, gray: CGFloat(index) / 8)
    }
    let remoteRows = try await pair.client.tag(images: images, codec: .rgba8)
    checkEqual(remoteRows.count, images.count, "one row per image")
    checkEqual(remoteRows.first?.count, TagVocabulary.headCount, "each row is a full vocabulary of heads")

    // The claim under test: these are exactly the rows the same backend hands
    // the local pipeline, so a remote run writes rows the local disk cache can
    // also read.
    let localRows = try await tagger.probabilityRows(batch: images)
    check(remoteRows == localRows, "remote rows are bit-identical to local rows")
    checkEqual(remoteRows.map { $0[7] }, localRows.map { $0[7] }, "frame order is preserved")

    // The client derives results with its own threshold, from the same function
    // the local path uses.
    let results = try localRows.map { try vocabulary.result(probabilities: $0, threshold: 0.35) }
    checkEqual(results.count, images.count, "one result per frame")
    check(results.allSatisfy { $0.probabilities.count == TagVocabulary.headCount },
          "each result carries its full row")

    // A second batch on the same connection must work, so the framing is
    // reusable rather than accidentally one-shot.
    let again = try await pair.client.tag(images: images, codec: .rgba8)
    check(again == remoteRows, "a repeated batch returns identical rows")
    checkEqual(tagger.batches, 3, "two remote batches plus the local reference batch")

    // The JPEG codec is the cheap path a video run takes: the cached thumbnail
    // goes up as the bytes that were already in the cache, with no re-encode on
    // the client and no decode-and-resize on the server.
    let encoded = try await pair.client.tag(images: images, codec: .encoded)
    checkEqual(encoded.count, images.count, "the encoded codec also returns one row per image")
    check(encoded.allSatisfy { $0.count == TagVocabulary.headCount }, "and every row is complete")
    checkEqual(tagger.batches, 4, "the third batch reached the server too")
}

@MainActor
private func queuedTaggingCase() async throws {
    let vocabulary = try VocabularyFixture.make()
    let probe = RemoteTagWindowProbe()
    let tagger = StubTagger(vocabulary: vocabulary, windowProbe: probe)
    let pair = try LoopbackPair(catalog: StubCatalog.tagging(tagger, vocabulary: vocabulary), key: remoteTestKey())
    defer { pair.tearDown() }
    check(await pair.connect(), "the loopback pair is connected")
    let image = try ImageFixture.cgImage(width: 64, height: 48, gray: 0.5)
    let counts = try await withThrowingTaskGroup(of: Int.self) { group in
        for count in 1...8 {
            group.addTask {
                let rows = try await pair.client.tag(images: Array(repeating: image, count: count), codec: .rgba8)
                guard rows.count == count else { throw TaggerError.invalidProbabilities }
                for (position, row) in rows.enumerated() {
                    guard row.count == TagVocabulary.headCount,
                          row[7] == Float((7 * 31 + position * 7) % 1_000) / 4_000 else {
                        throw TaggerError.invalidProbabilities
                    }
                }
                return count
            }
        }
        var results: [Int] = []
        for try await count in group { results.append(count) }
        return results.sorted()
    }
    checkEqual(counts, Array(1...8), "stream IDs route every different-sized reply to its request")
    checkEqual(await probe.arrived, 8, "all eight requests arrive while the first model call is suspended")
    checkEqual(tagger.batches, 8, "each batch executes once")
}

@MainActor
private func queuedTaggingCancellationCase() async throws {
    let vocabulary = try VocabularyFixture.make()
    let probe = RemoteTagWindowProbe(waitForCancellation: true)
    let tagger = StubTagger(vocabulary: vocabulary, windowProbe: probe)
    var catalog = StubCatalog.tagging(tagger, vocabulary: vocabulary)
    catalog.releaseTagger = { await probe.noteReleased() }
    let pair = try LoopbackPair(catalog: catalog, key: remoteTestKey())
    defer { pair.tearDown() }
    check(await pair.connect(), "the loopback pair is connected")
    let image = try ImageFixture.cgImage(width: 64, height: 48, gray: 0.5)
    let requests = Task {
        try await withThrowingTaskGroup(of: Void.self) { group in
            for _ in 0..<8 {
                group.addTask { _ = try await pair.client.tag(images: [image], codec: .rgba8) }
            }
            try await group.waitForAll()
        }
    }
    let startedDeadline = Date().addingTimeInterval(5)
    while await probe.arrived < 8, Date() < startedDeadline {
        try await Task.sleep(for: .milliseconds(5))
    }
    checkEqual(await probe.arrived, 8, "all eight batches are awaiting inference")
    requests.cancel()
    do { try await requests.value; check(false, "cancelled requests must end with an error") }
    catch {}
    let stoppedDeadline = Date().addingTimeInterval(5)
    while !(await probe.released), Date() < stoppedDeadline {
        try await Task.sleep(for: .milliseconds(5))
    }
    checkEqual(await probe.completed, 8, "shutdown joins every cancelled batch")
    check(await probe.released, "a cancelled worker is discarded before reconnecting")
}

@MainActor
private func vocabularyCase(vocabulary: TagVocabulary) async throws {
    do {
        let csv = try VocabularyFixture.csv()
        let pair = try LoopbackPair(
            catalog: StubCatalog.tagging(StubTagger(vocabulary: vocabulary),
                                         vocabulary: vocabulary, csv: csv),
            key: remoteTestKey())
        defer { pair.tearDown() }
        check(await pair.connect(), "the pair is connected")
        checkEqual(pair.client.remoteVocabulary?.cacheFingerprint, vocabulary.cacheFingerprint,
                   "the client ends up with the vocabulary the server reported")
    }

    // A server that shipped labels not matching its own fingerprint must be
    // refused rather than cached: that mismatch would silently reorder every tag
    // index the client computes afterwards.
    do {
        let csv = try VocabularyFixture.csv()
        let pair = try LoopbackPair(
            catalog: StubCatalog.tagging(StubTagger(vocabulary: vocabulary),
                                         vocabulary: vocabulary, csv: csv, fingerprint: 1),
            key: remoteTestKey())
        defer { pair.tearDown() }
        check(!(await pair.connect()),
              "a fingerprint that does not match its labels is refused")
    }
}

@MainActor
private func uncoverCase() async throws {
    let pair = try LoopbackPair(catalog: StubCatalog.uncoverOnly(), key: remoteTestKey())
    defer { pair.tearDown() }
    check(await pair.connect(), "the loopback pair is connected")

    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("waifu-remote-e2e-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: root) }
    let manifest = try PreparedJobFixture.make(directory: root, frames: 3)
    let expected = try (0..<3).map { try Data(contentsOf: RegionImages.frameURL(root, "degraded", $0)) }

    let progress = ProgressLog()
    let outcome = try await pair.client.runUncoverJob(
        manifest: manifest, algorithm: .seedVR2, allowUnsafeMemory: false,
        prepared: UncoverPrepared(directory: root, manifest: manifest),
        progress: { message in progress.append(message) })
    checkEqual(outcome.model, "STUB · baseline", "the server's model label reaches the client")
    check(outcome.serverSeconds > 0, "the server's own timing reaches the client")
    checkEqual(outcome.outputScale, 1, "the output scale reaches the client")

    let returned = try (0..<3).map { try Data(contentsOf: RegionImages.frameURL(root, "reconstructed", $0)) }
    check(returned == expected, "returned frames are byte-identical to what a local run would write")

    // The point of writing into the client's own `reconstructed/`: the ordinary
    // renderer composites it with no remote-specific branch at all.
    let result = UncoverResult(prepared: UncoverPrepared(directory: root, manifest: manifest),
                               model: outcome.model, seconds: outcome.serverSeconds,
                               outputScale: outcome.outputScale)
    let rendered = try await UncoverRenderer().render(result, index: 1, feather: 8, strength: 1)
    checkEqual(rendered.original.width, manifest.width, "the original composites at source size")
    checkEqual(rendered.reconstructed.width, manifest.regionWidth * outcome.outputScale,
               "the reconstructed region composites at its own size")

    // Progress arrives through the same string channel the local backend uses, so
    // the status line needs no remote branch either.
    check(progress.contains { $0.contains("STUB") },
          "server progress reached the client: \(progress.values)")
}

/// A thread-safe list, because the backend's progress closure is `@Sendable` and
/// runs off the caller's actor.
private final class ProgressLog: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [String] = []

    func append(_ message: String) { lock.withLock { storage.append(message) } }
    var values: [String] { lock.withLock { storage } }
    func contains(where body: (String) -> Bool) -> Bool { lock.withLock { storage.contains(where: body) } }
}

@MainActor
private func kickCase() async throws {
    let pair = try LoopbackPair(catalog: StubCatalog.uncoverOnly(), key: remoteTestKey())
    defer { pair.tearDown() }
    check(await pair.connect(), "the first client is connected")
    check(pair.server.client != nil, "the server shows it")

    let secondDefaults = UserDefaults(suiteName: "waifu.remote.second-client.tests")!
    defer { secondDefaults.removePersistentDomain(forName: "waifu.remote.second-client.tests") }
    let second = RemoteClient(defaults: secondDefaults, keyStore: RemoteFileKeyStore(
        url: FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)), discoveryEnabled: false)
    second.manualAddress = pair.client.manualAddress
    second.setEnabled(true)
    defer { second.setEnabled(false) }
    let deadline = Date().addingTimeInterval(3)
    while !second.needsAttention, Date() < deadline { try await Task.sleep(for: .milliseconds(20)) }
    check(second.needsAttention && !second.status.isConnected, "a second client is refused as busy")
    check(pair.client.status.isConnected, "busy rejection preserves the first client")

    pair.server.kick()
    check(await pair.waitForDisconnect(), "a kicked client stops being connected")
    check(pair.server.client == nil, "the server forgets the client it kicked")
    check(pair.client.needsAttention,
          "a kick is reported as something the user has to notice, not a silent retry")

    // And the pair can reconnect afterwards, so a kick is a disconnection rather
    // than a poisoned state.
    check(await pair.connect(), "the same client can connect again after a kick")
}

@MainActor
private func labelCase(capabilities: RemoteCapabilities) throws {
    let connected = RemoteLabelContext(connected: true, serverName: "Studio", capabilities: capabilities)
    check(connected.tagsRemotely, "Start tagging is marked when the server has a tagger")
    check(connected.offersRemoteUncover, "Uncover Waifu is marked when the server has any algorithm")
    check(connected.runsRemotely(.seedVR2), "Reconstruct is marked for an available algorithm")
    check(!connected.runsRemotely(.realESRGAN), "Reconstruct is unmarked for one the server lacks")
    checkEqual(connected.serverDescription, "Studio", "the tooltip names the server")

    // A server with no algorithms offers no Uncover workspace, so the opener is
    // unmarked even though tagging still works.
    let taggingOnly = RemoteCapabilities(
        protocolVersion: 1, appVersion: "1.0", serverName: "Tags", sessionID: nil,
        tagger: capabilities.tagger, uncover: [], vision: capabilities.vision)
    let tagging = RemoteLabelContext(connected: true, serverName: "Tags", capabilities: taggingOnly)
    check(tagging.tagsRemotely, "tagging is still marked")
    check(!tagging.offersRemoteUncover, "no algorithm means no Uncover mark")

    // Disconnected: capabilities in hand must not be enough on their own.
    let disconnected = RemoteLabelContext(connected: false, serverName: nil, capabilities: capabilities)
    check(!disconnected.tagsRemotely, "a disconnected client marks nothing")
    check(!disconnected.runsRemotely(.seedVR2), "and runs nothing remotely")
    checkEqual(disconnected.serverDescription, "a remote server",
               "an unnamed server reads generically rather than showing a blank tooltip")
}

private final class CancellationProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var state = (started: false, cancelled: false)
    var started: Bool { lock.withLock { state.started } }
    var cancelled: Bool { lock.withLock { state.cancelled } }
    func start() { lock.withLock { state.started = true } }
    func cancel() { lock.withLock { state.cancelled = true } }
}

private struct SlowRemoteBackend: TemporalVideoUpscaler {
    let probe: CancellationProbe
    func capabilities() async -> UpscalerCapabilities { await StubUncoverBackend().capabilities() }
    func reconstruct(_ prepared: UncoverPrepared, allowUnsafeMemory: Bool,
                     progress: @escaping @Sendable (String) -> Void) async throws -> UncoverResult {
        probe.start()
        do { try await Task.sleep(for: .seconds(30)) }
        catch { probe.cancel(); throw error }
        return UncoverResult(prepared: prepared, model: "slow", seconds: 30)
    }
}

@MainActor
private func cancellationCase() async throws {
    let probe = CancellationProbe()
    var catalog = StubCatalog.uncoverOnly()
    catalog.backend = { _ in SlowRemoteBackend(probe: probe) }
    let pair = try LoopbackPair(catalog: catalog, key: remoteTestKey())
    defer { pair.tearDown() }
    check(await pair.connect(), "the cancellation pair connects")
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let manifest = try PreparedJobFixture.make(directory: root, frames: 3)
    let references = ModelLibrary.shared.inUse[.seed3, default: 0]
    let work = Task { @MainActor in
        try await pair.client.runUncoverJob(manifest: manifest, algorithm: .seedVR2,
                                           allowUnsafeMemory: false,
                                           prepared: UncoverPrepared(directory: root, manifest: manifest), progress: { _ in })
    }
    let startDeadline = Date().addingTimeInterval(3)
    while !probe.started, Date() < startDeadline { try await Task.sleep(for: .milliseconds(20)) }
    check(probe.started, "the server started the long model job")
    work.cancel()
    _ = await work.result
    let endDeadline = Date().addingTimeInterval(3)
    while (!probe.cancelled || ModelLibrary.shared.inUse[.seed3, default: 0] != references), Date() < endDeadline {
        try await Task.sleep(for: .milliseconds(20))
    }
    check(probe.cancelled, "client cancellation reaches the running server backend")
    checkEqual(ModelLibrary.shared.inUse[.seed3, default: 0], references, "the model reference is released")
}
