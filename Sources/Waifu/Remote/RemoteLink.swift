import CryptoKit
import Foundation
import Network

/// One reassembled message: a whole request or response, never a fragment.
struct RemoteEnvelope: Sendable {
    var opcode: RemoteOpcode
    var streamID: UInt32
    var payload: [UInt8]
    var flags: RemoteFlags

    var data: Data { Data(payload) }
    func control() throws -> RemoteControl { try RemoteControl.decode(payload) }
    var isError: Bool { flags.contains(.isError) }

    init(opcode: RemoteOpcode, streamID: UInt32, payload: [UInt8], flags: RemoteFlags = [.isFinal]) {
        self.opcode = opcode
        self.streamID = streamID
        self.payload = payload
        self.flags = flags
    }
}

/// A payload wrapped for the `.zlib` flag: an exact uncompressed length,
/// then the compressed bytes.
///
/// The length is stated rather than discovered because
/// `compression_decode_buffer` needs a destination size up front, and guessing
/// one is how a decompression bomb turns into an allocation.
enum RemoteCompressedPayload {
    static func encode(_ bytes: [UInt8]) -> [UInt8] {
        var writer = WireWriter(capacity: bytes.count + 8)
        writer.u64(UInt64(bytes.count))
        writer.append(RemoteCrypto.compress(bytes))
        return writer.bytes
    }

    static func decode(_ bytes: [UInt8]) throws -> [UInt8] {
        var reader = WireReader(bytes)
        let declared = try reader.u64()
        guard declared <= UInt64(RemoteLimits.maximumMessagePayload) else { throw RemoteError.tooLarge }
        let expected = Int(declared)
        guard expected <= RemoteLimits.maximumMessagePayload else {
            throw RemoteError.framing("declared decompressed size \(expected) is implausible")
        }
        return try RemoteCrypto.decompress(reader.remainingBytes(), expected: expected)
    }
}

/// One framed, sealed connection.
///
/// Both ends use this. The read loop pulls bytes, decodes 24-byte headers,
/// reassembles messages and opens them; the send side splits a payload into
/// frames and seals each one. Everything above it deals in whole
/// `(opcode, streamID, payload)` messages and never sees a socket.
///
/// `NWConnection` is not `Sendable`, so this is an actor: it owns the connection
/// and every mutable byte of framing state, so no lock is needed anywhere below.
actor RemoteLink {
    enum State: Sendable {
        case connecting
        case open
        case closed(String)
    }

    private let connection: NWConnection
    private let socketQueue = DispatchQueue(label: "waifu.remote.link.socket")
    private var assembler = RemoteMessageAssembler()
    private var inbox: [RemoteEnvelope] = []
    private var inboxBytes = 0
    private var readWaiters: [CheckedContinuation<RemoteEnvelope?, Never>] = []
    private var state: State = .connecting
    private var openWaiters: [CheckedContinuation<Void, Error>] = []
    private var sealer: RemoteCrypto.Sealer?
    private var opener: RemoteCrypto.Opener?
    /// Socket bytes read but not yet consumed by a whole frame.
    private var buffer: [UInt8] = []
    private var lastReceived = Date()
    var isIdle: Bool { Date().timeIntervalSince(lastReceived) > RemoteLimits.idleTimeout }
    private var started = false
    private var parsingPaused = false

    init(endpoint: RemoteAddress.Endpoint) {
        let tcp = NWProtocolTCP.Options()
        // Inference is many small request/response exchanges. Nagle would hold a
        // reply behind an unacknowledged control frame for far longer than the
        // round trip it is waiting for.
        tcp.noDelay = true
        tcp.enableKeepalive = true
        tcp.keepaliveIdle = Int(RemoteLimits.idleTimeout / 2)
        let parameters = NWParameters(tls: nil, tcp: tcp)
        parameters.allowLocalEndpointReuse = true
        connection = NWConnection(to: endpoint.networkEndpoint(), using: parameters)
    }

    init(accepting connection: NWConnection) { self.connection = connection }

    /// Installs the session keys once the handshake has completed, so the
    /// handshake itself travels in the clear and an unauthenticated peer never
    /// sees a single key-derived byte.
    func seal(with keys: (clientToServer: SymmetricKey, serverToClient: SymmetricKey), role: RemoteDirection) {
        sealer = RemoteCrypto.Sealer(
            key: role == .clientToServer ? keys.clientToServer : keys.serverToClient, direction: role)
        opener = RemoteCrypto.Opener(
            key: role == .clientToServer ? keys.serverToClient : keys.clientToServer, direction: role)
        parsingPaused = false
        ingest([])
    }

    /// Install the receive cipher before the server can answer the clear auth.
    func authenticateClient(proof: [UInt8], keys: (clientToServer: SymmetricKey, serverToClient: SymmetricKey)) async throws {
        seal(with: keys, role: .clientToServer)
        try await sendRaw([RemoteFrame(opcode: .auth, payload: proof)], unsealed: true)
    }

    var currentState: State { state }
    /// Why the link ended, for a status line. `nil` while it is healthy.
    var disconnectReason: String? { if case .closed(let reason) = state { return reason } else { return nil } }

    func start() {
        guard !started else { return }
        started = true
        connection.stateUpdateHandler = { [weak self] update in
            guard let self else { return }
            switch update {
            case .ready:
                Task { await self.markOpen() }
                self.readLoop()
            case .failed(let error):
                Task { await self.terminate(Self.describe(error)) }
            case .cancelled:
                Task { await self.terminate("closed") }
            default:
                break
            }
        }
        connection.start(queue: socketQueue)
    }

    /// Ends the link and wakes everything waiting on it.
    func close(reason: String) {
        connection.cancel()
        terminate(reason)
    }

    private static func describe(_ error: NWError) -> String {
        if case .posix(let code) = error {
            switch code {
            case .ECONNREFUSED: return "refused"
            case .ETIMEDOUT: return "timed out"
            case .ENETDOWN, .ENETUNREACH, .EHOSTUNREACH: return "unreachable"
            default: break
            }
        }
        return error.localizedDescription
    }

    private func markOpen() {
        if case .connecting = state { state = .open }
        let waiting = openWaiters
        openWaiters.removeAll()
        for waiter in waiting { waiter.resume() }
    }

    private func terminate(_ reason: String) {
        if case .closed = state { return }
        state = .closed(reason)
        while !readWaiters.isEmpty { readWaiters.removeFirst().resume(returning: nil) }
        let waiting = openWaiters
        openWaiters.removeAll()
        for waiter in waiting { waiter.resume(throwing: RemoteError.framing(reason)) }
    }

    /// The peer address on the live socket, for a status line.
    func remoteEndpointDescription() -> String {
        switch connection.currentPath?.remoteEndpoint {
        case .hostPort(let host, let port): "\(host.debugDescription):\(port.rawValue)"
        default: "connected"
        }
    }

    /// Suspends until the link is open, so the handshake never writes to a
    /// socket that is not yet usable.
    func waitUntilOpen() async throws {
        if case .open = state { return }
        if case .closed(let reason) = state { throw RemoteError.framing(reason) }
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            switch state {
            case .open: continuation.resume()
            case .closed(let reason): continuation.resume(throwing: RemoteError.framing(reason))
            case .connecting: openWaiters.append(continuation)
            }
        }
    }

    // MARK: Sending

    /// Sends one message, split into frames and sealed per frame.
    ///
    /// Returns once the socket has accepted every frame. That is a backpressure
    /// signal, not a delivery acknowledgement.
    func send(_ opcode: RemoteOpcode, streamID: UInt32 = 0, payload: [UInt8],
              extraFlags: RemoteFlags = []) async throws {
        guard payload.count <= RemoteLimits.maximumMessagePayload else { throw RemoteError.tooLarge }
        try await sendRaw(RemoteMessage.frames(opcode: opcode, streamID: streamID,
                                               payload: payload, extraFlags: extraFlags))
    }

    /// Sends already-framed bytes, for the one caller that needs per-frame flag
    /// control (an error message, where every frame must carry the flag because
    /// the reader checks each frame it takes).
    func sendRaw(_ frames: [RemoteFrame], unsealed: Bool = false) async throws {
        guard !frames.isEmpty else { return }
        // Seal and submit every frame before suspending. Actor reentrancy must
        // never interleave fragments of two messages or reorder nonce counters.
        var output = Data()
        for var frame in frames {
            if !unsealed, var active = sealer {
                frame.payload = try active.seal(frame.header, payload: frame.payload)
                sealer = active
            }
            output.append(contentsOf: frame.encoded())
        }
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            connection.send(content: output, completion: .contentProcessed { error in
                if let error { continuation.resume(throwing: error) }
                else { continuation.resume() }
            })
        }
    }

    /// Sends a control payload as JSON, zlib-wrapped when that is actually
    /// smaller. Control payloads are small text, so this is the only place the
    /// compression flag is used for anything other than the vocabulary.
    func send(_ opcode: RemoteOpcode, control: RemoteControl) async throws {
        let encoded = try RemoteControl.encode(control)
        let compressed = RemoteCompressedPayload.encode(encoded)
        try await send(opcode, payload: encoded.count <= compressed.count ? encoded : compressed,
                       extraFlags: encoded.count <= compressed.count ? [] : [.zlib])
    }

    // MARK: Receiving

    /// Waits for the next message. Returns `nil` once the link has ended, which
    /// is how the caller's read loop terminates rather than by throwing.
    func next() async -> RemoteEnvelope? {
        if !inbox.isEmpty {
            let envelope = inbox.removeFirst()
            inboxBytes -= envelope.payload.count
            return envelope
        }
        if case .closed = state { return nil }
        return await withCheckedContinuation { continuation in
            if !inbox.isEmpty { continuation.resume(returning: inbox.removeFirst()) }
            else if case .closed = state { continuation.resume(returning: nil) }
            else { readWaiters.append(continuation) }
        }
    }

    private func deliver(_ envelope: RemoteEnvelope) throws {
        if !readWaiters.isEmpty { readWaiters.removeFirst().resume(returning: envelope) }
        else {
            guard inbox.count < 1024, inboxBytes + envelope.payload.count <= RemoteLimits.maximumMessagePayload else {
                throw RemoteError.tooLarge
            }
            inbox.append(envelope)
            inboxBytes += envelope.payload.count
        }
    }

    private nonisolated func readLoop() {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 512 * 1024) { [weak self] data, _, complete, error in
            guard let self else { return }
            Task {
                if let data, !data.isEmpty { await self.ingest([UInt8](data)) }
                if error != nil || complete {
                    await self.terminate(error.map { Self.describe($0) } ?? "closed")
                } else { self.readLoop() }
            }
        }
    }

    /// Feeds socket bytes through the framer.
    ///
    /// Serialised by the actor, so the header/payload state machine cannot be
    /// re-entered by a second read completing while the first is mid-frame.
    private func ingest(_ bytes: [UInt8]) {
        if case .closed = state { return }
        if !bytes.isEmpty { lastReceived = Date() }
        buffer.append(contentsOf: bytes)
        guard !parsingPaused else { return }
        do {
            while true {
                guard buffer.count >= RemoteLimits.headerBytes else { return }
                let header = try RemoteHeader.decode(Array(buffer[0..<RemoteLimits.headerBytes]))
                // The sealed payload always carries a 16-byte Poly1305 tag.
                let frameLength = RemoteLimits.headerBytes + header.payloadByteCount + (opener == nil ? 0 : 16)
                guard buffer.count >= frameLength else { return }
                let body = Array(buffer[RemoteLimits.headerBytes..<frameLength])
                buffer.removeFirst(frameLength)
                let plaintext: [UInt8]
                if var opener {
                    plaintext = try opener.open(header, sealed: body)
                    self.opener = opener
                } else {
                    plaintext = body
                }
                if let message = try assembler.append(RemoteFrame(header: header, payload: plaintext)) {
                    let payload = header.flags.contains(.zlib)
                        ? try RemoteCompressedPayload.decode(message.2) : message.2
                    try deliver(RemoteEnvelope(opcode: message.0, streamID: message.1,
                                           payload: payload, flags: header.flags))
                    if message.0 == .auth {
                        parsingPaused = true
                        return
                    }
                }
            }
        } catch {
            // A framing or authentication failure is unrecoverable: once the
            // stream is misaligned there is nothing to resynchronise to, so the
            // connection ends rather than guessing.
            let reason = error.localizedDescription
            connection.cancel()
            terminate(reason)
        }
    }
}

