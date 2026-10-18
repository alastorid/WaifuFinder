import Foundation
import CryptoKit

/// Routes messages to per-stream inboxes over one connection.
///
/// Frames arrive in order and carry the `streamID` of the request they belong
/// to, so a client can keep more than one request in flight without a second
/// connection. Remote tagging keeps a bounded window of batches in flight while
/// preparation and result commits overlap server inference.
///
/// Everything is expressed as **one primitive** — "the next message on this
/// stream". A one-shot request/response is that primitive read once; a
/// reconstruction, which is a manifest, then N inputs, then N results, is the
/// same primitive read N times. A second mechanism for the multi-message case
/// would be one more thing the two ends can disagree about.
actor RemoteConnection {
    private let link: RemoteLink
    /// Buffered frames per stream, in arrival order.
    private var queues: [UInt32: [RemoteEnvelope]] = [:]
    /// Callers suspended on a stream.
    private var readers: [UInt32: [CheckedContinuation<RemoteEnvelope?, Never>]] = [:]
    private var nextStreamID: UInt32 = 1
    /// Callers waiting on `nextAny()`.
    private var anyReaders: [CheckedContinuation<RemoteEnvelope?, Never>] = []
    private var pump: Task<Void, Never>?
    private var onConnectionMessage: (@Sendable (RemoteEnvelope) -> Void)?
    private var onDisconnect: (@Sendable (String) -> Void)?
    private var closed = false
    private var arrivalOrder: [UInt32] = []
    private var bufferedBytes = 0
    private var activeRequests = 0

    func beginRequest() throws -> UInt32 {
        guard !closed else { throw RemoteError.framing("Connection closed") }
        guard activeRequests < RemoteLimits.maximumOutstandingRequests else { throw RemoteError.busy }
        activeRequests += 1
        return allocateStreamID()
    }

    func endRequest() { activeRequests = max(0, activeRequests - 1) }

    private func removeQueued(_ streamID: UInt32) -> RemoteEnvelope? {
        guard let envelope = queues[streamID]?.first else { return nil }
        queues[streamID]?.removeFirst()
        if queues[streamID]?.isEmpty == true { queues[streamID] = nil }
        if let index = arrivalOrder.firstIndex(of: streamID) { arrivalOrder.remove(at: index) }
        bufferedBytes -= envelope.payload.count
        return envelope
    }

    init(link: RemoteLink) { self.link = link }

    func start(onConnectionMessage: @escaping @Sendable (RemoteEnvelope) -> Void,
               onDisconnect: @escaping @Sendable (String) -> Void) {
        self.onConnectionMessage = onConnectionMessage
        self.onDisconnect = onDisconnect
        guard pump == nil else { return }
        pump = Task { [weak self] in await self?.readLoop() }
    }

    func stop() async {
        pump?.cancel()
        pump = nil
        closed = true
        await link.close(reason: "closed")
        releaseAllReaders()
    }

    /// A fresh stream id. Stream 0 is reserved for connection-level messages.
    func allocateStreamID() -> UInt32 {
        defer { nextStreamID = nextStreamID == UInt32.max ? 1 : nextStreamID + 1 }
        return nextStreamID
    }

    // MARK: Sending

    func send(_ opcode: RemoteOpcode, payload: [UInt8], streamID: UInt32 = 0,
              extraFlags: RemoteFlags = []) async throws {
        try await link.send(opcode, streamID: streamID, payload: payload, extraFlags: extraFlags)
    }

    func send(_ opcode: RemoteOpcode, control: RemoteControl, streamID: UInt32 = 0) async throws {
        try await link.send(opcode, streamID: streamID, payload: try RemoteControl.encode(control))
    }

    /// Sends a reply carrying an error, flagged so the peer's reader throws the
    /// server's own message rather than a generic one.
    func sendError(_ replyTo: RemoteEnvelope, _ error: RemoteError) async throws {
        let encoded = (try? RemoteControl.encode(.error(code: error.code,
                                                         message: error.errorDescription ?? ""))) ?? []
        var frames = RemoteMessage.frames(opcode: replyTo.opcode, streamID: replyTo.streamID,
                                           payload: encoded, extraFlags: [.isError])
        // `frames` marks only the final one `isError`; every frame of an error
        // message must carry it, because the reader checks each frame it takes.
        for index in frames.indices { frames[index].header.flags.insert(.isError) }
        try await link.sendRaw(frames)
    }

    // MARK: Receiving

    /// The next message on `streamID`, or `nil` once the connection has ended.
    func next(streamID: UInt32) async -> RemoteEnvelope? {
        if let envelope = removeQueued(streamID) { return envelope }
        if closed { return nil }
        return await withCheckedContinuation { continuation in
            readers[streamID, default: []].append(continuation)
        }
    }

    /// The next message on any stream, in arrival order across streams.
    ///
    /// This is what the server's loop uses: it dispatches work as it arrives
    /// rather than waiting for one particular stream, so a long reconstruction
    /// does not sit in front of a tagging batch. A per-stream reader would put
    /// the server's dispatch order at the mercy of whichever request registered
    /// first.
    func nextAny() async -> RemoteEnvelope? {
        if let envelope = dequeueAny() { return envelope }
        if closed { return nil }
        return await withCheckedContinuation { continuation in
            if let envelope = dequeueAny() { continuation.resume(returning: envelope) }
            else if closed { continuation.resume(returning: nil) }
            else { anyReaders.append(continuation) }
        }
    }

    private func dequeueAny() -> RemoteEnvelope? {
        guard let stream = arrivalOrder.first else { return nil }
        return removeQueued(stream)
    }

    /// Sends a request and returns the reply, throwing the server's error when
    /// the reply is a failure.
    func request(_ opcode: RemoteOpcode, payload: [UInt8],
                 maximumOutstanding: Int = RemoteLimits.maximumOutstandingRequests) async throws -> RemoteEnvelope {
        let streamID = try beginRequest()
        defer { endRequest() }
        return try await withTaskCancellationHandler {
        try Task.checkCancellation()
        try await send(opcode, payload: payload, streamID: streamID)
        guard let reply = await next(streamID: streamID) else {
            throw RemoteError.framing("The connection ended before the request was answered.")
        }
        if reply.isError { throw Self.remoteError(from: reply) }
        try Task.checkCancellation()
        return reply
        } onCancel: { Task { await self.stop() } }
    }

    func request(_ opcode: RemoteOpcode, control: RemoteControl) async throws -> RemoteEnvelope {
        try await request(opcode, payload: try RemoteControl.encode(control))
    }

    /// An error frame carries the same `error` control shape the server would
    /// have thrown locally, so the client's message is the server's message.
    static func remoteError(from envelope: RemoteEnvelope) -> RemoteError {
        guard let control = try? envelope.control(), case .error(let code, let message) = control else {
            return .internalFailure("The request failed.")
        }
        return RemoteError(code: code, message: message)
    }

    // MARK: Read loop

    private func readLoop() async {
        while !Task.isCancelled {
            guard let envelope = await link.next() else {
                let reason = await link.disconnectReason ?? "closed"
                closed = true
                releaseAllReaders()
                onDisconnect?(reason)
                return
            }
            if envelope.streamID == 0 && ![RemoteOpcode.hello, .challenge, .auth, .authResult, .error].contains(envelope.opcode) {
                onConnectionMessage?(envelope)
                continue
            }
            await deliver(envelope)
        }
    }

    private func deliver(_ envelope: RemoteEnvelope) async {
        // A stream that already has a waiter takes priority; otherwise the frame
        // waits for either that stream's reader or the server's `nextAny`.
        var waiting = readers[envelope.streamID] ?? []
        if !waiting.isEmpty {
            waiting.removeFirst().resume(returning: envelope)
            readers[envelope.streamID] = waiting.isEmpty ? nil : waiting
            return
        }
        var anyone = anyReaders
        if !anyone.isEmpty {
            anyone.removeFirst().resume(returning: envelope)
            anyReaders = anyone
            return
        }
        guard bufferedBytes + envelope.payload.count <= RemoteLimits.maximumMessagePayload,
              arrivalOrder.count < 1024 else {
            await stop()
            return
        }
        bufferedBytes += envelope.payload.count
        arrivalOrder.append(envelope.streamID)
        queues[envelope.streamID, default: []].append(envelope)
    }

    private func releaseAllReaders() {
        for (_, waiting) in readers { for continuation in waiting { continuation.resume(returning: nil) } }
        readers.removeAll()
        for continuation in anyReaders { continuation.resume(returning: nil) }
        anyReaders.removeAll()
        queues.removeAll()
        arrivalOrder.removeAll()
        bufferedBytes = 0
    }
}
