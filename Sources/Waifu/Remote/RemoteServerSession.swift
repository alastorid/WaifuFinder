import Foundation
import Network

/// The per-connection state: the handshake, then the serial loop that runs the
/// client's requests in arrival order.
actor RemoteServerSession {
    /// Everything the handshake needs from the serving app, read in one hop.
    struct ServerInfo: Sendable {
        let name: String
        let version: String
        let keyID: Data?
        let key: [UInt8]?
    }

    nonisolated let id = UUID()
    private var finished = false
    private var maintenanceTask: Task<Void, Never>?
    private var timeoutTask: Task<Void, Never>?
    private let link: RemoteLink
    let connection: RemoteConnection
    /// The serving `RemoteServer`. Weak because the server owns this session and
    /// clears it on disconnect; holding it strongly would keep a stopped server
    /// alive for as long as its last session.
    private weak var server: RemoteServer?
    private var authenticated = false
    private var clientNonce = Data()
    private var serverNonce = Data()
    private var sessionID = ""
    private var handshakeTask: Task<Void, Never>?
    let resources = RemoteServerResources()
    let catalog: RemoteModelCatalog

    init(link: RemoteLink, server: RemoteServer, catalog: RemoteModelCatalog) {
        self.link = link
        self.connection = RemoteConnection(link: link)
        self.server = server
        self.catalog = catalog
    }

    // MARK: Lifecycle

    func run() async {
        await link.start()
        do { try await link.waitUntilOpen() } catch {
            await finish(reason: "unreachable")
            return
        }
        timeoutTask = Task { [weak self] in
            do { try await Task.sleep(for: .seconds(RemoteLimits.handshakeTimeout)) } catch { return }
            await self?.shutdown(reason: "Handshake timed out")
        }
        await connection.start(onConnectionMessage: { [weak self] envelope in
            Task { await self?.handleConnectionMessage(envelope) }
        }, onDisconnect: { [weak self] reason in
            Task { await self?.finish(reason: reason) }
        })
        // The handshake reads connection-level messages directly: nothing may be
        // routed to a stream, or answered, before the client proves it holds the
        // key.
        handshakeTask = Task { await handshake() }
    }

    func shutdown(reason: String) async {
        try? await connection.send(.close, control: .close(reason: reason))
        await finish(reason: reason)
    }

    private func finish(reason: String) async {
        guard !finished else { return }
        finished = true
        timeoutTask?.cancel()
        timeoutTask = nil
        maintenanceTask?.cancel()
        maintenanceTask = nil
        handshakeTask?.cancel()
        handshakeTask = nil
        authenticated = false
        await resources.releaseAll()
        await link.close(reason: reason)
        // The reference is read here, on this actor, and the hop to the main
        // actor carries only that: the main actor cannot read an actor's
        // property.
        let target = server
        await MainActor.run { target?.sessionEnded(reason, id: self.id) }
    }

    // MARK: Reaching the serving app

    private func serverInfo() async -> ServerInfo? {
        guard let server else { return nil }
        return await MainActor.run {
            ServerInfo(name: server.serverName, version: server.appVersion,
                       keyID: server.keyID, key: server.currentKeyValue)
        }
    }

    /// Reports what the connected client is doing, for the Settings panel.
    func noteJob(_ description: String?) async {
        let target = server
        await MainActor.run { target?.noteJob(description) }
    }

    /// Publishes the authenticated client row.
    private func markEstablished(name: String) async {
        let target = server
        let id = sessionID
        await MainActor.run { target?.sessionEstablished(sessionID: id, name: name) }
    }

    /// The client's address on this socket, for the row in the Settings panel.
    ///
    /// Read from the live socket rather than remembered from the handshake,
    /// because the connection may be over a peer-to-peer interface whose address
    /// is not the one that was dialled.
    func remoteAddress() async -> String { await link.remoteEndpointDescription() }

    // MARK: Handshake

    /// Four messages, then every frame after them is sealed.
    ///
    /// Shaped like `Noise_NNpsk0`: two ephemeral nonces and one HMAC from each
    /// side. The key itself is never written to a frame, and the session keys are
    /// installed only after both proofs have been checked, so an unauthenticated
    /// peer never sees a single key-derived byte.
    private func handshake() async {
        var generator = SystemRandomNumberGenerator()
        serverNonce = Data((0..<RemoteCrypto.nonceByteCount).map { _ in generator.next() })
        guard let info = await serverInfo(), let key = info.key else {
            return await fail("no key")
        }

        // 1. The client's hello.
        guard let hello = await connection.next(streamID: 0) else { return await fail("no hello") }
        guard hello.opcode == .hello, case .hello(let version, _, let name, let nonce)? = try? hello.control(),
              nonce.count == RemoteCrypto.nonceByteCount, version == Int(RemoteLimits.protocolVersion) else {
            return await fail("bad hello")
        }
        clientNonce = nonce

        // 2. The challenge, which also states which key this server expects, so
        // the client notices a rotation before it proves anything.
        do {
            try await connection.send(.challenge, control: .challenge(
                serverName: info.name, appVersion: info.version,
                serverNonce: serverNonce, keyID: info.keyID ?? Data()))
        } catch { return await fail("challenge undeliverable") }

        // 3. The client's proof.
        guard let auth = await connection.next(streamID: 0) else { return await fail("no auth") }
        guard auth.opcode == .auth else { return await fail("bad auth") }
        let expected = RemoteCrypto.clientProof(key: key, clientNonce: clientNonce, serverNonce: serverNonce)
        guard RemoteCrypto.constantTimeEquals(Data(auth.payload), expected) else {
            // One generic message, and no distinction between "wrong key" and
            // "wrong proof": over a 256-bit random secret there is nothing there
            // to learn, and one vague failure is the more useful message to read.
            try? await connection.send(.authResult, control: .authResult(
                ok: false, sessionID: nil, serverProof: nil, capabilities: nil))
            return await fail("bad key")
        }

        // 4. The server proves itself too, so the client also learns it is
        // talking to the key holder rather than to a listener replaying
        // challenges.
        sessionID = UUID().uuidString
        let proof = RemoteCrypto.serverProof(key: key, sessionID: sessionID,
                                            clientNonce: clientNonce, serverNonce: serverNonce)
        let capabilities = await catalog.capabilities(info.version, info.name, sessionID)
        let keys = RemoteCrypto.sessionKeys(key: key, clientNonce: clientNonce, serverNonce: serverNonce)
        await link.seal(with: keys, role: .serverToClient)
        do {
            try await connection.send(.authResult, control: .authResult(
                ok: true, sessionID: sessionID, serverProof: proof, capabilities: capabilities))
        } catch { return await fail("result undeliverable") }

        authenticated = true
        timeoutTask?.cancel()
        timeoutTask = nil
        maintenanceTask = Task { [weak self] in
            while !Task.isCancelled {
                do { try await Task.sleep(for: .seconds(5)) } catch { return }
                guard let self else { return }
                await self.resources.releaseIfIdle()
                if await self.link.isIdle { await self.shutdown(reason: "Idle timeout"); return }
            }
        }
        await markEstablished(name: name)
        await serve()
    }

    private func fail(_ reason: String) async {
        await finish(reason: reason)
    }

    // MARK: Serving

    /// Connection-level messages: pings, and cancellations.
    private func handleConnectionMessage(_ envelope: RemoteEnvelope) async {
        guard authenticated else { return }
        switch envelope.opcode {
        case .ping:
            if case .ping(let counter)? = try? envelope.control() {
                try? await connection.send(.pong, control: .pong(counter: counter))
            }
        case .cancelJob:
            // Cancellation arrives on stream 0 because it is about the job, not
            // about the frame stream the job happens to share.
            if case .cancelJob(let streamID)? = try? envelope.control() {
                await resources.cancelJob(streamID)
            }
        default:
            break
        }
    }

    /// The request loop.
    ///
    /// Streams are drained independently, so a long reconstruction does not sit
    /// in front of a tagging batch. The model work still serialises against the
    /// GPU itself; this only keeps the sockets moving.
    private func serve() async {
        while authenticated {
            guard let envelope = await connection.nextAny() else { return }
            switch envelope.opcode {
            case .tagBatch: await handleTagBatch(envelope)
            case .vocabularyRequest: await handleVocabularyRequest(envelope)
            case .uncoverBegin: await handleUncover(envelope)
            default:
                try? await connection.sendError(envelope, .invalidRequest(
                    "Unsupported request \(envelope.opcode.rawValue)."))
            }
            await resources.releaseIfIdle()
        }
    }

    /// Pushes a fresh capability snapshot, so the client's ᴿ markers follow
    /// model installs and removals live.
    func sendCapabilities() async {
        guard authenticated, let info = await serverInfo() else { return }
        let capabilities = await catalog.capabilities(info.version, info.name, sessionID)
        try? await connection.send(.capabilities, control: .capabilities(capabilities))
    }
}