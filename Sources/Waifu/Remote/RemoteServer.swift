import Foundation
import Network

/// The serving side: a listener, at most one connected client, and a serial
/// executor that runs its requests in arrival order.
///
/// WaifuFinder can be both a server and a client at once. Neither half knows the
/// other exists; they meet only at `RemoteConnection`.
@MainActor
final class RemoteServer: ObservableObject {
    /// What the Settings panel shows for the one connected client.
    struct Client: Identifiable, Equatable {
        var id: String { sessionID }
        var name: String
        var address: String
        var sessionID: String
        var connectedSince: Date
        var currentJob: String?
    }

    enum ServiceState: Equatable {
        case stopped
        case listening(port: UInt16)
        case failed(String)

        var label: String {
            switch self {
            case .stopped: "Stopped"
            case .listening(let port): "Advertising on port \(port)"
            case .failed(let reason): "Failed: \(reason)"
            }
        }
    }

    @Published private(set) var state: ServiceState = .stopped
    @Published private(set) var client: Client?
    /// The key in display form, regenerated on rotation.
    @Published private(set) var keyDisplay: String = ""
    @Published private(set) var keyGeneratedAt: Date?
    @Published private(set) var stopAfterCurrentJob = false

    private let keyStore: any RemoteKeyStore
    /// Injected so the protocol can be exercised without downloading a model;
    /// the app always uses `.local`.
    private let catalog: RemoteModelCatalog
    private let desiredPort: UInt16
    private let advertises: Bool
    /// Read by the session for the handshake and the capability snapshot.
    let appVersion: String
    let serverName: String

    private var listener: NWListener?
    private var beaconSender: RemoteBeaconSender?
    private var session: RemoteServerSession?
    private var modelsChangedObserver: NSObjectProtocol?
    private var currentKey: [UInt8]?

    init(keyStore: any RemoteKeyStore = RemoteDefaultKeyStore(),
         catalog: RemoteModelCatalog = .local,
         port: UInt16 = RemoteLimits.defaultPort,
         advertises: Bool = true,
         appVersion: String = RemoteAppInfo.version,
         serverName: String = Host.current().localizedName ?? "WaifuFinder") {
        self.keyStore = keyStore
        self.catalog = catalog
        desiredPort = port
        self.advertises = advertises
        self.appVersion = appVersion
        self.serverName = serverName
        loadOrCreateKey()
    }

    /// `stop()` clears the observer; the app holds the server for its whole life, so
    /// a deinit-based teardown is not needed and would have to be `nonisolated`.
    deinit {}

    /// The app's one server. Created eagerly so the key exists and is visible
    /// before the user decides to serve anything.
    static let shared = RemoteServer()

    // MARK: Key

    /// Generates the key on first start, and every time the user rotates it.
    ///
    /// The key is read fresh on every handshake rather than cached, so a rotation
    /// takes effect on the next connection even if this object is long-lived.
    private func loadOrCreateKey() {
        let key = keyStore.loadKey() ?? RemoteCrypto.generateKey()
        if keyStore.loadKey() == nil { try? keyStore.saveKey(key) }
        currentKey = key
        keyDisplay = RemoteCrypto.displayString(for: key)
        let url = RemoteDefaultKeyStore.fallbackURL
        keyGeneratedAt = (try? FileManager.default.attributesOfItem(atPath: url.path))?[.creationDate] as? Date
    }

    /// Generates a new key and drops the live session, because the client's next
    /// handshake would otherwise still succeed against keys already derived from
    /// the old secret.
    func rotateKey() {
        let key = RemoteCrypto.generateKey()
        do {
            try keyStore.saveKey(key)
        } catch {
            state = .failed(error.localizedDescription)
            return
        }
        loadOrCreateKey()
        keyGeneratedAt = Date()
        guard case .listening = state else { return }
        // The restart is deliberate rather than a side effect: it is the only way
        // to guarantee no session derived from the old key survives.
        let running = session
        session = nil
        client = nil
        Task { await running?.shutdown(reason: "keyRotated") }
    }

    /// The `keyID` a beacon advertises, so a client can tell whether it holds
    /// this server's current key without opening a socket.
    var keyID: Data? { keyStore.loadKey().map { RemoteCrypto.keyID(for: $0) } }

    // MARK: Lifecycle

    func start() {
        guard listener == nil else { return }
        let tcp = NWProtocolTCP.Options()
        tcp.noDelay = true
        tcp.enableKeepalive = true
        tcp.keepaliveIdle = Int(RemoteLimits.idleTimeout / 2)
        let parameters = NWParameters(tls: nil, tcp: tcp)
        parameters.allowLocalEndpointReuse = true
        // Peer-to-peer keeps a laptop and a desktop in the same room reachable
        // without waiting for the Wi-Fi network to route between them.
        parameters.includePeerToPeer = true

        do {
            let listener = try NWListener(using: parameters, on: NWEndpoint.Port(rawValue: desiredPort) ?? .any)
            self.listener = listener
            listener.stateUpdateHandler = { [weak self] update in
                Task { @MainActor in self?.handleListener(update) }
            }
            listener.newConnectionHandler = { [weak self] connection in
                Task { @MainActor in self?.accept(connection) }
            }
            listener.start(queue: .global(qos: .userInitiated))
        } catch {
            state = .failed(error.localizedDescription)
            return
        }

        modelsChangedObserver = NotificationCenter.default.addObserver(
            forName: .waifuModelsChanged, object: nil, queue: .main) { [weak self] _ in
                Task { @MainActor in self?.pushCapabilities() }
            }
    }

    func stop() {
        listener?.cancel()
        listener = nil
        beaconSender?.stop()
        beaconSender = nil
        modelsChangedObserver.map(NotificationCenter.default.removeObserver)
        modelsChangedObserver = nil
        // The session shuts itself down asynchronously so it can say goodbye
        // first; clearing the reference here stops it being adopted again.
        let running = session
        session = nil
        Task { await running?.shutdown(reason: "serverStopped") }
        client = nil
        state = .stopped
    }

    private func handleListener(_ update: NWListener.State) {
        switch update {
        case .ready:
            let port = listener?.port?.rawValue ?? desiredPort
            state = .listening(port: port)
            if advertises { startBeacon(port: port) }
        case .failed(let error):
            stop()
            state = .failed(error.localizedDescription)
        case .cancelled:
            if case .listening = state { state = .stopped }
        default:
            break
        }
    }

    private func startBeacon(port: UInt16) {
        let sender = RemoteBeaconSender()
        beaconSender = sender
        sender.start { [weak self] in
            await self?.makeBeacon(port: port)
        }
    }

    /// Built fresh each beat, because the flags reflect the live state: whether a
    /// client is connected, and which models are installed right now.
    private func makeBeacon(port: UInt16) async -> RemoteBeacon? {
        guard let keyID else { return nil }
        let capabilities = await catalog.capabilities(appVersion, serverName, nil)
        var flags: RemoteBeacon.BeaconFlags = [.accepting]
        if client != nil { flags.insert(.clientConnected) }
        if capabilities.offersRemoteTagging { flags.insert(.taggerReady) }
        for algorithm in UncoverAlgorithm.allCases where capabilities.offersRemote(algorithm) {
            let bit = 3 + (UncoverAlgorithm.allCases.firstIndex(of: algorithm) ?? 0)
            flags.insert(RemoteBeacon.BeaconFlags(rawValue: 1 << bit))
        }
        return RemoteBeacon(tcpPort: port, serverInstance: Self.instanceID, keyID: keyID,
                            sentAtMilliseconds: UInt32(truncatingIfNeeded: Int64(Date().timeIntervalSince1970 * 1000)),
                            flags: flags, name: serverName, appVersion: appVersion)
    }

    /// Stable for the life of the process, so a restarted app looks like a
    /// different server to a client that cached a beacon.
    static let instanceID: UInt64 = {
        var generator = SystemRandomNumberGenerator()
        return generator.next()
    }()

    private func accept(_ connection: NWConnection) {
        // One client at a time, as specified. A second connection is told so and
        // closed before the handshake, so it learns nothing and costs nothing.
        guard session == nil else {
            let link = RemoteLink(accepting: connection)
            Task {
                await link.start()
                try? await link.waitUntilOpen()
                let encoded = (try? RemoteControl.encode(
                    .error(code: RemoteError.busy.code, message: "Another client is already connected."))) ?? []
                try? await link.send(.error, payload: encoded)
                await link.close(reason: "busy")
            }
            return
        }
        let link = RemoteLink(accepting: connection)
        let session = RemoteServerSession(link: link, server: self, catalog: catalog)
        self.session = session
        Task { await session.run() }
    }

    func pushCapabilities() {
        guard let session else { return }
        Task { await session.sendCapabilities() }
        Task { await RemoteServerResources.shared.modelsChanged() }
    }

    /// The session authenticated and is ready to serve.
    func sessionEstablished(sessionID: String, name: String) {
        // The address arrives a moment later: reading it means asking the
        // session's actor, and the row should not wait on that.
        client = Client(name: name, address: "connected", sessionID: sessionID,
                        connectedSince: Date(), currentJob: nil)
        Task { [weak self] in
            guard let self else { return }
            let address = await self.clientAddress()
            self.client?.address = address
        }
    }

    /// The session ended, for any reason: kick, stop, key rotation, or a dropped
    /// socket. The session clears itself so a late disconnect from an old session
    /// cannot blank a newer one.
    func sessionEnded(_ reason: String, id: UUID) {
        guard session?.id == id else { return }
        session = nil
        client = nil
        stopAfterCurrentJob = false
        if case .listening = state { return }
        state = .stopped
    }

    /// Kicks the connected client.
    func kick() {
        guard let session else { return }
        Task { await session.shutdown(reason: "kickedByServer") }
    }

    func noteJob(_ description: String?) {
        client?.currentJob = description
        if stopAfterCurrentJob, description == nil {
            stopAfterCurrentJob = false
            stop()
        }
    }

    // MARK: Session internals

    /// The key the live handshake uses. Read fresh each time so a rotation takes
    /// effect on the next connection.
    var currentKeyValue: [UInt8]? { keyStore.loadKey() ?? currentKey }

    /// The remote endpoint of the live socket, for the client's row in the panel.
    ///
    /// Asked of the session asynchronously because it lives on the session's
    /// actor; the panel only redraws when the row actually changes.
    func clientAddress() async -> String {
        guard let session else { return "connected" }
        return await session.remoteAddress()
    }
}

/// The app version the beacon advertises and the handshake negotiates against.
///
/// Read from the bundle so a build and its advertisement cannot disagree; the
/// test bundle has none, so the fallback is a value that still round-trips.
enum RemoteAppInfo {
    static var version: String {
        let short = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String
        if let short, !short.isEmpty { return short }
        let bundle = Bundle.main.bundleURL.lastPathComponent
        return bundle.isEmpty ? "0" : bundle
    }
    static var name: String {
        let name = Bundle.main.object(forInfoDictionaryKey: "CFBundleName") as? String
        return (name?.isEmpty == false ? name : nil) ?? "WaifuFinder"
    }
}
