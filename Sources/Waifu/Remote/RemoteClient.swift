import Foundation
import CoreGraphics

/// The borrowing side: discovery, the connection state machine, and the two
/// remote backends.
///
/// Connection status *is* reachability. There is no separate health probe: one
/// state machine answers both "can I reach it" and "what can it do", and an
/// authentication failure does not retry on a timer because a key that is known
/// to be wrong will not become right.
@MainActor
final class RemoteClient: ObservableObject {
    enum Status: Equatable {
        case off
        case discovering
        case connecting(String)
        case authenticating(String)
        case connected(String)
        /// The server refused the connection; the user has to act.
        case refused(String)
        /// Transient. `retryAfter` is when the next attempt is due.
        case retrying(String, Date)

        var isConnected: Bool { if case .connected = self { return true } else { return false } }

        var label: String {
            switch self {
            case .off: "Not connected"
            case .discovering: "Looking for servers…"
            case .connecting(let host): "Connecting to \(host)…"
            case .authenticating(let host): "Verifying the access key with \(host)…"
            case .connected(let host): "Connected to \(host)"
            case .refused(let reason): reason
            case .retrying(let reason, let when):
                "Reconnecting to \(reason) at \(Self.clock(when))"
            }
        }

        /// Whether a reconnect timer should be running.
        var shouldRetry: Date? { if case .retrying(_, let when) = self { return when } else { return nil } }

        /// A failure the user has to resolve. These never retry on their own.
        var needsAttention: Bool { if case .refused = self { return true } else { return false } }

        private static func clock(_ date: Date) -> String {
            let formatter = DateFormatter()
            formatter.dateFormat = "HH:mm:ss"
            return formatter.string(from: date)
        }
    }

    // MARK: Published state

    @Published private(set) var status: Status = .off
    @Published private(set) var servers: [DiscoveredServer] = []
    @Published private(set) var capabilities: RemoteCapabilities? {
        didSet {
            if let capabilities {
                lastKnownCapabilities = capabilities
                defaults.set(try? JSONEncoder().encode(capabilities), forKey: capabilitiesDefaultsKey)
            }
        }
    }
    @Published private(set) var lastKnownCapabilities: RemoteCapabilities?
    @Published private(set) var isEnabled = false
    /// The tag vocabulary the server supplied, once downloaded.
    @Published private(set) var remoteVocabulary: TagVocabulary?

    @Published var manualAddress: String {
        didSet {
            defaults.set(manualAddress, forKey: addressDefaultsKey)
            objectWillChange.send()
        }
    }
    /// The key for `manualAddress`, stored per address so two servers on one
    /// network do not overwrite each other.
    @Published var manualKey: String = ""
    @Published private(set) var keyMessage: String?

    private let vocabularyCacheDirectory: URL?
    private let discoveryEnabled: Bool
    private var lastPong = Date()
    private let capabilitiesDefaultsKey: String
    private let addressDefaultsKey: String
    private let enabledDefaultsKey: String
    private let defaults: UserDefaults
    private let cache: RemoteBeaconCache
    private var receiver: RemoteBeaconReceiver?
    private var link: RemoteLink?
    /// Read by the remote backends, which live outside this file.
    var connection: RemoteConnection?
    private var endpoint: RemoteAddress.Endpoint?
    private var pingTask: Task<Void, Never>?
    private var retryTask: Task<Void, Never>?
    private var beaconPoll: Task<Void, Never>?
    private var backoff = 1.0
    private var generation = UUID()
    private var connectTask: Task<Void, Never>?

    /// A short identity for the connected server, readable synchronously.
    ///
    /// `TemporalVideoUpscaler.cacheIdentity` is a synchronous requirement called
    /// off the main actor, so the value is mirrored into a locked box when the
    /// capabilities change rather than read across actors on demand.
    private let identityBox = RemoteIdentityBox()

    static let shared = RemoteClient()

    init(defaults: UserDefaults = .standard,
         keyStore: (any RemoteKeyStore)? = nil, discoveryEnabled: Bool = true,
         vocabularyCacheDirectory: URL? = nil, configurationID: String? = nil) {
        self.vocabularyCacheDirectory = vocabularyCacheDirectory
        self.discoveryEnabled = discoveryEnabled
        self.defaults = defaults
        addressDefaultsKey = configurationID.map { "remoteClientAddress.\($0)" } ?? "remoteClientAddress"
        enabledDefaultsKey = configurationID.map { "remoteClientEnabled.\($0)" } ?? "remoteClientEnabled"
        capabilitiesDefaultsKey = "remoteClientCapabilities.\(configurationID ?? "primary")"
        lastKnownCapabilities = defaults.data(forKey: capabilitiesDefaultsKey)
            .flatMap { try? JSONDecoder().decode(RemoteCapabilities.self, from: $0) }
        keyStoreBox.value = keyStore
        manualAddress = defaults.string(forKey: addressDefaultsKey) ?? ""
        cache = RemoteBeaconCache()
        // The receiver is created lazily in `setEnabled`: binding the beacon port
        // triggers a local-network permission prompt, and that must not happen
        // merely because a settings pane was built.
    }

    func clearConfiguration() {
        setEnabled(false)
        disconnect()
        manualAddress = ""
        manualKey = ""
        keyMessage = nil
        lastKnownCapabilities = nil
        defaults.removeObject(forKey: capabilitiesDefaultsKey)
    }

    func restoreEnabled() { setEnabled(defaults.bool(forKey: enabledDefaultsKey)) }

    // MARK: Keys

    /// A stored key for a discovered server, so the panel can say Ready or Key
    /// needed without a round trip.
    func keyID(for address: String) -> Data? {
        (keyStoreBox.value ?? RemoteDefaultKeyStore(clientAddress: address)).loadKey().map { RemoteCrypto.keyID(for: $0) }
    }

    func storedKey(for endpoint: RemoteAddress.Endpoint) -> [UInt8]? {
        guard let stored = keyStore.loadKey() else { return nil }
        return stored
    }

    /// Swaps the store this client reads its key from.
    ///
    /// Exists so a test can put the client and the server in the
    /// key-mismatch and no-key states without reaching into either one's
    /// internals — the states themselves are what the refusal cases are about.
    func replaceKeyStore(_ store: any RemoteKeyStore) {
        keyStoreBox.value = store
    }

    /// The key store is read from both the main actor and the settings panel, so
    /// it is held in a locked box rather than as an immutable property.
    private let keyStoreBox = RemoteKeyStoreBox()

    private var keyStore: any RemoteKeyStore {
        keyStoreBox.value ?? RemoteDefaultKeyStore(clientAddress: RemoteAddress.parse(manualAddress)?.display ?? manualAddress)
    }

    private func storeManualKey() -> Bool {
        let trimmed = manualKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return true }
        do {
            let key = try RemoteCrypto.parseDisplayString(trimmed)
            try keyStore.saveKey(key)
            manualKey = ""
            keyMessage = "Access key saved for this address."
            return true
        } catch {
            keyMessage = "Could not save the access key. Check the key and its final check group."
            return false
        }
    }

    // MARK: Enabling

    func setEnabled(_ enabled: Bool) {
        guard enabled != isEnabled else { return }
        isEnabled = enabled
        defaults.set(enabled, forKey: enabledDefaultsKey)
        if enabled {
            status = .discovering
            startReceiving()
            if !manualAddress.isEmpty { connect(manual: true) }
        } else {
            stop()
        }
    }

    private func startReceiving() {
        guard discoveryEnabled, receiver == nil else { return }
        let receiver = RemoteBeaconReceiver { [weak self] beacon, source in
            Task { await self?.record(beacon: beacon, from: source) }
        }
        self.receiver = receiver
        receiver.start()
        // The cached list is refreshed on a timer as well as on arrival, so an
        // expiry that happens between beacons still removes the row.
        beaconPoll = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(1))
                self?.refreshServers()
            }
        }
    }

    private func record(beacon: RemoteBeacon, from source: String) async {
        await cache.record(beacon, from: source)
        refreshServers()
    }

    func refreshServers() {
        let live = Task.detached { [cache] in await cache.live() }
        Task { [weak self] in
            let discovered = await live.value
            self?.servers = discovered
        }
    }

    // MARK: Connecting

    /// Connects to an address typed by hand, or to one of the discovered rows.
    func connect(to address: String? = nil) {
        let text = address ?? manualAddress
        guard let endpoint = RemoteAddress.parse(text) else {
            status = .refused("That address is not a host:port WaifuFinder can dial.")
            return
        }
        self.endpoint = endpoint
        manualAddress = endpoint.display
        guard storeManualKey() else { return }
        if let discovered = servers.first(where: { $0.address == endpoint.display }) {
            discoveredServersFirst = discovered
        } else {
            discoveredServersFirst = nil
        }
        attempt()
    }

    func connect(manual: Bool) { connect(to: nil) }

    private var discoveredServersFirst: DiscoveredServer?

    private func attempt() {
        guard let endpoint, isEnabled || !manualAddress.isEmpty else { return }
        retryTask?.cancel()
        pingTask?.cancel()
        generation = UUID()
        let token = generation
        connectTask?.cancel()
        let previous = self.connection
        capabilities = nil
        remoteVocabulary = nil
        publishIdentity(nil)
        status = .connecting(endpoint.display)
        let link = RemoteLink(endpoint: endpoint)
        let connection = RemoteConnection(link: link)
        self.link = link
        self.connection = connection
        let key = keyStore.loadKey()
        connectTask = Task {
            await previous?.stop()
            await link.start()
            let timeout = Task {
                do { try await Task.sleep(for: .seconds(RemoteLimits.handshakeTimeout)) } catch { return }
                await link.close(reason: "Handshake timed out")
            }
            defer { timeout.cancel() }
            do {
                try await link.waitUntilOpen()
                await connection.start(onConnectionMessage: { [weak self] envelope in
                    Task { @MainActor in
                        guard let self, self.generation == token else { return }
                        await self.handlePush(envelope)
                    }
                }, onDisconnect: { [weak self] reason in
                    Task { @MainActor in
                        guard let self, self.generation == token else { return }
                        await self.connectionEnded(reason)
                    }
                })
                try await handshake(connection: connection, link: link, key: key,
                                    displayName: discoveredServersFirst?.name ?? "remote")
            } catch {
                guard generation == token else { return }
                report(error, endpoint: endpoint)
                await link.close(reason: error.localizedDescription)
            }
        }
    }

    private func handshake(connection: RemoteConnection, link: RemoteLink,
                           key: [UInt8]?, displayName: String) async throws {
        await MainActor.run { self.status = .authenticating(endpoint?.display ?? displayName) }
        var generator = SystemRandomNumberGenerator()
        let clientNonce = Data((0..<RemoteCrypto.nonceByteCount).map { _ in generator.next() })
        try await connection.send(.hello, control: .hello(
            protocolVersion: Int(RemoteLimits.protocolVersion), appVersion: RemoteAppInfo.version,
            clientName: Host.current().localizedName ?? "WaifuFinder", clientNonce: clientNonce))

        guard let challenge = await connection.next(streamID: 0) else {
            throw RemoteError.framing("The server did not answer.")
        }
        if challenge.opcode == .error { throw RemoteConnection.remoteError(from: challenge) }
        guard let challengeControl = try? challenge.control(),
              case .challenge(_, let serverVersion, let serverNonce, let keyID) = challengeControl,
              serverNonce.count == RemoteCrypto.nonceByteCount else {
            throw RemoteError.framing("The server sent a malformed challenge.")
        }

        guard let key else {
            await MainActor.run {
                self.status = .refused("No access key for this server yet.")
                self.pendingServerKeyID = keyID
            }
            await connection.stop()
            return
        }
        // A keyID mismatch is reported as such rather than as a bad key: the
        // server rotated, and that is a different thing for the user to fix.
        guard keyID.isEmpty || keyID == RemoteCrypto.keyID(for: key) else {
            await MainActor.run {
                self.status = .refused("This server's access key has changed — paste the new key.")
                self.pendingServerKeyID = keyID
            }
            await connection.stop()
            return
        }

        let proof = RemoteCrypto.clientProof(key: key, clientNonce: clientNonce, serverNonce: serverNonce)
        let keys = RemoteCrypto.sessionKeys(key: key, clientNonce: clientNonce, serverNonce: serverNonce)
        try await link.authenticateClient(proof: [UInt8](proof), keys: keys)

        guard let result = await connection.next(streamID: 0) else {
            throw RemoteError.framing("The server did not answer.")
        }
        guard let resultControl = try? result.control(),
              case .authResult(let ok, let sessionID, let serverProof, let capabilities) = resultControl else {
            throw RemoteError.framing("The server sent a malformed result.")
        }
        guard ok else { throw RemoteError.badKey }

        // The server proves itself too, so a passive listener that replayed a
        // challenge is refused here rather than trusted with inference work.
        guard let sessionID, let serverProof, let capabilities,
              capabilities.protocolVersion == Int(RemoteLimits.protocolVersion) else { throw RemoteError.badKey }
        do {
            let expected = RemoteCrypto.serverProof(key: key, sessionID: sessionID,
                                                   clientNonce: clientNonce, serverNonce: serverNonce)
            guard RemoteCrypto.constantTimeEquals(serverProof, expected) else {
                throw RemoteError.badKey
            }
        }


        try Task.checkCancellation()
        await MainActor.run {
            self.capabilities = capabilities
            self.publishIdentity(capabilities)
            self.backoff = 1
        }
        await loadVocabularyIfNeeded()
        if capabilities.offersRemoteTagging, remoteVocabulary?.cacheFingerprint != capabilities.tagger?.vocabularyFingerprint {
            throw RemoteError.modelUnavailable("The server supplied invalid tag labels. Reconnect after fixing its vocabulary.")
        }
        try Task.checkCancellation()
        status = .connected(endpoint?.display ?? serverVersion)
        startPings(connection: connection)
    }

    /// The `keyID` of a server whose key this Mac does not hold, for the panel.
    @Published private(set) var pendingServerKeyID: Data?

    private func startPings(connection: RemoteConnection) {
        pingTask?.cancel()
        lastPong = Date()
        var counter: UInt64 = 0
        pingTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(RemoteLimits.clientPingInterval))
                guard !Task.isCancelled else { return }
                guard let self else { return }
                if Date().timeIntervalSince(self.lastPong) > RemoteLimits.clientPongTimeout {
                    await connection.stop()
                    return
                }
                counter &+= 1
                do { try await connection.send(.ping, control: .ping(counter: counter)) }
                catch { await self.connectionEnded("The server stopped answering."); return }
            }
        }
    }

    /// Server-pushed state: capability changes and job progress.
    private func handlePush(_ envelope: RemoteEnvelope) async {
        guard let control = try? envelope.control() else { return }
        switch control {
        case .pong:
            lastPong = Date()
        case .capabilities(let updated):
            await MainActor.run {
                self.capabilities = updated
                self.publishIdentity(updated)
            }
            await loadVocabularyIfNeeded()
        case .close(let reason):
            await MainActor.run {
                self.status = reason == "kickedByServer"
                    ? .refused("Disconnected — kicked by this Mac's owner.")
                    : .refused(Self.describeClose(reason))
            }
        default:
            break
        }
    }

    private static func describeClose(_ reason: String) -> String {
        switch reason {
        case "keyRotated": "The server's access key was rotated. Paste the new key."
        case "serverStopped": "The server stopped serving."
        case "bad key": "The server rejected the access key."
        case "badKey": "The server rejected the access key."
        case "busy": "The server is busy with another client."
        default: "Disconnected: \(reason)"
        }
    }

    private func report(_ error: Error, endpoint: RemoteAddress.Endpoint) {
        capabilities = nil
        remoteVocabulary = nil
        publishIdentity(nil)
        if let remote = error as? RemoteError, case .modelUnavailable(let message) = remote {
            status = .refused(message)
            return
        }
        if let remote = error as? RemoteError, case .busy = remote {
            status = .refused("The server is busy with another client.")
            return
        }
        if let remote = error as? RemoteError, case .badKey = remote {
            status = .refused("The server rejected the access key.")
            return
        }
        // A refused or unroutable address will not fix itself in one second, so
        // the backoff starts where a network problem would and grows from there.
        let delay = backoff
        backoff = min(backoff * 2, 30)
        status = .retrying(endpoint.display, Date().addingTimeInterval(delay))
        scheduleRetry(after: delay)
    }

    private func scheduleRetry(after delay: TimeInterval) {
        retryTask?.cancel()
        retryTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(delay))
            guard !Task.isCancelled else { return }
            await MainActor.run { self?.attempt() }
        }
    }

    private func connectionEnded(_ reason: String) async {
        guard endpoint != nil else { return }
        await MainActor.run {
            pingTask?.cancel()
            pingTask = nil
            capabilities = nil
            remoteVocabulary = nil
            publishIdentity(nil)
            if case .refused = status { return }
            if case .retrying = status { return }
            let display = endpoint?.display ?? "the server"
            let delay = backoff
            backoff = min(backoff * 2, 30)
            status = .retrying(display, Date().addingTimeInterval(delay))
            scheduleRetry(after: delay)
        }
    }

    func disconnect() {
        generation = UUID()
        connectTask?.cancel()
        retryTask?.cancel()
        pingTask?.cancel()
        backoff = 1
        capabilities = nil
        remoteVocabulary = nil
        publishIdentity(nil)
        let running = connection
        connection = nil
        link = nil
        endpoint = nil
        status = isEnabled ? .discovering : .off
        Task { await running?.stop() }
    }

    func forget(address: String) {
        do {
            try (keyStoreBox.value ?? RemoteDefaultKeyStore(clientAddress: address)).deleteKey()
            if endpoint?.display == address { disconnect() }
            Task {
                await cache.forget(address)
                refreshServers()
            }
        } catch { keyMessage = error.localizedDescription }
    }

    private func stop() {
        generation = UUID()
        connectTask?.cancel()
        retryTask?.cancel()
        pingTask?.cancel()
        beaconPoll?.cancel()
        receiver?.stop()
        receiver = nil
        backoff = 1
        capabilities = nil
        remoteVocabulary = nil
        publishIdentity(nil)
        let running = connection
        connection = nil
        link = nil
        endpoint = nil
        status = .off
        Task { await running?.stop() }
    }

    // MARK: Capabilities

    /// Whether a tagging request would run on the server.
    var offersRemoteTagging: Bool { status.isConnected && capabilities?.offersRemoteTagging == true }

    /// Whether an Uncover request for `algorithm` would run on the server.
    func offersRemote(_ algorithm: UncoverAlgorithm, scale: UncoverScale? = nil) -> Bool {
        guard status.isConnected else { return false }
        return capabilities?.offersRemote(algorithm, scale: scale) ?? false
    }

    var offersAnyRemoteUncover: Bool { status.isConnected && capabilities?.offersAnyUncover == true }

    /// A failure only the user can resolve — a refused key, a kicked session, a
    /// rotated key. These never retry on their own, so the panel can stop
    /// presenting them as a transient blip.
    var needsAttention: Bool { status.needsAttention }

    /// `"Studio@1.4.0"` while connected, `"unidentified"` otherwise.
    ///
    /// Distinct servers must not produce interchangeable cache entries, so this
    /// is part of a remote Uncover cache key.
    nonisolated var remoteIdentity: String { identityBox.value }

    private func publishIdentity(_ capabilities: RemoteCapabilities?) {
        identityBox.value = capabilities.map {
            let keyID = keyStore.loadKey().map { RemoteCrypto.keyID(for: $0).map { String(format: "%02x", $0) }.joined() } ?? "unknown"
            return "\($0.serverName)@\($0.appVersion):\(keyID)"
        } ?? "unidentified"
    }

    // MARK: Vocabulary

    /// Downloads the labels CSV once per fingerprint, and returns the vocabulary.
    ///
    /// Without it a client that has no local tagger could not filter, search or
    /// render a single tag, so this is what makes remote tagging usable rather
    /// than merely possible. Returns whatever is already held when no download
    /// is needed, so a caller can treat it as "the vocabulary, or nil".
    @discardableResult
    func loadVocabularyIfNeeded() async -> TagVocabulary? {
        guard let connection, capabilities?.offersRemoteTagging == true,
              let fingerprint = capabilities?.tagger?.vocabularyFingerprint else {
            return remoteVocabulary
        }
        if let remoteVocabulary, remoteVocabulary.cacheFingerprint == fingerprint { return remoteVocabulary }
        let token = generation
        do {
            let reply = try await connection.request(.vocabularyRequest, payload: [])
            let vocabulary = try Self.decodeVocabulary(reply.payload, cacheDirectory: vocabularyCacheDirectory)
            guard token == generation else { return nil }
            guard vocabulary.cacheFingerprint == fingerprint else { throw RemoteError.framing("Vocabulary fingerprint changed") }
            remoteVocabulary = vocabulary
            return vocabulary
        } catch {
            // A failure here is not fatal to the connection: the frames can still
            // be tagged, they just cannot be named. The tagging run reports it.
            return remoteVocabulary
        }
    }

    /// Decodes a `vocabularyResponse`, writing the CSV to a per-fingerprint cache
    /// directory so `TagVocabulary.load(from:)` — and everything built on it —
    /// works unchanged.
    nonisolated static func decodeVocabulary(_ payload: [UInt8], cacheDirectory: URL? = nil) throws -> TagVocabulary {
        var reader = WireReader(payload)
        let fingerprint = try reader.u64()
        let uncompressed = Int(try reader.u32())
        let csv = try RemoteCompressedPayload.decode(reader.remainingBytes())
        guard csv.count == uncompressed else {
            throw RemoteError.framing("the vocabulary is \(csv.count) bytes, expected \(uncompressed)")
        }
        let root = cacheDirectory ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Waifu/Remote/Vocabulary", isDirectory: true)
        let directory = root.appendingPathComponent("\(fingerprint)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let file = directory.appendingPathComponent("wd-tagger-labels.csv")
        try Data(csv).write(to: file, options: .atomic)
        let vocabulary = try TagVocabulary.load(from: file)
        // The fingerprint is what the client keyed its cache on, so it must be the
        // one the server reported — not merely one the CSV happens to produce.
        guard vocabulary.cacheFingerprint == fingerprint else {
            try? FileManager.default.removeItem(at: directory)
            throw RemoteError.framing("the vocabulary does not match the fingerprint the server reported")
        }
        return vocabulary
    }

    // MARK: Requests

    /// Sends a tagging batch. Returns one probability row per image.
    func tag(images: [CGImage], codec: RemoteTagImageCodec) async throws -> [QuantizedTagRow] {
        guard offersRemoteTagging, let connection else { throw RemoteError.framing("Not connected.") }
        guard (1...WDTaggerBackend.batchSize).contains(images.count) else { throw TaggerError.invalidBatch }
        let payload = try await Self.encodeTagBatch(images, codec: codec)
        let reply = try await connection.request(.tagBatch, payload: payload)
        let expected = capabilities?.tagger?.vocabularyFingerprint
        return try await Self.decodeTagResult(reply, count: images.count, fingerprint: expected)
    }

    // Nonisolated async helpers run image conversion and probability decoding
    // off the UI actor while other batches are travelling or running remotely.
    private nonisolated static func encodeTagBatch(_ images: [CGImage], codec: RemoteTagImageCodec) async throws -> [UInt8] {
        var payload = WireWriter()
        payload.u8(UInt8(images.count))
        payload.u8(codec.rawValue)
        payload.u16(0)
        for image in images {
            try Task.checkCancellation()
            switch codec {
            case .encoded: payload.data(try ThumbnailCache.encodeJPEG(image))
            case .rgba8: payload.raw(try RemoteTagImageDecoder.encodeRGBA8(image))
            }
        }
        return payload.bytes
    }

    private nonisolated static func decodeTagResult(_ reply: RemoteEnvelope, count expectedCount: Int,
                                                    fingerprint expected: UInt64?) async throws -> [QuantizedTagRow] {
        try Task.checkCancellation()
        guard reply.opcode == .tagResult else { throw TaggerError.invalidProbabilities }
        return try RemoteTagScorePayload.decode(reply.payload, count: expectedCount, fingerprint: expected)
    }

}
/// Holds the key store so `replaceKeyStore` can swap it.
final class RemoteKeyStoreBox: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: (any RemoteKeyStore)?

    var value: (any RemoteKeyStore)? {
        get { lock.withLock { storage } }
        set { lock.withLock { storage = newValue } }
    }
}

/// A string readable from any isolation domain.
///
/// Exists only for `remoteIdentity`, which `TemporalVideoUpscaler.cacheIdentity`
/// needs synchronously from a non-main-actor context. One lock-protected string,
/// written on the main actor when capabilities change.
final class RemoteIdentityBox: @unchecked Sendable {
    private let lock = NSLock()
    private var storage = "unidentified"

    var value: String {
        get { lock.withLock { storage } }
        set { lock.withLock { storage = newValue } }
    }
}
