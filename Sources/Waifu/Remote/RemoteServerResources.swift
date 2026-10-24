import Foundation
import CoreGraphics
import ImageIO

/// Where a serving WaifuFinder gets its models.
///
/// A value of closures rather than a direct read of `ModelLibrary`, for the same
/// reason `UncoverSession` takes a `backendFactory`: the wire protocol is worth
/// testing end to end, and it cannot be tested end to end behind a 225 MB
/// download. `.local` is the production path and is the only thing the app ever
/// constructs.
struct RemoteModelCatalog: Sendable {
    /// Resolves a tagger for a batch of rows.
    var tagger: @Sendable () async throws -> any TaggerBackend
    var backend: @Sendable (UncoverAlgorithm) -> any TemporalVideoUpscaler
    /// The labels CSV a client needs in order to name any tag at all.
    var vocabularyCSV: @Sendable () async throws -> Data
    /// The vocabulary fingerprint, without shipping the CSV.
    var vocabularyFingerprint: @Sendable () async throws -> UInt64
    /// What this Mac can currently do.
    var capabilities: @Sendable (String, String, String?) async -> RemoteCapabilities
    /// A cancelled native worker cannot be reused on the next connection.
    var releaseTagger: @Sendable () async -> Void = {}

    /// The same native backend is used by Photos and serving workers. Keeping
    /// this seam in the catalog allows real-model loopback tests without touching
    /// the user's configured model directory.
    var safire: @Sendable (RemoteSafireRequest) async throws -> SafireSpatialResult = { request in
        guard let config=SafireConfiguration.load(),config.fingerprint==request.modelFingerprint else {
            throw RemoteError.modelUnavailable("SAFIRE checkpoint fingerprint differs from the requested revision.")
        }
        return try await SafireBackend.shared.infer(png:request.png,photo:request.photo)
    }

    var trufor:@Sendable (RemoteTruForRequest) async throws -> TruForJobResult = { request in
        guard TruForConfiguration.load()?.fingerprint==request.fingerprint else { throw RemoteError.modelUnavailable("Install matching local TruFor weights on the server") }
        return try await TruForBackend.shared.submit(images:request.images,evidence:request.evidence)
    }

    static let local = RemoteModelCatalog(
        tagger: { try await RemoteServerResources.shared.tagger() },
        backend: { $0.makeBackend() },
        vocabularyCSV: {
            guard let configuration = WDTaggerConfiguration.load() else {
                throw RemoteError.modelUnavailable("Install WD Tagger in Settings → Models on this server.")
            }
            return try Data(contentsOf: configuration.labelsURL)
        },
        vocabularyFingerprint: { try await RemoteServerResources.shared.vocabularyFingerprint() },
        capabilities: { appVersion, serverName, sessionID in
            await RemoteCapabilities.current(appVersion: appVersion, serverName: serverName,
                                             sessionID: sessionID)
        },
        releaseTagger: { await RemoteServerResources.shared.releaseTagger() })
}

/// Everything a serving WaifuFinder holds on behalf of a remote client.
///
/// Kept apart from the session so a tag session and a reconstruction can be
/// awaited and cancelled independently, and so model references are released in
/// one place. Both live behind the same reference counting the Settings tab
/// already uses, so a model in remote use shows **In use** and cannot be removed
/// from under a running job.
actor RemoteServerResources {
    static let shared = RemoteServerResources()

    private var tagSession: RemoteTagSession?
    private var jobs: [UInt32: Task<Void, Never>] = [:]
    private var reserved: Set<UInt32> = []
    private var cancelled: Set<UInt32> = []
    private var lastUsed = Date()
    private var cachedFingerprint: UInt64?
    private var idleTask: Task<Void, Never>?

    /// How long the tagger is kept warm before being released.
    private static let idleSeconds: TimeInterval = 60

    // MARK: Tagging

    /// The tagger backend, held warm.
    ///
    /// The 225 MB safetensors load is far too expensive to repeat per
    /// eight-frame batch, so it is created on first use and dropped after an idle
    /// window — or immediately when the model set changes underneath it.
    ///
    /// Constructed with a neutral threshold because a remote server ships raw
    /// probability rows: the threshold is the client's preference, applied there.
    func tagger() async throws -> WDTaggerBackend {
        lastUsed = Date()
        idleTask?.cancel()
        idleTask = Task { [weak self] in
            do { try await Task.sleep(for: .seconds(Self.idleSeconds + 1)) } catch { return }
            await self?.releaseIfIdle()
        }
        if let existing = tagSession { return existing.model }
        guard let configuration = WDTaggerConfiguration.load() else {
            throw RemoteError.modelUnavailable("Install WD Tagger in Settings → Models on this server.")
        }
        await ModelLibrary.shared.acquireTagger()
        if let existing = tagSession {
            await ModelLibrary.shared.releaseTagger()
            return existing.model
        }
        do {
            let backend = try WDTaggerBackend(directory: configuration.directory, generalThreshold: 0.35)
            tagSession = RemoteTagSession(model: backend)
            return backend
        } catch {
            await ModelLibrary.shared.releaseTagger()
            throw RemoteError.modelUnavailable(error.localizedDescription)
        }
    }

    /// The vocabulary fingerprint, read from the labels once and remembered.
    ///
    /// Computing it walks 10 861 label rows, which is not something to redo for
    /// every batch just so the client can confirm it has the right CSV.
    func vocabularyFingerprint() async throws -> UInt64 {
        if let cachedFingerprint { return cachedFingerprint }
        guard let configuration = WDTaggerConfiguration.load() else {
            throw RemoteError.modelUnavailable("Install WD Tagger in Settings → Models on this server.")
        }
        let value = try TagVocabulary.load(from: configuration.labelsURL).cacheFingerprint
        cachedFingerprint = value
        return value
    }

    func touch() { lastUsed = Date() }

    /// Drops the warm tagger once it has been unused long enough to be worth the
    /// memory, and always when the model set changed.
    func releaseIfIdle(now: Date = Date()) async {
        guard tagSession != nil, now.timeIntervalSince(lastUsed) > Self.idleSeconds else { return }
        await releaseTagSession()
    }

    func modelsChanged() async {
        cachedFingerprint = nil
        await releaseTagSession()
    }

    private func releaseTagSession() async {
        guard tagSession != nil else { return }
        tagSession = nil
        await ModelLibrary.shared.releaseTagger()
    }

    func releaseTagger() async {
        idleTask?.cancel()
        idleTask = nil
        await releaseTagSession()
    }

    // MARK: Jobs

    /// Records that a stream is in flight, so a `cancelJob` arriving before the
    /// work starts is still honoured.
    func reserve(streamID: UInt32) throws {
        guard reserved.count < RemoteLimits.maximumOutstandingJobs, !reserved.contains(streamID) else { throw RemoteError.busy }
        cancelled.remove(streamID)
        reserved.insert(streamID)
        jobs[streamID] = nil
    }

    /// Runs a job and drops it when it ends.
    func run(streamID: UInt32, work: @escaping @Sendable () async -> Void) async {
        jobs[streamID]?.cancel()
        let job = Task { await work() }
        jobs[streamID] = job
        if cancelled.contains(streamID) { job.cancel() }
        await job.value
    }

    func finish(streamID: UInt32) {
        jobs[streamID] = nil
        reserved.remove(streamID)
        cancelled.remove(streamID)
    }

    func cancelJob(_ streamID: UInt32) {
        if reserved.contains(streamID) { cancelled.insert(streamID) }
        jobs[streamID]?.cancel()
        jobs[streamID] = nil
    }

    func releaseAll() async {
        for job in jobs.values { job.cancel() }
        jobs.removeAll()
        reserved.removeAll()
        cancelled.removeAll()
        await releaseTagSession()
    }
}

/// Keeps the warm tagger alive across requests.
actor RemoteTagSession {
    let model: WDTaggerBackend
    init(model: WDTaggerBackend) { self.model = model }
}

/// The images a tagging batch can carry.
///
/// Two codecs, and the choice is per frame rather than per batch: a video frame
/// already has a JPEG in the thumbnail cache and that JPEG is what travels,
/// while a Photos rendition is already at model size and is handed over as the
/// pixels the model would have received anyway. Neither path re-encodes.
enum RemoteTagImageCodec: UInt8 {
    /// A JPEG or PNG container, decoded by the server at analysis resolution.
    case encoded
    /// Raw RGBA8 at exactly `WDTaggerBackend.imageSize` square.
    case rgba8

    var name: String { self == .encoded ? "encoded" : "rgba8" }
}

/// Decodes one received image to a `CGImage` without an intermediate file.
enum RemoteTagImageDecoder {
    static func decode(codec: RemoteTagImageCodec, bytes: [UInt8]) throws -> CGImage {
        switch codec {
        case .encoded:
            return try WDTaggerBackend.thumbnailImage(data: Data(bytes))
        case .rgba8:
            let side = WDTaggerBackend.imageSize
            guard bytes.count == side * side * 4 else {
                throw TaggerError.preprocessing
            }
            let context = CGContext(data: nil, width: side, height: side, bitsPerComponent: 8,
                                    bytesPerRow: side * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            guard let context else { throw TaggerError.preprocessing }
            // Copied into the context's own buffer: a `CGImage` does not retain
            // the byte array it was drawn from, and the receive buffer is about
            // to be released.
            var mutable = bytes
            return try mutable.withUnsafeMutableBytes { source in
                guard let base = source.baseAddress, let target = context.data else {
                    throw TaggerError.preprocessing
                }
                target.copyMemory(from: base, byteCount: side * side * 4)
                guard let image = context.makeImage() else { throw TaggerError.preprocessing }
                return image
            }
        }
    }

    /// Renders a `CGImage` to the model's exact input, for the path where the
    /// client already holds model-size pixels.
    static func encodeRGBA8(_ image: CGImage) throws -> [UInt8] {
        try WDTaggerBackend.preprocessRGBA(image)
    }
}
