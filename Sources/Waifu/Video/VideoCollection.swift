import Foundation

/// A logical timeline of original videos. No composition or output media is made.
struct VideoCollection: Sendable {
    struct Member: Sendable {
        let video: VideoInfo
        let offset: Double
    }
    let url: URL
    let members: [Member]

    // Reserve a source slot independently of estimated FPS/frame counts. A VFR
    // estimate can be wrong, and changing analysis mode must not change identity.
    static let indexStride = 1 << 32
    static func sourceIndex(_ sampleIndex: Int) -> Int { sampleIndex / indexStride }
    static func localIndex(_ sampleIndex: Int) -> Int { sampleIndex % indexStride }
    static func frameIndex(source: Int, local: Int) throws -> Int {
        guard source >= 0, source < Int.max / indexStride, local >= 0, local < indexStride else {
            throw MediaImport.failure("This video collection has too many frames.")
        }
        return source * indexStride + local
    }

    var info: VideoInfo {
        VideoInfo(url: url, durationSeconds: members.reduce(0) { $0 + $1.video.durationSeconds },
                  width: members.map { $0.video.width }.max() ?? 1,
                  height: members.map { $0.video.height }.max() ?? 1,
                  frameRate: members.map { $0.video.frameRate }.max() ?? 30,
                  estimatedFrameCount: members.reduce(0) { $0 + $1.video.estimatedFrameCount })
    }

    func member(at seconds: Double) -> Member {
        var low = 0, high = members.count
        while low < high {
            let middle = (low + high) / 2
            if members[middle].offset <= seconds { low = middle + 1 } else { high = middle }
        }
        return members[max(0, low - 1)]
    }
    func resolve(seconds: Double) -> (url: URL, seconds: Double) {
        let member = member(at: seconds)
        return (member.video.url, max(0, min(seconds - member.offset, member.video.durationSeconds - 0.000001)))
    }
    func member(for frame: FrameID) -> Member? {
        let index = Self.sourceIndex(frame.sampleIndex)
        return members.indices.contains(index) ? members[index] : nil
    }
    func resolve(frame: FrameID, seconds: Double?) -> (url: URL, seconds: Double)? {
        guard let member = member(for: frame) else { return nil }
        let local = seconds.map { $0 - member.offset }
            ?? Double(Self.localIndex(frame.sampleIndex)) / max(1, member.video.frameRate)
        return (member.video.url, max(0, min(local, member.video.durationSeconds - 0.000001)))
    }
    func estimatedCount(mode: AnalysisMode) -> Int {
        members.reduce(0) { $0 + mode.analysedFrameCount(frameCount: $1.video.estimatedFrameCount,
            frameRate: $1.video.frameRate, durationSeconds: $1.video.durationSeconds) }
    }

    private final class Registry: @unchecked Sendable {
        let lock = NSLock()
        var collections: [URL: VideoCollection] = [:]
    }
    private static let registry = Registry()
    static func get(_ url: URL) -> VideoCollection? { registry.lock.withLock { registry.collections[url] } }
    static func register(_ collection: VideoCollection) {
        registry.lock.withLock { registry.collections[collection.url] = collection }
    }
    static func release(_ url: URL) { _ = registry.lock.withLock { registry.collections.removeValue(forKey: url) } }
    static func resolve(_ url: URL, seconds: Double) -> (url: URL, seconds: Double) {
        get(url)?.resolve(seconds: seconds) ?? (url, seconds)
    }
}

extension MediaImport {
    /// Keep original sources and their normal import lifetimes; only probe metadata.
    static func prepare(_ sources: [URL]) async throws -> PreparedMedia {
        guard !sources.isEmpty else { throw failure("No videos were supplied.") }
        if sources.count == 1 { return try await prepare(sources[0]) }
        var inputs: [PreparedMedia] = []
        var members: [VideoCollection.Member] = []
        var offset: Double = 0
        for source in sources {
            try Task.checkCancellation()
            do {
                let input = try await prepare(source)
                let video = try await VideoProbe.probe(input.url)
                inputs.append(input)
                members.append(.init(video: video, offset: offset))
                offset += video.durationSeconds
            } catch {
                if Task.isCancelled { throw CancellationError() }
                throw failure("Could not read \(source.lastPathComponent): \(error.localizedDescription)")
            }
        }
        try Task.checkCancellation()
        let url = URL(string: "waifu-videos://\(UUID().uuidString)/\(sources.count)%20videos")!
        VideoCollection.register(VideoCollection(url: url, members: members))
        return PreparedMedia(url: url, inputs: inputs)
    }
}

/// Pull one original reader at a time. There is no intermediate file, resizing,
/// re-encoding, shared FPS or eager buffering across videos.
actor VideoCollectionSampler {
    private let collection: VideoCollection
    private let width: Int
    private let mode: AnalysisMode
    private let onProgress: @Sendable (Int) -> Void
    private var memberIndex = 0
    private var processed = 0
    private var iterator: AsyncThrowingStream<SampledFrame, Error>.Iterator?

    init(collection: VideoCollection, width: Int, mode: AnalysisMode, onProgress: @escaping @Sendable (Int) -> Void) {
        self.collection = collection; self.width = width; self.mode = mode; self.onProgress = onProgress
    }
    func next() async throws -> SampledFrame? {
        try Task.checkCancellation()
        while memberIndex < collection.members.count {
            let member = collection.members[memberIndex]
            if iterator == nil {
                iterator = AVFoundationFrameSampler().sample(member.video.url, analysisWidth: width,
                    mode: mode, onProgress: { _ in }).makeAsyncIterator()
            }
            var active = iterator!
            let frame = try await active.next(isolation: self)
            iterator = active
            if let frame {
                processed += 1
                onProgress(processed)
                return SampledFrame(sampleIndex: try VideoCollection.frameIndex(source: memberIndex, local: frame.sampleIndex),
                    timeSeconds: member.offset + frame.timeSeconds, image: frame.image, nativeImage: frame.nativeImage)
            }
            iterator = nil
            memberIndex += 1
        }
        return nil
    }
}
