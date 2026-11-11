import Darwin
import Foundation

enum CollectionVideoAnalysis {
    static let readerLimit: Int = {
        var size = 0
        guard sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0) == 0, size > 0 else { return 1 }
        var bytes = [CChar](repeating: 0, count: size)
        guard sysctlbyname("machdep.cpu.brand_string", &bytes, &size, nil, 0) == 0 else { return 1 }
        return readerLimit(for: String(decoding: bytes.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self))
    }()

    /// One reader per published general-purpose video decode engine. ProRes
    /// engines are separate. Unknown chips use the conservative single-reader path.
    static func readerLimit(for chip: String) -> Int {
        switch chip {
        case "Apple M1 Ultra", "Apple M2 Ultra", "Apple M3 Ultra", "Apple M5 Ultra": 2
        default: 1
        }
    }

    static func schedule(count: Int, limit: Int = readerLimit, analyze: @escaping @Sendable
                         (Int, @Sendable (AnalyzedFrame) async -> Void) async throws -> Void) -> FrameScorePipeline.Run {
        let (frames, output) = AsyncThrowingStream<AnalyzedFrame, Error>.makeStream()
        let task = Task {
            do {
                try await withThrowingTaskGroup(of: Void.self) { group in
                    var next = 0
                    func submit() {
                        let index = next
                        next += 1
                        group.addTask {
                            try Task.checkCancellation()
                            try await analyze(index) { output.yield($0) }
                        }
                    }
                    for _ in 0..<min(max(1, limit), count) { submit() }
                    while try await group.next() != nil {
                        try Task.checkCancellation()
                        if next < count { submit() }
                    }
                }
                output.finish()
            } catch {
                output.finish(throwing: error)
            }
        }
        output.onTermination = { _ in task.cancel() }
        return .init(frames: frames, task: task)
    }

    static func analyze(_ member: VideoCollection.Member, index: Int, width: Int,
                        pipeline: FrameScorePipeline, minimumFaceAreaFraction: Float,
                        onIssue: @escaping @Sendable (VideoProcessingIssue) async -> Void,
                        mode: AnalysisMode, timeRange: ClosedRange<Double>? = nil,
                        exclusions: Set<Int> = [], indexMap: FrameSampleIndexMap? = nil,
                        emit: (AnalyzedFrame) async -> Void) async throws {
        let source = AVFoundationFrameSampler().sample(member.video.url, analysisWidth: width,
            mode: mode, timeRange: timeRange, excludingSampleIndices: exclusions,
            sampleIndexMaps: [0: indexMap].compactMapValues { $0 }, onProgress: { _ in }, onIssue: { issue in
                await onIssue(VideoProcessingIssue(source: member.originalSource ?? issue.source,
                    stage: issue.stage, reason: issue.reason, framesRetained: issue.framesRetained))
            })
        let indexed = IndexedFrames(source: source, index: index, offset: member.offset)
        let run = pipeline.start(frames: AsyncThrowingStream(unfolding: { try await indexed.next() }),
                                 minimumFaceAreaFraction: minimumFaceAreaFraction, onProgress: { _ in })
        try await withTaskCancellationHandler {
            do {
                for try await frame in run.frames {
                    try Task.checkCancellation()
                    await emit(frame)
                }
                await run.task.value
                try Task.checkCancellation()
            } catch {
                run.task.cancel()
                await run.task.value
                throw error
            }
        } onCancel: {
            run.task.cancel()
        }
    }

    private actor IndexedFrames {
        private var iterator: AsyncThrowingStream<SampledFrame, Error>.Iterator
        private let index: Int
        private let offset: Double

        init(source: AsyncThrowingStream<SampledFrame, Error>, index: Int, offset: Double) {
            iterator = source.makeAsyncIterator()
            self.index = index
            self.offset = offset
        }

        func next() async throws -> SampledFrame? {
            var active = iterator
            let frame = try await active.next(isolation: self)
            iterator = active
            guard let frame else { return nil }
            return SampledFrame(sampleIndex: try VideoCollection.frameIndex(source: index, local: frame.sampleIndex),
                timeSeconds: offset + frame.timeSeconds, image: frame.image, nativeImage: frame.nativeImage)
        }
    }
}
