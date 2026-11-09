import Foundation

struct VideoProcessingIssue: Identifiable, Sendable {
    let id = UUID()
    let source: URL
    let stage: String
    let reason: String
    let framesRetained: Int

    var sourceDescription: String { source.isFileURL ? source.path : source.absoluteString }
    var outcome: String {
        framesRetained == 0 ? "Not processed" : "Partially processed · \(framesRetained) decoded frames retained; remaining frames skipped"
    }
    var logText: String { "\(sourceDescription)\n\(stage): \(reason)\n\(outcome)" }
}

/// Convert only source-reading failures to end-of-input. Scoring and grouping
/// errors remain fatal, and cancellation never becomes a skipped-file warning.
actor RecoveringVideoReader {
    private var iterator: AsyncThrowingStream<SampledFrame, Error>.Iterator
    private let source: URL
    private let onIssue: @Sendable (VideoProcessingIssue) async -> Void
    private var count = 0
    private var finished = false

    init(frames: AsyncThrowingStream<SampledFrame, Error>, source: URL,
         onIssue: @escaping @Sendable (VideoProcessingIssue) async -> Void) {
        iterator = frames.makeAsyncIterator()
        self.source = source
        self.onIssue = onIssue
    }

    func next() async throws -> SampledFrame? {
        try Task.checkCancellation()
        guard !finished else { return nil }
        do {
            var active = iterator
            let frame = try await active.next(isolation: self)
            iterator = active
            try Task.checkCancellation()
            if frame != nil { count += 1 } else { finished = true }
            return frame
        } catch {
            if Task.isCancelled || error is CancellationError || (error as? FrameSamplingError) == .cancelled {
                throw CancellationError()
            }
            finished = true
            await onIssue(VideoProcessingIssue(source: source, stage: "Decoding video",
                reason: error.localizedDescription, framesRetained: count))
            return nil
        }
    }
}
