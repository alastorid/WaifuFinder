import Foundation

/// Keeps the decoded images of exactly one tagger batch alive. Scoring and tagging
/// share those images, and only compact frame records leave this stage.
struct TaggedFramePipeline: Sendable {
    let scorer: FrameScorePipeline
    let tagger: any TaggerBackend

    func run(frames: AsyncThrowingStream<SampledFrame, Error>,
             minimumFaceAreaFraction: Float) -> AsyncThrowingStream<AnalyzedFrame, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var iterator = frames.makeAsyncIterator()
                    while true {
                        try Task.checkCancellation()
                        var batch: [SampledFrame] = []
                        while batch.count < 8, let frame = try await iterator.next() {
                            batch.append(frame)
                        }
                        guard !batch.isEmpty else { break }
                        let input = AsyncThrowingStream<SampledFrame, Error> { inputContinuation in
                            for frame in batch { inputContinuation.yield(frame) }
                            inputContinuation.finish()
                        }
                        var scored: [AnalyzedFrame] = []
                        for try await frame in scorer.run(frames: input,
                                                         minimumFaceAreaFraction: minimumFaceAreaFraction,
                                                         onProgress: { _ in }) {
                            scored.append(frame)
                        }
                        try Task.checkCancellation()
                        let results = try await tagger.tag(batch: batch.map(\.image))
                        guard results.count == scored.count else {
                            throw ModelInstaller.fail("The tagger returned an unexpected frame count.")
                        }
                        for (var frame, result) in zip(scored, results) {
                            try Task.checkCancellation()
                            guard let encoded = result.encodedProbabilities ?? TagProbabilityCodec.encode(result.probabilities) else {
                                throw ModelInstaller.fail("The tagger returned invalid probabilities.")
                            }
                            frame.tagProbabilities = encoded
                            frame.tagRating = result.rating
                            frame.topTags = result.topTags
                            frame.tagGeneralScore = result.generalScore
                            continuation.yield(frame)
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
