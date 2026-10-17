import CoreGraphics
import Foundation

// The frame-scoring pipeline's bookkeeping, with Vision taken out.
//
// `FrameScorePipeline` runs several scorers concurrently but must hand the caller
// frames in the sampler's order, one per input frame, and drive a monotonic
// progress count. None of that depends on what the scorer actually computes, so
// this suite substitutes a scorer that returns the sample index as the score: the
// pipeline's output order and completeness are then observable, and Vision is not
// needed to check them.
//
// The case that matters most is the first one, and it is a *regression* case. The
// pipeline tracks how many frames it has handed to the task group (`submitted`) and
// how many it has emitted in order (`nextToEmit`), and after the group drains it
// flushes the range `nextToEmit..<submitted`. `submitted` was incremented only while
// priming the group, never when the drain loop submitted further work, so for any
// input longer than the worker count `nextToEmit` grew past `submitted` and that
// range trapped with "Range requires lowerBound <= upperBound" — the crash observed
// on a real 2,805-frame video, whose stack pointed at exactly that line. The worker
// count defaults to 4, so every input of five frames or more could reach it; the
// case therefore uses more frames than the default and pins the count and the order.
//
// `workerCount` is passed explicitly rather than left to the default: the number of
// workers is what makes the input outrun `submitted`, so the case names it instead
// of inheriting it from the machine.

/// A scorer with no Vision in it: the aesthetics score is the sample index, so the
/// emitted order is readable from the frames themselves. It can be told to fail on
/// one frame so error propagation is exercised.
private struct IndexScorer: FrameScoring {
    enum Failure: Error { case refused }

    /// The sample index that should throw, or `nil` for a scorer that always
    /// succeeds.
    let failAt: Int?

    init(failAt: Int? = nil) { self.failAt = failAt }

    func score(_ frame: SampledFrame) async throws -> FrameScoreResult {
        // Yield so the workers interleave: the pipeline must survive results
        // arriving out of order, which is the whole reason it buffers.
        await Task.yield()
        if frame.sampleIndex == failAt { throw Failure.refused }
        return FrameScoreResult(
            artScore: Float(frame.sampleIndex),
            faceScore: nil,
            faceCount: 0,
            faceAreaFraction: nil,
            featurePrint: nil
        )
    }
}

/// A thumbnail writer that keeps nothing. The pipeline only needs to be told the
/// write succeeded; the cache is not what this suite is about.
private struct DiscardingThumbnailWriter: ThumbnailWriter {
    func write(_ image: CGImage, key: String) async throws {}
}

/// `count` frames in ascending sample order, each with a distinct image so no two
/// are interchangeable.
private func makeSampledFrames(_ count: Int) throws -> [SampledFrame] {
    try (0..<count).map { index in
        let gray = count > 1 ? CGFloat(index) / CGFloat(count - 1) : 0.5
        return SampledFrame(
            sampleIndex: index,
            timeSeconds: Double(index),
            image: try ImageFixture.cgImage(width: 16, height: 16, gray: gray)
        )
    }
}

/// The `AnalyzedFrame` identifiers the pipeline must emit, in order.
private func expectedIDs(_ count: Int) -> [String] {
    (0..<count).map { FrameID(sampleIndex: $0).raw }
}

private func stream(of frames: [SampledFrame]) -> AsyncThrowingStream<SampledFrame, Error> {
    AsyncThrowingStream { continuation in
        for frame in frames { continuation.yield(frame) }
        continuation.finish()
    }
}

func registerFrameScorePipelineTests() {
    let suite = "frame score pipeline"

    Registry.shared.add(suite: suite, TestCase(name: "portrait crop aspect is retained before thumbnail loading", knownBug: nil) {
        let image = try ImageFixture.cgImage(width: 140, height: 200, gray: 0.5)
        let sampled = SampledFrame(sampleIndex: 0, timeSeconds: 0, image: image)
        let pipeline = FrameScorePipeline(scorer: IndexScorer(), thumbnailStore: DiscardingThumbnailWriter(), analysisWidth: 200, workerCount: 1)
        var output: [AnalyzedFrame] = []
        for try await frame in pipeline.run(frames: stream(of: [sampled]), minimumFaceAreaFraction: 0.01, onProgress: { _ in }) {
            output.append(frame)
        }
        checkEqual(output.first?.thumbnailAspectRatio, 0.7, "grid receives the crop ratio without waiting for JPEG IO")
        let encoded = try JSONEncoder().encode(output[0])
        checkEqual(try JSONDecoder().decode(AnalyzedFrame.self, from: encoded).thumbnailAspectRatio, 0.7, "saved metadata keeps portrait geometry")
        var old = try JSONSerialization.jsonObject(with: encoded) as! [String: Any]
        old.removeValue(forKey: "thumbnailAspectRatio")
        let restored = try JSONDecoder().decode(AnalyzedFrame.self, from: JSONSerialization.data(withJSONObject: old))
        checkNil(restored.thumbnailAspectRatio, "old records can use the loaded image size")
    })

    Registry.shared.add(suite: suite, TestCase(name: "quick-mode gapped ordinals keep input order and actual timestamps", knownBug: nil) {
        let ordinals = [0, 30, 60, 90, 120, 150, 180]
        let image = try ImageFixture.cgImage(width: 16, height: 16, gray: 0.5)
        let frames = ordinals.enumerated().map { position, ordinal in
            SampledFrame(sampleIndex: ordinal, timeSeconds: Double(position) * 1.001, image: image)
        }
        let pipeline = FrameScorePipeline(scorer: IndexScorer(), thumbnailStore: DiscardingThumbnailWriter(),
                                          analysisWidth: 16, workerCount: 4)
        var output: [AnalyzedFrame] = []
        for try await frame in pipeline.run(frames: stream(of: frames), minimumFaceAreaFraction: 0.01, onProgress: { _ in }) {
            output.append(frame)
        }
        checkEqual(output.map(\.id.sampleIndex), ordinals, "gaps do not strand the emission queue")
        checkEqual(output.map(\.timeSeconds), frames.map(\.timeSeconds), "actual PTS survives gapped IDs")
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "a stream longer than the worker count emits every frame, in order",
        knownBug: nil) {
        // Twelve frames through four workers: eight more submissions than the
        // primer makes, which is exactly the window in which `submitted` used to
        // fall behind and the final flush range inverted.
        let frames = try makeSampledFrames(12)
        let pipeline = FrameScorePipeline(
            scorer: IndexScorer(),
            thumbnailStore: DiscardingThumbnailWriter(),
            analysisWidth: 16,
            workerCount: 4
        )

        var emitted: [AnalyzedFrame] = []
        let progress = TickLog()
        for try await frame in pipeline.run(
            frames: stream(of: frames),
            minimumFaceAreaFraction: 0.01,
            onProgress: { progress.record($0) }
        ) {
            emitted.append(frame)
        }

        checkEqual(emitted.map(\.id.raw), expectedIDs(12),
                   "every input frame is emitted exactly once, in sample order")
        checkEqual(emitted.map(\.timeSeconds), frames.map(\.timeSeconds),
                   "and the timestamps travel with them")
        checkEqual(progress.all, Array(1...12),
                   "progress is one tick per emitted frame, in order")
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "a single worker emits a short stream in order", knownBug: nil) {
        // The degenerate concurrency: one worker means results arrive in order
        // anyway, so this pins that the bookkeeping does not depend on that.
        let frames = try makeSampledFrames(5)
        let pipeline = FrameScorePipeline(
            scorer: IndexScorer(),
            thumbnailStore: DiscardingThumbnailWriter(),
            analysisWidth: 16,
            workerCount: 1
        )

        var emitted: [AnalyzedFrame] = []
        for try await frame in pipeline.run(
            frames: stream(of: frames),
            minimumFaceAreaFraction: 0.01,
            onProgress: { _ in }
        ) {
            emitted.append(frame)
        }

        checkEqual(emitted.map(\.id.raw), expectedIDs(5), "every frame, in order, with one worker")
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "a scorer failure fails the stream instead of hanging it", knownBug: nil) {
        // Error propagation is part of the contract the UI relies on: a failed
        // frame surfaces as a thrown stream error, not as a silently short result
        // and not as a stream that never finishes. The failure is mid-stream, so at
        // least one frame has already been yielded and buffered.
        let frames = try makeSampledFrames(12)
        let pipeline = FrameScorePipeline(
            scorer: IndexScorer(failAt: 3),
            thumbnailStore: DiscardingThumbnailWriter(),
            analysisWidth: 16,
            workerCount: 4
        )

        let thrown = await checkThrowsAsync("a failing scorer must fail the stream") {
            for try await _ in pipeline.run(
                frames: stream(of: frames),
                minimumFaceAreaFraction: 0.01,
                onProgress: { _ in }
            ) {}
        }
        checkNotNil(thrown, "the stream throws rather than finishing short")
        if let failure = thrown as? IndexScorer.Failure {
            checkEqual(failure, .refused, "and the error is the scorer's own")
        } else {
            check(false, "expected IndexScorer.Failure.refused, got \(String(describing: thrown))")
        }
    })
}
