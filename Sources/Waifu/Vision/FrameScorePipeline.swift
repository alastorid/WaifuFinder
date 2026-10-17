import CoreGraphics
import Foundation

/// A bounded-concurrency pipeline that runs a `FrameScoring` implementation over a
/// stream of `SampledFrame`s, emitting `AnalyzedFrame`s in **deterministic input
/// order** (not completion order), with progress callback and cancellation.
///
/// ## Why deterministic order
///
/// The sampling layer produces frames in ascending timestamp order. The UI and
/// downstream layers (grouping, ranking) expect `AnalyzedFrame`s in that same
/// order — `FrameID` is derived from `sampleIndex`, and `frames` in
/// `WaifuSession` is documented as "always sorted by `timeSeconds` ascending".
/// If the pipeline emitted in completion order, the session would have to re-sort,
/// which is both wasteful and a source of subtle bugs when two frames have the
/// same timestamp (the sampler guarantees unique timestamps, but the invariant
/// is cheaper to keep than to prove). A quick pass hands over indices with
/// gaps (0, 30, 60…), so the pipeline's order is the order frames were
/// submitted — a queue — not the arithmetic of consecutive indices.
///
/// Emitting in order means a slow frame blocks later frames from being delivered,
/// but the three Vision requests per frame run concurrently *within* the frame,
/// and the worker count is chosen so the head-of-line effect is small in practice.
///
/// ## Worker count and memory
///
/// The default is up to four scoring workers. The app chooses a source-size-aware
/// count with FrameProcessingConcurrency, allowing for native-resolution images
/// and the sampler's preparation window. Scoring and JPEG encoding overlap with
/// decoding and raster preparation; native frames are retained only while those
/// bounded stages need them.
///
/// This pipeline pulls another input only when a scoring slot is free. Together
/// with the sampler's bounded preparation window, image residency depends on
/// concurrency and source resolution rather than video duration. The output
/// stream contains compact analysis records, never decoded pixels.
///
/// Two consequences worth stating, because both are easy to break by accident:
///
///   * An input that is *already* materialised in an array is legal and scores
///     correctly, but gives up the bound — it holds the whole file's images at once.
///     The session feeds the sampler's stream in directly for this reason.
///   * Because one pulled frame is one scored frame, `onProgress`'s count is also
///     the number of frames the sampler produced; the session reports it as the
///     sampling *and* scoring progress rather than minting a second, equal number.
final class FrameScorePipeline: Sendable {
    /// Number of concurrent scoring workers.
    /// The session supplies its native-frame memory estimate; other callers
    /// default to at most four workers.
    private let workerCount: Int

    /// The scorer to use for each frame. Injected so tests can supply a fake.
    private let scorer: FrameScoring

    /// Store for retaining analysis-resolution thumbnails in memory.
    private let thumbnailStore: ThumbnailWriter

    /// Width the thumbnails are encoded at (the analysis width).
    private let analysisWidth: Int
    private let cacheSuffix: String
    /// Compressed analysis payloads keyed by decoded sample index. A payload is
    /// decompressed only when its frame reaches a scoring worker.
    private let cachedResults: [Int: Data]

    /// Creates a pipeline with the given scorer and thumbnail store.
    ///
    /// - Parameters:
    ///   - scorer: The `FrameScoring` implementation (typically `VisionFrameScorer`).
    ///   - thumbnailStore: Where analysis-resolution thumbnails are retained.
    ///   - analysisWidth: The width frames were decoded at; used for the thumbnail
    ///     key suffix so a change of analysis width invalidates old thumbnails.
    ///   - workerCount: Override for testing. Defaults to 4 (see class docs).
    init(
        scorer: FrameScoring,
        thumbnailStore: ThumbnailWriter,
        analysisWidth: Int,
        workerCount: Int? = nil,
        cacheSuffix: String = "",
        cachedResults: [Int: Data] = [:]
    ) {
        self.scorer = scorer
        self.thumbnailStore = thumbnailStore
        self.analysisWidth = analysisWidth
        self.cacheSuffix = cacheSuffix
        self.cachedResults = cachedResults
        self.workerCount = workerCount ?? min(4, ProcessInfo.processInfo.activeProcessorCount)
    }

    /// Runs the pipeline over the given frames.
    ///
    /// - Parameters:
    ///   - frames: An `AsyncThrowingStream` of `SampledFrame` in ascending
    ///     `timeSeconds` order (the sampling layer guarantees this).
    ///   - minimumFaceAreaFraction: Passed through to the scorer (already set on
    ///     `VisionFrameScorer`; kept here for progress reporting context).
    ///   - onProgress: Called with the number of frames *completed and emitted*,
    ///     so a progress bar advances in lockstep with the visible results.
    /// - Returns: An `AsyncThrowingStream` of `AnalyzedFrame` in the same order
    ///   as the input frames.
    func run(
        frames: AsyncThrowingStream<SampledFrame, Error>,
        minimumFaceAreaFraction: Float,
        onProgress: @escaping @Sendable (Int) -> Void
    ) -> AsyncThrowingStream<AnalyzedFrame, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var buffer: [Int: AnalyzedFrame] = [:] // sampleIndex → frame
                    var submitted: [Int] = []  // every submitted index, in order
                    var emitted = 0             // how many have been handed on
                    var completed = 0

                    // Process frames with bounded concurrency using a task group.
                    // We submit up to `workerCount` frames at a time, then wait
                    // for the next one to complete before submitting another.
                    // This keeps at most `workerCount` frames in flight.
                    try await withThrowingTaskGroup(of: (Int, AnalyzedFrame).self) { group in
                        var frameIterator = frames.makeAsyncIterator()

                        // Prime the group with up to workerCount frames.
                        while submitted.count < workerCount {
                            guard let frame = try await frameIterator.next() else { break }
                            let sampleIndex = frame.sampleIndex
                            submitted.append(sampleIndex)
                            group.addTask {
                                let result = try await self.scoreAndStore(frame: frame)
                                return (sampleIndex, result)
                            }
                        }

                        // Drain results, emitting in order, and submit new work.
                        while let (sampleIndex, analyzedFrame) = try await group.next() {
                            // Check cancellation between frames.
                            if Task.isCancelled { throw CancellationError() }

                            buffer[sampleIndex] = analyzedFrame

                            // Emit any frames that are now next in order. The
                            // submission queue *is* the order: a quick pass's
                            // indices have gaps, so the next frame to emit is
                            // the queue's head, not the last index plus one.
                            // Counting every submission, primed or not, is what
                            // keeps the final flush's bound honest.
                            while emitted < submitted.count, let frame = buffer.removeValue(forKey: submitted[emitted]) {
                                continuation.yield(frame)
                                emitted += 1
                                completed += 1
                                onProgress(completed)
                            }

                            // Submit the next frame if any remain.
                            if let nextFrame = try await frameIterator.next() {
                                let nextSampleIndex = nextFrame.sampleIndex
                                submitted.append(nextSampleIndex)
                                group.addTask {
                                    let result = try await self.scoreAndStore(frame: nextFrame)
                                    return (nextSampleIndex, result)
                                }
                            }
                        }

                        // After the group finishes, emit any remaining buffered
                        // frames — the tail of the submission queue whose
                        // results landed after the last drain.
                        while emitted < submitted.count, let frame = buffer.removeValue(forKey: submitted[emitted]) {
                            continuation.yield(frame)
                            emitted += 1
                            completed += 1
                            onProgress(completed)
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

    /// Scores one frame and retains its thumbnail in memory, returning the `AnalyzedFrame`.
    private func scoreAndStore(frame: SampledFrame) async throws -> AnalyzedFrame {
        let result: FrameScoreResult
        if let encoded = cachedResults[frame.sampleIndex],
           let cached = AnalysisResultCacheCodec.decode(encoded) {
            result = cached
        } else {
            result = try await scorer.score(frame)
        }

        // Store the thumbnail and get its key. The store uses the sampleIndex-based
        // key we pass; the suffix includes analysisWidth so changing the width
        // invalidates old thumbnails automatically.
        let thumbnailKey = "thumb-\(FrameID(sampleIndex: frame.sampleIndex).raw)-\(analysisWidth)\(cacheSuffix)"
        try await thumbnailStore.write(
            frame.image,
            key: thumbnailKey
        )

        return AnalyzedFrame(
            sampleIndex: frame.sampleIndex,
            timeSeconds: frame.timeSeconds,
            artScore: result.artScore,
            faceScore: result.faceScore,
            faceCount: result.faceCount,
            faceAreaFraction: result.faceAreaFraction,
            featurePrint: result.featurePrint,
            thumbnailKey: thumbnailKey,
            faceDefinitionScore: result.faceDefinitionScore,
            thumbnailAspectRatio: Double(frame.image.width) / Double(frame.image.height)
        )
    }
}

/// Protocol for writing analysis-resolution thumbnails to a cache.
///
/// The pipeline does not know the cache layout; it only knows keys. This keeps
/// the pipeline testable with an in-memory store and lets production retain
/// thumbnails without coupling scoring to storage details.
///
/// Named `ThumbnailWriter` to avoid collision with the export `ThumbnailStore` protocol
/// (which uses `NSImage` and a different API). The engine layer bridges between
/// the pipeline's `CGImage` and the store's `NSImage`.
protocol ThumbnailWriter: Sendable {
    /// Writes `image` at `key`. The image is the analysis-resolution decode
    /// (width = `analysisWidth`, height aspect-preserved).
    func write(_ image: CGImage, key: String) async throws
}
