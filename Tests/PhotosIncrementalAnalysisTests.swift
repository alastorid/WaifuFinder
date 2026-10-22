import CoreGraphics
import Foundation

func registerPhotosIncrementalAnalysisTests() {
    let suite = "Photos incremental analysis"
    Registry.shared.add(suite: suite, TestCase(name: "enabling downloads reads and scores only unfinished photos", knownBug: nil) {
        let image = try ImageFixture.cgImage(width: 16, height: 16)
        let log = PhotosAnalysisTestLog()
        let checkpoint = PhotosAnalysisCheckpoint(sourceURL: URL(string: "waifu-photos://album/retry")!, settings: .default)
        let items = (0..<9).map { index in
            PhotosAlbumSampler.Item(url: URL(string: "waifu-photos://asset/\(index)")!, offset: Double(index), isStill: true)
        }
        let missing: Set<Int> = [0, 3, 7]
        let load: @Sendable (URL, Int, Bool) async throws -> CGImage = { url, _, network in
            let index = Int(url.lastPathComponent)!
            log.read(index, network: network)
            if !network && missing.contains(index) { throw PhotosMedia.localUnavailable() }
            return image
        }
        let first = PhotosAlbumSampler(items: items, width: 512, mode: .full, allowNetwork: false, photoRendition: load)
        let pipeline = FrameScorePipeline(scorer: PhotosAnalysisTestScorer(log: log),
                                          thumbnailStore: PhotosAnalysisTestWriter(), analysisWidth: 512, workerCount: 2)
        let initialStream = AsyncThrowingStream<SampledFrame, Error>(unfolding: { try await first.next() })
        let initialRun = pipeline.start(frames: initialStream, minimumFaceAreaFraction: 0.01,
                                        onProgress: { _ in }, onFrame: { await checkpoint.retain($0) })
        // Complete without consuming the UI stream: even buffered scores must
        // survive a switch change before the grid has shown its first batch.
        await initialRun.task.value
        checkEqual(await first.unavailableItemCount, 3, "the local pass reports only its own missing photos")
        var retained = await checkpoint.frames()
        checkEqual(retained.count, 6, "available scores are checkpointed before any grid publication")
        retained[0].topTags = ["kept tag"]
        await checkpoint.retain(retained[0])
        let retry = PhotosAlbumSampler(items: items, width: 512, mode: .full,
                                      completedIndices: Set(retained.map { $0.id.sampleIndex }),
                                      allowNetwork: true, photoRendition: load)
        let retryStream = AsyncThrowingStream<SampledFrame, Error>(unfolding: { try await retry.next() })
        let retryRun = pipeline.start(frames: retryStream, minimumFaceAreaFraction: 0.01,
                                      onProgress: { _ in }, onFrame: { await checkpoint.retain($0) })
        var combined: [AnalyzedFrame] = []
        for try await frame in PhotosAnalysisCheckpoint.merging(retained, with: retryRun.frames) {
            combined.append(frame)
        }
        checkEqual(combined.map(\.timeSeconds), (0..<9).map(Double.init), "earlier missing photos return to chronological order")
        checkEqual(log.networkReads.sorted(), missing.sorted(), "completed photos are never requested again")
        checkEqual(log.scores.sorted(), (0..<9).map { $0 * 1_000_000 }, "every photo is scored exactly once across both passes")
        checkEqual(combined[1].topTags, ["kept tag"], "retained tagging metadata survives the merge")
        checkEqual(combined[1].thumbnailKey, retained[0].thumbnailKey, "existing thumbnail identities remain stable")
        checkEqual(await retry.unavailableItemCount, 0, "a retry does not inherit the previous pass's skipped count")
        checkEqual(await checkpoint.frames(), combined, "the checkpoint holds the complete merged result")
    })
    Registry.shared.add(suite: suite, TestCase(name: "cancelling a download cancels the entire prefetched PhotoKit window", knownBug: nil) {
        let log = PhotosAnalysisTestLog()
        let starts = AsyncStream<Int>.makeStream()
        let items = (0..<10).map { index in
            PhotosAlbumSampler.Item(url: URL(string: "waifu-photos://asset/\(index)")!, offset: Double(index), isStill: true)
        }
        let completed = Set([1, 2, 4, 5, 7, 8].map { $0 * 1_000_000 })
        let sampler = PhotosAlbumSampler(items: items, width: 512, mode: .full,
            completedIndices: completed, allowNetwork: true,
            photoRendition: { url, _, _ in
                let index = Int(url.lastPathComponent)!
                let request = PhotosRequest<CGImage>(cancel: { id in log.cancel(Int(id)) })
                return try await request.perform { _ in
                    starts.continuation.yield(index)
                    return Int32(index)
                }
            })
        let task = Task { try await sampler.next() }
        var iterator = starts.stream.makeAsyncIterator()
        for _ in 0..<4 { _ = await iterator.next() }
        task.cancel()
        do {
            _ = try await task.value
            check(false, "cancellation must escape the sampler")
        } catch is CancellationError { }
        starts.continuation.finish()
        checkEqual(log.cancelled.sorted(), [0, 3, 6, 9], "all four sparse missing-photo requests overlap and are cancelled")
        checkEqual(await sampler.unavailableItemCount, 0, "cancellation is never reported as a missing local photo")
    })
    Registry.shared.add(suite: suite, TestCase(name: "an interrupted gap fill retains the tail and retries without duplicates", knownBug: nil) {
        let checkpoint = PhotosAnalysisCheckpoint(sourceURL: URL(string: "waifu-photos://album/interrupted")!, settings: .default)
        for index in [0, 4, 7] { await checkpoint.retain(photosAnalysisTestFrame(index)) }
        let interrupted = AsyncThrowingStream<AnalyzedFrame, Error> { continuation in
            continuation.yield(photosAnalysisTestFrame(1))
            continuation.finish(throwing: PhotosMedia.failure("Download interrupted"))
        }
        do {
            for try await frame in PhotosAnalysisCheckpoint.merging(await checkpoint.frames(), with: interrupted) {
                await checkpoint.retain(frame)
            }
            check(false, "download errors reach the caller")
        } catch { }
        let retained = await checkpoint.frames()
        checkEqual(retained.map(\.timeSeconds), [0, 1, 4, 7], "tail scores survive failure before the merge reaches them")
        let incoming = AsyncThrowingStream<AnalyzedFrame, Error> { continuation in
            for index in [2, 3, 5, 6] { continuation.yield(photosAnalysisTestFrame(index)) }
            continuation.finish()
        }
        let grouping = IncrementalFrameGrouper(settings: WaifuSettings.default.grouping, distance: { _, _ in 0 })
        var completed: [AnalyzedFrame] = []
        for try await frame in PhotosAnalysisCheckpoint.merging(retained, with: incoming) {
            try grouping.append(frame)
            completed.append(frame)
        }
        checkEqual(completed.map(\.timeSeconds), (0..<8).map(Double.init), "retries emit each retained and new score exactly once")
        let groups = try grouping.finish()
        checkEqual(groups.count, 1, "grouping includes old scores on both sides of the filled gaps")
        checkEqual(groups.first?.memberIDs.count, 8, "all photos participate in the updated group")
    })
}

private func photosAnalysisTestFrame(_ index: Int) -> AnalyzedFrame {
    AnalyzedFrame(sampleIndex: index * 1_000_000, timeSeconds: Double(index), artScore: Float(index),
                  faceScore: nil, faceCount: 0, faceAreaFraction: nil, featurePrint: Data([1]))
}

private final class PhotosAnalysisTestLog: @unchecked Sendable {
    private let lock = NSLock()
    private var reads: [(Int, Bool)] = []
    private var scored: [Int] = []
    private var cancellations: [Int] = []
    func read(_ index: Int, network: Bool) { lock.withLock { reads.append((index, network)) } }
    func score(_ index: Int) { lock.withLock { scored.append(index) } }
    func cancel(_ index: Int) { lock.withLock { cancellations.append(index) } }
    var networkReads: [Int] { lock.withLock { reads.filter(\.1).map(\.0) } }
    var scores: [Int] { lock.withLock { scored } }
    var cancelled: [Int] { lock.withLock { cancellations } }
}

private struct PhotosAnalysisTestScorer: FrameScoring {
    let log: PhotosAnalysisTestLog
    func score(_ frame: SampledFrame) async throws -> FrameScoreResult {
        log.score(frame.sampleIndex)
        return FrameScoreResult(artScore: Float(frame.sampleIndex), faceScore: nil,
                                faceCount: 0, faceAreaFraction: nil, featurePrint: nil)
    }
}

private struct PhotosAnalysisTestWriter: ThumbnailWriter {
    func write(_ image: CGImage, key: String) async throws { }
}
