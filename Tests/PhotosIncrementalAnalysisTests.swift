import CoreGraphics
import Foundation

func registerPhotosIncrementalAnalysisTests() {
    let suite = "Photos incremental analysis"
    Registry.shared.add(suite: suite, TestCase(name: "cloud-only first photo cannot block cached photos later in the album", knownBug: nil) {
        let tiny = try ImageFixture.cgImage(width: 64, height: 64)
        let log = PhotosAnalysisTestLog()
        let items = (0..<6).map {
            PhotosAlbumSampler.Item(url: URL(string: "waifu-photos://asset/\($0)")!, offset: Double($0), isStill: true)
        }
        let sampler = PhotosAlbumSampler(items: items, width: 512, mode: .full, allowNetwork: false,
            photoRendition: { url, _, network in
                let index = Int(url.lastPathComponent)!
                log.read(index, network: network)
                if index == 0 && !network { throw PhotosMedia.localUnavailable() }
                return tiny
            })
        for expected in 1..<6 {
            let frame = try await sampler.next()
            checkEqual(frame?.timeSeconds, Double(expected), "cached photos, even tiny ones, load before any cloud request")
            check(log.networkReads.isEmpty, "cloud cannot start during the local pass")
        }
        checkNil(try await sampler.next(), "local pass finishes without waiting for cloud")
        checkEqual(await sampler.unavailableItemCount, 1, "only the missing item needs a later recovery pass")
    })
    Registry.shared.add(suite: suite, TestCase(name: "manual refresh reuses moved photos and requests only new edited or unfinished photos", knownBug: nil) {
        func revision(_ id: String, _ modified: Double = 10) -> PhotosMedia.Revision {
            PhotosMedia.Revision(identifier: id, modified: modified, width: 100, height: 80, duration: 1)
        }
        let before = [revision("A"), revision("B"), revision("C"), revision("deleted"), revision("unfinished")]
        let after = [revision("new"), revision("C"), revision("B", 12), revision("A", 11), revision("unfinished")]
        var completed = (0..<4).map(photosAnalysisTestFrame)
        completed[0].topTags = ["keep tags"]
        let retained = PhotosAnalysisCheckpoint.reusableFrames(completed,
            before: before, offsetsBefore: [0, 1, 2, 3, 4], after: after, offsetsAfter: [0, 1, 2, 3, 4],
            keywordBaselines: ["A": 11, "B": 11])
        checkEqual(retained.map(\.timeSeconds), [1, 3], "moves and our keyword writes reuse scores; edits and deleted photos do not")
        checkEqual(retained.map(\.artScore), [2, 0], "scores follow asset identity rather than its old position")
        checkEqual(retained.last?.topTags, ["keep tags"], "tagging metadata survives a move")
        checkEqual(retained.last?.thumbnailKey, completed[0].thumbnailKey, "existing thumbnails remain reusable")
        let log = PhotosAnalysisTestLog()
        let image = try ImageFixture.cgImage(width: 16, height: 16)
        let items = after.indices.map { PhotosAlbumSampler.Item(url: URL(string: "waifu-photos://asset/\($0)")!, offset: Double($0), isStill: true) }
        let sampler = PhotosAlbumSampler(items: items, width: 512, mode: .full,
            completedIndices: Set(retained.map { $0.id.sampleIndex }), allowNetwork: false,
            photoRendition: { url, _, network in
                log.read(Int(url.lastPathComponent)!, network: network)
                return image
            })
        var requested: [Double] = []
        while let frame = try await sampler.next() { requested.append(frame.timeSeconds) }
        checkEqual(requested, [0, 2, 4], "only new, edited, and unfinished photos request pixels")
        let allCompleted = PhotosAnalysisCheckpoint.reusableFrames(completed,
            before: before, offsetsBefore: [0, 1, 2, 3, 4], after: before, offsetsAfter: [0, 1, 2, 3, 4], keywordBaselines: [:])
        checkEqual(allCompleted.count, completed.count, "an unchanged refresh retains every completed photo")
    })
    Registry.shared.add(suite: suite, TestCase(name: "real edits after a recorded keyword baseline invalidate completed scores", knownBug: nil) {
        let before = PhotosMedia.Revision(identifier: "A", modified: 11, width: 100, height: 80, duration: 1)
        let after = PhotosMedia.Revision(identifier: "A", modified: 12, width: 100, height: 80, duration: 1)
        check(!PhotosMedia.revisionsMatch([before], [after], keywordBaselines: ["A": 11]),
              "an old snapshot equal to the keyword baseline must not hide a newer real edit")
    })
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
    Registry.shared.add(suite: suite, TestCase(name: "retry snapshots advance without hiding retained photos", knownBug: nil) {
        var retained = [0, 4, 7].map(photosAnalysisTestFrame)
        retained[2].topTags = ["retained tag"]
        var prefix = [photosAnalysisTestFrame(0), photosAnalysisTestFrame(1)]
        prefix[0].groupID = FrameGroupID(raw: "updated-group")
        let first = PhotosAnalysisCheckpoint.includingRetainedTail(retained, after: prefix)
        checkEqual(first.map(\.timeSeconds), [0, 1, 4, 7], "newly loaded gaps appear before completion and the retained tail stays visible")
        checkEqual(first[0].groupID, prefix[0].groupID, "the grouped prefix replaces its retained copy")
        checkEqual(first.last, retained.last, "the unvisited tail retains all completed metadata")
        let secondPrefix = (0...4).map(photosAnalysisTestFrame)
        let second = PhotosAnalysisCheckpoint.includingRetainedTail(retained, after: secondPrefix)
        checkEqual(second.map(\.timeSeconds), [0, 1, 2, 3, 4, 7], "visible loaded count advances on the next snapshot")
        checkEqual(Set(second.map(\.id)).count, second.count, "retained photos are never duplicated")
        checkEqual(PhotosAnalysisCheckpoint.includingRetainedTail(retained, after: []), retained, "an empty prefix preserves existing results")
        checkEqual(PhotosAnalysisCheckpoint.includingRetainedTail([], after: secondPrefix), secondPrefix, "initial loads retain their normal prefix publication")
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
