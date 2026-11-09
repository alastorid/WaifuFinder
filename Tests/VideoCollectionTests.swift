import AVFoundation
import CoreGraphics
import Foundation

private func collectionFrames(_ url: URL, mode: AnalysisMode = .full) async throws -> [SampledFrame] {
    var frames: [SampledFrame] = []
    for try await frame in AVFoundationFrameSampler().sample(url, analysisWidth: 96, mode: mode, onProgress: { _ in }) {
        frames.append(frame)
    }
    return frames
}

private func collectionFrame(index: Int, time: Double, score: Float = 0.75) -> AnalyzedFrame {
    AnalyzedFrame(sampleIndex: index, timeSeconds: time, artScore: score, faceScore: nil,
                  faceCount: 0, faceAreaFraction: nil, featurePrint: nil)
}

private actor VideoIssueRecorder {
    var issues: [VideoProcessingIssue] = []
    func append(_ issue: VideoProcessingIssue) { issues.append(issue) }
}

func registerVideoCollectionTests() {
    let suite = "video collection"
    Registry.shared.add(suite: suite, TestCase(name: "original videos share a timeline without creating media or resampling", knownBug: nil) {
        let directory = try Temporary.directory("video-collection")
        let first = try VideoFixture.makeMP4(in: directory, durationSeconds: 1, width: 64, height: 48, frameRate: 4)
        let second = try VideoFixture.makeMP4(in: directory, durationSeconds: 2, width: 96, height: 64, frameRate: 7)
        let filesBefore = try FileManager.default.contentsOfDirectory(atPath: directory.path).sorted()
        var prepared: PreparedMedia? = try await MediaImport.prepare([first, second])
        let url = prepared!.url
        checkNil(prepared!.directory, "no intermediate media directory exists")
        check(!url.isFileURL, "workspace is a logical identifier, not a media file")
        let collection = VideoCollection.get(url)!
        checkEqual(collection.members.map { $0.video.url }, [first, second], "source order and files are retained")
        let info = try await VideoProbe.probe(url)
        check(abs(info.durationSeconds - 3) < 0.001, "workspace includes both durations")
        let frames = try await collectionFrames(url)
        checkEqual(frames.count, 18, "each video's own frame count is preserved, without FPS conversion")
        checkEqual(Set(frames.map { FrameID(sampleIndex: $0.sampleIndex) }).count, 18, "IDs cannot collide across sources")
        checkEqual(frames[4].timeSeconds, 1, "second video's first frame starts at the timeline boundary")
        checkEqual(VideoCollection.sourceIndex(frames[4].sampleIndex), 1, "frame retains its source slot")
        checkEqual(VideoCollection.localIndex(frames[4].sampleIndex), 0, "frame retains its local ordinal")
        checkEqual(FrameID(sampleIndex: frames[4].sampleIndex).sampleIndex, frames[4].sampleIndex, "large source IDs round trip without truncation")
        let fullIDs = Set(frames.map(\.sampleIndex))
        for mode in [AnalysisMode.quick, .keyframe] {
            let subset = try await collectionFrames(url, mode: mode)
            check(subset.allSatisfy { fullIDs.contains($0.sampleIndex) }, "source IDs are stable across analysis modes")
            check(subset.contains { VideoCollection.sourceIndex($0.sampleIndex) == 1 }, "every mode includes both videos")
        }
        let resolved = collection.resolve(seconds: 1.5)
        checkEqual(resolved.url, second, "the selected frame chooses the second video")
        check(abs(resolved.seconds - 0.5) < 0.000001, "further processing uses local seconds")
        checkEqual(try FileManager.default.contentsOfDirectory(atPath: directory.path).sorted(), filesBefore, "loading writes no media")
        prepared = nil
        checkNil(VideoCollection.get(url), "release removes the workspace registry entry")
        check(FileManager.default.fileExists(atPath: first.path), "original videos survive release")
    })
    Registry.shared.add(suite: suite, TestCase(name: "preview regeneration export and restoration use the selected original", knownBug: nil) {
        let directory = try Temporary.directory("collection-routing")
        let first = try VideoFixture.makeMP4(in: directory, durationSeconds: 1, width: 64, height: 48, frameRate: 4)
        let source = try VideoFixture.makeMP4(in: directory, durationSeconds: 1, width: 96, height: 48, frameRate: 4)
        let rotated = directory.appendingPathComponent("portrait.mov")
        try await VideoFixture.rotate(source, to: rotated)
        let prepared = try await MediaImport.prepare([first, rotated])
        let image = try await PhotosMedia.shared.image(prepared.url, seconds: 1.5)
        checkEqual(image.width, 48, "preview keeps the original portrait width")
        checkEqual(image.height, 96, "preview keeps the original portrait height")
        let jpeg = try await ThumbnailRegenerator.shared.jpegData(for: prepared.url, timeSeconds: 1.5, analysisWidth: 96)
        check(!jpeg.isEmpty, "evicted thumbnails regenerate from the correct source")
        let id = FrameID(sampleIndex: try VideoCollection.frameIndex(source: 1, local: 2))
        let output = try Temporary.directory("collection-export")
        let request = ExportRequest(frameIDs: [id], destinationDirectory: output, frameTimes: [id: 1.5])
        var results: [ExportResult] = []
        for try await result in FrameExporter().export(request, sourceURL: prepared.url) { results.append(result) }
        checkEqual(results.count, 1, "selected frame exports")
        checkEqual(results[0].url.lastPathComponent, "portrait-000002.png", "export names the original video and local frame")
        let session = await MainActor.run { UncoverSession(video: VideoCollection.get(prepared.url)!.member(at: 1.5).video, seconds: 0.5) }
        await MainActor.run {
            checkEqual(session.source, rotated, "restoration receives the original source")
            checkEqual(session.sourceDuration, 1, "restoration stays inside that source")
            session.dispose()
        }
    })
    Registry.shared.add(suite: suite, TestCase(name: "tag cache keys survive single loads reordering and membership changes", knownBug: nil) {
        let directory = try Temporary.directory("collection-cache")
        let first = try VideoFixture.makeMP4(in: directory, durationSeconds: 1, width: 64, height: 48, frameRate: 4)
        let second = try VideoFixture.makeMP4(in: directory, durationSeconds: 1, width: 96, height: 64, frameRate: 4)
        let vocabulary = try tagVocabularyFixture()
        let batch = try await MediaImport.prepare([first, second])
        let reordered = try await MediaImport.prepare([second, first, second])
        let local = collectionFrame(index: 2, time: 0.5)
        let batchFrame = collectionFrame(index: try VideoCollection.frameIndex(source: 1, local: 2), time: 1.5)
        let moved = collectionFrame(index: 2, time: 0.5)
        let duplicate = collectionFrame(index: try VideoCollection.frameIndex(source: 2, local: 2), time: 2.5)
        let singleTags = TagResultCacheKeys.make(videoURL: second, frames: [local], vocabulary: vocabulary, analysisWidth: 96, analysisMode: .full)
        let batchTags = TagResultCacheKeys.make(videoURL: batch.url, frames: [batchFrame], vocabulary: vocabulary, analysisWidth: 96, analysisMode: .full)
        let movedTags = TagResultCacheKeys.make(videoURL: reordered.url, frames: [moved, duplicate], vocabulary: vocabulary, analysisWidth: 96, analysisMode: .full)
        checkEqual(batchTags.key(for: batchFrame), singleTags.key(for: local), "tag key is the original video")
        checkEqual(batchTags.recordIndex(for: batchFrame), 2, "tag cache uses local ordinal")
        checkEqual(movedTags.key(for: duplicate), singleTags.key(for: local), "membership/order cannot invalidate tags")
        let memory = await MainActor.run { TagResultMemory(disk: TagResultDiskCache(fileURL: directory.appendingPathComponent("tags.db"))) }
        var tagged = batchFrame
        var probabilities = [Float](repeating: 0, count: TagVocabulary.headCount)
        probabilities[1] = 0.9
        probabilities[vocabulary.generalIndices[0]] = 0.8
        tagged.tagProbabilities = TagProbabilityCodec.encode(probabilities)
        tagged.tagVocabularyFingerprint = vocabulary.cacheFingerprint
        await memory.remember(frames: [tagged], keys: batchTags)
        let reopenedTags = await MainActor.run { TagResultMemory(disk: TagResultDiskCache(fileURL: directory.appendingPathComponent("tags.db"))) }
        let restored = await reopenedTags.restore([local], keys: singleTags, vocabulary: vocabulary, threshold: 0.35)
        checkEqual(restored.restoredCount, 1, "tags saved in a batch restore when the source is opened alone")
        let movedRestored = await reopenedTags.restore([moved, duplicate], keys: movedTags, vocabulary: vocabulary, threshold: 0.35)
        checkEqual(movedRestored.restoredCount, 2, "cached tags restore after reordering and duplication")
        let otherVideoFrame = collectionFrame(index: 2, time: 0.5)
        let otherVideoTags = TagResultCacheKeys.make(videoURL: first, frames: [otherVideoFrame], vocabulary: vocabulary,
            analysisWidth: 96, analysisMode: .full)
        let unrelated = await reopenedTags.restore([otherVideoFrame], keys: otherVideoTags, vocabulary: vocabulary, threshold: 0.35)
        checkEqual(unrelated.restoredCount, 0, "equal frame numbers in different videos never share cached tags")
        var taggedAlone = collectionFrame(index: 1, time: 0.25)
        taggedAlone.tagProbabilities = tagged.tagProbabilities
        taggedAlone.tagVocabularyFingerprint = vocabulary.cacheFingerprint
        await reopenedTags.remember(frames: [taggedAlone], keys: singleTags)
        let laterBatchFrame = collectionFrame(index: try VideoCollection.frameIndex(source: 1, local: 1), time: 1.25)
        let laterKeys = TagResultCacheKeys.make(videoURL: batch.url, frames: [laterBatchFrame], vocabulary: vocabulary,
            analysisWidth: 96, analysisMode: .full)
        let fromSingle = await reopenedTags.restore([laterBatchFrame], keys: laterKeys, vocabulary: vocabulary, threshold: 0.35)
        checkEqual(fromSingle.restoredCount, 1, "tags saved alone restore inside a collection")
        let changedDate = Date().addingTimeInterval(60)
        try FileManager.default.setAttributes([.modificationDate: changedDate], ofItemAtPath: second.path)
        let changedTags = TagResultCacheKeys.make(videoURL: batch.url, frames: [batchFrame], vocabulary: vocabulary,
            analysisWidth: 96, analysisMode: .full)
        check(changedTags.key(for: batchFrame) != batchTags.key(for: batchFrame), "edited video invalidates its tag cache")

    })
    Registry.shared.add(suite: suite, TestCase(name: "unreadable middle video logs failure and keeps both surrounding sources", knownBug: nil) {
        let directory = try Temporary.directory("collection-recovery")
        let first = try VideoFixture.makeMP4(in: directory, durationSeconds: 1, width: 64, height: 48, frameRate: 4)
        let broken = try VideoFixture.makeMP4(in: directory, durationSeconds: 1, width: 64, height: 48, frameRate: 4)
        let last = try VideoFixture.makeMP4(in: directory, durationSeconds: 1, width: 64, height: 48, frameRate: 4)
        let prepared = try await MediaImport.prepare([first, broken, last])
        try FileManager.default.removeItem(at: broken)
        let recorder = VideoIssueRecorder()
        var frames: [SampledFrame] = []
        for try await frame in AVFoundationFrameSampler().sample(prepared.url, analysisWidth: 96,
            mode: .full, timeRange: nil, excludingSampleIndices: [], onProgress: { _ in },
            onIssue: { await recorder.append($0) }) { frames.append(frame) }
        checkEqual(frames.count, 8, "all frames from working videos survive")
        checkEqual(Set(frames.map { VideoCollection.sourceIndex($0.sampleIndex) }), Set([0, 2]), "skipping does not renumber sources")
        let issues = await recorder.issues
        checkEqual(issues.count, 1, "one failure is logged")
        checkEqual(issues.first?.source, broken, "log identifies the failed original")
        check(!(issues.first?.reason.isEmpty ?? true), "log includes the cause")
        checkEqual(issues.first?.framesRetained, 0, "unreadable file is reported unprocessed")
    })
    Registry.shared.add(suite: suite, TestCase(name: "mixed invalid imports continue and all invalid inputs fail with logs", knownBug: nil) {
        let directory = try Temporary.directory("collection-import-recovery")
        let good = try VideoFixture.makeMP4(in: directory, durationSeconds: 1, width: 64, height: 48, frameRate: 4)
        let bad = directory.appendingPathComponent("invalid.mp4")
        try Data("invalid video".utf8).write(to: bad)
        let recorder = VideoIssueRecorder()
        let prepared = try await MediaImport.prepare([bad, good], onIssue: { await recorder.append($0) })
        checkEqual(try await collectionFrames(prepared.url).count, 4, "readable source still runs")
        let issues = await recorder.issues
        checkEqual(issues.count, 1, "failed import logs once")
        checkEqual(issues.first?.source, bad, "import log preserves original path")
        _ = await checkThrowsAsync("all invalid inputs cannot succeed") {
            try await MediaImport.prepare([bad, bad], onIssue: { await recorder.append($0) })
        }
        checkEqual(await recorder.issues.count, 3, "all invalid files have reasons")
    })
    Registry.shared.add(suite: suite, TestCase(name: "late cannot decode retains prefix and cancellation is never logged", knownBug: nil) {
        let directory = try Temporary.directory("reader-recovery")
        let source = try VideoFixture.makeMP4(in: directory, durationSeconds: 1, width: 64, height: 48, frameRate: 4)
        let original = try await collectionFrames(source)
        let recorder = VideoIssueRecorder()
        let stream = AsyncThrowingStream<SampledFrame, Error> { continuation in
            continuation.yield(original[0])
            continuation.yield(original[1])
            continuation.finish(throwing: FrameSamplingError.assetReaderFailed("Cannot Decode"))
        }
        let reader = RecoveringVideoReader(frames: stream, source: source, onIssue: { await recorder.append($0) })
        var kept: [Int] = []
        while let frame = try await reader.next() { kept.append(frame.sampleIndex) }
        checkEqual(kept, original.prefix(2).map(\.sampleIndex), "completed prefix survives decoding failure")
        checkNil(try await reader.next(), "failed reader stays exhausted")
        let issues = await recorder.issues
        checkEqual(issues.count, 1, "failure logs once")
        checkEqual(issues.first?.reason, "Asset reader failed: Cannot Decode", "decoder reason is preserved")
        checkEqual(issues.first?.framesRetained, 2, "log distinguishes partial processing")
        let cancelled = AsyncThrowingStream<SampledFrame, Error> { $0.finish(throwing: CancellationError()) }
        let stopped = RecoveringVideoReader(frames: cancelled, source: source, onIssue: { await recorder.append($0) })
        let error = await checkThrowsAsync("cancellation propagates") { try await stopped.next() }
        check(error is CancellationError, "cancellation retains its type")
        checkEqual(await recorder.issues.count, 1, "cancel is not a file error")
    })
    Registry.shared.add(suite: suite, TestCase(name: "single input is unchanged and invalid or cancelled batches fail", knownBug: nil) {
        let directory = try Temporary.directory("collection-errors")
        let source = try VideoFixture.makeMP4(in: directory, durationSeconds: 1, width: 64, height: 48, frameRate: 4)
        let single = try await MediaImport.prepare([source])
        checkEqual(single.url, source, "single-video source stays unchanged")
        let bad = directory.appendingPathComponent("invalid.mp4")
        try Data("not a video".utf8).write(to: bad)
        _ = await checkThrowsAsync("invalid members fail visibly") { try await MediaImport.prepare([source, bad]) }
        _ = await checkThrowsAsync("empty batch fails") { try await MediaImport.prepare([URL]()) }
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await MediaImport.prepare([source, source])
        }
        switch await task.result {
        case .success: check(false, "cancelled batches cannot produce results")
        case .failure(let error): check(error is CancellationError, "cancellation is preserved")
        }
    })
}
