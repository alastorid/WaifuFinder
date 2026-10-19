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
    Registry.shared.add(suite: suite, TestCase(name: "analysis and tag cache keys survive single loads reordering and membership changes", knownBug: nil) {
        let directory = try Temporary.directory("collection-cache")
        let first = try VideoFixture.makeMP4(in: directory, durationSeconds: 1, width: 64, height: 48, frameRate: 4)
        let second = try VideoFixture.makeMP4(in: directory, durationSeconds: 1, width: 96, height: 64, frameRate: 4)
        let vocabulary = try tagVocabularyFixture()
        let batch = try await MediaImport.prepare([first, second])
        let reordered = try await MediaImport.prepare([second, first, second])
        let singleAnalysis = VideoAnalysisCacheKeys.make(videoURL: second, analysisWidth: 96,
            minimumFaceAreaFraction: 0.05, analysisMode: .full)!
        let batchAnalysis = VideoAnalysisCacheKeys.make(videoURL: batch.url, analysisWidth: 96,
            minimumFaceAreaFraction: 0.05, analysisMode: .full)!
        let reorderedAnalysis = VideoAnalysisCacheKeys.make(videoURL: reordered.url, analysisWidth: 96,
            minimumFaceAreaFraction: 0.05, analysisMode: .full)!
        let local = collectionFrame(index: 2, time: 0.5)
        let batchFrame = collectionFrame(index: try VideoCollection.frameIndex(source: 1, local: 2), time: 1.5)
        let moved = collectionFrame(index: 2, time: 0.5)
        let duplicate = collectionFrame(index: try VideoCollection.frameIndex(source: 2, local: 2), time: 2.5)
        checkEqual(batchAnalysis.identity(for: batchFrame.id.sampleIndex), singleAnalysis.identity(for: 2), "analysis key is the original video")
        checkEqual(reorderedAnalysis.identity(for: duplicate.id.sampleIndex), singleAnalysis.identity(for: 2), "duplicate/reordered sources reuse the same analysis")
        let records = batchAnalysis.records(from: [batchFrame])
        checkEqual(records[singleAnalysis.sources[0]!]!.map(\.sampleIndex), [2], "saved analysis keeps the original ordinal")
        let analysisFile = directory.appendingPathComponent("analysis.db")
        let analysisMemory = await MainActor.run { AnalysisResultMemory(disk: AnalysisResultDiskCache(fileURL: analysisFile)) }
        await analysisMemory.remember(identity: batchAnalysis, frames: [batchFrame])
        let reopenedAnalysis = await MainActor.run { AnalysisResultMemory(disk: AnalysisResultDiskCache(fileURL: analysisFile)) }
        let singleCached = await reopenedAnalysis.cachedResults(for: singleAnalysis)
        checkEqual(singleCached[2].flatMap(AnalysisResultCacheCodec.decode)?.artScore, 0.75,
                   "analysis saved in a batch restores from disk for a single video")
        let reorderedCached = await reopenedAnalysis.cachedResults(for: reorderedAnalysis)
        checkNotNil(reorderedCached[duplicate.id.sampleIndex], "analysis follows the video after reordering and duplication")
        await reopenedAnalysis.remember(identity: singleAnalysis, frames: [collectionFrame(index: 1, time: 0.25, score: 0.25)])
        let batchCached = await reopenedAnalysis.cachedResults(for: batchAnalysis)
        let workspaceIndex = try VideoCollection.frameIndex(source: 1, local: 1)
        checkEqual(batchCached[workspaceIndex].flatMap(AnalysisResultCacheCodec.decode)?.artScore, 0.25,
                   "analysis saved alone restores inside a collection")
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
        let previousFirst = batchAnalysis.sources[0]
        let changedDate = Date().addingTimeInterval(60)
        try FileManager.default.setAttributes([.modificationDate: changedDate], ofItemAtPath: second.path)
        let changedAnalysis = VideoAnalysisCacheKeys.make(videoURL: batch.url, analysisWidth: 96,
            minimumFaceAreaFraction: 0.05, analysisMode: .full)!
        check(changedAnalysis.sources[1] != singleAnalysis.sources[0], "edited video invalidates its own analysis cache")
        checkEqual(changedAnalysis.sources[0], previousFirst, "editing one video cannot invalidate another")
        let changedTags = TagResultCacheKeys.make(videoURL: batch.url, frames: [batchFrame], vocabulary: vocabulary,
            analysisWidth: 96, analysisMode: .full)
        check(changedTags.key(for: batchFrame) != batchTags.key(for: batchFrame), "edited video invalidates its tag cache")

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
