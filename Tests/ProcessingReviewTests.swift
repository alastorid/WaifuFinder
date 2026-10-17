import AVFoundation
import CoreGraphics
import Foundation

func registerProcessingReviewTests() {
    let suite = "Processing review"
    Registry.shared.add(suite: suite, TestCase(name: "older saved settings keep preferences when analysis mode is introduced", knownBug: nil) {
        let legacy = Data(#"{"analysisWidth":256,"similarityThreshold":0.12,"maximumGroupSize":20,"faceWeight":0.3,"minimumFaceAreaFraction":0.02,"perGroupFrameCount":2,"exportFormat":"jpeg","exportJPEGQuality":0.75,"samplingPlan":{"framesPerSecond":1}}"#.utf8)
        var settings = try JSONDecoder().decode(WaifuSettings.self, from: legacy)
        checkEqual(settings.analysisMode, .full, "missing field gets the new default")
        checkEqual(settings.analysisWidth, 256, "saved decode width retained")
        checkEqual(settings.perGroupFrameCount, 2, "saved cap retained")
        checkEqual(settings.exportFormat, .jpeg, "saved format retained")
        checkEqual(settings.similarityThreshold, 0.12, "saved grouping retained")
        settings.analysisMode = .quick
        checkEqual(try JSONDecoder().decode(WaifuSettings.self, from: JSONEncoder().encode(settings)), settings,
                   "new preferences round trip with old ones")
        let largeID = FrameID(sampleIndex: 1_234_567)
        checkEqual(FrameID(raw: largeID.raw), largeID, "long full passes keep valid frame identifiers")
    })
    Registry.shared.add(suite: suite, TestCase(name: "joined shots skip redundant distances without changing split groups", knownBug: nil) {
        let calls = TickLog()
        let frames = (0..<200).map { index in
            AnalyzedFrame(sampleIndex: index, timeSeconds: Double(index) / 30, artScore: 0,
                          faceScore: nil, faceCount: 0, faceAreaFraction: nil, featurePrint: Data([1]))
        }
        let grouper = FrameGrouper(distance: { _, _ in calls.record(1); return 0 })
        let groups = try grouper.groupCancellable(frames, settings: .default)
        checkEqual(calls.all.count, 199, "199 useful edges replace 19,900 redundant distances")
        checkEqual(groups.map(\.memberIDs.count).sorted(), [20, 60, 60, 60], "same contiguous group size limits")
        checkEqual(Set(groups.flatMap(\.memberIDs)), Set(frames.map(\.id.raw)), "no frames lost")
    })
    Registry.shared.add(suite: suite, TestCase(name: "cancelled grouping exits before comparing frames", knownBug: nil) {
        let calls = TickLog()
        let grouper = FrameGrouper(distance: { _, _ in calls.record(1); return 0 })
        let task = Task.detached {
            withUnsafeCurrentTask { $0?.cancel() }
            return try grouper.groupCancellable([], settings: .default)
        }
        do { _ = try await task.value; check(false, "cancellation must throw") }
        catch is CancellationError { }
        check(calls.all.isEmpty, "no expensive distance on a cancelled pass")
    })
    Registry.shared.add(suite: suite, TestCase(name: "source seek timing matches every decoded VFR frame", knownBug: nil) {
        let directory = try Temporary.directory("source-vfr-timeline")
        let source = directory.appendingPathComponent("vfr.mp4")
        let times: [Double] = (0..<24).map { index in
            if index < 12 { return Double(index) / 12.0 }
            return 1.0 + Double(index - 12) / 6.0
        }
        try VideoFixture.writeVideo(to: source, duration: 3, width: 128, height: 96, frameRate: 12, timestamps: times)
        let timeline = try await SourceFrameTimeline.load(source: source)
        var decoded: [Double] = []
        for try await frame in AVFoundationFrameSampler().sample(source, analysisWidth: 64, mode: .full, onProgress: { _ in }) {
            decoded.append(frame.timeSeconds)
        }
        checkEqual(timeline.timestamps, decoded, "metadata seeking follows actual presentation order")
        checkEqual(timeline.timestamps.count, 24, "every source frame is addressable")
        guard timeline.timestamps.count == 24 else { return }
        checkEqual(timeline.index(at: -10), 0, "backward clamp")
        checkEqual(timeline.index(at: 100), 23, "forward clamp")
        let lastGap = timeline.timestamps[23] - timeline.timestamps[22]
        let firstGap = timeline.timestamps[1] - timeline.timestamps[0]
        check(lastGap > firstGap * 1.5, "fixture has genuinely variable timing")
        let exporter = FrameExporter()
        let id = FrameID(sampleIndex: 18)
        let request = ExportRequest(frameIDs: [id], destinationDirectory: directory.appendingPathComponent("exports"),
                                    frameTimes: [id: timeline.timestamps[18]])
        var exports: [ExportResult] = []
        for try await result in exporter.export(request, sourceURL: source) { exports.append(result) }
        checkEqual(exports.count, 1, "one requested VFR frame exported")
        guard let exported = exports.first else { return }
        let actual = try RegionImages.read(exported.url)
        let expected = try await exporter.nativeImage(at: timeline.timestamps[18], in: AVURLAsset(url: source))
        checkEqual(reviewPixels(actual), reviewPixels(expected), "export uses decoded PTS, not ordinal divided by nominal FPS")
    })
}

private func reviewPixels(_ image: CGImage) -> [UInt8] {
    var pixels = [UInt8](repeating: 0, count: image.width * image.height * 4)
    pixels.withUnsafeMutableBytes { bytes in
        let context = CGContext(data: bytes.baseAddress, width: image.width, height: image.height,
                                bitsPerComponent: 8, bytesPerRow: image.width * 4,
                                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
    }
    return pixels
}
