import AVFoundation
import Foundation

func registerVideoClipExporterTests() {
    let suite = "clip export"
    Registry.shared.add(suite: suite, TestCase(name: "clip windows merge overlaps and preserve gaps and source bounds", knownBug: nil) {
        let url = URL(fileURLWithPath: "/synthetic.mov")
        let video = VideoInfo(url: url, durationSeconds: 40, width: 64, height: 48,
                              frameRate: 4, estimatedFrameCount: 160)
        let frames = [38.0, 2, 8, 13, 19, 29].enumerated().map {
            FrameFixture.make(id: "f\($0.offset)", time: $0.element)
        }
        checkEqual(VideoClipExporter.ranges(video: video, frames: frames), [
            VideoClipRange(sourceURL: url, seconds: 0...22),
            VideoClipRange(sourceURL: url, seconds: 26...32),
            VideoClipRange(sourceURL: url, seconds: 35...40)
        ], "selected coverage includes padding once per merged interval without bridging gaps")
    })
    Registry.shared.add(suite: suite, TestCase(name: "clip ranges stay in each selected original video", knownBug: nil) {
        let first = VideoInfo(url: URL(fileURLWithPath: "/first.mov"), durationSeconds: 10,
                              width: 64, height: 48, frameRate: 4, estimatedFrameCount: 40)
        let second = VideoInfo(url: URL(fileURLWithPath: "/second.mov"), durationSeconds: 10,
                               width: 64, height: 48, frameRate: 4, estimatedFrameCount: 40)
        let url = URL(string: "waifu-videos://clip-tests-\(UUID().uuidString)")!
        let collection = VideoCollection(url: url, members: [.init(video: first, offset: 0), .init(video: second, offset: 10)])
        VideoCollection.register(collection)
        defer { VideoCollection.release(url) }
        let secondIndex = try VideoCollection.frameIndex(source: 1, local: 4)
        let frames = [FrameFixture.make(id: "f36", time: 9), FrameFixture.make(id: "f\(secondIndex)", time: 11)]
        checkEqual(VideoClipExporter.ranges(video: collection.info, frames: frames), [
            VideoClipRange(sourceURL: first.url, seconds: 6...10),
            VideoClipRange(sourceURL: second.url, seconds: 0...4)
        ], "adjacent collection timestamps yield separate clips with original local timestamps")
    })
    Registry.shared.add(suite: suite, TestCase(name: "native clip exports retain audio, cut duration and avoid overwriting", knownBug: nil) {
        let directory = try Temporary.directory("video-clips")
        let silent = try VideoFixture.makeMP4(in: directory, durationSeconds: 12, width: 64, height: 48, frameRate: 4)
        let source = directory.appendingPathComponent("source.mov")
        try await VideoFixture.addAudio(to: silent, output: source, duration: 12)
        let destination = directory.appendingPathComponent("exported")
        let range = VideoClipRange(sourceURL: source, seconds: 1.25...4.75)
        let first = try await VideoClipExporter.export(range, to: destination)
        let originalBytes = try Data(contentsOf: first)
        let second = try await VideoClipExporter.export(range, to: destination)
        check(first != second, "repeated exports receive distinct names")
        checkEqual(try Data(contentsOf: first), originalBytes, "existing clip bytes remain intact")
        let later = try await VideoClipExporter.export(.init(sourceURL: source, seconds: 8...11), to: destination)
        for (url, duration) in [(first, 3.5), (second, 3.5), (later, 3.0)] {
            let asset = AVURLAsset(url: url)
            let actual = try await asset.load(.duration).seconds
            check(abs(actual - duration) <= 0.05, "clip has requested duration: \(actual) vs \(duration)")
            checkEqual(try await asset.loadTracks(withMediaType: .video).count, 1, "one video track is retained")
            checkEqual(try await asset.loadTracks(withMediaType: .audio).count, 1, "source audio is retained")
        }
        let files = try FileManager.default.contentsOfDirectory(at: destination, includingPropertiesForKeys: nil)
        checkEqual(files.count, 3, "only completed clips are left in the destination")
    })
}
