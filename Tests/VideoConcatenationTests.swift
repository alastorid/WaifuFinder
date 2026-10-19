import AVFoundation
import CoreGraphics
import Foundation

private func joinedPixel(_ image: CGImage) -> [UInt8] {
    let rect = CGRect(x: image.width * 3 / 4, y: image.height * 3 / 4, width: 1, height: 1)
    guard let crop = image.cropping(to: rect) else { return [] }
    var pixel = [UInt8](repeating: 0, count: 4)
    pixel.withUnsafeMutableBytes { buffer in
        let context = CGContext(data: buffer.baseAddress, width: 1, height: 1, bitsPerComponent: 8,
                                bytesPerRow: 4, space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        context?.draw(crop, in: CGRect(x: 0, y: 0, width: 1, height: 1))
    }
    return pixel
}

func registerVideoConcatenationTests() {
    let suite = "video concatenation"
    Registry.shared.add(suite: suite, TestCase(name: "mixed sizes join in order and every frame is sampled", knownBug: nil) {
        let directory = try Temporary.directory("joined-video")
        let first = try VideoFixture.makeMP4(in: directory, durationSeconds: 1, width: 64, height: 48, frameRate: 4)
        let second = directory.appendingPathComponent("gray.mp4")
        try VideoFixture.writeVideo(to: second, duration: 2, width: 96, height: 64, frameRate: 4,
                                    picture: CGRect(x: 0, y: 0, width: 96, height: 64))
        var prepared: PreparedMedia? = try await MediaImport.prepare([first, second])
        let url = prepared!.url
        let info = try await VideoProbe.probe(url)
        check(abs(info.durationSeconds - 3) < 0.02, "duration includes both clips")
        checkEqual(info.width, 96, "canvas accommodates the widest clip")
        checkEqual(info.height, 64, "canvas accommodates the tallest clip")
        var frames: [SampledFrame] = []
        for try await frame in AVFoundationFrameSampler().sample(url, analysisWidth: 96, mode: .full, onProgress: { _ in }) {
            frames.append(frame)
        }
        checkEqual(frames.count, 12, "full analysis includes frames from both clips")
        check(frames.contains { $0.timeSeconds >= 1 }, "analysis passes the join")
        let generator = AVAssetImageGenerator(asset: VideoAssetFactory.asset(for: url))
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        let start = try await generator.image(at: CMTime(seconds: 0.5, preferredTimescale: 600)).image
        let end = try await generator.image(at: CMTime(seconds: 1.5, preferredTimescale: 600)).image
        let color = joinedPixel(start), gray = joinedPixel(end)
        check(color.count == 4 && abs(Int(color[0]) - Int(color[1])) > 30, "first clip remains first")
        check(gray.count == 4 && abs(Int(gray[0]) - Int(gray[1])) < 5, "second clip follows on the shared timeline")
        prepared = nil
        check(!FileManager.default.fileExists(atPath: url.path), "joined temporary media is removed after release")
        check(FileManager.default.fileExists(atPath: first.path), "source videos survive cleanup")
    })
    Registry.shared.add(suite: suite, TestCase(name: "rotation is preserved when joining clips", knownBug: nil) {
        let directory = try Temporary.directory("joined-rotation")
        let source = try VideoFixture.makeMP4(in: directory, durationSeconds: 1, width: 96, height: 48, frameRate: 4)
        let rotated = directory.appendingPathComponent("rotated.mov")
        try await VideoFixture.rotate(source, to: rotated)
        let prepared = try await MediaImport.prepare([source, rotated])
        let info = try await VideoProbe.probe(prepared.url)
        checkEqual(info.width, 96, "landscape width fits")
        checkEqual(info.height, 96, "rotated portrait height fits")
        let generator = AVAssetImageGenerator(asset: VideoAssetFactory.asset(for: prepared.url))
        let image = try await generator.image(at: CMTime(seconds: 1.5, preferredTimescale: 600)).image
        checkEqual(image.height, 96, "frames after the join decode on the oriented canvas")
    })
    Registry.shared.add(suite: suite, TestCase(name: "audio after a silent clip keeps its place on the joined timeline", knownBug: nil) {
        let directory = try Temporary.directory("joined-audio")
        let source = try VideoFixture.makeMP4(in: directory, durationSeconds: 1, width: 64, height: 48, frameRate: 4)
        let withAudio = directory.appendingPathComponent("with-audio.mov")
        try await VideoFixture.addAudio(to: source, output: withAudio, duration: 1)
        let prepared = try await MediaImport.prepare([source, withAudio])
        let asset = VideoAssetFactory.asset(for: prepared.url)
        let tracks = try await asset.loadTracks(withMediaType: .audio)
        checkEqual(tracks.count, 1, "audio survives concatenation")
        if let audio = tracks.first {
            let range = try await audio.load(.timeRange)
            check(range.end.seconds > 1.9, "audio reaches the end of the second clip")
        }
        let info = try await VideoProbe.probe(prepared.url)
        check(abs(info.durationSeconds - 2) < 0.02, "audio does not alter the video duration")
    })
    Registry.shared.add(suite: suite, TestCase(name: "single input stays native and invalid members fail the whole import", knownBug: nil) {
        let directory = try Temporary.directory("joined-errors")
        let source = try VideoFixture.makeMP4(in: directory, durationSeconds: 1, width: 64, height: 48, frameRate: 4)
        let single = try await MediaImport.prepare([source])
        checkEqual(single.url, source, "single video does not need concatenation")
        let bad = directory.appendingPathComponent("invalid.mp4")
        try Data("not a video".utf8).write(to: bad)
        _ = await checkThrowsAsync("an invalid member must not silently drop part of the batch") {
            try await MediaImport.prepare([source, bad])
        }
        _ = await checkThrowsAsync("an empty batch fails") { try await MediaImport.prepare([URL]()) }
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await MediaImport.prepare([source, source])
        }
        switch await task.result {
        case .success: check(false, "a cancelled import must not produce media")
        case .failure(let error): check(error is CancellationError, "cancellation is preserved")
        }
    })
}
