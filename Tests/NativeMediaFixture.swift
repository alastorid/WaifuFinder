import AVFoundation
import CoreGraphics
import CoreMedia
import CoreVideo
import Foundation

/// Original synthetic media, encoded exclusively with Apple system frameworks.
/// This helper is also compiled by tools/make-synthetic-video.sh.
enum VideoFixture {
    static func makeMP4(in directory: URL, durationSeconds: Double,
                        width: Int, height: Int, frameRate: Int) throws -> URL {
        let output = directory.appendingPathComponent("fixture-\(UUID().uuidString).mp4")
        try writeVideo(to: output, duration: durationSeconds, width: width, height: height, frameRate: frameRate)
        return output
    }

    static func makeFixedGOPMP4(in directory: URL, durationSeconds: Double,
                                width: Int, height: Int, frameRate: Int, gop: Int) throws -> URL {
        let output = directory.appendingPathComponent("fixture-gop\(gop)-\(UUID().uuidString).mp4")
        try writeVideo(to: output, duration: durationSeconds, width: width, height: height, frameRate: frameRate, gop: gop)
        return output
    }

    static func writeVideo(to output: URL, duration: Double, width: Int, height: Int, frameRate: Int,
                           gop: Int? = nil, timestamps: [Double]? = nil, picture: CGRect? = nil) throws {
        guard duration > 0, width > 0, height > 0, frameRate > 0 else { throw failure("Invalid synthetic video geometry or timing") }
        let writer = try AVAssetWriter(outputURL: output, fileType: .mp4)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width, AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [
                AVVideoMaxKeyFrameIntervalKey: gop ?? frameRate * 2,
                AVVideoAllowFrameReorderingKey: false,
                AVVideoExpectedSourceFrameRateKey: frameRate,
                AVVideoAverageBitRateKey: max(500_000, width * height * 6)
            ]
        ])
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32ARGB,
            kCVPixelBufferWidthKey as String: width, kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
        ])
        guard writer.canAdd(input) else { throw failure("Native H.264 writer rejected video input") }
        writer.add(input)
        guard writer.startWriting() else { throw writer.error ?? failure("Native video writer did not start") }
        writer.startSession(atSourceTime: .zero)
        let times = timestamps ?? (0..<max(1, Int(ceil(duration * Double(frameRate))))).map { Double($0) / Double(frameRate) }
        let deadline = Date().addingTimeInterval(60)
        do {
            for (index, seconds) in times.enumerated() {
                while !input.isReadyForMoreMediaData {
                    guard Date() < deadline, writer.status == .writing else { throw writer.error ?? failure("Native fixture encoder timed out") }
                    Thread.sleep(forTimeInterval: 0.001)
                }
                var buffer: CVPixelBuffer?
                guard let pool = adaptor.pixelBufferPool,
                      CVPixelBufferPoolCreatePixelBuffer(nil, pool, &buffer) == kCVReturnSuccess,
                      let buffer else { throw failure("Cannot allocate synthetic video pixels") }
                CVPixelBufferLockBaseAddress(buffer, [])
                guard let context = CGContext(data: CVPixelBufferGetBaseAddress(buffer), width: width, height: height,
                    bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                    space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue) else {
                    CVPixelBufferUnlockBaseAddress(buffer, [])
                    throw failure("Cannot draw synthetic video")
                }
                context.setFillColor(CGColor(gray: picture == nil ? 0.2 : 0, alpha: 1))
                context.fill(CGRect(x: 0, y: 0, width: width, height: height))
                if let picture {
                    context.setFillColor(CGColor(gray: 0.5, alpha: 1)); context.fill(picture)
                } else {
                    for stripe in 0..<8 {
                        context.setFillColor(CGColor(red: CGFloat(stripe) / 9, green: CGFloat(8 - stripe) / 10, blue: 0.4, alpha: 1))
                        context.fill(CGRect(x: stripe * width / 8, y: 0, width: (width + 7) / 8, height: height))
                    }
                    context.setFillColor(CGColor(gray: 0.95, alpha: 1))
                    context.fill(CGRect(x: (index * 3) % max(1, width / 2), y: height / 3, width: max(4, width / 5), height: max(4, height / 4)))
                }
                CVPixelBufferUnlockBaseAddress(buffer, [])
                guard adaptor.append(buffer, withPresentationTime: CMTime(seconds: seconds, preferredTimescale: 60_000)) else {
                    throw writer.error ?? failure("Native fixture encoder rejected a frame")
                }
            }
            writer.endSession(atSourceTime: CMTime(seconds: duration, preferredTimescale: 60_000))
            input.markAsFinished()
            let finished = DispatchSemaphore(value: 0)
            writer.finishWriting { finished.signal() }
            guard finished.wait(timeout: .now() + 60) == .success, writer.status == .completed else {
                throw writer.error ?? failure("Native fixture encoder did not finish")
            }
        } catch { writer.cancelWriting(); throw error }
    }

    static func makeAudioOnly(in directory: URL, durationSeconds: Double) throws -> URL {
        let output = directory.appendingPathComponent("fixture-audio-\(UUID().uuidString).caf")
        guard let format = AVAudioFormat(standardFormatWithSampleRate: 48_000, channels: 1),
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(durationSeconds * 48_000)) else {
            throw failure("Cannot allocate synthetic audio")
        }
        buffer.frameLength = buffer.frameCapacity
        for sample in 0..<Int(buffer.frameLength) {
            buffer.floatChannelData![0][sample] = Float(sin(Double(sample) * 2 * .pi * 440 / 48_000)) * 0.2
        }
        let file = try AVAudioFile(forWriting: output, settings: format.settings)
        try file.write(from: buffer)
        return output
    }

    static func rotate(_ source: URL, to output: URL) async throws {
        let asset = AVURLAsset(url: source)
        guard let video = try await asset.loadTracks(withMediaType: .video).first else { throw failure("Rotation needs a video track") }
        let composition = AVMutableComposition()
        guard let destination = composition.addMutableTrack(withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid) else { throw failure("Cannot create rotated track") }
        try destination.insertTimeRange(CMTimeRange(start: .zero, duration: try await asset.load(.duration)), of: video, at: .zero)
        let size = try await video.load(.naturalSize)
        destination.preferredTransform = CGAffineTransform(a: 0, b: 1, c: -1, d: 0, tx: size.height, ty: 0)
        try await export(composition, to: output)
    }

    static func addAudio(to videoURL: URL, output: URL, duration: Double) async throws {
        let audioURL = try makeAudioOnly(in: output.deletingLastPathComponent(), durationSeconds: duration)
        let composition = AVMutableComposition()
        for (url, type) in [(videoURL, AVMediaType.video), (audioURL, AVMediaType.audio)] {
            let asset = AVURLAsset(url: url)
            guard let track = try await asset.loadTracks(withMediaType: type).first,
                  let destination = composition.addMutableTrack(withMediaType: type, preferredTrackID: kCMPersistentTrackID_Invalid) else {
                throw failure("Cannot compose synthetic audio/video")
            }
            try destination.insertTimeRange(CMTimeRange(start: .zero, duration: CMTime(seconds: duration, preferredTimescale: 60_000)), of: track, at: .zero)
        }
        try await export(composition, to: output)
    }

    private static func export(_ composition: AVComposition, to output: URL) async throws {
        guard let session = AVAssetExportSession(asset: composition, presetName: AVAssetExportPresetPassthrough) else { throw failure("Cannot create native fixture exporter") }
        try await session.export(to: output, as: output.pathExtension == "mp4" ? .mp4 : .mov)
    }

    static func failure(_ message: String) -> NSError {
        NSError(domain: "NativeMediaFixture", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }
}
