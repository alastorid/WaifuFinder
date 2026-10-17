import AVFoundation
import CoreImage
import Foundation

struct UncoverVideoExporter {
    func export(_ result: UncoverResult, feather: Double, strength: Double, to destination: URL,
                progress: @escaping @Sendable (String) -> Void) async throws {
        let manifest = result.prepared.manifest
        guard UncoverOutputLimits.accepts(width: manifest.width, height: manifest.height, scale: result.outputScale) else {
            throw UncoverError.invalid("The upscaled video would exceed the supported output dimensions. Choose a smaller source video.")
        }
        let outputWidth = result.outputWidth, outputHeight = result.outputHeight
        guard destination.standardizedFileURL != manifest.source.standardizedFileURL,
              !FileManager.default.fileExists(atPath: destination.path) else {
            throw UncoverError.invalid("Choose a new output file; existing files and the source are never overwritten.")
        }
        if manifest.source.isFileURL {
            let attributes = try FileManager.default.attributesOfItem(atPath: manifest.source.path)
            guard (attributes[.size] as? NSNumber)?.int64Value == manifest.sourceBytes,
                  (attributes[.modificationDate] as? Date)?.timeIntervalSince1970 == manifest.sourceModified else {
                throw UncoverError.invalid("The source video changed after reconstruction. Reconstruct again before exporting.")
            }
        }
        let temporary = destination.deletingLastPathComponent().appendingPathComponent(".uncover-\(UUID().uuidString).mov")
        let videoOnly = destination.deletingLastPathComponent().appendingPathComponent(".uncover-video-\(UUID().uuidString).mov")
        defer {
            try? FileManager.default.removeItem(at: temporary)
            try? FileManager.default.removeItem(at: videoOnly)
        }
        let source: AVAsset = if PhotosMedia.shared.isStill(manifest.source) { AVMutableComposition() } else { try await PhotosMedia.asset(manifest.source).value }
        let videoTrack = try await source.loadTracks(withMediaType: .video).first
        var colors: [String: Any] = [:]
        if let videoTrack, let format = try await videoTrack.load(.formatDescriptions).first,
           let extensions = CMFormatDescriptionGetExtensions(format) as NSDictionary? {
            for (sourceKey, outputKey) in [
                (kCMFormatDescriptionExtension_ColorPrimaries, AVVideoColorPrimariesKey),
                (kCMFormatDescriptionExtension_TransferFunction, AVVideoTransferFunctionKey),
                (kCMFormatDescriptionExtension_YCbCrMatrix, AVVideoYCbCrMatrixKey),
            ] {
                if let value = extensions[sourceKey] as? String { colors[outputKey] = value }
            }
        }
        let writer = try AVAssetWriter(outputURL: videoOnly, fileType: .mov)
        writer.movieTimeScale = manifest.frames[0].timestamp.timescale
        var settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: outputWidth, AVVideoHeightKey: outputHeight,
            AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: 16_000_000 * result.outputScale * result.outputScale],
        ]
        if colors.count == 3 { settings[AVVideoColorPropertiesKey] = colors }
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.mediaTimeScale = manifest.frames[0].timestamp.timescale
        input.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: outputWidth, kCVPixelBufferHeightKey as String: outputHeight,
        ])
        guard writer.canAdd(input) else { throw UncoverError.invalid("Cannot encode the output geometry.") }
        writer.add(input)
        guard writer.startWriting() else { throw UncoverError.invalid(writer.error?.localizedDescription ?? "Cannot start export.") }
        writer.startSession(atSourceTime: .zero)
        let context = CIContext(options: [.cacheIntermediates: false])
        let renderer = UncoverRenderer()
        let first = manifest.frames[0].timestamp.time
        do {
            for frame in manifest.frames {
                try Task.checkCancellation()
                let rendered = try await renderer.render(result, index: frame.index, feather: feather, strength: strength)
                while !input.isReadyForMoreMediaData {
                    try Task.checkCancellation()
                    guard writer.status == .writing else { throw UncoverError.invalid(writer.error?.localizedDescription ?? "Video encoding failed.") }
                    try await Task.sleep(nanoseconds: 10_000_000)
                }
                guard let pool = adaptor.pixelBufferPool else { throw UncoverError.invalid("Cannot allocate an export buffer.") }
                var buffer: CVPixelBuffer?
                guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &buffer) == kCVReturnSuccess, let buffer else {
                    throw UncoverError.invalid("Cannot allocate an export frame.")
                }
                context.render(CIImage(cgImage: rendered.reconstructed), to: buffer)
                guard adaptor.append(buffer, withPresentationTime: CMTimeSubtract(frame.timestamp.time, first)) else {
                    throw UncoverError.invalid(writer.error?.localizedDescription ?? "Could not encode an output frame.")
                }
                progress("Exporting · \(frame.index + 1)/\(manifest.frames.count)")
            }
            let last = manifest.frames[manifest.frames.count - 1]
            let duration = CMTimeSubtract(CMTimeAdd(last.timestamp.time, last.duration.time), first)
            writer.endSession(atSourceTime: duration)
            input.markAsFinished()
            await writer.finishWriting()
            guard writer.status == .completed else { throw UncoverError.invalid(writer.error?.localizedDescription ?? "Export did not finish.") }
            try Task.checkCancellation()
            let composition = AVMutableComposition()
            let encoded = AVURLAsset(url: videoOnly)
            guard let encodedVideo = try await encoded.loadTracks(withMediaType: .video).first,
                  let outputVideo = composition.addMutableTrack(withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid) else {
                throw UncoverError.invalid("The encoded video is unreadable.")
            }
            try outputVideo.insertTimeRange(CMTimeRange(start: .zero, duration: duration), of: encodedVideo, at: .zero)
            for audio in try await source.loadTracks(withMediaType: .audio) {
                let trackRange = try await audio.load(.timeRange)
                let requested = CMTimeRange(start: first, duration: duration)
                let intersection = CMTimeRangeGetIntersection(trackRange, otherRange: requested)
                guard !intersection.isEmpty else { continue }
                guard let outputAudio = composition.addMutableTrack(withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid) else {
                    throw UncoverError.invalid("Could not preserve the audio track.")
                }
                try outputAudio.insertTimeRange(intersection, of: audio, at: CMTimeSubtract(intersection.start, first))
            }
            guard let exporter = AVAssetExportSession(asset: composition, presetName: AVAssetExportPresetPassthrough) else {
                throw UncoverError.invalid("Cannot mux the output clip.")
            }
            let cancellation = ExportCancellation(exporter)
            try await withTaskCancellationHandler {
                try await exporter.export(to: temporary, as: .mov)
            } onCancel: { cancellation.cancel() }
            try Task.checkCancellation()
            let verified = AVURLAsset(url: temporary)
            guard try await verified.loadTracks(withMediaType: .video).count == 1 else {
                throw UncoverError.invalid("The exported clip could not be verified.")
            }
            // POSIX link publishes without an overwrite race; both paths share a
            // filesystem, and observers can only see the finished container.
            guard link(temporary.path, destination.path) == 0 else {
                throw UncoverError.invalid("Could not publish export; the destination may already exist.")
            }
        } catch {
            writer.cancelWriting()
            throw error
        }
    }
}

private final class ExportCancellation: @unchecked Sendable {
    let exporter: AVAssetExportSession
    init(_ exporter: AVAssetExportSession) { self.exporter = exporter }
    func cancel() { exporter.cancelExport() }
}
