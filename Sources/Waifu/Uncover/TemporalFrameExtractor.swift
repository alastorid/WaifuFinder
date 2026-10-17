import AVFoundation
import CoreImage
import Foundation

/// AVFoundation's track objects are loaded together off actor and then used by
/// one extractor actor. AVAsset and AVAssetTrack support this pattern, but the
/// SDK does not yet declare them Sendable.
private struct TemporalAssetTrack: @unchecked Sendable {
    let asset: AVAsset
    let track: AVAssetTrack
    let transform: CGAffineTransform
    let rate: Float
}

/// Owns one sequential native-resolution reader. Only the current image is resident.
actor TemporalFrameExtractor {
    private let context = CIContext(options: [.cacheIntermediates: false])

    /// Keep AVFoundation's non-Sendable async load results in a nonisolated
    /// helper, then transfer the loaded immutable references as one Sendable unit.
    nonisolated private static func loadTrack(source: URL) async throws -> TemporalAssetTrack {
        let asset = try await PhotosMedia.asset(source).value
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw UncoverError.invalid("The file contains no video track.")
        }
        let formatDescriptions = try await track.load(.formatDescriptions)
        for format in formatDescriptions {
            let extensions = (CMFormatDescriptionGetExtensions(format) as NSDictionary?) ?? [:]
            let transfer = extensions[kCMFormatDescriptionExtension_TransferFunction] as? String ?? ""
            guard !transfer.contains("2084"), !transfer.contains("HLG") else {
                throw UncoverError.invalid("Uncover currently supports SDR video; HDR processing is not supported.")
            }
            if let aspect = extensions[kCMFormatDescriptionExtension_PixelAspectRatio] as? [String: Any],
               let h = aspect[kCMFormatDescriptionKey_PixelAspectRatioHorizontalSpacing as String] as? NSNumber,
               let v = aspect[kCMFormatDescriptionKey_PixelAspectRatioVerticalSpacing as String] as? NSNumber,
               h != v {
                throw UncoverError.invalid("Uncover currently requires square video pixels.")
            }
        }
        let transform = try await track.load(.preferredTransform)
        let rate = try await track.load(.nominalFrameRate)
        return TemporalAssetTrack(asset: asset, track: track, transform: transform, rate: rate)
    }

    func prepare(source: URL, roi: UncoverROI, scale: UncoverScale, start: Double, end: Double,
                 directory: URL, minimumFrames: Int = 5, progress: @escaping @Sendable (String) -> Void) async throws -> UncoverPrepared {
        guard start.isFinite, end.isFinite, start >= 0, end > start, end - start <= 5.01 else {
            throw UncoverError.invalid("Choose an interval between 0 and 5 seconds long.")
        }
        if PhotosMedia.shared.isStill(source) {
            guard minimumFrames <= 1 else { throw UncoverError.invalid("This algorithm requires multiple frames and cannot process a static photo.") }
            return try await prepareStill(source: source, roi: roi, scale: scale, directory: directory)
        }
        let loadedTrack = try await Self.loadTrack(source: source)
        let asset = loadedTrack.asset
        let track = loadedTrack.track
        let transform = loadedTrack.transform
        let rate = loadedTrack.rate
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        ])
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw UncoverError.invalid("Cannot decode the video.") }
        reader.add(output)
        // Reader timeRange clips the opening sample's PTS to its boundary. Decode
        // preroll and filter actual PTS instead, or a partial preceding frame would
        // become a fabricated extra frame at the requested start.
        reader.timeRange = CMTimeRange(start: CMTime(seconds: max(0, start - 1), preferredTimescale: 60000),
                                      end: CMTime(seconds: end + 1, preferredTimescale: 60000))
        guard reader.startReading() else { throw UncoverError.invalid(reader.error?.localizedDescription ?? "Cannot start decoding.") }
        defer { reader.cancelReading() }
        for name in ["original", "degraded", "baseline", "reconstructed"] {
            try FileManager.default.createDirectory(at: directory.appendingPathComponent(name), withIntermediateDirectories: true)
        }
        var frames: [UncoverFrame] = []
        var videoSize = CGSize.zero
        var pixelRect = CGRect.zero
        while let buffer = output.copyNextSampleBuffer() {
            try Task.checkCancellation()
            let pts = CMSampleBufferGetPresentationTimeStamp(buffer)
            guard pts.isNumeric else { continue }
            if pts.seconds >= end { break }
            guard pts.seconds >= start else { continue }
            guard frames.count < 300 else { throw UncoverError.invalid("The interval exceeds the 300-frame processing limit.") }
            guard let pixels = CMSampleBufferGetImageBuffer(buffer) else { throw UncoverError.invalid("The video frame could not be decoded.") }
            let oriented = CIImage(cvPixelBuffer: pixels).transformed(by: transform)
            let normalized = oriented.transformed(by: CGAffineTransform(translationX: -oriented.extent.minX, y: -oriented.extent.minY))
            videoSize = normalized.extent.size
            pixelRect = try roi.pixels(in: videoSize)
            let cropRect = CGRect(x: pixelRect.minX, y: videoSize.height - pixelRect.maxY,
                                  width: pixelRect.width, height: pixelRect.height)
            guard let crop = context.createCGImage(normalized, from: cropRect) else { throw UncoverError.invalid("Could not extract the region.") }
            // Source degradation is optional. Model-specific input transforms
            // happen in the backend and never resize the retained full frame.
            let degraded = scale == .one ? crop : try RegionImages.resize(crop, to: scale.degradedSize(pixelRect.size), context: context)
            let index = frames.count
            guard let original = context.createCGImage(normalized, from: normalized.extent) else {
                throw UncoverError.invalid("Could not prepare the original frame.")
            }
            try RegionImages.write(original, to: RegionImages.frameURL(directory, "original", index))
            try RegionImages.write(degraded, to: RegionImages.frameURL(directory, "degraded", index))
            let baseline = scale == .one ? crop : try RegionImages.resize(degraded, to: pixelRect.size, context: context)
            try RegionImages.write(baseline, to: RegionImages.frameURL(directory, "baseline", index))
            let duration = CMSampleBufferGetDuration(buffer)
            let fallback = CMTime(seconds: 1 / Double(rate > 0 ? rate : 30), preferredTimescale: 60000)
            frames.append(UncoverFrame(index: index, timestamp: UncoverTime(pts),
                                       duration: UncoverTime(duration.isNumeric && duration.seconds > 0 ? duration : fallback)))
            progress("Preparing frames · \(frames.count)")
        }
        if reader.status == .failed { throw UncoverError.invalid(reader.error?.localizedDescription ?? "Video decoding failed.") }
        guard frames.count >= max(1,minimumFrames) else { throw UncoverError.invalid("Select at least \(minimumFrames) consecutive source frames.") }
        let attributes = (try? FileManager.default.attributesOfItem(atPath: source.path)) ?? [:]
        let manifest = UncoverManifest(source: source,
            sourceBytes: (attributes[.size] as? NSNumber)?.int64Value ?? 0,
            sourceModified: (attributes[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0,
            roi: .from(pixels: pixelRect, in: videoSize), width: Int(videoSize.width), height: Int(videoSize.height),
            regionX: Int(pixelRect.minX), regionY: Int(pixelRect.minY), regionWidth: Int(pixelRect.width),
            regionHeight: Int(pixelRect.height), scale: scale, frames: frames)
        try JSONEncoder().encode(manifest).write(to: directory.appendingPathComponent("manifest.json"), options: .atomic)
        return UncoverPrepared(directory: directory, manifest: manifest)
    }
    private func prepareStill(source: URL, roi: UncoverROI, scale: UncoverScale, directory: URL) async throws -> UncoverPrepared {
        let original = try await PhotosMedia.shared.image(source, seconds: 0)
        let size = CGSize(width: original.width, height: original.height)
        let rect = try roi.pixels(in: size)
        guard let crop = original.cropping(to: rect) else { throw UncoverError.invalid("Could not extract the photo region.") }
        let degraded = scale == .one ? crop : try RegionImages.resize(crop, to: scale.degradedSize(rect.size), context: context)
        for name in ["original", "degraded", "baseline", "reconstructed"] {
            try FileManager.default.createDirectory(at: directory.appendingPathComponent(name), withIntermediateDirectories: true)
        }
        try RegionImages.write(original, to: RegionImages.frameURL(directory, "original", 0))
        try RegionImages.write(degraded, to: RegionImages.frameURL(directory, "degraded", 0))
        let baseline = scale == .one ? crop : try RegionImages.resize(degraded, to: rect.size, context: context)
        try RegionImages.write(baseline, to: RegionImages.frameURL(directory, "baseline", 0))
        let frame = UncoverFrame(index: 0, timestamp: UncoverTime(.zero), duration: UncoverTime(CMTime(seconds: 1, preferredTimescale: 60000)))
        let manifest = UncoverManifest(source: source, sourceBytes: 0, sourceModified: 0,
            roi: .from(pixels: rect, in: size), width: original.width, height: original.height,
            regionX: Int(rect.minX), regionY: Int(rect.minY), regionWidth: Int(rect.width), regionHeight: Int(rect.height), scale: scale, frames: [frame])
        try JSONEncoder().encode(manifest).write(to: directory.appendingPathComponent("manifest.json"), options: .atomic)
        return UncoverPrepared(directory: directory, manifest: manifest)
    }

}
