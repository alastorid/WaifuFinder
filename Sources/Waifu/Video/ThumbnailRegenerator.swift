import AVFoundation
import CoreImage
import CoreMedia
import Foundation

/// Rebuilds evicted thumbnails from the source video without retaining decoded frames.
final class ThumbnailRegenerator: @unchecked Sendable {
    static let shared = ThumbnailRegenerator()

    private init() {}

    func jpegData(for url: URL, timeSeconds: Double, analysisWidth: Int) async throws -> Data {
        let (url, timeSeconds) = VideoCollection.resolve(url, seconds: timeSeconds)
        if PhotosMedia.isPhotos(url) {
            let image = try await PhotosMedia.shared.analysisImage(url, seconds: timeSeconds, maxPixelSize: analysisWidth)
            try Task.checkCancellation()
            let picture = image.cropping(to: FramePictureArea.bounds(image)) ?? image
            let input = CIImage(cgImage: picture)
            let scale = min(1, CGFloat(analysisWidth) / max(input.extent.width, input.extent.height))
            let scaled = input.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
            guard let output = CIContext(options: [.cacheIntermediates: false]).createCGImage(scaled, from: scaled.extent) else {
                throw PhotosMedia.failure("Could not regenerate the Photos thumbnail.")
            }
            try Task.checkCancellation()
            return try ThumbnailCache.encodeJPEG(output)
        }
        try Task.checkCancellation()
        // Each request owns its decoder: cancelling an offscreen frame must not
        // cancel another visible frame from the same video.
        let image = try await PreviewFrameReader(
            asset: VideoAssetFactory.asset(for: url), maximumPixelSize: analysisWidth
        ).image(at: timeSeconds)
        try Task.checkCancellation()
        let picture = image.cropping(to: FramePictureArea.bounds(image)) ?? image
        let scale = min(1, CGFloat(analysisWidth) / CGFloat(max(picture.width, picture.height)))
        let scaled = CIImage(cgImage: picture)
            .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        guard let thumbnail = CIContext(options: [.cacheIntermediates: false])
            .createCGImage(scaled, from: scaled.extent) else {
            throw ThumbnailStoreError.writeFailed("Could not resize a regenerated thumbnail")
        }
        try Task.checkCancellation()
        return try ThumbnailCache.encodeJPEG(thumbnail)
    }
}
