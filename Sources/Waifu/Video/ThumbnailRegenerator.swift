import AVFoundation
import CoreImage
import CoreMedia
import Foundation

/// Rebuilds evicted thumbnails from the source video without retaining decoded frames.
final class ThumbnailRegenerator: @unchecked Sendable {
    static let shared = ThumbnailRegenerator()

    private struct GeneratorKey: Hashable {
        let path: String
        let width: Int
    }

    private let queue = DispatchQueue(label: "Waifu.thumbnail.regeneration", qos: .userInitiated)
    private let context = CIContext()
    /// Accessed only on `queue`.
    private var generators: [GeneratorKey: AVAssetImageGenerator] = [:]

    private init() {}

    func jpegData(for url: URL, timeSeconds: Double, analysisWidth: Int) async throws -> Data {
        if PhotosMedia.isPhotos(url) {
            let image = try await PhotosMedia.shared.analysisImage(url, seconds: timeSeconds, maxPixelSize: analysisWidth)
            let picture = image.cropping(to: FramePictureArea.bounds(image)) ?? image
            let input = CIImage(cgImage: picture)
            let scale = min(1, CGFloat(analysisWidth) / max(input.extent.width, input.extent.height))
            let scaled = input.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
            guard let output = CIContext(options: [.cacheIntermediates: false]).createCGImage(scaled, from: scaled.extent) else {
                throw PhotosMedia.failure("Could not regenerate the Photos thumbnail.")
            }
            return try ThumbnailCache.encodeJPEG(output)
        }
        return try await withCheckedThrowingContinuation { continuation in
            queue.async { [self] in
                let key = GeneratorKey(path: url.standardizedFileURL.path, width: analysisWidth)
                let generator: AVAssetImageGenerator
                if let existing = generators[key] {
                    generator = existing
                } else {
                    generator = AVAssetImageGenerator(asset: VideoAssetFactory.asset(for: url))
                    generator.appliesPreferredTrackTransform = true
                    generator.maximumSize = CGSize(width: analysisWidth, height: analysisWidth)
                    generator.requestedTimeToleranceBefore = .zero
                    generator.requestedTimeToleranceAfter = .zero
                    generators[key] = generator
                }

                let timestamp = CMTime(seconds: timeSeconds, preferredTimescale: 60_000)
                generator.generateCGImagesAsynchronously(forTimes: [NSValue(time: timestamp)]) {
                    [self] _, image, _, _, error in
                    queue.async { [self] in
                        do {
                            guard let source = image else {
                                throw error ?? ThumbnailStoreError.writeFailed("Could not regenerate a thumbnail")
                            }
                            let picture = source.cropping(to: FramePictureArea.bounds(source)) ?? source
                            let scale = min(
                                1, CGFloat(analysisWidth) / CGFloat(max(picture.width, picture.height)))
                            let scaledImage = CIImage(cgImage: picture)
                                .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
                            guard let thumbnail = context.createCGImage(scaledImage, from: scaledImage.extent)
                            else {
                                throw ThumbnailStoreError.writeFailed("Could not resize a regenerated thumbnail")
                            }
                            continuation.resume(returning: try ThumbnailCache.encodeJPEG(thumbnail))
                        } catch {
                            continuation.resume(throwing: error)
                        }
                    }
                }
            }
        }
    }

    /// Drops AVFoundation's source readers when a different analysis begins.
    func reset() {
        queue.async { [self] in
            for generator in generators.values { generator.cancelAllCGImageGeneration() }
            generators.removeAll(keepingCapacity: false)
        }
    }
}
