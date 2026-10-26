import CoreGraphics
import Foundation

/// Local PhotoKit reads may overlap. Only a missing/requested-resolution
/// upgrade needs a cloud lane, even when the global cloud switch is enabled.
enum PhotosRenditionLoader {
    static func load(allowNetwork: Bool, requestedSize: CGSize?,
                     local: @Sendable () async throws -> CGImage,
                     cloud: @Sendable () async throws -> CGImage) async throws -> CGImage {
        do {
            let image = try await local()
            try Task.checkCancellation()
            if !allowNetwork || requestedSize.map({ PhotosMedia.hasRequestedResolution(image, targetSize: $0) }) != false {
                return image
            }
        } catch {
            if error is CancellationError || Task.isCancelled { throw CancellationError() }
            if !allowNetwork || !PhotosMedia.isUnavailableLocally(error) { throw error }
        }
        try Task.checkCancellation()
        return try await cloud()
    }
}
