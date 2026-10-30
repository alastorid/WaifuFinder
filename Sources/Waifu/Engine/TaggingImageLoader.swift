import CoreGraphics
import Foundation

enum TaggingImageSource: Sendable {
    case photo(URL)
    /// Regenerate an evicted Photos analysis JPEG after a server lane is reserved.
    case photoJPEG(URL)
    case thumbnail(Data)
}

/// Keeps at most four image requests active per batch. The tagging scheduler
/// bounds the number of batches preparing and running across all server lanes.
enum TaggingImageLoader {
    /// Remote Photos tagging reuses the analysis JPEG just like video tagging.
    /// Only a cache miss should need another PhotoKit request or JPEG encoding.
    /// Local-only Photos tagging retains its model-sized lossless rendition.
    static func photoSource(_ photoURL: URL, preferCachedJPEG: Bool, cachedJPEG: Data? = nil) -> TaggingImageSource {
        if !preferCachedJPEG { return .photo(photoURL) }
        if let cachedJPEG { return .thumbnail(cachedJPEG) }
        return .photoJPEG(photoURL)
    }

    static func load(
        _ sources: [TaggingImageSource],
        photoRendition: @escaping @Sendable (URL, Int) async throws -> CGImage = { url, size in
            try await PhotosMedia.shared.rendition(url, maxPixelSize: size,
                                                  allowNetwork: PhotosMedia.downloadFromICloud)
        }
    ) async throws -> [CGImage] {
        try Task.checkCancellation()
        let loadImage: @Sendable (Int) async throws -> (Int, CGImage) = { index in
            try Task.checkCancellation()
            let image: CGImage
            switch sources[index] {
            case .photo(let url):
                image = try await photoRendition(url, WDTaggerBackend.imageSize)
            case .photoJPEG(let url):
                let data = try await ThumbnailRegenerator.shared.jpegData(for: url, timeSeconds: 0,
                                                                           analysisWidth: PhotosMedia.analysisPixelSize)
                image = try autoreleasepool { try WDTaggerBackend.thumbnailImage(data: data) }
            case .thumbnail(let data):
                image = try autoreleasepool { try WDTaggerBackend.thumbnailImage(data: data) }
            }
            try Task.checkCancellation()
            return (index, image)
        }
        return try await loadBounded(count: sources.count, operation: loadImage)
    }

    static func jpegData(_ sources: [TaggingImageSource],
                         regenerate: @escaping @Sendable (URL) async throws -> Data = { url in
                             try await ThumbnailRegenerator.shared.jpegData(for: url, timeSeconds: 0,
                                                                            analysisWidth: PhotosMedia.analysisPixelSize)
                         }) async throws -> [Data] {
        try await loadBounded(count: sources.count) { index in
            switch sources[index] {
            case .thumbnail(let data): return (index, data)
            case .photoJPEG(let url): return (index, try await regenerate(url))
            case .photo: throw TaggerError.preprocessing
            }
        }
    }

    private static func loadBounded<Value: Sendable>(count: Int,
        operation: @escaping @Sendable (Int) async throws -> (Int, Value)) async throws -> [Value] {
        try Task.checkCancellation()
        return try await withThrowingTaskGroup(of: (Int, Value).self) { group in
            let initialCount = min(4, count)
            for index in 0..<initialCount {
                group.addTask { try await operation(index) }
            }
            var nextIndex = initialCount
            var images: [Int: Value] = [:]
            for try await (index, image) in group {
                images[index] = image
                if nextIndex < count {
                    let next = nextIndex
                    group.addTask { try await operation(next) }
                    nextIndex += 1
                }
            }
            return (0..<count).map { images[$0]! }
        }
    }
}
