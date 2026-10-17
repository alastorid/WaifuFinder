import CoreGraphics
import Foundation

enum TaggingImageSource: Sendable {
    case photo(URL)
    case thumbnail(Data)
}

/// Keeps at most four image requests active. The batch pipeline retains only the
/// current batch and one prepared following batch, even for large Photos albums.
enum TaggingImageLoader {
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
            case .thumbnail(let data):
                image = try autoreleasepool { try WDTaggerBackend.thumbnailImage(data: data) }
            }
            try Task.checkCancellation()
            return (index, image)
        }
        return try await withThrowingTaskGroup(of: (Int, CGImage).self) { group in
            let initialCount = min(4, sources.count)
            for index in 0..<initialCount {
                group.addTask { try await loadImage(index) }
            }
            var nextIndex = initialCount
            var images: [Int: CGImage] = [:]
            for try await (index, image) in group {
                images[index] = image
                if nextIndex < sources.count {
                    let next = nextIndex
                    group.addTask { try await loadImage(next) }
                    nextIndex += 1
                }
            }
            return sources.indices.map { images[$0]! }
        }
    }
}
