import Foundation
import CoreImage
import ImageIO

struct PhotosFilmstripFrame: Identifiable, @unchecked Sendable {
    let index: Int
    let image: CGImage
    var id: Int { index }
}

/// An animated PhotoKit image kept in memory. Frames are decoded on demand and
/// thumbnails are bounded so a long GIF cannot pin every full-size frame.
final class PhotosAnimatedImage: @unchecked Sendable {
    private final class CachedFrame: NSObject {
        let image: CGImage
        init(_ image: CGImage) { self.image = image }
    }

    let identity: String
    let frameDurations: [TimeInterval]
    private let source: CGImageSource
    private let orientation: CGImagePropertyOrientation
    private let lock = NSLock()
    private static let imageContext = CIContext(options: [.cacheIntermediates: false])
    private let cache: NSCache<NSString, CachedFrame> = {
        let cache = NSCache<NSString, CachedFrame>()
        cache.totalCostLimit = 96 * 1024 * 1024
        cache.countLimit = 96
        return cache
    }()

    init(source: CGImageSource, identity: String, orientation: CGImagePropertyOrientation) {
        self.source = source
        self.identity = identity
        self.orientation = orientation
        let count = CGImageSourceGetCount(source)
        self.frameDurations = (0..<count).map { index in
            let properties = CGImageSourceCopyPropertiesAtIndex(source, index, nil) as? [CFString: Any]
            let gif = properties?[kCGImagePropertyGIFDictionary] as? [CFString: Any]
            let png = properties?[kCGImagePropertyPNGDictionary] as? [CFString: Any]
            let delay = (gif?[kCGImagePropertyGIFUnclampedDelayTime] as? NSNumber)?.doubleValue
                ?? (gif?[kCGImagePropertyGIFDelayTime] as? NSNumber)?.doubleValue
                ?? (png?[kCGImagePropertyAPNGDelayTime] as? NSNumber)?.doubleValue
                ?? 0.1
            return min(10, max(0.02, delay))
        }
    }

    var frameCount: Int { frameDurations.count }
    var duration: TimeInterval { frameDurations.reduce(0, +) }

    func elapsed(before index: Int) -> TimeInterval {
        frameDurations.prefix(max(0, min(index, frameCount))).reduce(0, +)
    }

    func duration(at index: Int) -> TimeInterval {
        frameDurations.indices.contains(index) ? frameDurations[index] : 0.1
    }

    func frame(at index: Int, maximumPixelSize: Int) -> CGImage? {
        guard frameDurations.indices.contains(index) else { return nil }
        let pixelSize = max(1, maximumPixelSize)
        let key = "\(index):\(pixelSize)" as NSString
        if let cached = cache.object(forKey: key) { return cached.image }
        while !lock.try() {
            if Task.isCancelled { return nil }
            Thread.sleep(forTimeInterval: 0.002)
        }
        defer { lock.unlock() }
        guard !Task.isCancelled else { return nil }
        if let cached = cache.object(forKey: key) { return cached.image }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: false,
            kCGImageSourceThumbnailMaxPixelSize: pixelSize,
            kCGImageSourceShouldCacheImmediately: true
        ]
        guard let decoded = CGImageSourceCreateThumbnailAtIndex(source, index, options as CFDictionary) else { return nil }
        let image: CGImage
        if orientation != .up {
            let oriented = CIImage(cgImage: decoded).oriented(orientation)
            guard let result = Self.imageContext.createCGImage(oriented, from: oriented.extent) else { return nil }
            image = result
        } else {
            image = decoded
        }
        cache.setObject(CachedFrame(image), forKey: key, cost: image.bytesPerRow * image.height)
        return image
    }

    func frameAsync(at index: Int, maximumPixelSize: Int) async -> CGImage? {
        let work = Task.detached(priority: .userInitiated) { self.frame(at: index, maximumPixelSize: maximumPixelSize) }
        return await withTaskCancellationHandler { await work.value } onCancel: { work.cancel() }
    }

    func filmstripFrames(maximumCount: Int = 48, maximumPixelSize: Int = 112) async -> [PhotosFilmstripFrame] {
        let sampleCount = min(frameCount, max(1, maximumCount))
        let indices = (0..<sampleCount).map { slot in
            sampleCount == 1 ? 0 : slot * (frameCount - 1) / (sampleCount - 1)
        }
        let work = Task.detached(priority: .utility) { () -> [PhotosFilmstripFrame] in
            return indices.compactMap { index in
                guard !Task.isCancelled else { return nil }
                return self.frame(at: index, maximumPixelSize: maximumPixelSize).map { PhotosFilmstripFrame(index: index, image: $0) }
            }
        }
        return await withTaskCancellationHandler { await work.value } onCancel: { work.cancel() }
    }
}

enum PhotosPreviewContent: @unchecked Sendable {
    case still(CGImage)
    case animated(PhotosAnimatedImage)
}
