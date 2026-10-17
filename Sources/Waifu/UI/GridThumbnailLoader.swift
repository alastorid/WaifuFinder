import Foundation
import ImageIO

/// Decode off the UI thread and retain a bounded set of recently visible images.
final class GridThumbnailLoader: @unchecked Sendable {
    static let shared = GridThumbnailLoader()
    private final class Entry: NSObject {
        let image: CGImage
        init(_ image: CGImage) { self.image = image }
    }
    private let cache = NSCache<NSString, Entry>()
    private let queue = DispatchQueue(label: "Waifu.grid.thumbnails", qos: .userInitiated)
    private final class Request: @unchecked Sendable {
        private let lock = NSLock()
        private var cancelled = false
        var isCancelled: Bool { lock.withLock { cancelled } }
        func cancel() { lock.withLock { cancelled = true } }
    }

    private init() {
        cache.totalCostLimit = 64 * 1024 * 1024
        cache.countLimit = 256
    }

    /// Fast path for a thumbnail already loaded by a visible grid cell.
    /// Preview entry can paint it immediately without queueing behind other cells.
    func cachedImage(identity: String) -> CGImage? {
        cache.object(forKey: identity as NSString)?.image
    }

    /// Decode memory-cached thumbnail bytes and retain only recently used pixels.
    func image(data: Data, identity: String) async -> CGImage? {
        if Task.isCancelled { return nil }
        if let cached = cache.object(forKey: identity as NSString) { return cached.image }
        let request = Request()
        return await withTaskCancellationHandler {
          await withCheckedContinuation { continuation in
            queue.async { [self] in
                guard !request.isCancelled else { continuation.resume(returning: nil); return }
                if let cached = cache.object(forKey: identity as NSString) {
                    continuation.resume(returning: cached.image)
                    return
                }
                let image: CGImage? = autoreleasepool {
                    guard let source = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
                    return CGImageSourceCreateImageAtIndex(source, 0, [
                        kCGImageSourceShouldCacheImmediately: true
                    ] as CFDictionary)
                }
                if let image {
                    cache.setObject(Entry(image), forKey: identity as NSString,
                                    cost: image.bytesPerRow * image.height)
                }
                continuation.resume(returning: image)
            }
          }
        } onCancel: {
            request.cancel()
        }
    }

    /// Legacy URL reader retained for tests and callers with external image files.
    func image(at url: URL, identity: String) async -> CGImage? {
        if Task.isCancelled { return nil }
        if let cached = cache.object(forKey: identity as NSString) { return cached.image }
        let request = Request()
        return await withTaskCancellationHandler {
          await withCheckedContinuation { continuation in
            queue.async { [self] in
                guard !request.isCancelled else { continuation.resume(returning: nil); return }
                if let cached = cache.object(forKey: identity as NSString) {
                    continuation.resume(returning: cached.image)
                    return
                }
                let image: CGImage? = autoreleasepool {
                    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
                    return CGImageSourceCreateImageAtIndex(source, 0, [
                        kCGImageSourceShouldCacheImmediately: true
                    ] as CFDictionary)
                }
                if let image {
                    cache.setObject(Entry(image), forKey: identity as NSString,
                                    cost: image.bytesPerRow * image.height)
                }
                continuation.resume(returning: image)
            }
          }
        } onCancel: {
            request.cancel()
        }
    }
}
