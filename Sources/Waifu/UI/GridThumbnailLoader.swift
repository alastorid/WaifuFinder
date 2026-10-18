import Foundation
import ImageIO

/// Decode off the UI thread and retain a bounded set of recently visible images.
final class GridThumbnailLoader: @unchecked Sendable {
    static let shared = GridThumbnailLoader()
    private final class Entry: NSObject {
        let image: CGImage
        // Accessed only under visibleLock. Decoding/prefetching is not display.
        var firstPresentationTime: TimeInterval?
        init(_ image: CGImage) { self.image = image }
    }
    private let cache = NSCache<NSString, Entry>()
    private struct VisibleImage {
        let entry: Entry
        var owners: Set<UUID>
    }
    private let visibleLock = NSLock()
    private var visibleImages: [String: VisibleImage] = [:]
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
        visibleLock.withLock { visibleImages[identity]?.entry.image }
            ?? cache.object(forKey: identity as NSString)?.image
    }

    /// Visible cells lease their decoded pixels so cache eviction cannot make a
    /// moving/recreated cell flash empty. Leases end shortly after disappearance.
    func retainVisibleImage(_ image: CGImage, identity: String, owner: UUID) {
        visibleLock.withLock {
            if var entry = visibleImages[identity] {
                entry.owners.insert(owner)
                visibleImages[identity] = entry
            } else {
                let cached = cache.object(forKey: identity as NSString)
                let entry: Entry
                if let cached, cached.image === image { entry = cached }
                else { entry = Entry(image) }
                visibleImages[identity] = VisibleImage(entry: entry, owners: [owner])
            }
        }
    }

    /// Start the reveal only when pixels reach the viewport. Recreated cells
    /// share its clock, so a handoff during the fade cannot jump to full opacity.
    /// This record lives with the bounded decoded cache/visibility lease.
    func revealStart(identity: String, at time: TimeInterval) -> TimeInterval {
        visibleLock.withLock {
            guard let entry = visibleImages[identity]?.entry
                    ?? cache.object(forKey: identity as NSString) else { return time }
            if let start = entry.firstPresentationTime { return start }
            entry.firstPresentationTime = time
            return time
        }
    }

    func releaseVisibleImage(identity: String, owner: UUID) {
        visibleLock.withLock {
            guard var entry = visibleImages[identity] else { return }
            entry.owners.remove(owner)
            if entry.owners.isEmpty { visibleImages.removeValue(forKey: identity) }
            else { visibleImages[identity] = entry }
        }
    }

    /// Decode memory-cached thumbnail bytes and retain only recently used pixels.
    func image(data: Data, identity: String) async -> CGImage? {
        if Task.isCancelled { return nil }
        if let cached = cachedImage(identity: identity) { return cached }
        let request = Request()
        return await withTaskCancellationHandler {
          await withCheckedContinuation { continuation in
            queue.async { [self] in
                guard !request.isCancelled else { continuation.resume(returning: nil); return }
                if let cached = cachedImage(identity: identity) {
                    continuation.resume(returning: cached)
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
        if let cached = cachedImage(identity: identity) { return cached }
        let request = Request()
        return await withTaskCancellationHandler {
          await withCheckedContinuation { continuation in
            queue.async { [self] in
                guard !request.isCancelled else { continuation.resume(returning: nil); return }
                if let cached = cachedImage(identity: identity) {
                    continuation.resume(returning: cached)
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
