import CoreGraphics
import CryptoKit
import Foundation

/// The image being inspected is also the input to preview evidence. A local
/// rendition remains usable when PhotoKit cannot supply original image data.
struct PreviewPhotoSource: Sendable {
    let content: PhotosPreviewContent
    let isOriginal: Bool

    /// Compare decoded pixels, never NSImage's point size. This is only used
    /// between renditions of the same photo/revision, not across navigation.
    static func isSharper(_ image: CGImage, than previous: CGImage?) -> Bool {
        guard let previous else { return true }
        return Int64(image.width) * Int64(image.height) > Int64(previous.width) * Int64(previous.height)
    }

    /// PhotoKit may return a tiny successful rendition even when another cache
    /// already has larger pixels. Success alone never makes a candidate best.
    static func sharpest(_ candidates: [CGImage?]) -> CGImage? {
        candidates.compactMap { $0 }.reduce(nil as CGImage?) { best, image in
            isSharper(image, than: best) ? image : best
        }
    }

    private static func still(_ image: CGImage, isOriginal: Bool,
                              cached: CGImage?, cachedIsOriginal: Bool) -> Self {
        if let cached, isSharper(cached, than: image)
            || (cachedIsOriginal && !isOriginal && cached.width == image.width && cached.height == image.height) {
            return Self(content: .still(cached), isOriginal: cachedIsOriginal)
        }
        return Self(content: .still(image), isOriginal: isOriginal)
    }

    static func load(original: @Sendable () async throws -> PhotosPreviewContent,
                     rendition: @Sendable () async throws -> CGImage,
                     cached: CGImage? = nil, cachedIsOriginal: Bool = false) async throws -> Self {
        do {
            let content = try await original()
            try Task.checkCancellation()
            if case .still(let image) = content {
                return still(image, isOriginal: true, cached: cached, cachedIsOriginal: cachedIsOriginal)
            }
            return Self(content: content, isOriginal: true)
        } catch {
            if error is CancellationError || Task.isCancelled { throw CancellationError() }
        }
        do {
            let image = try await rendition()
            try Task.checkCancellation()
            return still(image, isOriginal: false, cached: cached, cachedIsOriginal: cachedIsOriginal)
        } catch {
            if error is CancellationError || Task.isCancelled { throw CancellationError() }
            guard let cached else { throw error }
            return Self(content: .still(cached), isOriginal: cachedIsOriginal)
        }
    }
}

struct PreviewEvidenceInput: Sendable {
    let id = UUID()
    let photoURL: URL
    let image: CGImage
    let isOriginal: Bool
    let thumbnail: CGImage?

    init(photoURL: URL, image: CGImage, isOriginal: Bool) {
        self.photoURL = photoURL
        self.image = image
        self.isOriginal = isOriginal
        self.thumbnail = EvidenceThumbnailRaster.make(image)
    }

    /// Preview renditions must never reuse, or populate, original-image results.
    func contentIdentity(assetIdentity: String, png: Data) -> String {
        let digest = SHA256.hash(data: png).map { String(format: "%02x", $0) }.joined()
        return LineageIdentity.hash("preview-evidence-v1|\(assetIdentity)|\(isOriginal)|\(digest)")
    }

    func photo(_ photo: LineagePhoto, png: Data) -> LineagePhoto {
        LineagePhoto(id: photo.id, contentIdentity: contentIdentity(assetIdentity: photo.contentIdentity, png: png),
                     name: photo.name, width: image.width, height: image.height, format: "PNG",
                     url: photo.url, offset: photo.offset, photosDate: photo.photosDate)
    }
}

/// Recently decoded inspection pixels, bounded independently of encoded originals.
/// Keys include the source revision; stale edits and other albums cannot alias.
final class PreviewImageCache: @unchecked Sendable {
    static let shared = PreviewImageCache()
    final class Entry: NSObject {
        let image: CGImage
        let isOriginal: Bool
        init(_ image: CGImage, isOriginal: Bool) {
            self.image = image; self.isOriginal = isOriginal
        }
    }
    private let cache = NSCache<NSString, Entry>()
    private let lock = NSLock()
    init() {
        cache.countLimit = 9
        cache.totalCostLimit = 128 * 1024 * 1024
    }
    func cached(_ key: String) -> Entry? { cache.object(forKey: key as NSString) }
    func store(_ image: CGImage, isOriginal: Bool, key: String) {
        lock.withLock {
            if let previous = cached(key),
               PreviewPhotoSource.isSharper(previous.image, than: image)
                || (previous.isOriginal && !isOriginal && !PreviewPhotoSource.isSharper(image, than: previous.image)) { return }
            cache.setObject(Entry(image, isOriginal: isOriginal), forKey: key as NSString,
                            cost: image.bytesPerRow * image.height)
        }
    }
}
