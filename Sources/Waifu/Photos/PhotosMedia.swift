import AppKit
import AVFoundation
import CoreImage
import CryptoKit
import ImageIO
import Photos
import SwiftUI

struct PhotosAlbum: Identifiable, Equatable, Sendable {
    let id: String
    let title: String
    let count: Int
    /// Built-in collections are listed separately from the user's album tree.
    let isSmartAlbum: Bool
    /// nil is a leaf album; an empty array is an empty folder.
    var children: [PhotosAlbum]? = nil

    var isFolder: Bool { children != nil }

    func find(_ identifier: String) -> PhotosAlbum? {
        if id == identifier { return self }
        for child in children ?? [] {
            if let match = child.find(identifier) { return match }
        }
        return nil
    }
}

/// PhotoKit supplies decoded photos and AVAssets directly. Originals are never
/// exported, copied, or opened through the Photos library's private filesystem.
final class PhotosMedia: @unchecked Sendable {
    static let shared = PhotosMedia()
    static let analysisPixelSize = 512
    static var downloadFromICloud: Bool { UserDefaults.standard.bool(forKey: "photosDownloadFromICloud") }
    private var skippedItems: [URL: Int] = [:]
    func resetSkippedItems(_ url: URL) { lock.withLock { skippedItems[url] = 0 } }
    func noteUnavailable(_ url: URL) { lock.withLock { skippedItems[url, default: 0] += 1 } }
    func skippedItemCount(_ url: URL) -> Int { lock.withLock { skippedItems[url] ?? 0 } }
    static func isUnavailableLocally(_ error: Error) -> Bool {
        let error = error as NSError
        return error.domain == "Waifu.Photos" && error.code == 2
    }
    static func localUnavailable() -> NSError {
        NSError(domain: "Waifu.Photos", code: 2, userInfo: [NSLocalizedDescriptionKey: "Not available locally. Enable Download from iCloud to load this item."])
    }
    /// Stable across missing items and sampling modes; microsecond precision is
    /// finer than the timestamps of supported photo/video sources.
    static func frameIndex(seconds: Double) -> Int { Int((max(0, seconds) * 1_000_000).rounded()) }

    private let renditionManager = PHCachingImageManager()
    private let renditionCache: NSCache<NSString, RenditionBox> = {
        let cache = NSCache<NSString, RenditionBox>()
        cache.totalCostLimit = 64 * 1024 * 1024
        cache.countLimit = 512
        return cache
    }()
    private final class RenditionBox {
        let image: CGImage
        let isComplete: Bool
        init(_ image: CGImage, isComplete: Bool) { self.image = image; self.isComplete = isComplete }
    }
    struct Item: @unchecked Sendable {
        let asset: PHAsset
        let url: URL
        let offset: Double
        var duration: Double { asset.mediaType == .video ? max(asset.duration, 0.001) : 1 }
        var isStill: Bool { asset.mediaType == .image }
        /// This photo's own cache identity, or nil for album video items, which
        /// have no per-photo rendition to key.
        var assetCacheKey: AssetCacheKey? {
            guard isStill else { return nil }
            let date = asset.modificationDate ?? asset.creationDate ?? .distantPast
            return AssetCacheKey(identifier: asset.localIdentifier,
                                 modifiedMilliseconds: UInt64(max(0, date.timeIntervalSince1970 * 1_000)),
                                 pixelWidth: UInt32(clamping: asset.pixelWidth),
                                 pixelHeight: UInt32(clamping: asset.pixelHeight))
        }
    }
    struct Album: @unchecked Sendable {
        let title: String
        let collectionIdentifier: String
        let canReorder: Bool
        let items: [Item]
        let offsets: [Double]
        let url: URL
        let fingerprint: String
        // Retain the fetch so PhotoKit continues reporting membership changes.
        let fetchResult: PHFetchResult<PHAsset>
        var duration: Double { items.last.map { $0.offset + $0.duration } ?? 1 }
    }
    /// One photo's own cache identity. Photos exposes no file to stat, so the
    /// asset's identifier, revision and pixel geometry stand in for a path, a
    /// size and a modification time. Album membership and album order are
    /// deliberately absent: PhotoKit does not promise the member order of an
    /// unsorted fetch, so anything derived from album position changes between
    /// sessions even when no photo did.
    struct AssetCacheKey: Sendable, Hashable {
        let identifier: String
        let modifiedMilliseconds: UInt64
        let pixelWidth: UInt32
        let pixelHeight: UInt32
    }

    struct VideoAsset: @unchecked Sendable { let value: AVAsset }
    private let lock = NSLock()
    private var albums: [URL: Album] = [:]
    private var items: [URL: Item] = [:]
    private var videoAssets: [URL: VideoAsset] = [:]
    // Keep sidebar fetches alive so new albums, folder changes and counts also
    // remain part of PhotoKit's observed results between refreshes.
    private var sidebarCollections: [PHFetchResult<PHCollection>] = []
    private var sidebarAssets: [PHFetchResult<PHAsset>] = []
    private var sidebarSmartAlbums: PHFetchResult<PHAssetCollection>?


    static func isPhotos(_ url: URL) -> Bool { url.scheme == "waifu-photos" }
    func album(_ url: URL) -> Album? { lock.withLock { albums[url] } }
    func item(_ url: URL) -> Item? { lock.withLock { items[url] } }
    /// PhotoKit's identifier fetch is synchronous and may wait on photolibraryd.
    /// Keep that wait off SwiftUI's main actor when refreshing a favorite badge.
    func favoriteStatus(_ url: URL) async -> Bool? {
        guard let identifier = item(url)?.asset.localIdentifier else { return nil }
        return await Task.detached(priority: .utility) {
            PHAsset.fetchAssets(withLocalIdentifiers: [identifier], options: nil).firstObject?.isFavorite
        }.value
    }
    func photoItemURL(for sourceURL: URL, seconds: Double) -> URL? {
        guard Self.isPhotos(sourceURL) else { return nil }
        let (resolved, _) = resolve(sourceURL, seconds: seconds)
        guard item(resolved)?.isStill == true else { return nil }
        return resolved
    }
    func collectionIdentifier(for sourceURL: URL) -> String? {
        album(sourceURL)?.collectionIdentifier
    }
    func assetIdentifier(for sourceURL: URL, seconds: Double) -> String? {
        let (resolved, _) = resolve(sourceURL, seconds: seconds)
        return item(resolved)?.asset.localIdentifier
    }

    /// Ask Photos to bring a library item into view using its AppleScript
    /// `spotlight` command. This passes only the PhotoKit identifier; it never
    /// exports or opens the original image file.
    func revealInPhotos(_ url: URL) async throws {
        guard let item = item(url), item.isStill else {
            throw Self.failure("This photo is no longer available in Photos.")
        }
        let identifier = item.asset.localIdentifier
        try await Task.detached(priority: .userInitiated) {
            let escapedIdentifier = identifier
                .replacingOccurrences(of: "\\", with: "\\\\")
                .replacingOccurrences(of: "\"", with: "\\\"")
            let source = """
            tell application id "com.apple.Photos"
                activate
                spotlight media item id "\(escapedIdentifier)"
            end tell
            """
            guard let script = NSAppleScript(source: source) else {
                throw Self.failure("Could not prepare the Photos reveal request.")
            }
            var errorInfo: NSDictionary?
            _ = script.executeAndReturnError(&errorInfo)
            if let errorInfo {
                let message = errorInfo["NSAppleScriptErrorMessage"] as? String
                    ?? "Allow WaifuFinder to control Photos in System Settings → Privacy & Security → Automation."
                throw Self.failure(message)
            }
        }.value
    }

    func resolve(_ url: URL, seconds: Double) -> (URL, Double) {
        guard let album = album(url), let index = Self.itemIndex(offsets: album.offsets, seconds: seconds) else { return (url, seconds) }
        let item = album.items[index]
        return (item.url, max(0, min(seconds - item.offset, item.duration - 0.000001)))
    }
    static func itemIndex(offsets: [Double], seconds: Double) -> Int? {
        guard !offsets.isEmpty else { return nil }
        var low = 0, high = offsets.count
        while low < high {
            let middle = (low + high) / 2
            if offsets[middle] <= seconds { low = middle + 1 } else { high = middle }
        }
        return max(0, low - 1)
    }
    func isStill(_ url: URL) -> Bool { item(url)?.isStill == true }

    /// The album photo behind one frame, and its own cache identity. Returns nil
    /// for a non-Photos source, for an album video item, and for a timestamp
    /// that no longer falls inside a member — so a caller caching per photo
    /// stores nothing rather than the wrong photo's results.
    func photoAssetCacheKey(for sourceURL: URL, seconds: Double) -> AssetCacheKey? {
        guard Self.isPhotos(sourceURL) else { return nil }
        let (resolved, _) = resolve(sourceURL, seconds: seconds)
        return item(resolved)?.assetCacheKey
    }

    func setFavorite(_ urls: [URL], isFavorite: Bool) async throws {
        var assets: [PHAsset] = []
        var identifiers = Set<String>()
        for url in urls {
            guard let item = item(url), item.isStill else {
                throw Self.failure("This photo is no longer available in Photos.")
            }
            if identifiers.insert(item.asset.localIdentifier).inserted {
                assets.append(item.asset)
            }
        }
        guard !assets.isEmpty else {
            throw Self.failure("No photos are available to update.")
        }
        let status = PHPhotoLibrary.authorizationStatus(for: .readWrite)
        guard status == .authorized || status == .limited else {
            throw Self.failure("Allow WaifuFinder to make changes to Photos in System Settings → Privacy & Security → Photos.")
        }
        try await PHPhotoLibrary.shared().performChanges {
            for asset in assets {
                PHAssetChangeRequest(for: asset).isFavorite = isFavorite
            }
        }
    }

    /// Commit the displayed order to the real album using PhotoKit's tracked
    /// snapshot, including media outside this app's still-photo view.
    func syncOrder(collectionIdentifier: String, assetIdentifiers: [String]) async throws {
        try await PhotosOrderSync.run {
            try await self.syncOrderAttempt(collectionIdentifier: collectionIdentifier,
                                            assetIdentifiers: assetIdentifiers)
        }
    }

    private func syncOrderAttempt(collectionIdentifier: String, assetIdentifiers: [String]) async throws {
        try Task.checkCancellation()
        let status = PHPhotoLibrary.authorizationStatus(for: .readWrite)
        guard status == .authorized || status == .limited else {
            throw Self.failure("Allow WaifuFinder to make changes in System Settings → Privacy & Security → Photos.")
        }
        guard let collection = PHAssetCollection.fetchAssetCollections(
            withLocalIdentifiers: [collectionIdentifier], options: nil).firstObject else {
            throw Self.failure("This album is no longer available in Photos.")
        }
        guard collection.canPerform(.rearrangeContent) else {
            throw Self.failure("Photos does not allow custom ordering in this album. Choose a regular album under My Albums.")
        }
        let assets = PHAsset.fetchAssets(in: collection, options: nil)
        var identifiers: [String] = []
        assets.enumerateObjects { asset, _, _ in identifiers.append(asset.localIdentifier) }
        let moves = try PhotosAlbumOrder.moves(current: identifiers, ordered: assetIdentifiers)
        guard !moves.isEmpty else { return }
        try await PHPhotoLibrary.shared().performChanges {
            // canPerform was checked above: unsupported requests can raise an
            // Objective-C exception instead of returning a Swift error.
            guard let request = PHAssetCollectionChangeRequest(for: collection, assets: assets) else { return }
            for move in moves {
                request.moveAssets(at: IndexSet(integer: move.from), to: move.to)
            }
        }
        let updated = PHAsset.fetchAssets(in: collection, options: nil)
        let requested = Set(assetIdentifiers)
        var synced: [String] = []
        updated.enumerateObjects { asset, _, _ in
            if requested.contains(asset.localIdentifier) { synced.append(asset.localIdentifier) }
        }
        guard synced == assetIdentifiers else {
            throw PhotosOrderSync.VerificationError.orderNotVisible
        }
    }

    func listAlbums() async throws -> [PhotosAlbum] {
        let status = await PHPhotoLibrary.requestAuthorization(for: .readWrite)
        guard status == .authorized || status == .limited else {
            throw Self.failure("Allow Photos access in System Settings → Privacy & Security → Photos, then try again.")
        }
        PhotosLibraryChanges.shared.start()
        // A flat fetchAssetCollections(.album) is not the user's sidebar order.
        // Read the ordered root collection list and each folder's child list.
        // Do not apply a sort descriptor or flatten albums out of their folders.
        var visited = Set<String>()
        var collectionFetches: [PHFetchResult<PHCollection>] = []
        var assetFetches: [PHFetchResult<PHAsset>] = []
        func readCollections(_ collections: PHFetchResult<PHCollection>) -> [PhotosAlbum] {
            collectionFetches.append(collections)
            var rows: [PhotosAlbum] = []
            for index in 0..<collections.count {
                let collection = collections.object(at: index)
                guard visited.insert(collection.localIdentifier).inserted else { continue }
                if let folder = collection as? PHCollectionList {
                    let children = readCollections(PHCollection.fetchCollections(in: folder, options: nil))
                    rows.append(PhotosAlbum(id: folder.localIdentifier,
                                            title: folder.localizedTitle ?? "Folder",
                                            count: children.reduce(0) { $0 + $1.count },
                                            isSmartAlbum: false, children: children))
                } else if let album = collection as? PHAssetCollection {
                    let assets = PHAsset.fetchAssets(in: album, options: Self.fetchOptions())
                    assetFetches.append(assets)
                    rows.append(PhotosAlbum(id: album.localIdentifier,
                                            title: album.localizedTitle ?? "Album",
                                            count: assets.count,
                                            isSmartAlbum: false))
                }
            }
            return rows
        }
        var rows = readCollections(PHCollection.fetchTopLevelUserCollections(with: nil))
        let smartAlbums = PHAssetCollection.fetchAssetCollections(with: .smartAlbum, subtype: .any, options: nil)
        smartAlbums.enumerateObjects { collection, _, _ in
            guard visited.insert(collection.localIdentifier).inserted else { return }
            let assets = PHAsset.fetchAssets(in: collection, options: Self.fetchOptions())
            assetFetches.append(assets)
            let count = assets.count
            if count > 0 {
                rows.append(PhotosAlbum(id: collection.localIdentifier,
                                        title: collection.localizedTitle ?? "Album",
                                        count: count, isSmartAlbum: true))
            }
        }
        lock.withLock {
            sidebarCollections = collectionFetches
            sidebarAssets = assetFetches
            sidebarSmartAlbums = smartAlbums
        }
        return rows
    }

    func openAlbum(_ selection: PhotosAlbum, allowEmpty: Bool = false) throws -> URL {
        guard !selection.isFolder else { throw Self.failure("Choose an album inside this folder.") }
        guard let collection = PHAssetCollection.fetchAssetCollections(withLocalIdentifiers: [selection.id], options: nil).firstObject else {
            throw Self.failure("This album is no longer available in Photos.")
        }
        let session = UUID().uuidString
        let url = URL(string: "waifu-photos://album/\(session)")!
        var entries: [Item] = [], offset = 0.0
        let fetched = PHAsset.fetchAssets(in: collection, options: Self.fetchOptions())
        fetched.enumerateObjects { asset, _, _ in
            let item = Item(asset: asset, url: URL(string: "waifu-photos://asset/\(session)-\(entries.count)")!, offset: offset)
            entries.append(item)
            offset += item.duration
        }
        guard allowEmpty || !entries.isEmpty else { throw Self.failure("This album contains no accessible photos.") }
        lock.withLock {
            albums[url] = Album(title: selection.title, collectionIdentifier: collection.localIdentifier,
                                canReorder: collection.canPerform(.rearrangeContent),
                                items: entries, offsets: entries.map(\.offset), url: url,
                                fingerprint: Self.fingerprint(entries), fetchResult: fetched)
            for entry in entries { items[entry.url] = entry }
        }
        return url
    }

    struct Revision: Sendable {
        let identifier: String
        let modified: Double
        let width: Int
        let height: Int
        let duration: Double
    }
    static func fingerprint(_ revisions: [Revision]) -> String {
        var hash = SHA256()
        for revision in revisions {
            // Length prefixes distinguish identifiers containing delimiters.
            let text = "\(revision.identifier.utf8.count):\(revision.identifier)|\(revision.modified)|\(revision.width)x\(revision.height)|\(revision.duration);"
            hash.update(data: Data(text.utf8))
        }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }
    private static func fingerprint(_ entries: [Item]) -> String {
        fingerprint(entries.map { Revision(identifier: $0.asset.localIdentifier,
            modified: ($0.asset.modificationDate ?? $0.asset.creationDate ?? .distantPast).timeIntervalSince1970,
            width: $0.asset.pixelWidth, height: $0.asset.pixelHeight, duration: $0.duration) })
    }
    func cacheSource(_ url: URL, mode: AnalysisMode) -> (path: String, count: UInt64)? {
        guard let album = album(url) else { return nil }
        return ("photos-rendition-v3/\(mode.rawValue)/\(album.fingerprint)", UInt64(album.items.count))
    }

    static func renditionSize(width: Int, height: Int, maxPixelSize: Int) -> CGSize {
        let w = CGFloat(max(1, width)), h = CGFloat(max(1, height))
        let scale = min(1, CGFloat(max(1, min(analysisPixelSize, maxPixelSize))) / max(w, h))
        return CGSize(width: max(1, (w * scale).rounded()), height: max(1, (h * scale).rounded()))
    }

    func cachedRendition(_ url: URL, maxPixelSize: Int) -> CGImage? {
        guard let asset = item(url)?.asset else { return nil }
        let size = Self.renditionSize(width: asset.pixelWidth, height: asset.pixelHeight, maxPixelSize: maxPixelSize)
        let revision = (asset.modificationDate ?? asset.creationDate ?? .distantPast).timeIntervalSince1970
        let key = "\(asset.localIdentifier)|\(revision)|\(Int(size.width))x\(Int(size.height))" as NSString
        return renditionCache.object(forKey: key)?.image
    }

    /// PhotoKit's fast delivery can return a nearby cached size. Enforce the
    /// pixel bound ourselves without requesting or decoding the original.
    static func boundedRendition(_ image: CGImage, maxPixelSize: Int) throws -> CGImage {
        let size = renditionSize(width: image.width, height: image.height, maxPixelSize: maxPixelSize)
        guard image.width > Int(size.width) || image.height > Int(size.height) else { return image }
        guard let context = CGContext(data: nil, width: Int(size.width), height: Int(size.height),
                                      bitsPerComponent: 8, bytesPerRow: 0,
                                      space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw failure("Could not resize the Photos thumbnail.")
        }
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(origin: .zero, size: size))
        guard let result = context.makeImage() else { throw failure("Could not resize the Photos thumbnail.") }
        return result
    }

    static func hasRequestedResolution(_ image: CGImage, targetSize: CGSize) -> Bool {
        max(image.width, image.height) >= Int(max(targetSize.width, targetSize.height))
    }

    /// NSImage.size is measured in points. Choose the actual bitmap pixels rather
    /// than letting a point-sized drawing request select a smaller representation.
    static func renditionPixels(_ image: NSImage) -> CGImage? {
        if let bitmap = image.representations.compactMap({ $0 as? NSBitmapImageRep }).max(by: {
            $0.pixelsWide * $0.pixelsHigh < $1.pixelsWide * $1.pixelsHigh
        }), let pixels = bitmap.cgImage { return pixels }
        return image.cgImage(forProposedRect: nil, context: nil, hints: nil)
    }

    private func storeRendition(_ image: CGImage, key: String, isComplete: Bool) {
        lock.withLock {
            if let previous = renditionCache.object(forKey: key as NSString) {
                if previous.isComplete { return }
                if previous.image.width * previous.image.height > image.width * image.height { return }
            }
            renditionCache.setObject(RenditionBox(image, isComplete: isComplete), forKey: key as NSString,
                                     cost: image.bytesPerRow * image.height)
        }
    }

    /// Show degraded cached pixels immediately, then replace them with the exact
    /// requested rendition. Tiny previews never satisfy the completed cache key.
    func renditionPreviews(_ url: URL, maxPixelSize: Int = analysisPixelSize,
                           allowNetwork: Bool) -> AsyncThrowingStream<CGImage, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let final = try await rendition(url, maxPixelSize: maxPixelSize, allowNetwork: allowNetwork,
                                                    onPreview: { continuation.yield($0) })
                    continuation.yield(final)
                    continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Analysis requests 512px; tagging requests its own model-sized rendition.
    /// Prefer the final requested pixels, with a smaller cached thumbnail as a
    /// fallback if Photos cannot supply a larger local rendition.
    func rendition(_ url: URL, maxPixelSize: Int = analysisPixelSize, allowNetwork: Bool,
                   onPreview: (@Sendable (CGImage) -> Void)? = nil) async throws -> CGImage {
        try Task.checkCancellation()
        guard let item = item(url) else { throw Self.failure("This Photos item is no longer available.") }
        let asset = item.asset
        let size = Self.renditionSize(width: asset.pixelWidth, height: asset.pixelHeight, maxPixelSize: maxPixelSize)
        let revision = (asset.modificationDate ?? asset.creationDate ?? .distantPast).timeIntervalSince1970
        let key = "\(asset.localIdentifier)|\(revision)|\(Int(size.width))x\(Int(size.height))"
        let cached = renditionCache.object(forKey: key as NSString)
        if let cached, cached.isComplete { return cached.image }
        if let cached { onPreview?(cached.image) }
        let delivery = PhotosRenditionDelivery(initial: cached?.image)
        let options = PHImageRequestOptions()
        options.deliveryMode = .opportunistic
        options.resizeMode = .exact
        options.version = .current
        options.isNetworkAccessAllowed = allowNetwork
        options.isSynchronous = false
        let manager = renditionManager
        let request = PhotosRequest<CGImage>(
            timeout: allowNetwork ? 120 : 10,
            timeoutError: allowNetwork ? Self.failure("Photos did not return this rendition. Try again.") : Self.localUnavailable(),
            timeoutFallback: { delivery.bestImage },
            // Capture the PhotosMedia owner (already audited as Sendable) rather
            // than PhotosKit's non-Sendable manager directly.
            cancel: { [self] requestID in renditionManager.cancelImageRequest(requestID) })
        let image = try await request.perform { completion in
            manager.requestImage(for: asset, targetSize: size, contentMode: .aspectFit, options: options) { image, info in
                if (info?[PHImageCancelledKey] as? Bool) == true { completion(.failure(CancellationError())); return }
                let degraded = (info?[PHImageResultIsDegradedKey] as? Bool) == true
                guard let image else {
                    let message = (info?[PHImageResultIsInCloudKey] as? Bool) == true
                        ? "Available in iCloud" : "Preview unavailable"
                    if let result = delivery.receive(nil, isDegraded: degraded,
                        error: !allowNetwork ? Self.localUnavailable() : (info?[PHImageErrorKey] as? Error ?? Self.failure(message))) {
                        completion(result)
                    }
                    return
                }
                guard let cgImage = Self.renditionPixels(image) else {
                    completion(.failure(Self.failure("Photos returned an unreadable rendition."))); return
                }
                do {
                    let bounded = try Self.boundedRendition(cgImage, maxPixelSize: maxPixelSize)
                    self.storeRendition(bounded, key: key,
                                        isComplete: !degraded && Self.hasRequestedResolution(bounded, targetSize: size))
                    let result = delivery.receive(bounded, isDegraded: degraded)
                    if let preview = delivery.bestImage { onPreview?(preview) }
                    if let result { completion(result) }
                } catch { completion(.failure(error)) }
            }
        }
        try Task.checkCancellation()
        return image
    }

    /// Regenerating an analysis/tagging thumbnail must not reopen the original.
    func analysisImage(_ url: URL, seconds: Double, maxPixelSize: Int = analysisPixelSize) async throws -> CGImage {
        let (source, time) = resolve(url, seconds: seconds)
        if isStill(source) { return try await rendition(source, maxPixelSize: maxPixelSize, allowNetwork: Self.downloadFromICloud) }
        let asset = try await videoAsset(source).value
        return try await PreviewFrameReader(asset: asset, maximumPixelSize: min(Self.analysisPixelSize, maxPixelSize)).image(at: time)
    }

    func release(_ url: URL) {
        lock.withLock {
            skippedItems.removeValue(forKey: url)
            guard let album = albums.removeValue(forKey: url) else { return }
            for item in album.items {
                items.removeValue(forKey: item.url)
                videoAssets.removeValue(forKey: item.url)
            }
        }
    }

    private static func fetchOptions() -> PHFetchOptions {
        let options = PHFetchOptions()
        options.predicate = NSPredicate(format: "mediaType == %d", PHAssetMediaType.image.rawValue)
        // Keep PhotoKit's collection sequence, including user-defined album order.
        options.sortDescriptors = nil
        return options
    }

    func videoAsset(_ url: URL) async throws -> VideoAsset {
        throw Self.failure("Photos integration supports still photos only. Open video files or URLs from the main loading panel.")
    }

    /// Reads a Photos still directly into memory and detects animated formats for
    /// preview playback. Ordinary analysis continues to use small PhotoKit renditions.
    func previewContent(_ url: URL) async throws -> PhotosPreviewContent {
        guard let item = item(url), item.isStill else {
            throw Self.failure("This photo is no longer available in Photos.")
        }
        let options = PHImageRequestOptions()
        options.version = .current
        options.deliveryMode = .highQualityFormat
        options.isNetworkAccessAllowed = Self.downloadFromICloud
        let allowNetwork = options.isNetworkAccessAllowed
        let request = PhotosRequest<(Data, CGImagePropertyOrientation)>(
            timeout: allowNetwork ? 120 : 10,
            timeoutError: allowNetwork ? Self.failure("Photos did not return this image. Try again.") : Self.localUnavailable())
        let result = try await request.perform { completion in
            PHImageManager.default().requestImageDataAndOrientation(for: item.asset, options: options) { data, _, orientation, info in
                guard let data else {
                    completion(.failure(!allowNetwork ? Self.localUnavailable() : (info?[PHImageErrorKey] as? Error ?? Self.failure("Photos could not read this image."))))
                    return
                }
                completion(.success((data, orientation)))
            }
        }
        try Task.checkCancellation()
        let (imageData, orientation) = result
        guard let source = CGImageSourceCreateWithData(imageData as CFData, nil) else {
            throw Self.failure("Photos returned an unreadable image.")
        }
        let count = CGImageSourceGetCount(source)
        let type = CGImageSourceGetType(source) as String?
        if count > 1, ["com.compuserve.gif", "public.png", "org.webmproject.webp"].contains(type ?? "") {
            let modified = item.asset.modificationDate?.timeIntervalSince1970 ?? 0
            return .animated(PhotosAnimatedImage(source: source, identity: "\(item.asset.localIdentifier):\(modified)", orientation: orientation))
        }
        guard let input = CIImage(data: imageData)?.oriented(orientation),
              let image = CIContext(options: [.cacheIntermediates: false]).createCGImage(input, from: input.extent) else {
            throw Self.failure("Photos returned an unreadable image.")
        }
        return .still(image)
    }

    func image(_ url: URL, seconds: Double) async throws -> CGImage {
        let (resolved, time) = resolve(url, seconds: seconds)
        guard let item = item(resolved) else {
            return try await PreviewFrameReader(asset: AVURLAsset(url: resolved)).image(at: time)
        }
        if !item.isStill { return try await PreviewFrameReader(asset: try await videoAsset(resolved).value).image(at: time) }
        let options = PHImageRequestOptions()
        options.isNetworkAccessAllowed = Self.downloadFromICloud
        options.deliveryMode = .highQualityFormat
        options.version = .current
        let request = PhotosRequest<CGImage>()
        let allowNetwork = options.isNetworkAccessAllowed
        let image: CGImage = try await request.perform { completion in
            PHImageManager.default().requestImageDataAndOrientation(for: item.asset, options: options) { data, _, orientation, info in
                guard let data else {
                    completion(.failure(!allowNetwork ? Self.localUnavailable() : (info?[PHImageErrorKey] as? Error ?? Self.failure("Photos could not read this image. Check your iCloud connection.")))); return
                }
                // Apply EXIF orientation without serializing the original to disk.
                guard let input = CIImage(data: data)?.oriented(orientation),
                      let image = CIContext(options: [.cacheIntermediates: false]).createCGImage(input, from: input.extent) else {
                    completion(.failure(Self.failure("Photos returned an unreadable image."))); return
                }
                completion(.success(image))
            }
        }
        try Task.checkCancellation()
        return image
    }

    static func asset(_ url: URL) async throws -> VideoAsset {
        if isPhotos(url) { return try await shared.videoAsset(url) }
        return VideoAsset(value: VideoAssetFactory.asset(for: url))
    }
    static func failure(_ message: String) -> NSError { NSError(domain: "Waifu.Photos", code: 1, userInfo: [NSLocalizedDescriptionKey: message]) }
}

/// A degraded callback is a placeholder, not the end of an opportunistic request.
/// Keep the best delivered pixels as a fallback if the final callback fails.
final class PhotosRenditionDelivery: @unchecked Sendable {
    private let lock = NSLock()
    private var best: CGImage?

    init(initial: CGImage? = nil) { best = initial }
    var bestImage: CGImage? { lock.withLock { best } }

    func receive(_ image: CGImage?, isDegraded: Bool,
                 error: Error = PhotosMedia.localUnavailable()) -> Result<CGImage, Error>? {
        lock.withLock {
            if let image, best == nil || image.width * image.height >= best!.width * best!.height {
                best = image
            }
            if isDegraded { return nil }
            return best.map { .success($0) } ?? .failure(error)
        }
    }
}

/// Cancellation completes the awaiting task immediately and cancels the PhotoKit
/// request, including iCloud downloads. Callback races resume each waiter once.
final class PhotosRequest<Value: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Value, Error>?
    private var result: Result<Value, Error>?
    private var requestID: PHImageRequestID?
    private let cancelRequest: @Sendable (PHImageRequestID) -> Void
    private let timeout: TimeInterval?
    private let timeoutError: Error
    private let timeoutFallback: (@Sendable () -> Value?)?
    init(timeout: TimeInterval? = nil, timeoutError: Error = PhotosMedia.failure("Photos did not answer this request."),
         timeoutFallback: (@Sendable () -> Value?)? = nil,
         cancel: @escaping @Sendable (PHImageRequestID) -> Void = { PHImageManager.default().cancelImageRequest($0) }) {
        cancelRequest = cancel
        self.timeout = timeout
        self.timeoutError = timeoutError
        self.timeoutFallback = timeoutFallback
    }
    func perform(_ begin: (@escaping @Sendable (Result<Value, Error>) -> Void) -> PHImageRequestID) async throws -> Value {
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                let previous = lock.withLock { () -> Result<Value, Error>? in
                    if let result { return result }
                    self.continuation = continuation
                    return nil
                }
                if let previous { continuation.resume(with: previous); return }
                let id = begin { [self] in finish($0) }
                let cancelled = lock.withLock { requestID = id; return result != nil }
                if cancelled { cancelRequest(id) }
                if let timeout {
                    DispatchQueue.global().asyncAfter(deadline: .now() + timeout) { [weak self] in
                        guard let self else { return }
                        self.finish(self.timeoutFallback?().map { .success($0) } ?? .failure(self.timeoutError))
                    }
                }
            }
        } onCancel: { self.finish(.failure(CancellationError())) }
    }
    private func finish(_ value: Result<Value, Error>) {
        let state = lock.withLock { () -> (CheckedContinuation<Value, Error>?, PHImageRequestID?) in
            guard result == nil else { return (nil, nil) }
            result = value
            let waiter = continuation; continuation = nil
            return (waiter, requestID)
        }
        state.0?.resume(with: value)
        if let id = state.1 { cancelRequest(id) }
    }
}

