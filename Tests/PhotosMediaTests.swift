import Foundation
import AVFoundation
import Photos

func registerPhotosMediaTests() {
    Registry.shared.add(suite: "Photos media", TestCase(name: "confirmed deletions update the loaded album without replacing surviving photos", knownBug: nil) {
        let source = URL(string: "waifu-photos://album/confirmed-delete")!
        let items = (0..<3).map { index in
            PhotosMedia.Item(asset: DeletedPhotoTestAsset("photo-\(index)"),
                url: URL(string: "waifu-photos://asset/delete-\(index)")!, offset: Double(index))
        }
        let album = PhotosMedia.Album(title: "Delete test", collectionIdentifier: "test-album", canReorder: true,
            items: items, offsets: items.map(\.offset), url: source, fingerprint: "original",
            fetchResult: PHFetchResult<PHAsset>())
        let media = PhotosMedia(albums: [album])
        media.removeDeletedAssets(["photo-1"], from: source)
        let survivors = media.album(source)!
        checkEqual(survivors.url, source, "the workspace source identity stays mounted")
        checkEqual(survivors.items.map(\.url), [items[0].url, items[2].url], "only the confirmed photo disappears")
        checkEqual(survivors.offsets, [0, 2], "surviving frame IDs and timestamps remain valid")
        checkEqual(media.resolve(source, seconds: 2).0, items[2].url, "surviving previews still resolve to their original photo")
        checkNil(media.item(items[1].url), "deleted media can no longer be requested")
        check(media.album(source)?.fingerprint != album.fingerprint, "source caches no longer describe the deleted photo")
        media.removeDeletedAssets(["photo-1"], from: source)
        checkEqual(media.album(source)?.fingerprint, survivors.fingerprint, "applying the same confirmation twice is harmless")
        media.removeDeletedAssets(["photo-0", "photo-2"], from: source)
        checkEqual(media.album(source)?.items.count, 0, "deleting the last photos leaves the same empty album open")
        check(media.album(source)!.duration > 0, "the empty source remains valid")
    })

    Registry.shared.add(suite: "Photos media", TestCase(name: "album frame timestamps resolve to their owning item", knownBug: nil) {
        let offsets: [Double] = [0, 1, 3.125, 4.125]
        checkEqual(PhotosMedia.itemIndex(offsets: offsets, seconds: 0), 0, "first photo")
        checkEqual(PhotosMedia.itemIndex(offsets: offsets, seconds: 1), 1, "exact video start")
        checkEqual(PhotosMedia.itemIndex(offsets: offsets, seconds: 3.1249), 1, "video last frame")
        checkEqual(PhotosMedia.itemIndex(offsets: offsets, seconds: 3.125), 2, "following photo start")
        checkEqual(PhotosMedia.itemIndex(offsets: offsets, seconds: -1), 0, "negative time clamps to first")
        checkEqual(PhotosMedia.itemIndex(offsets: offsets, seconds: 100), 3, "end clamps to last")
        checkNil(PhotosMedia.itemIndex(offsets: [], seconds: 0), "empty album has no item")
    })
    Registry.shared.add(suite: "Photos media", TestCase(name: "temporal restoration algorithms are unavailable for still photos", knownBug: nil) {
        check(UncoverAlgorithm.realESRGAN.supportsStillPhotos, "spatial upscaling remains available")
        for algorithm in [UncoverAlgorithm.seedVR2, .seedVR2SevenB, .flashVSR11, .pgtFormer] {
            check(!algorithm.supportsStillPhotos, "temporal models require source frames")
        }
    })
    Registry.shared.add(suite: "Photos media", TestCase(name: "shared native frame reader preserves local media behavior", knownBug: nil) {
        let dir = try Temporary.directory("photos-reader-local")
        let clip = try VideoFixture.makeMP4(in: dir, durationSeconds: 1, width: 64, height: 48, frameRate: 5)
        let image = try await PhotosMedia.shared.image(clip, seconds: 0)
        checkEqual(image.width, 64, "full resolution width")
        checkEqual(image.height, 48, "full resolution height")
        check(!PhotosMedia.isPhotos(clip), "local videos retain file source semantics")
        let resolved = PhotosMedia.shared.resolve(clip, seconds: 0.4)
        checkEqual(resolved.0, clip, "local source unchanged")
        checkEqual(resolved.1, 0.4, "local timestamp unchanged")
    })
    Registry.shared.add(suite: "Photos media", TestCase(name: "album player translates virtual times into the current video", knownBug: nil) {
        let dir = try Temporary.directory("photos-player-offset")
        let clip = try VideoFixture.makeMP4(in: dir, durationSeconds: 1, width: 64, height: 48, frameRate: 5)
        await MainActor.run {
            let preview = PreviewController()
            preview.load(asset: AVURLAsset(url: clip), offset: 10, mediaDuration: 1, timelineDuration: 20)
            preview.seek(to: 10.4)
            checkEqual(preview.currentTime, 10.4, "display time stays in album coordinates")
            checkEqual(preview.duration, 20, "timeline spans the entire album")
            preview.seek(to: 1)
            checkEqual(preview.currentTime, 10, "seek cannot escape the selected asset")
            preview.loadStill(time: 15, duration: 20)
            checkEqual(preview.currentTime, 15, "static photos retain album location")
            checkNil(preview.player, "static photos do not create a fake video")
            preview.unload()
        }
    })

    Registry.shared.add(suite: "Photos media", TestCase(name: "PhotoKit analysis renditions fit inside 512 pixels without upscaling", knownBug: nil) {
        for (width, height) in [(6000, 4000), (3000, 5000), (12000, 400), (64, 48)] {
            let size = PhotosMedia.renditionSize(width: width, height: height, maxPixelSize: 4096)
            check(size.width <= 512 && size.height <= 512, "large request stays bounded")
            check(size.width <= Double(width) && size.height <= Double(height), "small photos stay native sized")
            let thumb = PhotosMedia.renditionSize(width: width, height: height, maxPixelSize: 256)
            check(thumb.width <= 256 && thumb.height <= 256, "browser asks for smaller renditions")
        }
    })
    Registry.shared.add(suite: "Photos media", TestCase(name: "album cache revisions are stable and invalidate edited or reordered assets", knownBug: nil) {
        let first = PhotosMedia.Revision(identifier: "asset-A", modified: 10, width: 4000, height: 3000, duration: 1)
        let second = PhotosMedia.Revision(identifier: "asset-B", modified: 20, width: 4000, height: 3000, duration: 1)
        let edited = PhotosMedia.Revision(identifier: "asset-A", modified: 11, width: 4000, height: 3000, duration: 1)
        let key = PhotosMedia.fingerprint([first, second])
        checkEqual(key, PhotosMedia.fingerprint([first, second]), "reopening the same assets reuses analysis")
        check(key != PhotosMedia.fingerprint([edited, second]), "an edit invalidates old analysis")
        check(key != PhotosMedia.fingerprint([second, first]), "order affects frame correspondence")
        check(key != PhotosMedia.fingerprint([first]), "album membership affects frame correspondence")
        checkEqual(PhotosMedia.frameIndex(seconds: 2), 2_000_000, "frame identity is independent of skipped items")
    })

    Registry.shared.add(suite: "Photos media", TestCase(name: "deleting the last photo leaves a valid empty sampling source", knownBug: nil) {
        let url = URL(string: "waifu-photos://album/empty-test")!
        let album = PhotosMedia.Album(title: "Empty album", collectionIdentifier: "empty-test",
            canReorder: true, items: [], offsets: [], url: url,
            fingerprint: PhotosMedia.fingerprint([]), fetchResult: PHFetchResult<PHAsset>())
        let sampler = PhotosAlbumSampler(album: album, width: 512, mode: .full)
        checkNil(try await sampler.next(), "an emptied album completes without decoding deleted photos")
        check(album.duration > 0, "the empty source still has a valid probe duration")
        let photo = PhotosMedia.Revision(identifier: "last-photo", modified: 1, width: 100, height: 100, duration: 1)
        check(album.fingerprint != PhotosMedia.fingerprint([photo]), "deleting the last member invalidates the previous source")
    })

    Registry.shared.add(suite: "Photos media", TestCase(name: "an unanswered thumbnail request times out instead of blocking an album", knownBug: nil) {
        let request = PhotosRequest<Int>(timeout: 0.01, timeoutError: PhotosMedia.localUnavailable(), cancel: { _ in })
        let start = Date()
        let error = await checkThrowsAsync("missing local rendition skips") {
            try await request.perform { _ in 42 }
        }
        check(error.map { PhotosMedia.isUnavailableLocally($0) } == true, "timeout is classified as locally unavailable")
        check(Date().timeIntervalSince(start) < 1, "failed request does not strand the worker")
    })

}

/// PhotoKit metadata only; these fixtures never access or mutate a real library.
private final class DeletedPhotoTestAsset: PHAsset, @unchecked Sendable {
    private let identifier: String
    init(_ identifier: String) { self.identifier = identifier; super.init() }
    override var localIdentifier: String { identifier }
    override var mediaType: PHAssetMediaType { .image }
}
