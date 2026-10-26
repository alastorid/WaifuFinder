import AVFoundation
import Foundation

func registerPreviewFrameTests() {
    Registry.shared.add(suite: "Frame preview", TestCase(name: "a delayed local fallback never downgrades sharper cached photo pixels", knownBug: nil) {
        let sharp = try ImageFixture.cgImage(width: 2048, height: 1536)
        let small = try ImageFixture.cgImage(width: 512, height: 384)
        let result = try await PreviewPhotoSource.load(
            original: { throw PhotosMedia.localUnavailable() }, rendition: { small }, cached: sharp)
        guard case .still(let pixels) = result.content else { check(false, "still image required"); return }
        check(pixels === sharp, "the locally cached sharp photo survives a smaller Photos fallback")
        check(!result.isOriginal, "a large thumbnail is not mislabeled as original")
        check(!PreviewPhotoSource.isSharper(small, than: sharp), "same-photo thumbnail refresh cannot replace sharper pixels")
        check(PreviewPhotoSource.isSharper(small, than: nil), "a new photo can display its own smaller thumbnail")
        check(PreviewPhotoSource.isSharper(sharp, than: small), "a larger local rendition can still upgrade the photo")
    })
    Registry.shared.add(suite: "Frame preview", TestCase(name: "same-photo reload retains original provenance when Photos is unavailable", knownBug: nil) {
        let sharp = try ImageFixture.cgImage(width: 640, height: 480)
        let small = try ImageFixture.cgImage(width: 64, height: 48)
        for unavailable in [false, true] {
            let result = try await PreviewPhotoSource.load(
                original: { throw PhotosMedia.localUnavailable() }, rendition: {
                    if unavailable { throw PhotosMedia.localUnavailable() }
                    return small
                }, cached: sharp, cachedIsOriginal: true)
            guard case .still(let pixels) = result.content else { check(false, "still image required"); return }
            check(pixels === sharp && result.isOriginal, "retained original stays sharp and remains the evidence input")
        }
    })
    Registry.shared.add(suite: "Frame preview", TestCase(name: "photo preview upgrades smaller cached pixels to larger locally available pixels", knownBug: nil) {
        let sharp = try ImageFixture.cgImage(width: 640, height: 480)
        let small = try ImageFixture.cgImage(width: 64, height: 48)
        let result = try await PreviewPhotoSource.load(
            original: { throw PhotosMedia.localUnavailable() }, rendition: { sharp }, cached: small)
        guard case .still(let pixels) = result.content else { check(false, "still image required"); return }
        check(pixels === sharp, "larger fallback upgrades the thumbnail")
        let original = try await PreviewPhotoSource.load(original: { .still(sharp) }, rendition: { small }, cached: small)
        guard case .still(let native) = original.content else { check(false, "original required"); return }
        check(native === sharp && original.isOriginal, "available local original upgrades the thumbnail")
    })
    Registry.shared.add(suite: "Frame preview", TestCase(name: "cloud original failure keeps a local image available to both evidence models", knownBug: nil) {
        let image = try ImageFixture.cgImage(width: 96, height: 64)
        let source = try await PreviewPhotoSource.load(
            original: { throw NSError(domain: "CloudPhotoLibraryErrorDomain", code: 1005) },
            rendition: { image })
        guard case .still(let pixels) = source.content else { check(false, "still fallback required"); return }
        check(pixels === image, "the same decoded rendition feeds preview and evidence")
        check(!source.isOriginal, "rendition analysis is identified explicitly")
        let original = try await PreviewPhotoSource.load(
            original: { .still(image) }, rendition: { throw PhotosMedia.localUnavailable() })
        check(original.isOriginal, "available originals remain the preferred input")
    })
    Registry.shared.add(suite: "Frame preview", TestCase(name: "failed Photos reads retain the target thumbnail and cancellation never starts fallback", knownBug: nil) {
        let image = try ImageFixture.cgImage(width: 64, height: 96)
        let source = try await PreviewPhotoSource.load(
            original: { throw PhotosMedia.localUnavailable() },
            rendition: { throw PhotosMedia.localUnavailable() }, cached: image)
        guard case .still(let pixels) = source.content else { check(false, "cached image required"); return }
        check(pixels === image && !source.isOriginal, "local cached pixels survive both failed requests")
        do {
            _ = try await PreviewPhotoSource.load(original: { throw CancellationError() }, rendition: {
                check(false, "cancelled photos do not queue fallback work")
                return image
            }, cached: image)
            check(false, "cancelled navigation must terminate")
        } catch is CancellationError { }
    })
    Registry.shared.add(suite: "Frame preview", TestCase(name: "evidence cache identities separate preview quality and actual pixels", knownBug: nil) {
        let image = try ImageFixture.cgImage(width: 64, height: 64)
        let url = URL(string: "waifu-photos://preview-cache")!
        let original = PreviewEvidenceInput(photoURL: url, image: image, isOriginal: true)
        let preview = PreviewEvidenceInput(photoURL: url, image: image, isOriginal: false)
        let a = original.contentIdentity(assetIdentity: "asset", png: Data([1]))
        check(a != preview.contentIdentity(assetIdentity: "asset", png: Data([1])), "renditions never stand in for originals")
        check(a != original.contentIdentity(assetIdentity: "asset", png: Data([2])), "same-sized different pixels invalidate evidence")
        checkEqual(a, original.contentIdentity(assetIdentity: "asset", png: Data([1])), "unchanged input reuses cached evidence")
    })
    Registry.shared.add(suite: "Frame preview", TestCase(name: "original preview keeps native pixels and the source canvas instead of the analysis thumbnail", knownBug: nil) {
        let directory = try Temporary.directory("preview-original")
        let url = directory.appendingPathComponent("padded.mp4")
        try VideoFixture.writeVideo(to: url, duration: 0.5, width: 1920, height: 1080, frameRate: 12,
                                    picture: CGRect(x: 480, y: 270, width: 960, height: 540))
        var samples = AVFoundationFrameSampler().sample(url, analysisWidth: 512, mode: .full,
                                                      onProgress: { _ in }).makeAsyncIterator()
        guard let sample = try await samples.next() else { check(false, "fixture yields an analysis frame"); return }
        check(sample.image.width <= 512, "analysis deliberately uses a small image")
        let original = try await PreviewFrameReader(asset: AVURLAsset(url: url)).image(at: sample.timeSeconds)
        checkEqual(original.width, 1920, "preview reads native width independently of analysis size")
        checkEqual(original.height, 1080, "original canvas is not cropped to the grid thumbnail")
        let rotated = directory.appendingPathComponent("rotated.mp4")
        try await VideoFixture.rotate(url, to: rotated)
        let oriented = try await PreviewFrameReader(asset: AVURLAsset(url: rotated)).image(at: sample.timeSeconds)
        checkEqual(oriented.width, 1080, "source orientation is preserved at full resolution")
        checkEqual(oriented.height, 1920, "portrait previews keep every native row")
    })
    Registry.shared.add(suite: "Frame preview", TestCase(name: "held arrows repeat without OS repeat and stop on release", knownBug: nil) {
        try await Task { @MainActor in
            let repeater = PreviewArrowRepeater(delay: .milliseconds(20), interval: .milliseconds(10))
            let ticks = TickLog()
            repeater.start(1) { ticks.record($0) }
            try await Task.sleep(for: .milliseconds(300))
            check(ticks.all.count >= 2, "one initial key-down starts continuing navigation")
            repeater.stop()
            let stopped = ticks.all.count
            try await Task.sleep(for: .milliseconds(40))
            checkEqual(ticks.all.count, stopped, "release cancels all future ticks")
            repeater.start(-1) { ticks.record($0) }
            repeater.stop()
            try await Task.sleep(for: .milliseconds(40))
            checkEqual(ticks.all.count, stopped, "a brief tap does not repeat")
        }.value
    })
    Registry.shared.add(suite: "Frame preview", TestCase(name: "held arrows wait, fresh presses supersede, release stops", knownBug: nil) {
        var keys = PreviewArrowNavigation()
        checkEqual(keys.down(1, isRepeat: false, loading: true), 1, "fresh press can skip an unfinished read")
        for _ in 0..<100 { check(keys.down(1, isRepeat: true, loading: true) == nil, "repeat cannot skip a pending frame") }
        checkEqual(keys.loaded(), 1, "held repeat proceeds after decode")
        keys.up(1)
        check(keys.loaded() == nil, "release stops navigation")
        check(keys.down(1, isRepeat: true, loading: false) == nil, "late repeat cannot restart a released key")
        checkEqual(keys.down(-1, isRepeat: false, loading: true), -1, "new direction replaces pending read")
        keys.up(1)
        checkEqual(keys.held, -1, "old direction release cannot cancel new direction")
        keys.reset()
        check(keys.loaded() == nil, "focus loss stops navigation")
    })
    Registry.shared.add(suite: "Frame preview", TestCase(name: "rapid cancelled seeks still allow the final full-resolution frame", knownBug: nil) {
        let directory = try Temporary.directory("preview-seeks")
        let url = try VideoFixture.makeMP4(in: directory, durationSeconds: 4, width: 640, height: 360, frameRate: 30)
        let asset = AVURLAsset(url: url)
        var requests: [Task<Void, Never>] = []
        for index in 0..<100 {
            let request = Task {
                do {
                    _ = try await PreviewFrameReader(asset: asset).image(at: Double(index % 90) / 30)
                } catch { /* Superseded frames are expected to cancel. */ }
            }
            requests.append(request)
            await Task.yield()
            request.cancel()
        }
        for request in requests { await request.value }
        let image = try await PreviewFrameReader(asset: asset).image(at: 3)
        checkEqual(image.width, 640, "final seek remains full resolution")
        checkEqual(image.height, 360, "final seek completes after rapid cancellation")
        let cancelled = Task { try await PreviewFrameReader(asset: asset).image(at: 1) }
        cancelled.cancel()
        do { _ = try await cancelled.value; check(false, "pre-cancelled read must cancel") }
        catch is CancellationError { }
        catch { check(false, "expected cancellation, got \(error)") }
        do {
            _ = try await PreviewFrameReader(asset: asset).image(at: 2, timeout: 0)
        } catch { /* Immediate deadline or decoder error must terminate the request. */ }
        let recovered = try await PreviewFrameReader(asset: asset).image(at: 0.5)
        checkEqual(recovered.width, 640, "a deadline does not poison subsequent seeks")
    })
}
