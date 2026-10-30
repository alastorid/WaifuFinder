import AppKit
import SwiftUI
import Photos
import QuartzCore

func registerPhotosWorkflowTests() {
    let suite = "Photos workflow"
    Registry.shared.add(suite: suite, TestCase(name: "photo deletion recognizes only PhotoKit user cancellation", knownBug: nil) {
        let code = PHPhotosError.Code.userCancelled.rawValue
        check(PhotosMedia.isDeletionCancellation(NSError(domain: PHPhotosErrorDomain, code: code)), "user cancellation is silent")
        check(!PhotosMedia.isDeletionCancellation(NSError(domain: PHPhotosErrorDomain, code: PHPhotosError.Code.accessUserDenied.rawValue)), "permission failures remain errors")
        check(!PhotosMedia.isDeletionCancellation(NSError(domain: "OtherDomain", code: code)), "unrelated errors remain errors")
    })
    Registry.shared.add(suite: suite, TestCase(name: "sync retries transient PhotoKit interruptions before surfacing errors", knownBug: nil) {
        let probe = PhotosOrderSyncProbe(failures: 2,
            error: NSError(domain: PHPhotosErrorDomain, code: PHPhotosError.Code.operationInterrupted.rawValue))
        try await PhotosOrderSync.run(operation: { try await probe.attempt() }, wait: { await probe.wait($0) })
        checkEqual(await probe.attempts, 3, "temporary failures recover automatically")
        checkEqual(await probe.delays, [0.25, 0.5], "retries back off without blocking the UI")
    })
    Registry.shared.add(suite: suite, TestCase(name: "sync retry refetches order after a successful commit with delayed verification", knownBug: nil) {
        let probe = PhotosOrderSyncProbe(failures: 0, error: PhotosOrderSync.VerificationError.orderNotVisible)
        try await PhotosOrderSync.run(operation: { try await probe.commitThenVerify() }, wait: { await probe.wait($0) })
        checkEqual(await probe.attempts, 2, "verification recovers on a new snapshot")
        checkEqual(await probe.writes, 1, "the retry recognizes the committed order instead of replaying moves")
        checkEqual(await probe.order, ["C", "A", "B"], "the requested order is retained")
    })
    Registry.shared.add(suite: suite, TestCase(name: "sync stops after seven retries and never retries permanent errors", knownBug: nil) {
        let transient = NSError(domain: PHPhotosErrorDomain, code: PHPhotosError.Code.internalError.rawValue)
        let probe = PhotosOrderSyncProbe(failures: 10, error: transient)
        _ = await checkThrowsAsync("persistent errors reach the caller") {
            try await PhotosOrderSync.run(operation: { try await probe.attempt() }, wait: { await probe.wait($0) })
        }
        checkEqual(await probe.attempts, 8, "one initial attempt plus seven retries")
        checkEqual(await probe.delays, [0.25, 0.5, 1, 1.5, 2, 2, 2], "persistent transient failures use the full capped backoff")
        for error in [
            NSError(domain: PHPhotosErrorDomain, code: PHPhotosError.Code.accessUserDenied.rawValue),
            NSError(domain: PHPhotosErrorDomain, code: PHPhotosError.Code.changeNotSupported.rawValue),
            NSError(domain: PHPhotosErrorDomain, code: PHPhotosError.Code.userCancelled.rawValue),
            PhotosMedia.failure("The album changed.")
        ] {
            let permanent = PhotosOrderSyncProbe(failures: 10, error: error)
            _ = await checkThrowsAsync("permanent errors reach the caller immediately") {
                try await PhotosOrderSync.run(operation: { try await permanent.attempt() }, wait: { await permanent.wait($0) })
            }
            checkEqual(await permanent.attempts, 1, "permissions, unsupported edits, cancellation, and missing members do not retry")
            checkEqual(await permanent.delays, [], "no delay for permanent errors")
        }
    })
    Registry.shared.add(suite: suite, TestCase(name: "cancellation during sync backoff prevents another write", knownBug: nil) {
        let probe = PhotosOrderSyncProbe(failures: 10, error: PhotosOrderSync.VerificationError.orderNotVisible)
        do {
            try await PhotosOrderSync.run(operation: { try await probe.attempt() }, wait: { _ in throw CancellationError() })
            check(false, "cancellation must propagate")
        } catch is CancellationError { }
        checkEqual(await probe.attempts, 1, "cancellation prevents the retry")
    })
    Registry.shared.add(suite: suite, TestCase(name: "tagging candidates grow from the first photo to the entire scored album", knownBug: nil) {
        let directory = try Temporary.directory("photos-tagging-growth")
        await Task { @MainActor in
            let session = WaifuSession(settingsURL: directory.appendingPathComponent("settings.json"))
            let photos = (0..<37).map { index in
                AnalyzedFrame(sampleIndex: index, timeSeconds: Double(index), artScore: Float(index),
                              faceScore: nil, faceCount: 0, faceAreaFraction: nil, featurePrint: Data())
            }
            session.updateAnalyzedFrames(Array(photos.prefix(1)))
            checkEqual(session.matchedFrameCountForTagging, 1, "first live snapshot has one candidate")
            session.updateAnalyzedFrames(Array(photos.prefix(9)))
            checkEqual(session.matchedFrameCountForTagging, 9, "later snapshots invalidate cached candidate IDs")
            session.updateAnalyzedFrames(photos)
            checkEqual(session.matchedFrameCountForTagging, 37, "the final snapshot includes every photo")
            checkEqual(session.untaggedFrameCountForTagging, 37, "all thumbnails need tagging, regardless of grid caps")
            session.frameLimits.art.low = 20
            checkEqual(session.matchedFrameCountForTagging, 17, "video filter behavior remains intact")
            session.updateAnalyzedFrames(photos.map { frame in
                AnalyzedFrame(sampleIndex: frame.id.sampleIndex, timeSeconds: frame.timeSeconds, artScore: 30,
                              faceScore: nil, faceCount: 0, faceAreaFraction: nil, featurePrint: Data())
            })
            checkEqual(session.matchedFrameCountForTagging, 37, "rescoring with the same IDs also invalidates eligibility")
        }.value
    })
    Registry.shared.add(suite: suite, TestCase(name: "a degraded Photos preview waits for the actual 512px rendition", knownBug: nil) {
        let tiny = try ImageFixture.cgImage(width: 64, height: 48)
        let full = try ImageFixture.cgImage(width: 512, height: 384)
        let delivery = PhotosRenditionDelivery()
        checkNil(delivery.receive(tiny, isDegraded: true), "a tiny degraded callback must not complete or cancel the PhotoKit request")
        checkEqual(delivery.bestImage?.width, 64, "the cached preview can be displayed immediately")
        check(!PhotosMedia.hasRequestedResolution(tiny, targetSize: CGSize(width: 512, height: 384)), "tiny pixels do not satisfy the requested size")
        guard let result = delivery.receive(full, isDegraded: false) else {
            throw PhotosMedia.failure("The final rendition did not complete the request.")
        }
        let final = try result.get()
        checkEqual(final.width, 512, "the final image replaces the tiny preview")
        checkEqual(final.height, 384, "aspect ratio is preserved")
        check(PhotosMedia.hasRequestedResolution(final, targetSize: CGSize(width: 512, height: 384)), "the final pixels satisfy the requested size")
        _ = delivery.receive(tiny, isDegraded: true)
        checkEqual(delivery.bestImage?.width, 512, "a late degraded callback cannot downgrade the image")
    })
    Registry.shared.add(suite: suite, TestCase(name: "unavailable final rendition retains locally cached pixels without upscaling", knownBug: nil) {
        let tiny = try ImageFixture.cgImage(width: 96, height: 64)
        let delivery = PhotosRenditionDelivery()
        _ = delivery.receive(tiny, isDegraded: true)
        let fallback = try delivery.receive(nil, isDegraded: false)?.get()
        checkEqual(fallback?.width, 96, "failed upgrade preserves the locally available thumbnail")
        let request = PhotosRequest<CGImage>(timeout: 0.01, timeoutFallback: { delivery.bestImage }, cancel: { _ in })
        let timedOut = try await request.perform { _ in 42 }
        checkEqual(timedOut.width, 96, "a stalled upgrade falls back to actual pixels instead of fabricating 512px")
        let empty = PhotosRenditionDelivery()
        guard let failure = empty.receive(nil, isDegraded: false) else {
            throw PhotosMedia.failure("A missing image did not complete the request.")
        }
        _ = checkThrows("missing previews still report failure") { try failure.get() }
    })
    Registry.shared.add(suite: suite, TestCase(name: "Photos bitmap resolution is independent of NSImage point size", knownBug: nil) {
        try await MainActor.run {
            let tiny = try ImageFixture.cgImage(width: 64, height: 48)
            let full = try ImageFixture.cgImage(width: 512, height: 384)
            let image = NSImage(size: NSSize(width: 128, height: 96))
            image.addRepresentation(NSBitmapImageRep(cgImage: tiny))
            image.addRepresentation(NSBitmapImageRep(cgImage: full))
            let pixels = PhotosMedia.renditionPixels(image)
            checkEqual(pixels?.width, 512, "the larger cached bitmap is used even when logical point size is small")
            checkEqual(pixels?.height, 384, "bitmap height uses actual pixels")
        }
    })
    Registry.shared.add(suite: suite, TestCase(name: "nested album selection preserves folder membership and resolves duplicate titles by ID", knownBug: nil) {
        let first = PhotosAlbum(id: "first", title: "Zebra", count: 1, isSmartAlbum: false)
        let second = PhotosAlbum(id: "second", title: "Alpha", count: 2, isSmartAlbum: false)
        let nested = PhotosAlbum(id: "nested", title: "Zebra", count: 3, isSmartAlbum: false)
        let empty = PhotosAlbum(id: "empty-folder", title: "Empty", count: 0, isSmartAlbum: false, children: [])
        let inner = PhotosAlbum(id: "inner", title: "Inner", count: 3, isSmartAlbum: false, children: [nested, empty])
        let root = PhotosAlbum(id: "root", title: "Folder", count: 6, isSmartAlbum: false, children: [first, inner, second])
        checkEqual(root.children?.map(\.id), ["first", "inner", "second"], "user sequence keeps folders interleaved with albums")
        checkEqual(root.find("nested")?.count, 3, "nested album is selectable by stable ID despite its duplicate title")
        checkEqual(root.find("first")?.count, 1, "top-level child retains its own identity")
        check(root.find("empty-folder")?.isFolder == true, "empty folders remain folders")
        check(!first.isFolder, "leaf albums are not disclosure rows")
        checkNil(root.find("removed"), "removed albums cannot resolve to another album with the same title")
    })
    Registry.shared.add(suite: suite, TestCase(name: "sync preserves hidden photos and videos while reordering the visible selection", knownBug: nil) {
        let original = ["hidden", "A", "video", "B", "C", "other"]
        let moves = try PhotosAlbumOrder.moves(current: original, ordered: ["C", "A", "B"])
        var result = original
        for move in moves { result.insert(result.remove(at: move.from), at: move.to) }
        checkEqual(result, ["hidden", "C", "video", "A", "B", "other"], "only selected album slots change")
        checkEqual(Set(result), Set(original), "membership is preserved")
        checkEqual(try PhotosAlbumOrder.moves(current: result, ordered: ["C", "A", "B"]), [], "already synced order needs no edits")
        _ = checkThrows("stale membership refuses the write") {
            try PhotosAlbumOrder.moves(current: original, ordered: ["C", "missing"])
        }
        _ = checkThrows("duplicate input refuses the write") {
            try PhotosAlbumOrder.moves(current: original, ordered: ["C", "C"])
        }
    })
    Registry.shared.add(suite: suite, TestCase(name: "album order handles a reversed ten-thousand-photo view", knownBug: nil) {
        let original = (0..<10_000).map(String.init)
        let desired = Array(original.reversed())
        var result = original
        for move in try PhotosAlbumOrder.moves(current: original, ordered: desired) {
            result.insert(result.remove(at: move.from), at: move.to)
        }
        checkEqual(result, desired, "PhotoKit moves reproduce the displayed order")
    })
    Registry.shared.add(suite: suite, TestCase(name: "square and full-aspect thumbnails render different sizes and crops", knownBug: nil) {
        try await MainActor.run {
            // Blue center with red edges makes a center crop distinguishable
            // from simply squeezing the full image into a square.
            guard let context = CGContext(data: nil, width: 512, height: 256,
                                          bitsPerComponent: 8, bytesPerRow: 0,
                                          space: CGColorSpaceCreateDeviceRGB(),
                                          bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
                throw PhotosMedia.failure("Could not create the thumbnail fixture.")
            }
            context.setFillColor(NSColor.red.cgColor)
            context.fill(CGRect(x: 0, y: 0, width: 512, height: 256))
            context.setFillColor(NSColor.blue.cgColor)
            context.fill(CGRect(x: 128, y: 0, width: 256, height: 256))
            context.setFillColor(NSColor.green.cgColor)
            context.fill(CGRect(x: 0, y: 0, width: 512, height: 32))
            guard let source = context.makeImage() else { throw PhotosMedia.failure("Could not create the thumbnail fixture.") }
            for (square, expectedHeight) in [(false, 80), (true, 160)] {
                let size = GridThumbnailLayout.size(width: 160, sourceAspect: 2, square: square)
                let view = GridThumbnailImageView(frame: NSRect(origin: .zero, size: size))
                view.configure(image: source, identity: UUID().uuidString, cropToFill: square, isVisible: true)
                view.layoutSubtreeIfNeeded()
                guard let target = CGContext(data: nil, width: Int(size.width), height: Int(size.height),
                                             bitsPerComponent: 8, bytesPerRow: 0,
                                             space: CGColorSpaceCreateDeviceRGB(),
                                             bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
                      let layer = view.layer else { throw PhotosMedia.failure("Could not render the thumbnail control.") }
                layer.render(in: target)
                guard let pixels = target.makeImage() else { throw PhotosMedia.failure("Could not render the thumbnail control.") }
                checkEqual(pixels.width, 160, "thumbnail width stays fixed")
                checkEqual(pixels.height, expectedHeight, "shape changes the actual image container")
                let edge = NSBitmapImageRep(cgImage: pixels).colorAt(x: 8, y: expectedHeight / 2)?.usingColorSpace(.deviceRGB)
                check(edge != nil, "rendered pixels are readable")
                check((square ? edge?.blueComponent : edge?.redComponent) ?? 0 > 0.8,
                      "square crops the red edges; full aspect keeps them")
                let sourceTop = NSBitmapImageRep(cgImage: source).colorAt(x: 256, y: 8)?.usingColorSpace(.deviceRGB)
                let renderedTop = NSBitmapImageRep(cgImage: pixels).colorAt(x: 80, y: max(1, expectedHeight / 32))?.usingColorSpace(.deviceRGB)
                check(abs((sourceTop?.greenComponent ?? -1) - (renderedTop?.greenComponent ?? -2)) < 0.05,
                      "the native layer keeps the photo's vertical orientation")
            }
            let portrait = GridThumbnailLayout.size(width: 160, sourceAspect: 0.5, square: false)
            checkEqual(portrait.height, 320, "full aspect also preserves portrait height before loading")

        }
    })
    Registry.shared.add(suite: suite, TestCase(name: "cached thumbnails feed the tagger without originals or model weights", knownBug: nil) {
        for (width, height) in [(512, 384), (256, 128), (2048, 1024), (64, 48)] {
            let source = try ImageFixture.cgImage(width: width, height: height)
            let data = try ThumbnailCache.encodeJPEG(source)
            let thumbnail = try WDTaggerBackend.thumbnailImage(data: data)
            check(thumbnail.width <= 512 && thumbnail.height <= 512, "tagging decode stays bounded")
            check(thumbnail.width <= width && thumbnail.height <= height, "small cached thumbnails stay small until model preprocessing")
            let input = try WDTaggerBackend.preprocessRGBA(thumbnail)
            checkEqual(input.count, WDTaggerBackend.imageSize * WDTaggerBackend.imageSize * 4, "cached pixels produce the model input")
            let rendition = try PhotosMedia.boundedRendition(source, maxPixelSize: 512)
            check(rendition.width <= 512 && rendition.height <= 512, "PhotoKit's larger cached results are bounded too")
            check(rendition.width <= width && rendition.height <= height, "renditions never upscale")
        }
    })

    Registry.shared.add(suite: suite, TestCase(name: "album comparison ignores only the modification bumps our keyword writes made", knownBug: nil) {
        func revision(_ id: String, _ modified: Double, width: Int = 100, height: Int = 80) -> PhotosMedia.Revision {
            PhotosMedia.Revision(identifier: id, modified: modified, width: width, height: height, duration: 1)
        }
        let before = [revision("A", 10), revision("B", 20), revision("C", 30)]
        let baselines = ["A": 11.0, "B": 21.0]  // keyword writes bumped A and B
        let after = [revision("A", 11), revision("B", 21), revision("C", 30)]
        check(PhotosMedia.revisionsMatch(before, after, keywordBaselines: baselines),
              "keyword-only bumps compare as unchanged, so nothing is reloaded or re-analyzed")
        check(PhotosMedia.revisionsMatch(after, before, keywordBaselines: baselines),
              "the comparison is direction-independent")
        check(!PhotosMedia.revisionsMatch(before, after, keywordBaselines: [:]),
              "without a baseline the bump reads as a real change")
        let edited = [revision("A", 11), revision("B", 25), revision("C", 30)]
        check(!PhotosMedia.revisionsMatch(before, edited, keywordBaselines: baselines),
              "a later real edit moves past the baseline and is detected")
        check(!PhotosMedia.revisionsMatch([revision("A", 10)],
                                          [revision("A", 10), revision("B", 20)],
                                          keywordBaselines: baselines),
              "a membership change never compares as unchanged")
        check(!PhotosMedia.revisionsMatch([revision("A", 10, width: 200)],
                                          [revision("A", 11, width: 100)],
                                          keywordBaselines: ["A": 11.0]),
              "geometry changes are not absorbed by a baseline")
    })

    Registry.shared.add(suite: suite, TestCase(name: "deletion-only snapshots preserve survivors and reject other content changes", knownBug: nil) {
        func revision(_ id: String, _ modified: Double = 10) -> PhotosMedia.Revision {
            PhotosMedia.Revision(identifier: id, modified: modified, width: 100, height: 80, duration: 1)
        }
        let before = [revision("A"), revision("B"), revision("C"), revision("D")]
        checkEqual(PhotosMedia.removedAssetIDs(from: before, to: [revision("B"), revision("D")], keywordBaselines: [:]),
                   Set(["A", "C"]), "removals at the beginning and middle are incremental")
        checkEqual(PhotosMedia.removedAssetIDs(from: before, to: [], keywordBaselines: [:]),
                   Set(["A", "B", "C", "D"]), "deleting the last photos preserves an empty workspace")
        checkEqual(PhotosMedia.removedAssetIDs(from: before, to: [revision("B", 11)], keywordBaselines: ["B": 11]),
                   Set(["A", "C", "D"]), "keyword echoes do not force reanalysis of survivors")
        for after in [[revision("D"), revision("B")], [revision("B", 12)], [revision("X")], before] {
            checkNil(PhotosMedia.removedAssetIDs(from: before, to: after, keywordBaselines: [:]),
                     "moves, edits, additions and unchanged snapshots do not take the deletion path")
        }
    })

    Registry.shared.add(suite: suite, TestCase(name: "deleting the preview advances through the captured order and skips deleted photos", knownBug: nil) {
        let directory = try Temporary.directory("preview-delete-next")
        await Task { @MainActor in
            let session = WaifuSession(settingsURL: directory.appendingPathComponent("settings.json"))
            session.frameOrder = .only([.time])
            let photos = (0..<5).map { index in
                AnalyzedFrame(sampleIndex: index, timeSeconds: Double(index), artScore: Float(index),
                    faceScore: nil, faceCount: 0, faceAreaFraction: nil, featurePrint: Data())
            }
            session.updateAnalyzedFrames(photos)
            let ordered = [photos[2], photos[0], photos[3], photos[1], photos[4]]
            session.commitGridNavigation(FrameNavigationOrder(frames: ordered, rows: [], groupRows: false),
                                         generation: session.thumbnailGeneration)
            session.showPreview(photos[2])
            // A later grid commit must not change the sequence already being inspected.
            session.commitGridNavigation(FrameNavigationOrder(frames: photos, rows: [], groupRows: false),
                                         generation: session.thumbnailGeneration)
            session.removeAnalyzedFrames([photos[2].id, photos[0].id])
            checkEqual(session.previewFrame?.id, photos[3].id, "successful deletion skips all removed targets and keeps preview open")
            checkEqual(session.focusedFrameID, photos[3].id, "focus follows the replacement")
            checkEqual(session.selectedFrameIDs, [photos[3].id], "the next photo becomes the sole selected target")
            session.focusNeighbour(1)
            checkEqual(session.previewFrame?.id, photos[1].id, "right arrow retains the captured order after deletion")
            session.removeAnalyzedFrames([photos[1].id])
            checkEqual(session.previewFrame?.id, photos[4].id, "repeated deletion advances again")
            session.removeAnalyzedFrames([photos[4].id])
            checkEqual(session.previewFrame?.id, photos[3].id, "deleting the last photo falls back to the previous survivor")
            session.removeAnalyzedFrames([photos[3].id])
            checkNil(session.previewFrame, "preview closes when no displayed photos survive")
            checkNil(session.focusedFrameID, "empty results clear focus")
            check(session.selectedFrameIDs.isEmpty, "empty results clear selection")
        }.value
    })

    Registry.shared.add(suite: suite, TestCase(name: "removing analyzed photos preserves surviving results and workspace state", knownBug: nil) {
        let directory = try Temporary.directory("photos-incremental-deletion")
        await Task { @MainActor in
            let session = WaifuSession(settingsURL: directory.appendingPathComponent("settings.json"))
            let photos = (0..<4).map { index in
                var frame = AnalyzedFrame(sampleIndex: index, timeSeconds: Double(index), artScore: Float(index),
                              faceScore: nil, faceCount: 0, faceAreaFraction: nil, featurePrint: Data())
                frame.topTags = ["photo-\(index)"]
                return frame
            }
            session.updateAnalyzedFrames(photos)
            session.selectFrame(photos[1].id)
            session.selectFrame(photos[2].id, command: true)
            session.previewFrame = photos[2]
            let generation = session.thumbnailGeneration
            let progress = session.progress
            session.removeAnalyzedFrames([photos[0].id, photos[1].id])
            checkEqual(session.frames, Array(photos.suffix(2)), "survivors retain IDs, timestamps, scores and metadata")
            checkEqual(session.selectedFrameIDs, Set([photos[2].id]), "only deleted selections disappear")
            checkEqual(session.previewFrame, photos[2], "a surviving preview stays open")
            checkEqual(session.focusedFrameID, photos[2].id, "surviving focus stays put")
            checkEqual(session.thumbnailGeneration, generation, "thumbnail identity is unchanged")
            checkEqual(session.progress.phase, progress.phase, "no new analysis starts")
            session.removeAnalyzedFrames(Set(photos.map(\.id)))
            check(session.frames.isEmpty && session.selectedFrameIDs.isEmpty, "all-deleted results clear")
            checkNil(session.previewFrame, "a deleted preview closes")
            checkNil(session.focusedFrameID, "deleted focus clears")
        }.value
    })

    Registry.shared.add(suite: suite, TestCase(name: "library change digests recognize keyword-write echoes precisely", knownBug: nil) {
        let written: Set<String> = ["A", "B"]
        var echo = PhotosLibraryChanges.Digest()
        echo.albumAffected = true
        echo.changedAssetIDs = ["A", "B"]
        check(echo.isEcho(ofWritten: written), "changed-only notifications confined to written photos are our echoes")
        var structural = echo
        structural.structural = true
        check(!structural.isEcho(ofWritten: written), "insertions, deletions and moves are never keyword echoes")
        var foreign = echo
        foreign.changedAssetIDs = ["A", "X"]
        check(!foreign.isEcho(ofWritten: written), "a photo we never wrote needs a real refresh")
        var untouched = PhotosLibraryChanges.Digest()
        untouched.changedAssetIDs = ["A"]
        check(!untouched.isEcho(ofWritten: written), "changes outside the observed albums are not keyword echoes")
        var empty = echo
        empty.changedAssetIDs = []
        check(!empty.isEcho(ofWritten: written), "an empty digest never claims to be an echo")
        checkEqual(PhotosLibraryChanges.shared.consumeDigest(), PhotosLibraryChanges.Digest(),
                   "consumption clears the accumulated digest")
    })
}

private actor PhotosOrderSyncProbe {
    var attempts = 0
    var delays: [TimeInterval] = []
    var writes = 0
    var order = ["A", "B", "C"]
    private var failures: Int
    private let error: Error

    init(failures: Int, error: Error) { self.failures = failures; self.error = error }

    func attempt() throws {
        attempts += 1
        if failures > 0 { failures -= 1; throw error }
    }

    func wait(_ delay: TimeInterval) { delays.append(delay) }

    func commitThenVerify() throws {
        attempts += 1
        let moves = try PhotosAlbumOrder.moves(current: order, ordered: ["C", "A", "B"])
        guard !moves.isEmpty else { return }
        for move in moves { order.insert(order.remove(at: move.from), at: move.to) }
        writes += 1
        throw error
    }
}
