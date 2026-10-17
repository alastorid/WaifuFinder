import AppKit
import Foundation

// Thumbnail cache: the bounded, in-memory store for analysis-resolution JPEGs.
//
// The cache holds the JPEG bytes of every analysed frame for the length of one
// session. It is deliberately *not* a durable artefact: every frame can be
// re-derived by re-running the pipeline, so writing thousands of JPEGs to disk
// bought nothing that outlived the run. That makes the interesting failures
// corruption rather than durability — a key that resolves to the wrong bytes, or
// a bounded cache that drops a frame it should have kept. Three properties keep
// that from happening, and each is pinned here:
//
//   * **Round trip.** `save` encodes `image.cgImage(forProposedRect:)` as JPEG and
//     `thumbnailData` reads it back for decoding. The grid holds `(key, videoStem)`
//     pairs, so both halves have to name the same entry or a cached frame loses its
//     picture. The case asserts the decoded *size*, not the bytes: `NSBitmapImageRep`'s JPEG
//     encoder is not byte-stable across OS releases, so a byte comparison would
//     fail for reasons that have nothing to do with the cache.
//   * **Key sanitisation.** Today's keys are derived from sample indices, but they
//     still name an entry, so a separator or a traversal sequence in a key must
//     be refused outright. Production refuses rather than mangles — silently
//     stripping `../` would let two distinct keys collide onto one entry, which
//     is worse than an error. `.pathTraversalAttempt` and `.invalidKey` are
//     asserted by *case*, not merely "it throws": an assertion that accepted any
//     error would keep passing if the safety check were deleted and the store
//     failed for an unrelated reason.
//   * **No filesystem footprint.** Nothing in the store writes a file. The
//     on-disk cache this replaced wrote `<key>.jpg.tmp.<uuid>` and replaced it
//     into place, and its cases could only observe atomicity from the outside —
//     assert that no `.tmp.` outlived a successful write, and that
//     `cacheDirectory(for:)` materialised on demand. Memory-only makes both
//     claims stronger and checkable directly: after any number of saves the
//     throwaway root is still empty, so there is no half-written artefact to
//     trip over and no directory lifecycle to get wrong. `cacheDirectory(for:)`
//     stays as a stable *logical* namespace, and the cases below pin that it
//     creates nothing.
//
// Rejected alternative: driving the refusal cases through `load`. `load` swallows
// every error into `nil` (by design, so the UI can show a placeholder), so it
// cannot distinguish a refused key from an entry that was simply never written.
// The refusals are therefore observed on `save`, where the error is still typed,
// and each case runs against a fresh throwaway root so that a regression which
// started writing to disk again would be caught against a directory the suite
// owns, never the real `~/Library/Application Support/Waifu` cache.

/// Asserts that a `save` refusal is `.pathTraversalAttempt` specifically.
///
/// `checkThrowsAsync` has already recorded the "nothing was thrown" failure when
/// `error` is nil, so this helper only classifies the error it was handed — the
/// two together mean "it threw, and it threw the right case".
private func checkPathTraversalRefusal(_ error: Error?, _ what: String) {
    guard let error else { return }
    guard let storeError = error as? ThumbnailStoreError else {
        check(false, "\(what): expected a ThumbnailStoreError, got \(error)")
        return
    }
    if case .pathTraversalAttempt = storeError { return }
    check(false, "\(what): expected .pathTraversalAttempt, got \(storeError)")
}

/// The `.invalidKey` counterpart of `checkPathTraversalRefusal`.
private func checkInvalidKeyRefusal(_ error: Error?, _ what: String) {
    guard let error else { return }
    guard let storeError = error as? ThumbnailStoreError else {
        check(false, "\(what): expected a ThumbnailStoreError, got \(error)")
        return
    }
    if case .invalidKey = storeError { return }
    check(false, "\(what): expected .invalidKey, got \(storeError)")
}

func registerThumbnailCacheTests() {
    let suite = "thumbnail cache"
    Registry.shared.add(suite: suite, TestCase(name: "grid thumbnail loads reuse decoded pixels and new analyses invalidate cached images", knownBug: nil) {
        let directory = try Temporary.directory("grid-thumbnail")
        let url = directory.appendingPathComponent("frame.png")
        let identity = UUID().uuidString
        try RegionImages.write(try ImageFixture.cgImage(width: 64, height: 48, gray: 0.5), to: url)
        guard let first = await GridThumbnailLoader.shared.image(at: url, identity: identity) else {
            check(false, "thumbnail decodes"); return
        }
        checkEqual(first.width, 64, "decoded geometry")
        try FileManager.default.removeItem(at: url)
        let cached = await GridThumbnailLoader.shared.image(at: url, identity: identity)
        check(cached === first, "revisiting uses the decoded cache without reading disk")
        let missing = await GridThumbnailLoader.shared.image(at: url, identity: UUID().uuidString)
        checkNil(missing, "missing thumbnail finishes without a fabricated image")
        try RegionImages.write(try ImageFixture.cgImage(width: 128, height: 96, gray: 0.8), to: url)
        let refreshed = await GridThumbnailLoader.shared.image(at: url, identity: UUID().uuidString)
        checkEqual(refreshed?.width, 128, "new analysis generation loads replacement pixels")
    })
    let stem = "test-video"

    // MARK: - Round trip

    Registry.shared.add(suite: suite, TestCase(
        name: "save then load round-trips the image at its original size", knownBug: nil) {
        let root = try Temporary.directory("thumb-round-trip")
        let cache = ThumbnailCache(rootDirectory: root)

        let image = ImageFixture.solid(width: 512, height: 288, color: .red)
        let key = "thumb-f000001"

        // `solid` builds an `NSImage` whose *nominal* size is 512×288 points, but
        // AppKit renders the locked focus into a backing bitmap at the display's
        // scale — 2× on a Retina Mac — and `save` encodes exactly that backing
        // CGImage (`cgImage(forProposedRect:)`), not the nominal point size. The
        // contract worth pinning is therefore that the decoded thumbnail carries the
        // same *pixels* as the image that was encoded; asserting a literal 512×288
        // would fail on a Retina display for a reason that has nothing to do with
        // the store, and would be wrong on a non-Retina one for the opposite reason.
        guard let source = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            check(false, "the fixture produced no CGImage to save")
            return
        }

        try await cache.save(image: image, forKey: key, videoStem: stem)

        guard let loadedData = checkNotNil(await cache.thumbnailData(forKey: key, videoStem: stem),
                                           "a saved thumbnail loads back"),
              let decoded = NSBitmapImageRep(data: loadedData) else {
            check(false, "the cached JPEG decodes to an image")
            return
        }
        checkEqual(decoded.pixelsWide, source.width, "the JPEG preserves the encoded width in pixels")
        checkEqual(decoded.pixelsHigh, source.height, "and height in pixels")
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "loading an unsaved key reads back nil, not an error", knownBug: nil) {
        let root = try Temporary.directory("thumb-missing")
        let cache = ThumbnailCache(rootDirectory: root)

        // `load` is deliberately non-throwing: a cache miss is the normal state
        // before a frame has been analysed, so the UI treats it as "no picture yet"
        // rather than surfacing an error. The miss touches nothing on disk.
        checkNil(await cache.thumbnailData(forKey: "thumb-nonexistent", videoStem: stem),
                 "an unsaved key reads back as nil")
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "ten thumbnails for one stem each read back", knownBug: nil) {
        let root = try Temporary.directory("thumb-many")
        let cache = ThumbnailCache(rootDirectory: root)

        // A scan writes thousands of frames into one stem's directory; ten distinct
        // hues prove the keys address distinct files rather than overwriting a
        // single slot.
        for i in 0..<10 {
            let image = ImageFixture.solid(width: 512, height: 288,
                                           color: NSColor(hue: CGFloat(i) / 10.0, saturation: 1,
                                                          brightness: 1, alpha: 1))
            try await cache.save(image: image, forKey: "thumb-f\(String(format: "%06d", i))", videoStem: stem)
        }

        for i in 0..<10 {
            let key = "thumb-f\(String(format: "%06d", i))"
            checkNotNil(await cache.thumbnailData(forKey: key, videoStem: stem),
                        "frame \(i) is cached alongside its siblings")
        }
    })

    // MARK: - Key sanitisation

    Registry.shared.add(suite: suite, TestCase(
        name: "a key containing ../ is refused as a traversal attempt", knownBug: nil) {
        let root = try Temporary.directory("thumb-dotdot")
        let cache = ThumbnailCache(rootDirectory: root)

        // Both spell a traversal the same way to the filesystem; the first is the
        // bare `../` and the second hides it behind a directory name.
        checkPathTraversalRefusal(
            await checkThrowsAsync("\"thumb-../escape\" is refused") {
                try await cache.save(image: ImageFixture.solid(width: 100, height: 100),
                                     forKey: "thumb-../escape", videoStem: stem)
            },
            "\"thumb-../escape\"")
        checkPathTraversalRefusal(
            await checkThrowsAsync("\"thumb-folder/../escape\" is refused") {
                try await cache.save(image: ImageFixture.solid(width: 100, height: 100),
                                     forKey: "thumb-folder/../escape", videoStem: stem)
            },
            "\"thumb-folder/../escape\"")
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "a key containing a forward slash is refused as a traversal attempt", knownBug: nil) {
        let root = try Temporary.directory("thumb-slash")
        let cache = ThumbnailCache(rootDirectory: root)

        checkPathTraversalRefusal(
            await checkThrowsAsync("\"thumb-f000001/extra\" is refused") {
                try await cache.save(image: ImageFixture.solid(width: 100, height: 100),
                                     forKey: "thumb-f000001/extra", videoStem: stem)
            },
            "\"thumb-f000001/extra\"")
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "a key containing a backslash is refused as a traversal attempt", knownBug: nil) {
        let root = try Temporary.directory("thumb-backslash")
        let cache = ThumbnailCache(rootDirectory: root)

        // A backslash is not a separator on APFS but is on the volumes a cache
        // directory might be shared with, so it is refused for the same reason.
        checkPathTraversalRefusal(
            await checkThrowsAsync("\"thumb-f000001\\extra\" is refused") {
                try await cache.save(image: ImageFixture.solid(width: 100, height: 100),
                                     forKey: "thumb-f000001\\extra", videoStem: stem)
            },
            "\"thumb-f000001\\extra\"")
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "a key that ends in a dot is refused as a traversal attempt",
        knownBug: nil) {
        let root = try Temporary.directory("thumb-trailing-dot")
        let cache = ThumbnailCache(rootDirectory: root)

        checkPathTraversalRefusal(
            await checkThrowsAsync("\"thumb-f000001.\" is refused") {
                try await cache.save(image: ImageFixture.solid(width: 100, height: 100),
                                     forKey: "thumb-f000001.", videoStem: stem)
            },
            "\"thumb-f000001.\"")
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "an empty or whitespace-only key is refused as invalid", knownBug: nil) {
        let root = try Temporary.directory("thumb-empty")
        let cache = ThumbnailCache(rootDirectory: root)

        checkInvalidKeyRefusal(
            await checkThrowsAsync("an empty key is refused") {
                try await cache.save(image: ImageFixture.solid(width: 100, height: 100),
                                     forKey: "", videoStem: stem)
            },
            "\"\"")
        checkInvalidKeyRefusal(
            await checkThrowsAsync("a whitespace-only key is refused") {
                try await cache.save(image: ImageFixture.solid(width: 100, height: 100),
                                     forKey: "   ", videoStem: stem)
            },
            "\"   \"")
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "a key longer than 255 characters is refused as invalid", knownBug: nil) {
        let root = try Temporary.directory("thumb-oversized")
        let cache = ThumbnailCache(rootDirectory: root)

        // 255 is the filesystem's per-component limit; 256 would be truncated or
        // rejected by the FS itself, so the cache refuses it up front.
        let longKey = "thumb-" + String(repeating: "a", count: 250)  // 256 characters
        checkInvalidKeyRefusal(
            await checkThrowsAsync("a \(longKey.count)-character key is refused") {
                try await cache.save(image: ImageFixture.solid(width: 100, height: 100),
                                     forKey: longKey, videoStem: stem)
            },
            "a \(longKey.count)-character key")
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "keys made of dots, dashes, underscores and mixed case are accepted", knownBug: nil) {
        let root = try Temporary.directory("thumb-valid")
        let cache = ThumbnailCache(rootDirectory: root)

        // The alphabet the real `thumbnailKey` uses, plus a dot that is *not* a
        // component and a case variation — each must survive sanitisation both as a
        // save and as a load.
        let validKeys = [
            "thumb-f000001",
            "thumb-f000002",
            "thumb-frame-123",
            "thumb_frame_123",
            "thumb.f000001",
            "THUMB-F000001",
        ]
        for key in validKeys {
            try await cache.save(image: ImageFixture.solid(width: 100, height: 100),
                                 forKey: key, videoStem: stem)
            checkNotNil(await cache.thumbnailData(forKey: key, videoStem: stem), "\"\(key)\" is a legal key")
        }
    })

    // MARK: - No filesystem footprint

    Registry.shared.add(suite: suite, TestCase(
        name: "a successful save leaves no .tmp. file behind", knownBug: nil) {
        let root = try Temporary.directory("thumb-atomic")
        let cache = ThumbnailCache(rootDirectory: root)
        let image = ImageFixture.solid(width: 512, height: 288)
        let key = "thumb-atomic-test"

        // The store is memory-only, so the old claim — the write went to
        // `<key>.jpg.tmp.<uuid>` and was then replaced into place, and a crash
        // between the two stranded the temp file — no longer has a half-written
        // artefact to strand. The stronger statement is available instead: after
        // a save the root is exactly as empty as it started, so there is nothing
        // on disk at all, temporary or otherwise.
        try await cache.save(image: image, forKey: key, videoStem: stem)
        checkNotNil(await cache.thumbnailData(forKey: key, videoStem: stem), "the thumbnail is readable after the write")

        let leftovers = try FileManager.default.contentsOfDirectory(atPath: root.path)
        check(leftovers.isEmpty,
              "no files remain after a successful write, found \(leftovers)")
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "fifty concurrent saves all load back", knownBug: nil) {
        let root = try Temporary.directory("thumb-concurrent")
        let cache = ThumbnailCache(rootDirectory: root)

        // The actor serialises the writes, so 50 racing producers must not lose one
        // another's data. Each uses a distinct key; failures are collected rather
        // than ignored so a dropped write names itself.
        await withTaskGroup(of: Void.self) { group in
            for i in 0..<50 {
                group.addTask {
                    let image = ImageFixture.solid(width: 512, height: 288,
                                                   color: NSColor(hue: CGFloat(i) / 50.0, saturation: 1,
                                                                  brightness: 1, alpha: 1))
                    let key = "thumb-concurrent-\(String(format: "%06d", i))"
                    try? await cache.save(image: image, forKey: key, videoStem: stem)
                }
            }
        }

        var missing: [String] = []
        for i in 0..<50 {
            let key = "thumb-concurrent-\(String(format: "%06d", i))"
            if await cache.thumbnailData(forKey: key, videoStem: stem) == nil { missing.append(key) }
        }
        check(missing.isEmpty,
              "all 50 concurrently-written thumbnails load back; missing \(missing.count): \(missing.prefix(5))")
    })

    // MARK: - Clear

    Registry.shared.add(suite: suite, TestCase(
        name: "clear(for:) removes only the named stem", knownBug: nil) {
        let root = try Temporary.directory("thumb-clear-one")
        let cache = ThumbnailCache(rootDirectory: root)

        try await cache.save(image: ImageFixture.solid(width: 100, height: 100),
                             forKey: "thumb-1", videoStem: "video-one")
        try await cache.save(image: ImageFixture.solid(width: 100, height: 100),
                             forKey: "thumb-2", videoStem: "video-two")
        checkNotNil(await cache.thumbnailData(forKey: "thumb-1", videoStem: "video-one"), "video-one is cached")
        checkNotNil(await cache.thumbnailData(forKey: "thumb-2", videoStem: "video-two"), "video-two is cached")

        try await cache.clear(for: "video-one")

        checkNil(await cache.thumbnailData(forKey: "thumb-1", videoStem: "video-one"),
                 "clearing video-one drops its thumbnail")
        checkNotNil(await cache.thumbnailData(forKey: "thumb-2", videoStem: "video-two"),
                    "and leaves video-two's alone")
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "clear() empties the whole cache root", knownBug: nil) {
        let root = try Temporary.directory("thumb-clear-all")
        let cache = ThumbnailCache(rootDirectory: root)

        try await cache.save(image: ImageFixture.solid(width: 100, height: 100),
                             forKey: "thumb-1", videoStem: "video-a")
        try await cache.save(image: ImageFixture.solid(width: 100, height: 100),
                             forKey: "thumb-2", videoStem: "video-b")
        checkNotNil(await cache.thumbnailData(forKey: "thumb-1", videoStem: "video-a"), "video-a is cached")
        checkNotNil(await cache.thumbnailData(forKey: "thumb-2", videoStem: "video-b"), "video-b is cached")

        try await cache.clear()

        // Checked before the loads below purely so the emptiness assertion reads
        // the root before anything else could have touched it. Nothing here writes,
        // so the order is defensive rather than load-bearing.
        let rootSurvived = FileManager.default.fileExists(atPath: root.path)
        let residual = (try? FileManager.default.contentsOfDirectory(atPath: root.path)) ?? []
        check(!rootSurvived || residual.isEmpty,
              "clear() removes the root or leaves it empty, found \(residual)")

        checkNil(await cache.thumbnailData(forKey: "thumb-1", videoStem: "video-a"), "clear() drops video-a")
        checkNil(await cache.thumbnailData(forKey: "thumb-2", videoStem: "video-b"), "and video-b")
    })

    // MARK: - Directory layout

    Registry.shared.add(suite: suite, TestCase(
        name: "cacheDirectory(for:) is a stable logical path that creates nothing", knownBug: nil) {
        let root = try Temporary.directory("thumb-directory")
        let cache = ThumbnailCache(rootDirectory: root)

        // The cache is memory-only, so the per-video URL is a namespace rather
        // than a directory: it must not appear on disk when it is asked for.
        // (The original suite asserted the path contained "Waifu/Thumbnails",
        // which is only true of the default initialiser — this suite pins the
        // custom root it was given instead.)
        let expected = root.appendingPathComponent(stem, isDirectory: true)
        check(!FileManager.default.fileExists(atPath: expected.path),
              "the per-video directory does not exist before it is requested")

        let dir = try await cache.cacheDirectory(for: stem)
        checkEqual(dir.lastPathComponent, stem, "the directory is named after the video stem")
        checkEqual(dir.deletingLastPathComponent().path, root.path,
                   "and sits directly under the cache root")
        check(!FileManager.default.fileExists(atPath: dir.path),
              "requesting it did not create it — the store writes nothing to disk")
        check((try? FileManager.default.contentsOfDirectory(atPath: root.path))?.isEmpty == true,
              "and the root itself is still empty")

        let again = try await cache.cacheDirectory(for: stem)
        checkEqual(again, dir, "a second request returns the same directory, not a new one")
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "a stem containing path and glob characters flattens to one safe directory name", knownBug: nil) {
        let root = try Temporary.directory("thumb-stem")
        let cache = ThumbnailCache(rootDirectory: root)

        // A stem is a filename without its extension, so it can legitimately carry
        // characters a path component cannot. The `/` and `\` cases must lose their
        // separators (replaced with `_`); `:` and `*` are legal in an APFS name and
        // are left as-is. What matters in every case is that exactly one path
        // component is produced — asserting `deletingLastPathComponent() == root` is
        // the check that a separator did not silently add a directory level, since
        // `lastPathComponent` never contains `/` on its own.
        let stems: [(stem: String, expectedName: String?)] = [
            ("video/with/slashes", "video_with_slashes"),
            ("video\\with\\backslashes", "video_with_backslashes"),
            ("video:with:colons", nil),
            ("video*with*asterisks", nil),
        ]

        for (stem, expectedName) in stems {
            let dir = try await cache.cacheDirectory(for: stem)
            checkEqual(dir.deletingLastPathComponent().path, root.path,
                       "\"\(stem)\" stays one path component under the root")
            check(!dir.lastPathComponent.contains("\\"),
                  "\"\(stem)\": no backslash survives in the directory name")
            if let expectedName {
                checkEqual(dir.lastPathComponent, expectedName,
                           "\"\(stem)\" loses its path separators")
            }
        }

        // Flattening is for the logical namespace's benefit only: none of the four
        // stems may have materialised anything under the root.
        let created = (try? FileManager.default.contentsOfDirectory(atPath: root.path)) ?? []
        check(created.isEmpty, "asking for a directory name creates no directories, found \(created)")
    })
}
