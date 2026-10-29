import AppKit
import CoreImage
import ImageIO
import Vision
import Combine
import UniformTypeIdentifiers
import os

private func orbFrame(_ index: Int, feature: Data? = nil) -> AnalyzedFrame {
    AnalyzedFrame(sampleIndex: index, timeSeconds: Double(index), artScore: Float(index), faceScore: nil,
                  faceCount: 0, faceAreaFraction: nil, featurePrint: feature)
}

private func orbImage(_ variant: Int, width: Int = 128, height: Int = 96) throws -> CGImage {
    let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                            space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    context.scaleBy(x: CGFloat(width) / 128, y: CGFloat(height) / 96)
    context.setFillColor(CGColor(red: variant == 0 ? 0.15 : 0.8, green: 0.3, blue: 0.6, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: 128, height: 96))
    context.setFillColor(CGColor(red: 0.9, green: 0.7, blue: 0.1, alpha: 1))
    if variant == 0 { context.fillEllipse(in: CGRect(x: 10, y: 20, width: 70, height: 60)) }
    else { for x in stride(from: 0, to: 128, by: 16) { context.fill(CGRect(x: x, y: 0, width: 5, height: 96)) } }
    return context.makeImage()!
}

private func orbPixel(_ image: CGImage, x: Int = 0, y: Int = 0) -> [UInt8] {
    var bytes = [UInt8](repeating: 0, count: image.width * image.height * 4)
    bytes.withUnsafeMutableBytes { buffer in
        let context = CGContext(data: buffer.baseAddress, width: image.width, height: image.height, bitsPerComponent: 8,
                                bytesPerRow: image.width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
    }
    let offset = (y * image.width + x) * 4
    return Array(bytes[offset..<offset + 4])
}

func registerCrystalOrbTests() {
    let suite = "Crystal Orb"
    Registry.shared.add(suite: suite, TestCase(name: "ranking handles ties invalid distances and empty sets deterministically", knownBug: nil) {
        let matches = [CrystalOrbMatch(frame: orbFrame(8), distance: 1), CrystalOrbMatch(frame: orbFrame(2), distance: 1),
                       CrystalOrbMatch(frame: orbFrame(3), distance: 0), CrystalOrbMatch(frame: orbFrame(4), distance: .nan)]
        checkEqual(CrystalOrbMatch.ranked(matches).map(\.id.sampleIndex), [3, 2, 8], "ties use source identity")
        check(CrystalOrbMatch.ranked([]).isEmpty, "empty results safe")
    })
    Registry.shared.add(suite: suite, TestCase(name: "live ranking gate caps changed lists to once per second and skips unchanged lists", knownBug: nil) {
        let first = [CrystalOrbMatch(frame: orbFrame(0), distance: 1)]
        let next = [CrystalOrbMatch(frame: orbFrame(1), distance: 0)]
        var gate = CrystalOrbPublicationGate()
        checkEqual(gate.delay(for: first, at: 10), 0, "first ranking appears immediately")
        gate.published(first, at: 10)
        check(gate.delay(for: first, at: 11) == nil, "unchanged ranking never republishes")
        checkClose(gate.delay(for: next, at: 10.1) ?? -1, 0.9, tolerance: 0.0001, "early changes wait for the one-second boundary")
        checkEqual(gate.delay(for: next, at: 11), 0, "second changed ranking is eligible after one second")
        gate.published(next, at: 11)
        checkClose(gate.delay(for: first, at: 11.1) ?? -1, 0.9, tolerance: 0.0001, "cap applies to every subsequent publication")
    })
    Registry.shared.add(suite: suite, TestCase(name: "alignment preserves aspect and upscales smaller sources without cropping", knownBug: nil) {
        let a = CrystalOrbAlignment(target: CGSize(width: 4032, height: 3024), candidate: CGSize(width: 1920, height: 1080))
        checkEqual(a.size, CGSize(width: 4032, height: 3024), "native target retained")
        checkEqual(a.target, CGRect(x: 0, y: 0, width: 4032, height: 3024), "target is not cropped")
        checkClose(Double(a.candidate.width / a.candidate.height), 16.0 / 9, tolerance: 0.0001, "candidate not stretched")
        check(a.candidate.width > 1920 && a.candidate.minY > 0, "smaller candidate upscaled and letterboxed")
    })
    Registry.shared.add(suite: suite, TestCase(name: "identical differences and masks are zero; changed pixels appear in place", knownBug: nil) {
        let original = try orbImage(0)
        let same = try CrystalOrbDifferenceRenderer.render(target: original, candidate: original)
        checkEqual(orbPixel(same.difference, x: 40, y: 40), [0, 0, 0, 255], "identical normalized pixels are black")
        checkEqual(orbPixel(same.redMask, x: 40, y: 40)[3], 0, "identical pixels have transparent red mask")
        let black = try ImageFixture.cgImage(width: 32, height: 32, gray: 0)
        let white = try ImageFixture.cgImage(width: 32, height: 32, gray: 1)
        let changed = try CrystalOrbDifferenceRenderer.render(target: black, candidate: white)
        checkEqual(orbPixel(changed.difference), [255, 255, 255, 255], "full RGB absolute difference")
        checkEqual(orbPixel(changed.redMask), [255, 0, 0, 255], "maximum difference creates opaque red")
        let context = CGContext(data: nil, width: 32, height: 32, bitsPerComponent: 8, bytesPerRow: 0,
                                space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        context.draw(black, in: CGRect(x: 0, y: 0, width: 32, height: 32))
        context.setFillColor(CGColor(gray: 1, alpha: 1)); context.fill(CGRect(x: 4, y: 4, width: 8, height: 8))
        let localized = try CrystalOrbDifferenceRenderer.render(target: black, candidate: context.makeImage()!)
        checkEqual(orbPixel(localized.difference, x: 8, y: 24)[0], 255, "changed region detected")
        checkEqual(orbPixel(localized.difference, x: 20, y: 20)[0], 0, "unchanged region remains zero")
    })
    Registry.shared.add(suite: suite, TestCase(name: "different aspects mark unmatched regions and resized identical content matches", knownBug: nil) {
        let large = try ImageFixture.cgImage(width: 128, height: 96, gray: 0.5)
        let small = try ImageFixture.cgImage(width: 32, height: 24, gray: 0.5)
        let resized = try CrystalOrbDifferenceRenderer.render(target: large, candidate: small)
        check(orbPixel(resized.difference, x: 64, y: 48).prefix(3).allSatisfy { $0 <= 1 }, "upscaled same content matches")
        check(orbPixel(resized.difference, x: 0, y: 0).prefix(3).allSatisfy { $0 <= 1 }, "clamped resampling preserves matching edge pixels")
        let wide = try ImageFixture.cgImage(width: 128, height: 32, gray: 0.5)
        let letterbox = try CrystalOrbDifferenceRenderer.render(target: large, candidate: wide)
        checkEqual(orbPixel(letterbox.difference, x: 64, y: 2), [255, 0, 255, 255], "unmatched is magenta")
        checkEqual(orbPixel(letterbox.redMask, x: 64, y: 2)[3], 255, "unmatched mask explicit")
        checkEqual(orbPixel(letterbox.redMask, x: 64, y: 48)[3], 0, "overlap same content transparent")
    })
    Registry.shared.add(suite: suite, TestCase(name: "file orientation metadata and clipboard ownership stay independent of frames", knownBug: nil) {
        let image = try orbImage(0, width: 80, height: 40)
        let data = NSMutableData()
        let destination = CGImageDestinationCreateWithData(data, "public.tiff" as CFString, 1, nil)!
        CGImageDestinationAddImage(destination, image, [kCGImagePropertyOrientation: 6] as CFDictionary)
        check(CGImageDestinationFinalize(destination), "fixture encoded")
        let query = try CrystalOrbImageSource.read(data: data as Data)
        checkEqual(query.image.width, 40, "EXIF rotation applied")
        checkEqual(query.image.height, 80, "oriented native pixels")
        check(query.fileURL == nil && query.metadata.fileSizeBytes == nil, "clipboard has no fabricated source or file size")
        checkEqual(query.metadata.orientation, "Right (90° clockwise)", "actual embedded metadata retained")
    })
    Registry.shared.add(suite: suite, TestCase(name: "real Vision ranks identical and reencoded resized images above unrelated content and caches 2048 entries", knownBug: nil) {
        let query = try orbImage(0)
        let same = try await GenerateImageFeaturePrintRequest(.revision2).perform(on: query)
        let other = try await GenerateImageFeaturePrintRequest(.revision2).perform(on: orbImage(1))
        let resized = try orbImage(0, width: 256, height: 192)
        let encoded = NSMutableData()
        let dest = CGImageDestinationCreateWithData(encoded, "public.jpeg" as CFString, 1, nil)!
        CGImageDestinationAddImage(dest, resized, [kCGImageDestinationLossyCompressionQuality: 0.8] as CFDictionary)
        check(CGImageDestinationFinalize(dest), "JPEG encoded")
        let reencoded = try CrystalOrbImageSource.read(data: encoded as Data).image
        let resizedFeature = try await GenerateImageFeaturePrintRequest(.revision2).perform(on: reencoded)
        let frames = [orbFrame(0, feature: FeaturePrintCodec.encode(same)), orbFrame(1, feature: FeaturePrintCodec.encode(resizedFeature))]
            + (2..<2048).map { orbFrame($0, feature: FeaturePrintCodec.encode(other)) }
        let engine = CrystalOrbSearchService(), generation = UUID(), queryID = UUID()
        let start = Date()
        let result = try await engine.search(query: query, queryID: queryID, frames: frames, sourceURL: URL(fileURLWithPath: "/unused"), generation: generation) { _, _ in }
        let cold = Date().timeIntervalSince(start)
        checkEqual(result.first?.id.sampleIndex, 0, "exact match first, independent of art score")
        checkEqual(result[1].id.sampleIndex, 1, "resized JPEG outranks unrelated images")
        let warmStart = Date()
        let repeated = try await engine.search(query: query, queryID: queryID, frames: frames, sourceURL: URL(fileURLWithPath: "/unused"), generation: generation) { _, _ in }
        checkEqual(result, repeated, "deterministic repeat")
        checkEqual(await engine.cachedCount, 2048, "session features retained")
        checkEqual(await engine.computations, 0, "existing source features avoid decoding originals")
        checkEqual(await engine.queryComputations, 1, "one feature computation per query")
        print(String(format: "    Orb 2048 candidates: cold %.3fs, warm %.3fs", cold, Date().timeIntervalSince(warmStart)))
        let empty = try await engine.search(query: query, queryID: UUID(), frames: [], sourceURL: URL(fileURLWithPath: "/other"), generation: UUID()) { _, _ in }
        check(empty.isEmpty, "new source has no old matches")
        checkEqual(await engine.cachedCount, 0, "source generation clears old index")
        let cancelled = Task { try await engine.search(query: query, queryID: UUID(), frames: frames, sourceURL: URL(fileURLWithPath: "/unused"), generation: generation) { _, _ in } }
        cancelled.cancel()
        do { _ = try await cancelled.value; check(false, "cancelled searches must not return stale matches") } catch is CancellationError { }
    })
    Registry.shared.add(suite: suite, TestCase(name: "hidden candidate pin preserves filters and restores exact selected identity", knownBug: nil) {
        let directory = try Temporary.directory("orb-selection")
        await Task { @MainActor in
            let session = WaifuSession(settingsURL: directory.appendingPathComponent("settings.json"))
            session.updateAnalyzedFrames((0..<40).map { orbFrame($0) })
            session.frameLimits.art.low = 30
            let filters = session.frameLimits
            let candidate = FrameID(sampleIndex: 4)
            check(!session.displayedFrames.contains { $0.id == candidate }, "candidate is hidden")
            session.crystalOrbIsOpen = true
            session.selectCrystalOrbCandidate(candidate)
            checkEqual(session.selectedFrameIDs, [candidate], "exact selected ID synchronized while open")
            session.crystalOrbIsOpen = false
            session.finishCrystalOrbNavigation()
            check(session.displayedFrames.contains { $0.id == candidate }, "hidden candidate pinned for staged layout")
            checkEqual(session.gridNavigationFrameID, candidate, "existing scroll pipeline targets identity")
            checkEqual(session.frameLimits, filters, "filter preferences unchanged")
            session.focusFromPointer(FrameID(sampleIndex: 35))
            check(!session.displayedFrames.contains { $0.id == candidate }, "pin removed after normal browsing resumes")
            let model = WaifuCrystalOrbModel()
            model.open(session: session)
            model.redOverlay = true; model.opacity = 0.3; model.mode = .difference
            model.step(1)
            model.close()
            check(!model.isOpen && !session.crystalOrbIsOpen && model.pair == nil, "close releases buffers and keyboard ownership")
            check(model.redOverlay && model.opacity == 0.3, "overlay preference survives mode and lifecycle")
        }.value
    })
}

private func orbPNG(_ image: CGImage) -> Data {
    let bytes = NSMutableData()
    let destination = CGImageDestinationCreateWithData(bytes, "public.png" as CFString, 1, nil)!
    CGImageDestinationAddImage(destination, image, nil)
    _ = CGImageDestinationFinalize(destination)
    return bytes as Data
}

@MainActor
private func orbWait(_ predicate: () -> Bool) async throws {
    let deadline = Date().addingTimeInterval(15)
    while !predicate() {
        if Date() > deadline { throw NSError(domain: "CrystalOrbTest", code: 1, userInfo: [NSLocalizedDescriptionKey: "Timed out awaiting background work"]) }
        try await Task.sleep(for: .milliseconds(10))
    }
}

func registerCrystalOrbLifecycleTests() {
    let suite = "Crystal Orb lifecycle"
    Registry.shared.add(suite: suite, TestCase(name: "Safari archive and alternate image formats reach Orb from thumbnail mode", knownBug: nil) {
        let directory = try Temporary.directory("orb-safari-archive")
        let payload = orbPNG(try orbImage(2, width: 200, height: 150))
        let archive = try PropertyListSerialization.data(fromPropertyList: [
            "WebMainResource": ["WebResourceData": Data("<img src='photo.png'>".utf8), "WebResourceMIMEType": "text/html"],
            "WebSubresources": [["WebResourceData": payload, "WebResourceMIMEType": "image/png"]]
        ], format: .binary, options: 0)
        try await Task { @MainActor in
            checkEqual(CrystalOrbImport.archivedImage(archive), payload, "embedded image is extracted from Safari archive")
            check(CrystalOrbImport.archivedImage(Data("invalid".utf8)) == nil, "invalid archive is rejected")
            check(CrystalOrbImport.dropTypes.contains { $0.identifier == CrystalOrbImport.webArchiveType }, "window accepts the archive representation")
            let session = WaifuSession(settingsURL: directory.appendingPathComponent("settings.json"))
            let model = WaifuCrystalOrbModel()
            let provider = NSItemProvider(item: archive as NSData, typeIdentifier: CrystalOrbImport.webArchiveType)
            check(CrystalOrbImport.accept([provider], model: model, session: session), "archive-only Safari drag is accepted")
            try await orbWait { model.query != nil && !model.importing }
            checkEqual(model.query?.nativeSize, CGSize(width: 200, height: 150), "archive supplies the dragged pixels")
            model.close()
            let alternatives = NSItemProvider()
            alternatives.registerDataRepresentation(forTypeIdentifier: UTType.tiff.identifier, visibility: .all) { completion in
                completion(Data("unavailable TIFF".utf8), nil); return nil
            }
            alternatives.registerDataRepresentation(forTypeIdentifier: UTType.png.identifier, visibility: .all) { completion in
                completion(payload, nil); return nil
            }
            checkEqual(await CrystalOrbImport.imageData(from: alternatives), payload, "usable second image representation is tried")
            session.cancel()
        }.value
    })
    Registry.shared.add(suite: suite, TestCase(name: "browser image drops ignore unrelated page URLs and start Orb immediately", knownBug: nil) {
        let directory = try Temporary.directory("orb-browser-drop")
        let payload = orbPNG(try orbImage(2, width: 200, height: 150))
        let urlLoads = OSAllocatedUnfairLock(initialState: 0)
        try await Task { @MainActor in
            let session = WaifuSession(settingsURL: directory.appendingPathComponent("settings.json"))
            let model = WaifuCrystalOrbModel()
            let provider = NSItemProvider()
            provider.registerDataRepresentation(forTypeIdentifier: UTType.url.identifier, visibility: .all) { completion in
                urlLoads.withLock { $0 += 1 }
                completion(Data("https://example.test/page".utf8), nil)
                return nil
            }
            provider.registerDataRepresentation(forTypeIdentifier: UTType.png.identifier, visibility: .all) { completion in
                completion(payload, nil)
                return nil
            }
            check(CrystalOrbImport.accept([provider], model: model, session: session), "mixed drop accepted")
            check(model.isOpen && model.importing, "Orb opens before provider resolution")
            try await orbWait { model.query != nil && !model.importing }
            checkEqual(urlLoads.withLock { $0 }, 0, "unrelated page URL is not loaded")
            checkEqual(model.status, "Load a video or open a Photos album to search.", "import invokes search even when no search source is loaded")
            check(session.video == nil && !session.progress.isRunning, "image is not routed to video analysis")
            model.close(); session.cancel()
        }.value
    })
    Registry.shared.add(suite: suite, TestCase(name: "image drops load the provider's registered image type and stage a file when data is unusable", knownBug: nil) {
        let directory = try Temporary.directory("orb-drop")
        let payload = orbPNG(try orbImage(2, width: 200, height: 150))
        try await Task { @MainActor in
            let session = WaifuSession(settingsURL: directory.appendingPathComponent("settings.json"))
            let model = WaifuCrystalOrbModel()

            // Drag sources register the concrete type they can deliver; the import
            // must request that identifier, not the abstract public.image.
            let heicLike = NSItemProvider(item: payload as NSData, typeIdentifier: "public.jpeg")
            checkEqual(CrystalOrbImport.imageType(of: heicLike), "public.jpeg", "concrete registered type is requested")
            let abstract = NSItemProvider(item: payload as NSData, typeIdentifier: "public.image")
            checkEqual(CrystalOrbImport.imageType(of: abstract), "public.image", "abstract-only registration falls back to the supertype")
            let mixed = NSItemProvider()
            mixed.registerDataRepresentation(forTypeIdentifier: "public.url", visibility: .all) { completion in
                completion(Data("https://example.test/a.png".utf8), nil); return nil
            }
            mixed.registerDataRepresentation(forTypeIdentifier: "public.tiff", visibility: .all) { completion in
                completion(payload, nil); return nil
            }
            checkEqual(CrystalOrbImport.imageType(of: mixed), "public.tiff", "first registered image type wins over non-image types")
            checkEqual(CrystalOrbImport.imageType(of: NSItemProvider()), "public.image", "empty provider falls back to the supertype")

            check(CrystalOrbImport.isDecodable(payload), "image payload recognised")
            check(!CrystalOrbImport.isDecodable(Data("not an image".utf8)), "non-image payload rejected before import")

            // A drop that only offers a bitmap representation opens the orb and
            // imports through the concrete type on the first attempt.
            check(CrystalOrbImport.accept([heicLike], model: model, session: session), "bitmap drop accepted")
            try await orbWait { model.query != nil && !model.importing }
            check(model.query != nil, "first drop imports instead of failing")
            model.close()

            // When data coercion cannot serve the image, the file representation
            // is staged and imported through the same pipeline.
            let file = directory.appendingPathComponent("staged.png")
            try payload.write(to: file)
            let fileBacked = NSItemProvider(contentsOf: file)!
            let staged = await CrystalOrbImport.stageImageFile(from: fileBacked)
            if let staged { check(CrystalOrbImport.isDecodable(try Data(contentsOf: staged)), "file representation is copied out decodable") }
            else { check(false, "file representation is copied out decodable") }
            model.close()

            // A promised file can arrive before its type is derivable from the
            // path (no extension): the drag declared an image, so the orb — not
            // the video importer — must receive it.
            let untyped = directory.appendingPathComponent("promised-image")
            try payload.write(to: untyped)
            let promised = NSItemProvider()
            promised.registerDataRepresentation(forTypeIdentifier: UTType.fileURL.identifier, visibility: .all) { completion in
                completion(untyped.dataRepresentation, nil); return nil
            }
            promised.registerDataRepresentation(forTypeIdentifier: "public.png", visibility: .all) { completion in
                completion(payload, nil); return nil
            }
            check(CrystalOrbImport.accept([promised], model: model, session: session), "untyped promised image drop accepted")
            try await orbWait { model.query != nil && !model.importing }
            check(model.query != nil, "untyped image drop reaches the orb")
            check(session.video == nil && !session.progress.isRunning, "untyped image drop is never handed to the video importer")
            model.close(); session.cancel()
        }.value
    })
    Registry.shared.add(suite: suite, TestCase(name: "flaky drop representations are re-asked before the drop is declared unreadable", knownBug: nil) {
        let directory = try Temporary.directory("orb-drop-retry")
        let payload = orbPNG(try orbImage(3, width: 160, height: 120))
        // Registered load handlers run once per request, so a first failure
        // followed by success models the cross-process flakes DropRetry exists
        // for. Handlers are invoked on provider queues, so the counters lock.
        let dataAsks = OSAllocatedUnfairLock(initialState: 0)
        let urlAsks = OSAllocatedUnfairLock(initialState: 0)
        let fileAsks = OSAllocatedUnfairLock(initialState: 0)
        let transient = NSError(domain: NSItemProvider.errorDomain, code: -1000)
        try await Task { @MainActor in
            let session = WaifuSession(settingsURL: directory.appendingPathComponent("settings.json"))
            let model = WaifuCrystalOrbModel()

            // Corrupt bytes on the first data load, a real image on the second.
            let corruptFirst = NSItemProvider()
            corruptFirst.registerDataRepresentation(forTypeIdentifier: "public.png", visibility: .all) { completion in
                let first = dataAsks.withLock { asks -> Bool in asks += 1; return asks == 1 }
                completion(first ? Data("not an image".utf8) : payload, nil)
                return nil
            }
            check(CrystalOrbImport.accept([corruptFirst], model: model, session: session), "flaky bitmap drop accepted")
            try await orbWait { model.query != nil && !model.importing }
            check(model.query != nil, "corrupt first representation retries into a successful import")
            check(dataAsks.withLock { $0 } >= 2, "undecodable payload causes a second ask")
            model.close()

            // A URL resolution that fails once would otherwise lose the file
            // from the drop entirely.
            let dropped = directory.appendingPathComponent("retry-drop.png")
            try payload.write(to: dropped)
            let urlFlaky = NSItemProvider()
            urlFlaky.registerDataRepresentation(forTypeIdentifier: UTType.fileURL.identifier, visibility: .all) { completion in
                let first = urlAsks.withLock { asks -> Bool in asks += 1; return asks == 1 }
                if first { completion(nil, transient) } else { completion(dropped.dataRepresentation, nil) }
                return nil
            }
            checkEqual(await VideoDrop.urls(from: [urlFlaky]), [dropped], "transient URL failure is re-asked and resolves")

            // The file-promise fallback gets the same treatment.
            let fileFlaky = NSItemProvider()
            fileFlaky.registerFileRepresentation(forTypeIdentifier: "public.png", fileOptions: [], visibility: .all) { completion in
                let first = fileAsks.withLock { asks -> Bool in asks += 1; return asks == 1 }
                if first { completion(nil, false, transient); return nil }
                let promised = FileManager.default.temporaryDirectory
                    .appendingPathComponent("waifu-flaky-\(UUID().uuidString).png")
                do { try payload.write(to: promised); completion(promised, false, nil) }
                catch { completion(nil, false, error) }
                return nil
            }
            let staged = await CrystalOrbImport.stageImageFile(from: fileFlaky)
            check(staged != nil, "transient file-promise failure is re-asked and staged")
            if let staged { check(CrystalOrbImport.isDecodable(try Data(contentsOf: staged)), "staged retry decodes") }
            check(fileAsks.withLock { $0 } >= 2, "failed file representation causes a second ask")
        }.value
    })
    Registry.shared.add(suite: suite, TestCase(name: "replacing imports and closing cannot publish late query pixels", knownBug: nil) {
        let directory = try Temporary.directory("orb-lifecycle")
        let first = orbPNG(try orbImage(0, width: 1024, height: 768))
        let second = orbPNG(try orbImage(1, width: 96, height: 64))
        try await Task { @MainActor in
            let session = WaifuSession(settingsURL: directory.appendingPathComponent("settings.json"))
            let model = WaifuCrystalOrbModel()
            model.importImage(data: first, session: session)
            model.importImage(data: second, session: session)
            try await orbWait { model.query != nil }
            checkEqual(model.query?.image.width, 96, "replacement query wins")
            check(model.matches.isEmpty && !model.searching, "no source has explicit empty search state")
            let retained = model.query?.id
            model.importImage(data: Data("not an image".utf8), session: session)
            checkEqual(model.query?.id, retained, "query content stays mounted during replacement decoding")
            try await orbWait { !model.importing }
            checkEqual(model.query?.id, retained, "failed replacement preserves the previous usable query")
            model.importImage(data: first, session: session)
            checkEqual(model.query?.id, retained, "new decoding does not blank the canvas")
            model.close()
            // Let already-enqueued actor work run without interacting with a window.
            for _ in 0..<100 { await Task.yield() }
            check(model.query == nil && model.pair == nil && !model.isOpen, "late import cannot reopen or repopulate closed workspace")
        }.value
    })
    Registry.shared.add(suite: suite, TestCase(name: "real video query search selection comparison and source replacement share exact identities", knownBug: nil) {
        let directory = try Temporary.directory("orb-video-integration")
        let url = try VideoFixture.makeMP4(in: directory, durationSeconds: 2.1, width: 320, height: 240, frameRate: 6)
        let image = try await PhotosMedia.shared.image(url, seconds: 0)
        let data = orbPNG(image)
        try await Task { @MainActor in
            let session = WaifuSession(settingsURL: directory.appendingPathComponent("settings.json"))
            session.settings.analysisMode = .quick
            let videoProvider = NSItemProvider(contentsOf: url)!
            let model = WaifuCrystalOrbModel()
            check(CrystalOrbImport.accept([videoProvider], model: model, session: session), "video drop accepted")
            try await orbWait { session.video != nil && !session.progress.isRunning }
            check(!model.isOpen, "video drop remains in ordinary import workflow")
            check(session.errorMessage == nil, "native fixture analyzed")
            for frame in session.frames {
                if let thumbnail = await session.thumbnailData(for: frame) {
                    _ = await GridThumbnailLoader.shared.image(data: thumbnail, identity: session.thumbnailGeneration.uuidString + ":" + frame.thumbnailKey)
                }
            }
            var rankingTimes: [TimeInterval] = []
            let rankingObserver = model.$matches.sink { matches in
                if !matches.isEmpty { rankingTimes.append(ProcessInfo.processInfo.systemUptime) }
            }
            var previewAlignment: CrystalOrbAlignment?
            let observer = model.$pair.sink { pair in
                if let pair, !pair.candidateIsOriginal { previewAlignment = pair.alignment }
            }
            model.zoom = 2.75; model.pan = CGSize(width: 21, height: -13)
            model.swipe = 0.36; model.opacity = 0.47; model.redOverlay = true
            let imageProvider = NSItemProvider(item: data as NSData, typeIdentifier: "public.png")
            check(CrystalOrbImport.accept([imageProvider], model: model, session: session), "external image representation accepted")
            try await orbWait { model.query != nil && !model.searching && !model.matches.isEmpty }
            try await orbWait { !model.loading && model.pair != nil }
            check(model.pair?.candidateIsOriginal == true, "automatic best candidate upgrades to the original")
            check(!model.canLoadOriginal, "automatic upgrade does not require a load button")
            for (earlier, later) in zip(rankingTimes, rankingTimes.dropFirst()) {
                check(later - earlier >= 0.99, "live ranked list updates at most once a second")
            }
            check(rankingTimes.count >= 2, "first and completed rankings are presented")
            let id = model.selectedID
            checkEqual(session.selectedFrameIDs, Set(id.map { [$0] } ?? []), "search selected exact underlying frame")
            checkEqual(model.pair?.frame.id, id, "loaded original belongs to selected identity")
            checkEqual(model.pair?.candidate.width, 320, "native original loaded")
            check(previewAlignment != nil, "cached thumbnail paints before the original")
            checkEqual(model.pair?.alignment, previewAlignment, "preview and original share native geometry")
            checkEqual(model.selectedID, model.matches.first?.id, "rank one automatically fills candidate pane")
            checkEqual(model.zoom, 2.75, "search and quality upgrades preserve zoom")
            checkEqual(model.pan, CGSize(width: 21, height: -13), "quality upgrades preserve pan")
            checkEqual(model.swipe, 0.36, "quality upgrades preserve divider")
            checkEqual(model.opacity, 0.47, "quality upgrades preserve opacity")
            check(model.redOverlay, "quality upgrades preserve overlay preference")
            withExtendedLifetime((observer, rankingObserver)) {}
            let mask = model.pair?.difference?.redMask
            if let id, let other = model.matches.last?.id, other != id {
                let firstDifference = model.pair?.difference?.id
                model.select(other)
                checkEqual(model.pair?.frame.id, other, "click paints the cached preview immediately")
                check(model.pair?.candidateIsOriginal == false, "the immediate image is identified as a preview")
                try await orbWait { !model.loading && model.pair?.frame.id == other }
                checkEqual(session.selectedFrameIDs, [other], "different candidate selects exact grid identity")
                model.select(id)
                check(!model.loading, "revisiting a cached candidate is immediate")
                checkEqual(model.pair?.difference?.id, firstDifference, "revisit reuses original and difference without a second decode")
            }
            if let id { model.select(id) }
            model.step(-1)
            check(!model.loading && model.pair?.difference?.redMask === mask, "same candidate and strip boundary reuse originals and difference")
            model.opacity = 0; model.redOverlay = true; model.mode = .difference
            model.opacity = 1; model.mode = .swipe; model.swipe = 0.2
            check(model.pair?.difference?.redMask === mask, "opacity mode and divider reuse the computed mask")
            model.mode = .toggle; model.showCandidate.toggle(); model.pan = CGSize(width: 12, height: -3); model.zoom = 2
            model.showCandidate.toggle()
            checkEqual(model.zoom, 2, "toggle preserves linked zoom")
            checkEqual(model.pan, CGSize(width: 12, height: -3), "toggle preserves pan")
            model.close()
            checkEqual(session.gridNavigationFrameID, id, "close requests exact frame through staged navigation")
            model.importImage(data: data, session: session)
            try await orbWait { model.query != nil }
            session.loadVideo(at: url) // source revision changes even for the same path
            model.sourceChanged(session: session)
            check(model.matches.isEmpty && model.pair == nil && model.selectedID == nil, "source replacement clears stale matches and pixels synchronously")
            model.close(); session.cancel()
        }.value
    })
    Registry.shared.add(suite: suite, TestCase(name: "Photos inspection requests the largest useful local preview independently of analysis size", knownBug: nil) {
        checkEqual(PhotosMedia.inspectionRenditionSize(width: 6000, height: 4000), CGSize(width: 2048, height: 1365), "inspection gets more detail than a 512-pixel analysis thumbnail")
        checkEqual(PhotosMedia.inspectionRenditionSize(width: 3000, height: 6000), CGSize(width: 1024, height: 2048), "portrait preview preserves aspect")
        checkEqual(PhotosMedia.inspectionRenditionSize(width: 320, height: 240), CGSize(width: 320, height: 240), "small sources are never needlessly upscaled")
        checkEqual(PhotosMedia.renditionSize(width: 6000, height: 4000, maxPixelSize: 2048).width, 512, "analysis cache bounds stay unchanged")
    })
    Registry.shared.add(suite: suite, TestCase(name: "query previews upgrade to originals without changing identity or inspection settings", knownBug: nil) {
        let directory = try Temporary.directory("orb-query-upgrade")
        let image = try orbImage(0, width: 4096, height: 2304)
        let bytes = orbPNG(image)
        let preview = try CrystalOrbImageSource.read(data: bytes, maxPixelSize: 2048)
        checkEqual(preview.image.width, 2048, "best quick query preview is decoded without full-size buffers")
        checkEqual(preview.nativeSize, CGSize(width: 4096, height: 2304), "preview carries original coordinate size")
        check(!preview.isOriginal, "preview quality is explicit")
        try await Task { @MainActor in
            let session = WaifuSession(settingsURL: directory.appendingPathComponent("settings.json"))
            let model = WaifuCrystalOrbModel()
            var previews: [UUID] = []
            let observer = model.$query.sink { query in
                if let query, !query.isOriginal { previews.append(query.id) }
            }
            model.zoom = 3; model.pan = CGSize(width: 15, height: -8); model.mode = .swipe
            model.swipe = 0.31; model.opacity = 0.82; model.redOverlay = true; model.actualPixels = true
            model.importImage(data: bytes, session: session)
            try await orbWait { model.query?.isOriginal == true }
            checkEqual(previews, model.query.map { [$0.id] } ?? [], "thumbnail and original keep one query identity")
            checkEqual(model.query?.image.width, 4096, "background decode upgrades to original")
            checkEqual(model.zoom, 3, "zoom is never reset by importing or upgrading")
            checkEqual(model.pan, CGSize(width: 15, height: -8), "pan survives")
            checkEqual(model.mode, .swipe, "mode survives")
            checkEqual(model.swipe, 0.31, "divider survives")
            checkEqual(model.opacity, 0.82, "opacity survives")
            check(model.redOverlay && model.actualPixels, "overlay and actual-pixel settings survive")
            model.close(); withExtendedLifetime(observer) {}
        }.value
    })
    Registry.shared.add(suite: suite, TestCase(name: "progressive search publishes correctly ranked live top nineteen before completion", knownBug: nil) {
        let query = try orbImage(0)
        let other = try await GenerateImageFeaturePrintRequest(.revision2).perform(on: orbImage(1))
        let exact = try await GenerateImageFeaturePrintRequest(.revision2).perform(on: query)
        let otherData = FeaturePrintCodec.encode(other)
        let frames = (0..<2048).map { orbFrame($0, feature: $0 == 2047 ? FeaturePrintCodec.encode(exact) : otherData) }
        let snapshots = OrbSearchSnapshots()
        let result = try await CrystalOrbSearchService().search(query: query, queryID: UUID(), frames: frames,
            sourceURL: URL(fileURLWithPath: "/unused"), generation: UUID(),
            partial: { matches, done, total in await snapshots.record(matches, done: done, total: total) }) { _, _ in }
        let states = await snapshots.values
        check(states.count >= 2, "results appear before the whole source is scanned")
        checkEqual(states.first?.done, 1, "first readable candidate is published immediately")
        checkEqual(states.first?.matches.first?.id.sampleIndex, 0, "initial best can be displayed immediately")
        checkEqual(states.last?.matches.first?.id.sampleIndex, 2047, "new best replaces rank one during the search")
        for state in states {
            check(state.matches.count <= 19, "gallery never publishes more than nineteen")
            checkEqual(state.matches, CrystalOrbMatch.ranked(state.matches), "each partial list is correctly ranked")
            check(state.matches.allSatisfy { $0.id.sampleIndex < state.done }, "no unexamined candidates leak into a partial ranking")
        }
        checkEqual(states.last?.matches, Array(result.prefix(19)), "final live list equals exact top nineteen")
    })
    Registry.shared.add(suite: suite, TestCase(name: "large native comparisons use bounded previews and viewport tiles without a size rejection", knownBug: nil) {
        let large = try ImageFixture.cgImage(width: 8192, height: 6144, gray: 0)
        let small = try ImageFixture.cgImage(width: 32, height: 24, gray: 1)
        let start = Date()
        let result = try CrystalOrbDifferenceRenderer.render(target: large, candidate: small)
        checkEqual(result.nativeDifference.extent.size, CGSize(width: 8192, height: 6144), "over-40 MP comparison retains native coordinates")
        check(result.difference.width <= 1600 && result.redMask.width <= 1600, "preview buffers are bounded independently of original size")
        let tile = try await CrystalOrbViewportRenderer.shared.render(image: result.nativeDifference,
            canvas: result.alignment.size, viewport: CGSize(width: 128, height: 96), scale: 1, pan: .zero, backing: 2)
        checkEqual(tile.width, 256, "actual-pixel inspection renders only viewport width")
        checkEqual(tile.height, 192, "viewport raster accounts for backing resolution")
        checkEqual(orbPixel(tile, x: 64, y: 64), [255, 255, 255, 255], "native tile contains real full-resolution differences")
        print(String(format: "    Orb 50 MP comparison preview + native tile: %.3fs; preview %d×%d, tile %d×%d", Date().timeIntervalSince(start), result.difference.width, result.difference.height, tile.width, tile.height))
    })
    Registry.shared.add(suite: suite, TestCase(name: "subtle pixel changes produce a visible red overlay and unchanged pixels stay clear", knownBug: nil) {
        let target = try ImageFixture.cgImage(width: 32, height: 32, gray: 0.5)
        let candidate = try ImageFixture.cgImage(width: 32, height: 32, gray: 0.55)
        let result = try CrystalOrbDifferenceRenderer.render(target: target, candidate: candidate)
        let red = orbPixel(result.redMask, x: 16, y: 16)
        check(red[0] > 0 && red[1] == 0 && red[2] == 0, "mask is visibly red")
        check(red[3] >= 40, "display curve makes small differences legible")
        let tile = try await CrystalOrbViewportRenderer.shared.render(image: result.nativeRedMask,
            canvas: result.alignment.size, viewport: CGSize(width: 32, height: 32), scale: 1, pan: .zero, backing: 1)
        checkEqual(orbPixel(tile, x: 16, y: 16), red, "native overlay and preview have identical intensity and alignment")
        let identical = try CrystalOrbDifferenceRenderer.render(target: target, candidate: target)
        checkEqual(orbPixel(identical.redMask)[3], 0, "display curve never colors unchanged pixels")
    })
    Registry.shared.add(suite: suite, TestCase(name: "color profiles normalize and mask compositing respects zero and full opacity", knownBug: nil) {
        func colored(_ space: CGColorSpace) -> CGImage {
            let context = CGContext(data: nil, width: 16, height: 16, bitsPerComponent: 8, bytesPerRow: 0,
                                    space: space, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            let srgb = CGColorSpace(name: CGColorSpace.sRGB)!
            context.setFillColor(CGColor(colorSpace: srgb, components: [0.35, 0.6, 0.25, 1])!)
            context.fill(CGRect(x: 0, y: 0, width: 16, height: 16))
            return context.makeImage()!
        }
        let srgb = colored(CGColorSpace(name: CGColorSpace.sRGB)!)
        let p3 = colored(CGColorSpace(name: CGColorSpace.displayP3)!)
        let normalized = try CrystalOrbDifferenceRenderer.render(target: srgb, candidate: p3)
        check(orbPixel(normalized.difference, x: 8, y: 8).prefix(3).allSatisfy { $0 < 4 }, "same physical colors across profiles have near-zero difference")
        let black = try ImageFixture.cgImage(width: 16, height: 16, gray: 0)
        let white = try ImageFixture.cgImage(width: 16, height: 16, gray: 1)
        let rendered = try CrystalOrbDifferenceRenderer.render(target: black, candidate: white)
        func composite(_ opacity: CGFloat) -> CGImage {
            let context = CGContext(data: nil, width: 16, height: 16, bitsPerComponent: 8, bytesPerRow: 0,
                                    space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            context.draw(black, in: CGRect(x: 0, y: 0, width: 16, height: 16))
            context.setAlpha(opacity); context.draw(rendered.redMask, in: CGRect(x: 0, y: 0, width: 16, height: 16))
            return context.makeImage()!
        }
        checkEqual(orbPixel(composite(0)), [0, 0, 0, 255], "zero opacity leaves original unchanged")
        checkEqual(orbPixel(composite(1)), [255, 0, 0, 255], "full opacity applies full mask")
        let transparentContext = CGContext(data: nil, width: 16, height: 16, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        transparentContext.clear(CGRect(x: 0, y: 0, width: 16, height: 16))
        let transparent = transparentContext.makeImage()!
        let sameTransparent = try CrystalOrbDifferenceRenderer.render(target: transparent, candidate: transparent)
        checkEqual(orbPixel(sameTransparent.difference), [0, 0, 0, 255], "identical transparent pixels are valid black matches")
        checkEqual(orbPixel(sameTransparent.redMask)[3], 0, "transparent matches have no false overlay")
    })
}

private actor OrbSearchSnapshots {
    struct Snapshot: Sendable { let matches: [CrystalOrbMatch]; let done: Int; let total: Int }
    private(set) var values: [Snapshot] = []
    func record(_ matches: [CrystalOrbMatch], done: Int, total: Int) { values.append(Snapshot(matches: matches, done: done, total: total)) }
}
