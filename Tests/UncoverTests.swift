import CoreGraphics
import CoreImage
import Foundation

func registerUncoverTests() {
    let suite = "Uncover"
    Registry.shared.add(suite: suite, TestCase(name: "source seek works before reconstruction and rapid reads keep only the latest frame", knownBug: nil) {
        let directory = try Temporary.directory("uncover-source-seek")
        let source = try VideoFixture.makeMP4(in: directory, durationSeconds: 2, width: 128, height: 96, frameRate: 24)
        let video = try await VideoProbe.probe(source)
        let backend = UncoverCountingBackend()
        let session = await UncoverSession(video: video, seconds: 0, backend: backend)
        try await waitUntil { await MainActor.run { session.original != nil && session.sourceFrameCount == 48 && !session.readingSource } }
        let roi = await session.roi
        await MainActor.run {
            for index in 0..<100 { session.seekSource(index: (index * 7) % 48) }
            session.seekSource(index: 47)
        }
        try await waitUntil { await MainActor.run { !session.readingSource && session.sourceDisplayedIndex == 47 } }
        checkEqual(await session.start, 47.0 / 24, "selected source PTS sets the reconstruction start")
        checkEqual(await session.roi, roi, "seeking keeps the selected region")
        await MainActor.run { session.step(-1); session.step(-1) }
        try await waitUntil { await MainActor.run { !session.readingSource && session.sourceDisplayedIndex == 45 } }
        checkEqual(await session.sourceRequestedIndex, 45, "fresh presses move from the requested frame, even before its read completes")
        await session.seekSource(index: -100)
        try await waitUntil { await MainActor.run { !session.readingSource && session.sourceDisplayedIndex == 0 } }
        await session.seekSource(index: 100)
        try await waitUntil { await MainActor.run { !session.readingSource && session.sourceDisplayedIndex == 47 } }
        check(await session.error == nil, "navigation never leaves a stale read error")
        check(!(await session.canReconstruct), "the final frame cannot satisfy the five-frame interval minimum")
        checkEqual(await backend.count, 0, "source navigation never launches inference")
        await session.dispose()
    })
    Registry.shared.add(suite: suite, TestCase(name: "letterboxed selection maps to exact native pixels", knownBug: nil) {
        let fit = UncoverROI.aspectFit(video: CGSize(width: 1920, height: 1080), in: CGSize(width: 800, height: 600))
        checkEqual(fit, CGRect(x: 0, y: 75, width: 800, height: 450), "aspect fit")
        let roi = UncoverROI(x: 0.25, y: 0.25, width: 1, height: 1)
        checkEqual(try roi.pixels(in: CGSize(width: 1920, height: 1080)),
                   CGRect(x: 480, y: 270, width: 1440, height: 810), "clipped ROI")
        for x in stride(from: 1, to: 1600, by: 7) {
            let pixels = CGRect(x: x, y: 157, width: 251, height: 179)
            checkEqual(try UncoverROI.from(pixels: pixels, in: CGSize(width: 1920, height: 1080))
                .pixels(in: CGSize(width: 1920, height: 1080)), pixels, "round trip does not grow")
        }
    })
    Registry.shared.add(suite: suite, TestCase(name: "odd dimensions degrade once without zero-sized inputs", knownBug: nil) {
        checkEqual(UncoverScale.one.stages, 0, "1× never uses learned enlargement")
        checkEqual(UncoverScale.allCases.map(\.rawValue), [1, 2, 4, 8, 16, 32, 64], "existing factors retained")
        for factor in UncoverScale.allCases {
            checkEqual(factor.degradedSize(CGSize(width: 641, height: 479)),
                       CGSize(width: ceil(641 / Double(factor.rawValue)), height: ceil(479 / Double(factor.rawValue))), "odd size")
            checkEqual(factor.degradedSize(CGSize(width: 1, height: 1)), CGSize(width: 1, height: 1), "tiny ROI")
        }
    })
    Registry.shared.add(suite: suite, TestCase(name: "1x preparation preserves odd ROI pixels and original export geometry", knownBug: nil) {
        let directory = try Temporary.directory("uncover-original-resolution")
        let source = try VideoFixture.makeMP4(in: directory, durationSeconds: 1, width: 128, height: 96, frameRate: 24)
        let rect = CGRect(x: 13, y: 11, width: 65, height: 47)
        let prepared = try await TemporalFrameExtractor().prepare(source: source,
            roi: .from(pixels: rect, in: CGSize(width: 128, height: 96)), scale: .one,
            start: 0, end: 0.5, directory: directory.appendingPathComponent("job"), progress: { _ in })
        for frame in prepared.manifest.frames {
            let original = try RegionImages.read(RegionImages.frameURL(prepared.directory, "original", frame.index))
            let input = try RegionImages.read(RegionImages.frameURL(prepared.directory, "degraded", frame.index))
            let baseline = try RegionImages.read(RegionImages.frameURL(prepared.directory, "baseline", frame.index))
            checkEqual(input.width, 65, "no hidden reduction of odd width")
            checkEqual(input.height, 47, "no hidden reduction of odd height")
            guard let crop = original.cropping(to: rect) else { throw UncoverError.invalid("Fixture crop failed") }
            checkEqual(rgba(input), rgba(crop), "1× retains source ROI pixels")
            checkEqual(rgba(baseline), rgba(input), "no downsample/upsample baseline")
        }
        let result = try await UncoverTestBackend().reconstruct(prepared, allowUnsafeMemory: false, progress: { _ in })
        let rendered = try await UncoverRenderer().render(result, index: 0, feather: 0, strength: 1)
        checkEqual(rendered.reconstructed.width, 128, "preview original width")
        checkEqual(rendered.reconstructed.height, 96, "preview original height")
        let destination = directory.appendingPathComponent("original-resolution.mov")
        try await UncoverVideoExporter().export(result, feather: 0, strength: 1, to: destination, progress: { _ in })
        let track = try await AVURLAsset(url: destination).loadTracks(withMediaType: .video)[0]
        checkEqual(try await track.load(.naturalSize), CGSize(width: 128, height: 96), "export retains source dimensions")
        checkEqual(try await videoTimes(destination).count, prepared.manifest.frames.count, "no missing or duplicate frames")
    })
    Registry.shared.add(suite: suite, TestCase(name:"spatial restoration reads straight RGB at translucent crop edges",knownBug:nil) {
        let bytes:[UInt8] = [88,84,117,228,200,150,75,0]
        let provider = CGDataProvider(data:Data(bytes) as CFData)!
        let image = CGImage(width:2,height:1,bitsPerComponent:8,bitsPerPixel:32,bytesPerRow:8,
            space:CGColorSpaceCreateDeviceRGB(),bitmapInfo:CGBitmapInfo(rawValue:CGImageAlphaInfo.last.rawValue),
            provider:provider,decode:nil,shouldInterpolate:false,intent:.defaultIntent)!
        checkEqual(try NativeSpatialUpscaler.straightRGBPixels(image),[88,84,117,200,150,75],"alpha does not darken or erase model input channels")
    })
    Registry.shared.add(suite: suite, TestCase(name: "spatial upscaler padding stays inside partial edge tiles", knownBug: nil) {
        for tileLength in [1, 2, 33, 479, 480, 512] {
            for paddedIndex in 0..<522 {
                let sourceIndex = NativeSpatialUpscaler.reflected(paddedIndex, count: tileLength)
                check(sourceIndex >= 0 && sourceIndex < tileLength,
                      "reflected index \(sourceIndex) escapes a \(tileLength)-pixel tile")
            }
        }
    })
    Registry.shared.add(suite: suite, TestCase(name: "feather is inward and full strength uses reconstruction in the interior", knownBug: nil) {
        checkEqual(RegionImages.maskWeight(x: -1, y: 10, width: 32, height: 32, feather: 8, strength: 1), 0, "outside")
        checkEqual(RegionImages.maskWeight(x: 16, y: 16, width: 32, height: 32, feather: 8, strength: 1), 1, "interior")
        checkEqual(RegionImages.maskWeight(x: 16, y: 16, width: 32, height: 32, feather: 8, strength: 0), 0, "zero blend")
        let original = try ImageFixture.cgImage(width: 64, height: 64, gray: 0.25)
        let region = try ImageFixture.cgImage(width: 32, height: 32, gray: 0.75)
        let result = try RegionImages.composite(original: original, region: region,
            rect: CGRect(x: 16, y: 16, width: 32, height: 32), feather: 8, strength: 1, context: CIContext())
        checkEqual(result.width, 64, "composite width")
        checkEqual(result.height, 64, "composite height")
        let originalPixels = rgba(original), resultPixels = rgba(result)
        checkEqual(Array(resultPixels[0..<4]), Array(originalPixels[0..<4]), "outside pixels unchanged")
        let center = (32 * 64 + 32) * 4
        checkEqual(Array(resultPixels[center..<(center + 3)]), Array(rgba(region)[0..<3]), "full-strength interior is reconstructed")
        let zero = try RegionImages.composite(original: original, region: region,
            rect: CGRect(x: 16, y: 16, width: 32, height: 32), feather: 8, strength: 0, context: CIContext())
        checkEqual(rgba(zero), originalPixels, "zero strength preserves every pixel")
    })
    Registry.shared.add(suite: suite, TestCase(name: "rotated video preparation uses oriented native geometry", knownBug: nil) {
        let directory = try Temporary.directory("uncover-rotation")
        let source = try VideoFixture.makeMP4(in: directory, durationSeconds: 1, width: 128, height: 96, frameRate: 24)
        let rotated = directory.appendingPathComponent("rotated.mov")
        try await VideoFixture.rotate(source, to: rotated)
        let prepared = try await TemporalFrameExtractor().prepare(source: rotated, roi: .initial, scale: .two,
            start: 0, end: 0.5, directory: directory.appendingPathComponent("job"), progress: { _ in })
        checkEqual(prepared.manifest.width, 96, "oriented width")
        checkEqual(prepared.manifest.height, 128, "oriented height")
        checkEqual(prepared.manifest.regionX, 24, "oriented x")
        checkEqual(prepared.manifest.regionY, 32, "oriented y")
    })
    Registry.shared.add(suite: suite, TestCase(name: "comparison keeps original pixels and renders AI without a Lanczos baseline", knownBug: nil) {
        let directory = try Temporary.directory("uncover-original-comparison")
        let source = try VideoFixture.makeMP4(in: directory, durationSeconds: 1, width: 128, height: 96, frameRate: 24)
        let prepared = try await TemporalFrameExtractor().prepare(source: source, roi: .initial, scale: .four,
            start: 0, end: 0.25, directory: directory.appendingPathComponent("job"), progress: { _ in })
        let manifest = prepared.manifest
        let ai = try ImageFixture.cgImage(width: manifest.regionWidth, height: manifest.regionHeight, gray: 0.9)
        try RegionImages.write(ai, to: RegionImages.frameURL(prepared.directory, "reconstructed", 0))
        try FileManager.default.removeItem(at: prepared.directory.appendingPathComponent("baseline"))
        let result = UncoverResult(prepared: prepared, model: "TEST ONLY", seconds: 0)
        let original = try RegionImages.read(RegionImages.frameURL(prepared.directory, "original", 0))
        let renderer = UncoverRenderer()
        let rendered = try await renderer.render(result, index: 0, feather: 0, strength: 1)
        checkEqual(rgba(rendered.original), rgba(original), "reference is the untouched original")
        check(rgba(rendered.reconstructed) != rgba(original), "AI output is composited into the selected region")
        let blended = try await renderer.render(result, index: 0, feather: 8, strength: 0)
        checkEqual(rgba(blended.original), rgba(original), "blend controls never alter the reference")
        checkEqual(rgba(blended.reconstructed), rgba(original), "zero strength restores source pixels")
    })
    Registry.shared.add(suite: suite, TestCase(name: "interval export preserves every source PTS and matching audio", knownBug: nil) {
        let directory = try Temporary.directory("uncover-export")
        let source = directory.appendingPathComponent("audio-source.mov")
        let video = directory.appendingPathComponent("vfr-video.mp4")
        var times: [Double] = []
        for index in 0..<24 where index % 3 == 0 || index % 5 == 0 {
            times.append(Double(index) / 24.0)
        }
        try VideoFixture.writeVideo(to: video, duration: 1, width: 128, height: 96, frameRate: 24, timestamps: times)
        try await VideoFixture.addAudio(to: video, output: source, duration: 1)
        let prepared = try await TemporalFrameExtractor().prepare(source: source, roi: .initial, scale: .two,
            start: 0.01, end: 0.5, directory: directory.appendingPathComponent("job"), progress: { _ in })
        let result = try await UncoverTestBackend().reconstruct(prepared, allowUnsafeMemory: false, progress: { _ in })
        let output = directory.appendingPathComponent("export.mov")
        try await UncoverVideoExporter().export(result, feather: 8, strength: 1, to: output, progress: { _ in })
        let asset = AVURLAsset(url: output)
        checkEqual(try await asset.loadTracks(withMediaType: .audio).count, 1, "audio retained")
        let timestamps = try await videoTimes(output)
        checkEqual(timestamps.count, prepared.manifest.frames.count, "no missing or duplicate frames")
        for (actual, expected) in zip(timestamps, prepared.manifest.frames) {
            checkClose(actual, expected.timestamp.seconds - prepared.manifest.frames[0].timestamp.seconds,
                       tolerance: 0.0001, "source-relative PTS")
        }
        _ = await checkThrowsAsync("cannot overwrite the source", {
            try await UncoverVideoExporter().export(result, feather: 8, strength: 1, to: source, progress: { _ in })
        })
    })
    Registry.shared.add(suite: suite, TestCase(name: "cancellation terminates the inference process promptly", knownBug: nil) {
        let directory = try Temporary.directory("uncover-cancel")
        let process = UncoverProcess()
        let task = Task { try await process.run(executable: "/bin/sleep", arguments: ["30"], log: directory.appendingPathComponent("log")) }
        try await Task.sleep(nanoseconds: 100_000_000)
        let started = Date()
        task.cancel()
        do { try await task.value; check(false, "cancelled process returned success") }
        catch is CancellationError { }
        check(Date().timeIntervalSince(started) < 3, "cancellation bounded")
    })
    Registry.shared.add(suite: suite, TestCase(name: "display changes reuse inference and failed replacements retain the last success", knownBug: nil) {
        let directory = try Temporary.directory("uncover-session")
        let source = try VideoFixture.makeMP4(in: directory, durationSeconds: 1, width: 128, height: 96, frameRate: 24)
        let video = try await VideoProbe.probe(source)
        let backend = UncoverCountingBackend()
        let session = await UncoverSession(video: video, seconds: 0, backend: backend)
        try await waitUntil { await MainActor.run { session.original != nil && session.capabilities != nil } }
        await MainActor.run { session.duration = 0.5; session.showOriginal = true; session.reconstruct() }
        try await waitUntil { await MainActor.run { !session.busy } }
        check(await session.hasCurrentResult, "first result ready")
        check(!(await session.showOriginal), "completion reveals original versus AI comparison")
        check(await MainActor.run { session.imageA != nil && session.imageA === session.original }, "comparison reference shares the untouched source image")
        let savedROI = await session.roi
        let successful = await session.result?.directory
        await MainActor.run {
            session.feather = 12; session.strength = 0.5; session.comparison = .sideBySide
            session.zoom = 2; session.showOriginal = true; session.seek(index: 3)
        }
        try await Task.sleep(nanoseconds: 150_000_000)
        checkEqual(await backend.count, 1, "display adjustments do not infer")
        await backend.setFailure(true)
        await MainActor.run { session.roi = .initial; session.reconstruct() }
        try await waitUntil { await MainActor.run { !session.busy } }
        checkEqual(await session.result?.directory, successful, "previous success retained")
        check(!(await session.hasCurrentResult), "changed ROI cannot export previous result")
        await MainActor.run { session.roi = savedROI }
        check(await session.hasCurrentResult, "returning to identical inputs reuses success")
        await session.dispose()
    })
    Registry.shared.add(suite: suite, TestCase(name: "consecutive extraction keeps source timestamps and disk-backed geometry", knownBug: nil) {
        let directory = try Temporary.directory("uncover-extraction")
        let source = try VideoFixture.makeMP4(in: directory, durationSeconds: 1, width: 128, height: 96, frameRate: 24)
        let prepared = try await TemporalFrameExtractor().prepare(source: source,
            roi: UncoverROI(x: 0.25, y: 0.25, width: 0.5, height: 0.5), scale: .two,
            start: 0, end: 0.5, directory: directory.appendingPathComponent("job"), progress: { _ in })
        checkEqual(prepared.manifest.frames.count, 12, "consecutive frame count")
        checkEqual(prepared.manifest.regionWidth, 64, "ROI width")
        checkEqual(prepared.manifest.regionHeight, 48, "ROI height")
        checkEqual(try RegionImages.read(RegionImages.frameURL(prepared.directory, "degraded", 0)).width, 32, "degraded width")
        for (index, frame) in prepared.manifest.frames.enumerated() {
            checkClose(frame.timestamp.seconds, Double(index) / 24, tolerance: 0.0001, "source timestamp")
        }
        let fractional = try await TemporalFrameExtractor().prepare(source: source,
            roi: .initial, scale: .two, start: 0.01, end: 0.5,
            directory: directory.appendingPathComponent("fractional"), progress: { _ in })
        checkEqual(fractional.manifest.frames.count, 11, "partial opening frame excluded")
        checkClose(fractional.manifest.frames[0].timestamp.seconds, 1.0 / 24, tolerance: 0.0001, "actual first PTS")
    })
    Registry.shared.add(suite: suite, TestCase(name: "large selections stay enabled at every scale and retain native output geometry", knownBug: nil) {
        let directory = try Temporary.directory("uncover-large-region")
        let source = try VideoFixture.makeMP4(in: directory, durationSeconds: 1, width: 768, height: 640, frameRate: 24)
        let video = try await VideoProbe.probe(source)
        let session = await UncoverSession(video: video, seconds: 0, backend: UncoverTestBackend())
        try await waitUntil { await MainActor.run { session.original != nil && session.capabilities != nil } }
        await MainActor.run {
            session.roi = .from(pixels: CGRect(x: 100, y: 50, width: 528, height: 537), in: session.videoSize)
            session.duration = 0.25
        }
        let scale64 = try await TemporalFrameExtractor().prepare(source: source,
            roi: .from(pixels: CGRect(x: 100, y: 50, width: 528, height: 537), in: CGSize(width: 768, height: 640)),
            scale: .sixtyFour, start: 0, end: 0.25,
            directory: directory.appendingPathComponent("scale-64"), progress: { _ in })
        let smallestInput = try RegionImages.read(RegionImages.frameURL(scale64.directory, "degraded", 0))
        let baseline = try RegionImages.read(RegionImages.frameURL(scale64.directory, "baseline", 0))
        checkEqual(smallestInput.width, 9, "64× model input width")
        checkEqual(smallestInput.height, 9, "64× model input height rounds up")
        checkEqual(baseline.width, 528, "64× baseline returns to ROI width")
        checkEqual(baseline.height, 537, "64× baseline returns to ROI height")
        for scale in UncoverScale.allCases {
            await MainActor.run { session.scale = scale }
            check(await session.canReconstruct, "large region enabled at \(scale.rawValue)×")
            check(await session.limitation == nil, "no size restriction at \(scale.rawValue)×")
        }
        await MainActor.run { session.scale = .eight }
        await session.reconstruct()
        try await waitUntil { await MainActor.run { !session.busy } }
        check(await session.hasCurrentResult, "large result ready")
        guard let result = await session.result else { throw UncoverError.invalid("Large ROI reconstruction did not complete.") }
        let input = try RegionImages.read(RegionImages.frameURL(result.directory, "degraded", 0))
        let output = try RegionImages.read(RegionImages.frameURL(result.directory, "reconstructed", 0))
        checkEqual(input.width, 66, "8× degraded width")
        checkEqual(input.height, 68, "8× degraded height rounds up")
        checkEqual(output.width, 528, "native ROI width restored")
        checkEqual(output.height, 537, "native ROI height restored")
        await session.dispose()
    })
    Registry.shared.add(suite: suite, TestCase(name: "spatial output limit disables oversized regions and allows smaller selections", knownBug: nil) {
        let directory = try Temporary.directory("uncover-output-limit")
        let source = try VideoFixture.makeMP4(in: directory, durationSeconds: 1, width: 128, height: 96, frameRate: 24)
        let video = try await VideoProbe.probe(source)
        let session = await UncoverSession(video: video, seconds: 0, backend: UncoverOutputLimitedBackend())
        try await waitUntil { await MainActor.run { session.original != nil && session.capabilities != nil } }
        await MainActor.run {
            session.roi = .from(pixels: CGRect(x: 20, y: 20, width: 10, height: 10), in: session.videoSize)
            session.duration = 0.25
        }
        check(await session.canReconstruct, "region within the output limit is enabled")
        await MainActor.run {
            session.roi = .from(pixels: CGRect(x: 20, y: 20, width: 11, height: 10), in: session.videoSize)
        }
        check(!(await session.canReconstruct), "region above the output limit is disabled")
        check((await session.limitation)?.contains("output limit") == true, "the size restriction is explained")
        await session.dispose()
    })
    Registry.shared.add(suite: suite, TestCase(name: "algorithm switching reuses preparation and cached results without changing the experiment", knownBug: nil) {
        let directory = try Temporary.directory("uncover-algorithms")
        let source = try VideoFixture.makeMP4(in: directory, durationSeconds: 1, width: 128, height: 96, frameRate: 24)
        let video = try await VideoProbe.probe(source)
        let seed = UncoverCountingBackend(), flash = UncoverCountingBackend()
        let session = await UncoverSession(video: video, seconds: 0, backendFactory: { $0 == .seedVR2 ? seed : flash })
        try await waitUntil { await MainActor.run { session.original != nil && session.capabilities != nil } }
        checkEqual(await session.algorithm, .seedVR2, "SeedVR2 remains the default")
        await MainActor.run {
            session.scale = .four; session.duration = 0.5; session.zoom = 2
            session.feather = 12; session.strength = 0.6; session.comparison = .sideBySide; session.reconstruct()
        }
        try await waitUntil { await MainActor.run { !session.busy } }
        guard let originalResult = await session.result else { throw UncoverError.invalid("Missing seed test result") }
        let sharedInput = RegionImages.frameURL(originalResult.directory, "degraded", 0).resolvingSymlinksInPath()
        let roi = await session.roi, start = await session.start
        await MainActor.run { session.algorithm = .flashVSR11 }
        try await waitUntil { await MainActor.run { session.capabilities != nil } }
        check(!(await session.hasCurrentResult), "different algorithm cannot export the SeedVR2 result")
        await session.reconstruct()
        try await waitUntil { await MainActor.run { !session.busy } }
        guard let flashResult = await session.result else { throw UncoverError.invalid("Missing flash test result") }
        check(originalResult.directory != flashResult.directory, "output ownership is separate")
        checkEqual(RegionImages.frameURL(flashResult.directory, "degraded", 0).resolvingSymlinksInPath(), sharedInput, "no re-decode or re-downsample")
        await MainActor.run { session.algorithm = .seedVR2 }
        check(await session.hasCurrentResult, "cached seed result restored")
        checkEqual(await session.result?.directory, originalResult.directory, "same cached output")
        await MainActor.run { session.algorithm = .flashVSR11 }
        check(await session.hasCurrentResult, "cached flash result restored")
        checkEqual(await seed.count, 1, "SeedVR2 inferred once")
        checkEqual(await flash.count, 1, "FlashVSR inferred once")
        checkEqual(await session.roi, roi, "ROI preserved")
        checkEqual(await session.start, start, "interval preserved")
        checkEqual(await session.scale, .four, "downsampling preserved")
        checkEqual(await session.zoom, 2, "zoom preserved")
        checkEqual(await session.comparison, .sideBySide, "layout preserved")
        checkEqual(await session.feather, 12, "feather preserved")
        checkEqual(await session.strength, 0.6, "blend preserved")
        await session.dispose()
        check(!FileManager.default.fileExists(atPath: sharedInput.path), "shared prepared inputs released")
        check(!FileManager.default.fileExists(atPath: originalResult.directory.path), "seed result released")
        check(!FileManager.default.fileExists(atPath: flashResult.directory.path), "flash result released")
    })
    Registry.shared.add(suite: suite, TestCase(name: "FlashVSR accepts source reduction factors with fixed 4x output", knownBug: nil) {
        let backend = FlashVSRBackend(configuration: nil)
        let capabilities = await backend.capabilities()
        check(!capabilities.available, "missing weights do not enable inference")
        checkEqual(capabilities.scales, UncoverScale.allCases, "source downsampling stays adjustable")
        checkEqual(FlashVSRBackend.outputScale, 4, "native output remains fixed at four times source dimensions")
        check(!capabilities.requiresUnsafeMemoryOptIn, "native inference uses its own budget")
        check(capabilities.message.contains("Settings → Models"), "actionable installation instruction")
    })
    Registry.shared.add(suite: suite, TestCase(name: "algorithm changes cancel active inference and preserve unsupported degradation", knownBug: nil) {
        let directory = try Temporary.directory("uncover-switch-active")
        let source = try VideoFixture.makeMP4(in: directory, durationSeconds: 1, width: 128, height: 96, frameRate: 24)
        let video = try await VideoProbe.probe(source)
        let slow = UncoverDelayedBackend()
        let session = await UncoverSession(video: video, seconds: 0, backendFactory: {
            if $0 == .seedVR2 { return slow }
            return UncoverFourOnlyBackend()
        })
        try await waitUntil { await MainActor.run { session.original != nil && session.capabilities != nil } }
        await MainActor.run { session.duration = 0.5; session.reconstruct() }
        try await waitUntil { await slow.started }
        await MainActor.run { session.algorithm = .flashVSR11 }
        try await waitUntil { await MainActor.run { !session.busy && session.capabilities != nil } }
        check(await slow.cancelled, "inactive engine was cancelled")
        check(!(await session.hasCurrentResult), "inactive result never published")
        checkEqual(await session.scale, .two, "unsupported experiment is preserved")
        check(!(await session.canReconstruct), "unsupported factor cannot run")
        check(await session.limitation?.contains("4×") == true, "limitation explains supported factor")
        await MainActor.run { session.scale = .four }
        check(await session.canReconstruct, "supported experiment enabled")
        await session.dispose()
    })
    Registry.shared.add(suite: suite, TestCase(name: "SeedVR2 size selection requires optimized native weights", knownBug: nil) {
        let directory = try Temporary.directory("optimized-model-selection")
        func model(_ name: String, _ variant: String, fused: Bool = true) throws -> String {
            let path = directory.appendingPathComponent(name)
            try FileManager.default.createDirectory(at: path, withIntermediateDirectories: true)
            var manifest: [String: Any] = ["algorithm": variant, "precision": "mlx-4bit", "groupSize": 64]
            if fused { manifest["fusion"] = "packed-projections-v1"; manifest["vaeFusion"] = "qkv-packed-v1" }
            try JSONSerialization.data(withJSONObject: manifest).write(to: path.appendingPathComponent("manifest.json"))
            return path.path
        }
        let small = try model("custom-small", "seedvr2-3b")
        let large = try model("custom-large", "seedvr2-7b")
        var config = SeedVR2Configuration(python: "/usr/bin/python3", model: large, helper: "legacy.py", allowUnsafeMemory: true,
            nativeExecutable: "/usr/bin/true", optimizedModels: ["seedvr2-3b": small, "seedvr2-7b": large])
        let three = SeedVR2Backend(configuration: config)
        let seven = SeedVR2Backend(configuration: config, variant: "seedvr2-7b")
        checkEqual(three.configuration?.model, small, "3B resolves independently of last configured size")
        checkEqual(seven.configuration?.model, large, "7B resolves its own prepared weights")
        checkEqual(seven.configuration?.nativeEmbedding, large + "/pos_emb.safetensors", "embedding belongs to selected bundle")
        check(three.cacheIdentity != seven.cacheIdentity, "model sizes have distinct result identities")
        config.optimizedModels = ["seedvr2-3b": try model("original-small", "seedvr2-3b", fused: false)]
        check(config.optimized(variant: "seedvr2-3b") == nil, "original weights never substitute for optimized 3B")
        config.nativeExecutable = nil
        check(config.optimized(variant: "seedvr2-7b") == nil, "Python configuration cannot substitute for native inference")
        let absent = await SeedVR2Backend(configuration: nil, variant: "seedvr2-7b").capabilities()
        check(!absent.available && absent.message.contains("SeedVR2 7B") && absent.message.contains("Settings → Models"), "missing 7B points to its in-app setup")
        check(!absent.requiresUnsafeMemoryOptIn, "native models use the bounded worker")
    })
    Registry.shared.add(suite: suite, TestCase(name: "3B and 7B reuse inputs but retain separate cached results", knownBug: nil) {
        let directory = try Temporary.directory("uncover-model-sizes")
        let source = try VideoFixture.makeMP4(in: directory, durationSeconds: 1, width: 128, height: 96, frameRate: 24)
        let video = try await VideoProbe.probe(source)
        let small = UncoverCountingBackend(), large = UncoverCountingBackend()
        let session = await UncoverSession(video: video, seconds: 0, backendFactory: { $0 == .seedVR2 ? small : large })
        try await waitUntil { await MainActor.run { session.original != nil && session.capabilities != nil } }
        await MainActor.run { session.duration = 0.5; session.reconstruct() }
        try await waitUntil { await MainActor.run { !session.busy && session.hasCurrentResult } }
        guard let first = await session.result else { throw UncoverError.invalid("Missing 3B result") }
        let input = RegionImages.frameURL(first.directory, "degraded", 0).resolvingSymlinksInPath()
        await MainActor.run { session.algorithm = .seedVR2SevenB }
        try await waitUntil { await MainActor.run { session.capabilities != nil } }
        check(!(await session.hasCurrentResult), "3B output cannot be exported as 7B")
        await session.reconstruct()
        try await waitUntil { await MainActor.run { !session.busy && session.hasCurrentResult } }
        guard let second = await session.result else { throw UncoverError.invalid("Missing 7B result") }
        check(first.directory != second.directory, "separate output ownership")
        checkEqual(RegionImages.frameURL(second.directory, "degraded", 0).resolvingSymlinksInPath(), input, "size change does not repeat frame preparation")
        await MainActor.run { session.algorithm = .seedVR2 }
        checkEqual(await session.result?.directory, first.directory, "3B cache restored")
        await MainActor.run { session.algorithm = .seedVR2SevenB }
        checkEqual(await session.result?.directory, second.directory, "7B cache restored")
        checkEqual(await small.count, 1, "3B inferred once")
        checkEqual(await large.count, 1, "7B inferred once")
        await session.dispose()
    })
    Registry.shared.add(suite: "Uncover real inference", TestCase(name: "real native temporal video proof", knownBug: nil) {
        guard ProcessInfo.processInfo.environment["WAIFU_TEST_SEEDVR2"] == "1" else {
            throw SkipTest(reason: "opt in with WAIFU_TEST_SEEDVR2=1; downloads are separate")
        }
        let algorithm: UncoverAlgorithm = ProcessInfo.processInfo.environment["WAIFU_TEST_ALGORITHM"] == "flashvsr" ? .flashVSR11
            : (ProcessInfo.processInfo.environment["WAIFU_TEST_ALGORITHM"] == "seedvr2-7b" ? .seedVR2SevenB : .seedVR2)
        let config = SeedVR2Configuration.load()
        let backend = algorithm.makeBackend()
        guard await backend.capabilities().available else { throw UncoverError.backend("Install the selected native backend first.") }
        let directory = try Temporary.directory("seedvr2-proof")
        guard let sourcePath = ProcessInfo.processInfo.environment["WAIFU_TEST_VIDEO"] else {
            throw SkipTest(reason: "set WAIFU_TEST_VIDEO to a local video fixture")
        }
        let source = URL(fileURLWithPath: sourcePath)
        let scale = UncoverScale(rawValue: Int(ProcessInfo.processInfo.environment["WAIFU_TEST_SCALE"] ?? (algorithm == .flashVSR11 ? "4" : "2")) ?? 2) ?? .two
        let seconds = Double(ProcessInfo.processInfo.environment["WAIFU_TEST_SECONDS"] ?? "0.15") ?? 0.15
        let roiWidth = Double(ProcessInfo.processInfo.environment["WAIFU_TEST_ROI_WIDTH"] ?? "256") ?? 256
        let roiHeight = Double(ProcessInfo.processInfo.environment["WAIFU_TEST_ROI_HEIGHT"] ?? "256") ?? 256
        let prepared = try await TemporalFrameExtractor().prepare(source: source,
            roi: UncoverROI(x: 832.0 / 1920, y: 412.0 / 1080, width: roiWidth / 1920, height: roiHeight / 1080),
            scale: scale, start: 10, end: 10 + seconds,
            directory: directory.appendingPathComponent("job"), progress: { print($0) })
        let result = try await backend.reconstruct(prepared,
            allowUnsafeMemory: config?.allowUnsafeMemory ?? false, progress: { print($0) })
        if seconds == 0.15 { checkEqual(result.prepared.manifest.frames.count, 5, "proof frame count") }
        let exported = directory.appendingPathComponent("uncovered.mov")
        try await UncoverVideoExporter().export(result, feather: 8, strength: 1, to: exported, progress: { print($0) })
        checkEqual(try await videoTimes(exported).count, prepared.manifest.frames.count, "real exported frame count")
        let sourceAudio = try await AVURLAsset(url: source).loadTracks(withMediaType: .audio)
        let outputAudio = try await AVURLAsset(url: exported).loadTracks(withMediaType: .audio)
        checkEqual(outputAudio.count, sourceAudio.isEmpty ? 0 : 1, "real export preserves matching audio")
        for (actual, frame) in zip(try await videoTimes(exported), prepared.manifest.frames) {
            checkClose(actual, frame.timestamp.seconds - prepared.manifest.frames[0].timestamp.seconds,
                       tolerance: 0.00004, "real source-relative PTS")
        }
        let report = try Data(contentsOf: result.directory.appendingPathComponent("report.json"))
        if algorithm == .flashVSR11 || config?.nativeExecutable != nil {
            let metadata = try JSONSerialization.jsonObject(with: report) as? [String: Any]
            checkEqual(metadata?["backend"] as? String, algorithm == .flashVSR11 ? "Native Swift MLX FlashVSR 1.1" : "Native Swift MLX", "configured native backend runs")
            checkEqual(metadata?["pythonRuntime"] as? Bool, false, "no Python inference")
            if algorithm != .flashVSR11 {
                checkEqual(metadata?["modelVariant"] as? String,
                           algorithm == .seedVR2SevenB ? "seedvr2-7b" : "seedvr2-3b", "selected model size runs")
                checkEqual(metadata?["fusedVAEAttention"] as? Bool, true, "optimized VAE always runs")
                checkEqual(metadata?["fusedVAENormalization"] as? Bool, true, "native Metal normalization/activation always runs")
                checkEqual(metadata?["fusedVAECausalPreparation"] as? Bool, true, "native causal preparation enabled")
                checkEqual(metadata?["vaePointwiseMatmul"] as? Bool, true, "native pointwise convolution optimization enabled")
                checkEqual(metadata?["fusedVAEBiasResidual"] as? Bool, true, "native VAE bias/residual fusion enabled")
                checkEqual(metadata?["fusedMLP"] as? Bool, algorithm == .seedVR2, "selected architecture's fusion layout")
            }
            if let peak = metadata?["peakPhysicalFootprintBytes"] as? NSNumber,
               let budget = metadata?["memoryBudgetBytes"] as? NSNumber {
                check(peak.doubleValue <= budget.doubleValue, "native physical footprint stays inside budget")
            } else { check(false, "native memory evidence recorded") }
        }
        print(String(decoding: report, as: UTF8.self))
        // Preserve the proof independently of the test harness's temp-directory cleanup.
        if let output = ProcessInfo.processInfo.environment["WAIFU_PROOF_OUTPUT"] {
            try FileManager.default.copyItem(at: result.directory, to: URL(fileURLWithPath: output))
            try FileManager.default.copyItem(at: exported, to: URL(fileURLWithPath: output).appendingPathComponent("uncovered.mov"))
        }
    })
    Registry.shared.add(suite: "Uncover PGTFormer product", TestCase(name: "real face video selects native PGTFormer, previews and exports with audio", knownBug: nil) {
        guard let path = ProcessInfo.processInfo.environment["WAIFU_TEST_PGT_VIDEO"] else { throw SkipTest(reason: "set WAIFU_TEST_PGT_VIDEO to a real face video with audio") }
        let source = URL(fileURLWithPath:path), video = try await VideoProbe.probe(source)
        let session = await UncoverSession(video:video,seconds:0)
        await MainActor.run { session.algorithm = .pgtFormer; session.scale = .one; session.roi = UncoverROI(x:0.125,y:0.125,width:0.75,height:0.75); session.duration = 0.08 }
        try await waitUntil { await MainActor.run { session.original != nil && session.capabilities?.available == true && !session.readingSource } }
        await MainActor.run { session.roi = UncoverROI(x:0.125,y:0.125,width:0.75,height:0.75) }
        // Two source frames exercise replicated boundaries below model context.
        for (scale,duration) in [(UncoverScale.one,0.08),(UncoverScale.two,0.32)] {
            await MainActor.run { session.scale = scale; session.duration = duration; session.reconstruct() }
            try await waitUntil(timeout:180) { await MainActor.run { !session.busy } }
            guard let result = await session.result, await session.hasCurrentResult else { throw UncoverError.backend(await session.error ?? "PGTFormer failed") }
            check(await MainActor.run { session.imageB != nil }, "native result appeared in Uncover preview")
            let report = try JSONSerialization.jsonObject(with:Data(contentsOf:result.directory.appendingPathComponent("report.json"))) as! [String:Any]
            checkEqual(report["model"] as? String,"pgtformer","selected engine executed its own pretrained model")
            checkEqual(report["pythonRuntime"] as? Bool,false,"no Python runtime")
            checkEqual(report["encoderUnits"] as? String,"gpu","the product no longer forces the encoder onto the CPU")
            checkEqual(report["decoderUnits"] as? String,"gpu","both neural stages allow GPU execution")
            let windows = report["temporalWindows"] as! [[Int]], count = result.prepared.manifest.frames.count
            checkEqual(windows.first,[0,0,min(1,count-1)],"first boundary repeats first frame")
            checkEqual(windows.last,[max(0,count-2),count-1,count-1],"last boundary repeats last frame")
            let output = result.directory.appendingPathComponent("product-export.mov")
            await session.export(to:output)
            try await waitUntil(timeout:30) { await MainActor.run { !session.busy } }
            check(await session.error == nil,"export through product session succeeded")
            let times = try await videoTimes(output)
            checkEqual(times.count,count,"all source frames exported")
            let asset = AVURLAsset(url:output), track = try await asset.loadTracks(withMediaType:.video)[0]
            checkEqual(try await track.load(.naturalSize),CGSize(width:128,height:128),"source resolution preserved")
            checkEqual(try await asset.loadTracks(withMediaType:.audio).count,1,"audio preserved")
            for (time,frame) in zip(times,result.prepared.manifest.frames) { check(abs(time-(frame.timestamp.seconds-result.prepared.manifest.frames[0].timestamp.seconds)) < 0.0001,"source PTS preserved") }
            if let destination = ProcessInfo.processInfo.environment["WAIFU_PGT_PROOF_OUTPUT"] {
                let root = URL(fileURLWithPath:destination).appendingPathComponent("scale-\(scale.rawValue)")
                try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
                for name in ["original","degraded","baseline","reconstructed"] {
                    try FileManager.default.createDirectory(at:root.appendingPathComponent(name),withIntermediateDirectories:true)
                    for frame in result.prepared.manifest.frames {
                        try RegionImages.write(RegionImages.read(RegionImages.frameURL(result.directory,name,frame.index)),to:RegionImages.frameURL(root,name,frame.index))
                    }
                }
                for file in ["manifest.json","report.json","product-export.mov"] { try FileManager.default.copyItem(at:result.directory.appendingPathComponent(file),to:root.appendingPathComponent(file)) }
                let validation = result.directory.appendingPathComponent("validation-tensors")
                if FileManager.default.fileExists(atPath:validation.path) { try FileManager.default.copyItem(at:validation,to:root.appendingPathComponent("validation-tensors")) }
                let preview = try await UncoverRenderer().render(result,index:0,feather:8,strength:1)
                try RegionImages.write(preview.reconstructed,to:root.appendingPathComponent("product-preview.png"))
            }
            // Model switching returns the same usable cached product result.
            let directory = result.directory
            await MainActor.run { session.algorithm = .seedVR2; session.algorithm = .pgtFormer }
            try await waitUntil { await MainActor.run { session.hasCurrentResult } }
            checkEqual(await session.result?.directory,directory,"repeat selection reuses completed video")
        }
        await MainActor.run { session.duration = 0.8; session.reconstruct() }
        try await waitUntil { await session.busy }
        try await Task.sleep(nanoseconds:500_000_000)
        await session.cancel()
        try await waitUntil(timeout:10) { await MainActor.run { !session.busy } }
        check(await session.error == nil,"native PGTFormer cancellation is clean")
        await session.dispose()
    })
    Registry.shared.add(suite: "Uncover RealESRGAN product", TestCase(name: "optimized native restoration preserves 1x input, preview, audio export and cancellation", knownBug: nil) {
        guard let path = ProcessInfo.processInfo.environment["WAIFU_TEST_REALESRGAN_VIDEO"] else { throw SkipTest(reason:"set WAIFU_TEST_REALESRGAN_VIDEO to a real video with audio") }
        let video = try await VideoProbe.probe(URL(fileURLWithPath:path))
        let session = await UncoverSession(video:video,seconds:0)
        await MainActor.run { session.algorithm = .realESRGAN;session.scale = .one;session.roi = UncoverROI(x:0.125,y:0.125,width:0.75,height:0.75);session.duration = 0.08 }
        try await waitUntil { await MainActor.run { session.original != nil && session.capabilities?.available == true && !session.readingSource } }
        await MainActor.run { session.roi = UncoverROI(x:0.125,y:0.125,width:0.75,height:0.75) }
        for (scale,duration) in [(UncoverScale.one,0.08),(UncoverScale.two,0.32)] {
            await MainActor.run { session.scale = scale;session.duration = duration;session.reconstruct() }
            try await waitUntil(timeout:60) { await MainActor.run { !session.busy } }
            guard let result = await session.result,await session.hasCurrentResult else { throw UncoverError.backend(await session.error ?? "RealESRGAN failed") }
            check(await MainActor.run { session.imageB != nil },"RealESRGAN result appears in preview")
            let report = try JSONSerialization.jsonObject(with:Data(contentsOf:result.directory.appendingPathComponent("report.json"))) as! [String:Any]
            checkEqual(report["model"] as? String,"realesrgan-x2plus","selected network owns restoration")
            checkEqual(report["pythonRuntime"] as? Bool,false,"native inference has no Python")
            checkEqual(report["sourceDownsampling"] as? Int,scale.rawValue,"source degradation is independent of tensor padding")
            check((report["predictionSizes"] as? [Int])?.allSatisfy { $0 == 256 } == true,"small ROI uses the validated adaptive input size")
            if scale == .one {
                for frame in result.prepared.manifest.frames {
                    let original = try RegionImages.read(RegionImages.frameURL(result.directory,"original",frame.index))
                    let degraded = try RegionImages.read(RegionImages.frameURL(result.directory,"degraded",frame.index))
                    let manifest = result.prepared.manifest
                    guard let crop = original.cropping(to:CGRect(x:manifest.regionX,y:manifest.regionY,width:manifest.regionWidth,height:manifest.regionHeight)) else { throw UncoverError.invalid("ROI crop failed") }
                    checkEqual(crop.width,degraded.width,"1x has no source reduction")
                    checkEqual(crop.height,degraded.height,"1x retains source height")
                    checkEqual(rgba(crop),rgba(degraded),"1x preserves source pixels")
                }
            }
            let output = result.directory.appendingPathComponent("realesrgan-export.mov")
            await session.export(to:output);try await waitUntil(timeout:30) { await MainActor.run { !session.busy } }
            check(await session.error == nil,"RealESRGAN exports successfully")
            let times = try await videoTimes(output),asset = AVURLAsset(url:output),track = try await asset.loadTracks(withMediaType:.video)[0]
            checkEqual(times.count,result.prepared.manifest.frames.count,"all restored frames exported")
            checkEqual(try await track.load(.naturalSize),CGSize(width:128,height:128),"original video size retained")
            checkEqual(try await asset.loadTracks(withMediaType:.audio).count,1,"audio retained")
            for (time,frame) in zip(times,result.prepared.manifest.frames) { check(abs(time-(frame.timestamp.seconds-result.prepared.manifest.frames[0].timestamp.seconds))<0.0001,"original PTS retained") }
            if let destination = ProcessInfo.processInfo.environment["WAIFU_REAL_PROOF_OUTPUT"] {
                let directory = URL(fileURLWithPath:destination).appendingPathComponent("scale-\(scale.rawValue)")
                try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
                for name in ["original","degraded","reconstructed"] {
                    try FileManager.default.createDirectory(at:directory.appendingPathComponent(name),withIntermediateDirectories:true)
                    for frame in result.prepared.manifest.frames {
                        try RegionImages.write(RegionImages.read(RegionImages.frameURL(result.directory,name,frame.index)),to:RegionImages.frameURL(directory,name,frame.index))
                    }
                }
                for name in ["manifest.json","report.json","realesrgan-export.mov"] { try FileManager.default.copyItem(at:result.directory.appendingPathComponent(name),to:directory.appendingPathComponent(name)) }
            }
        }
        await MainActor.run { session.duration = 0.8;session.reconstruct() }
        try await waitUntil { await session.busy };try await Task.sleep(nanoseconds:100_000_000);await session.cancel()
        try await waitUntil(timeout:10) { await MainActor.run { !session.busy } }
        check(await session.error == nil,"native Core ML cancellation is clean")
        await session.dispose()
    })
    Registry.shared.add(suite: "Uncover real 1x inference", TestCase(name: "native 3B and 7B restore original ROI without enlargement", knownBug: nil) {
        guard ProcessInfo.processInfo.environment["WAIFU_TEST_NATIVE_1X"] == "1" else {
            throw SkipTest(reason: "opt in with WAIFU_TEST_NATIVE_1X=1; install native 3B and 7B first")
        }
        let directory = try Temporary.directory("native-1x")
        let source = try VideoFixture.makeMP4(in: directory, durationSeconds: 1, width: 128, height: 96, frameRate: 24)
        for algorithm in [UncoverAlgorithm.seedVR2, .seedVR2SevenB] {
            guard let config = SeedVR2Configuration.load() else { throw UncoverError.backend("Install native SeedVR2 first") }
            let backend = SeedVR2Backend(configuration: config, variant: algorithm == .seedVR2 ? "seedvr2-3b" : "seedvr2-7b")
            guard await backend.capabilities().available else { throw UncoverError.backend("Install \(algorithm.label) first") }
            checkEqual(await backend.capabilities().scales, UncoverScale.allCases, "SeedVR2 supports every output scale without RealESRGAN")
            let prepared = try await TemporalFrameExtractor().prepare(source: source, roi: .initial, scale: .one,
                start: 0, end: 0.21, directory: directory.appendingPathComponent(algorithm.rawValue), progress: { _ in })
            let result = try await backend.reconstruct(prepared, allowUnsafeMemory: false, progress: { print($0) })
            let reportData = try Data(contentsOf: result.directory.appendingPathComponent("report.json"))
            let report = try JSONSerialization.jsonObject(with: reportData) as? [String: Any]
            let spatial = try JSONSerialization.jsonObject(with: Data(contentsOf: result.directory.appendingPathComponent("spatial-upscale-report.json"))) as? [String: Any]
            checkEqual(spatial?["stagesPerFrame"] as? Int, 0, "no hidden learned enlargement")
            checkEqual(spatial?["spatialModel"] as? String, "SeedVR2 native", "SeedVR2 owns spatial reconstruction")
            checkEqual(report?["scale"] as? Int, 1, "native worker accepted 1×")
            checkEqual(report?["pythonRuntime"] as? Bool, false, "Swift MLX inference")
            for frame in prepared.manifest.frames {
                let restored = try RegionImages.read(RegionImages.frameURL(result.directory, "reconstructed", frame.index))
                checkEqual(restored.width, 64, "native original ROI width")
                checkEqual(restored.height, 48, "native original ROI height")
            }
            let destination = directory.appendingPathComponent("\(algorithm.rawValue).mov")
            try await UncoverVideoExporter().export(result, feather: 8, strength: 1, to: destination, progress: { _ in })
            checkEqual(try await videoTimes(destination).count, prepared.manifest.frames.count, "real restored export frame count")
            let track = try await AVURLAsset(url: destination).loadTracks(withMediaType: .video)[0]
            checkEqual(try await track.load(.naturalSize), CGSize(width: 128, height: 96), "real restored export source dimensions")
            print("\(algorithm.label) 1×: \(result.seconds) seconds; report: \(String(decoding: reportData, as: UTF8.self))")
        }
    })
}
import AVFoundation

private func rgba(_ image: CGImage) -> [UInt8] {
    var bytes = [UInt8](repeating: 0, count: image.width * image.height * 4)
    bytes.withUnsafeMutableBytes { data in
        let context = CGContext(data: data.baseAddress, width: image.width, height: image.height,
            bitsPerComponent: 8, bytesPerRow: image.width * 4, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
    }
    return bytes
}

private func videoTimes(_ url: URL) async throws -> [Double] {
    let asset = AVURLAsset(url: url)
    let track = try await asset.loadTracks(withMediaType: .video)[0]
    let reader = try AVAssetReader(asset: asset)
    let output = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
    reader.add(output)
    guard reader.startReading() else { throw UncoverError.invalid("Cannot reopen export.") }
    var times: [Double] = []
    while let sample = output.copyNextSampleBuffer() { times.append(CMSampleBufferGetPresentationTimeStamp(sample).seconds) }
    if reader.status == .failed { throw UncoverError.invalid("Reopened export is corrupt.") }
    return times
}

/// Tests only: never instantiated by the application or labelled SeedVR2.
private struct UncoverTestBackend: TemporalVideoUpscaler {
    func capabilities() async -> UpscalerCapabilities {
        UpscalerCapabilities(available: true, message: "TEST ONLY", scales: UncoverScale.allCases, minimumFrames: 5, requiresUnsafeMemoryOptIn: false)
    }
    func reconstruct(_ prepared: UncoverPrepared, allowUnsafeMemory: Bool,
                     progress: @escaping @Sendable (String) -> Void) async throws -> UncoverResult {
        for frame in prepared.manifest.frames {
            try Task.checkCancellation()
            try FileManager.default.copyItem(at: RegionImages.frameURL(prepared.directory, "baseline", frame.index),
                                            to: RegionImages.frameURL(prepared.directory, "reconstructed", frame.index))
        }
        return UncoverResult(prepared: prepared, model: "TEST ONLY", seconds: 0)
    }
}

private struct UncoverOutputLimitedBackend: TemporalVideoUpscaler {
    func capabilities() async -> UpscalerCapabilities {
        UpscalerCapabilities(available: true, message: "TEST ONLY", scales: UncoverScale.allCases,
            minimumFrames: 5, requiresUnsafeMemoryOptIn: false, maximumOutputPixels: 100)
    }
    func reconstruct(_ prepared: UncoverPrepared, allowUnsafeMemory: Bool,
                     progress: @escaping @Sendable (String) -> Void) async throws -> UncoverResult {
        try await UncoverTestBackend().reconstruct(prepared, allowUnsafeMemory: false, progress: progress)
    }
}

private actor UncoverCountingBackend: TemporalVideoUpscaler {
    private(set) var count = 0
    private var fail = false
    func setFailure(_ value: Bool) { fail = value }
    func capabilities() async -> UpscalerCapabilities { await UncoverTestBackend().capabilities() }
    func reconstruct(_ prepared: UncoverPrepared, allowUnsafeMemory: Bool,
                     progress: @escaping @Sendable (String) -> Void) async throws -> UncoverResult {
        count += 1
        if fail { throw UncoverError.backend("Expected test failure") }
        return try await UncoverTestBackend().reconstruct(prepared, allowUnsafeMemory: false, progress: progress)
    }
}

private struct UncoverFourOnlyBackend: TemporalVideoUpscaler {
    func capabilities() async -> UpscalerCapabilities {
        UpscalerCapabilities(available: true, message: "TEST ONLY", scales: [.four], minimumFrames: 5, requiresUnsafeMemoryOptIn: false)
    }
    func reconstruct(_ prepared: UncoverPrepared, allowUnsafeMemory: Bool,
                     progress: @escaping @Sendable (String) -> Void) async throws -> UncoverResult {
        try await UncoverTestBackend().reconstruct(prepared, allowUnsafeMemory: false, progress: progress)
    }
}

private actor UncoverDelayedBackend: TemporalVideoUpscaler {
    private(set) var started = false
    private(set) var cancelled = false
    func capabilities() async -> UpscalerCapabilities { await UncoverTestBackend().capabilities() }
    func reconstruct(_ prepared: UncoverPrepared, allowUnsafeMemory: Bool,
                     progress: @escaping @Sendable (String) -> Void) async throws -> UncoverResult {
        started = true
        do { try await Task.sleep(nanoseconds: 10_000_000_000) }
        catch { cancelled = true; throw error }
        return try await UncoverTestBackend().reconstruct(prepared, allowUnsafeMemory: false, progress: progress)
    }
}

private func waitUntil(timeout: Double = 5, _ predicate: () async -> Bool) async throws {
    let deadline = Date().addingTimeInterval(timeout)
    while !(await predicate()) {
        if Date() > deadline { throw UncoverError.invalid("Session did not reach the expected state.") }
        try await Task.sleep(nanoseconds: 10_000_000)
    }
}
