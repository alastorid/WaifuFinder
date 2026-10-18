import CoreGraphics
import CoreML
import Combine
import Foundation

private func taggerContractImage(width: Int, height: Int, pixel: (Int, Int) -> [UInt8]) -> CGImage {
    var bytes: [UInt8] = []
    bytes.reserveCapacity(width * height * 4)
    for y in 0..<height { for x in 0..<width { bytes.append(contentsOf: pixel(x, y)) } }
    return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                   space: CGColorSpace(name: CGColorSpace.sRGB)!,
                   bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                   provider: CGDataProvider(data: Data(bytes) as CFData)!, decode: nil,
                   shouldInterpolate: false, intent: .defaultIntent)!
}

private actor TagBatchPipelineProbe {
    var fetched = 0
    var committed: [[Int]] = []
    var sawOverlap = false
    var maxOutstanding = 0
    var drainedAfterCommit = false
    var emptyCommitCounts: [Int] = []
    let failSecondFetch: Bool
    let cancelConsumer: Bool
    let batches: [[Int]]
    let recheckInitialEmpty: Bool
    init(failSecondFetch: Bool = false, cancelConsumer: Bool = false,
         batches: [[Int]] = [[0, 1], [2, 3], [4], []], recheckInitialEmpty: Bool = false) {
        self.failSecondFetch = failSecondFetch
        self.cancelConsumer = cancelConsumer
        self.batches = batches
        self.recheckInitialEmpty = recheckInitialEmpty
    }
    func next() throws -> [Int]? {
        let index = fetched
        fetched += 1
        maxOutstanding = max(maxOutstanding, fetched - committed.count)
        if failSecondFetch && index == 1 { throw TaggerError.preprocessing }
        return batches[min(index, batches.count - 1)]
    }
    func consume(_ batch: [Int]) async throws {
        // Let a ready producer run while consumption is suspended.
        try await Task.sleep(for: .milliseconds(30))
        sawOverlap = sawOverlap || fetched > committed.count + 1
        committed.append(batch)
        if cancelConsumer { withUnsafeCurrentTask { $0?.cancel() } }
    }
    func empty() -> Bool {
        emptyCommitCounts.append(committed.count)
        drainedAfterCommit = committed == [[0, 1], [2, 3], [4]]
        return recheckInitialEmpty && fetched == 1
    }
}

/// Holds the first wave until the entire window has arrived, then lets its
/// first batch finish last. Deadlines turn a serial implementation into a
/// clear failure instead of hanging the regression runner.
private actor TagWindowProbe {
    var fetched = 0
    var active = 0
    var peakActive = 0
    var peakRetained = 0
    var finished: [Int] = []
    var committed: [Int] = []
    var drainedAfterCommit = false
    let window: Int
    var total: Int
    let failFetch: Int?
    let growWhenEmpty: Bool
    private var didGrow = false

    init(window: Int = 8, total: Int = 25, failFetch: Int? = nil, growWhenEmpty: Bool = false) {
        self.window = window
        self.total = total
        self.failFetch = failFetch
        self.growWhenEmpty = growWhenEmpty
    }
    func next() throws -> [Int]? {
        if fetched == failFetch { throw TaggerError.preprocessing }
        guard fetched < total else { return [] }
        let index = fetched
        fetched += 1
        peakRetained = max(peakRetained, fetched - committed.count)
        return [index]
    }
    func process(_ batch: [Int]) async throws -> Int {
        let index = batch[0]
        active += 1
        peakActive = max(peakActive, active)
        defer { active -= 1 }
        let deadline = Date().addingTimeInterval(5)
        if failFetch == nil {
            while peakActive < window || (index == 0 && finished.count < window - 1) {
                try Task.checkCancellation()
                guard Date() < deadline else { throw TaggerError.invalidBatch }
                try await Task.sleep(for: .milliseconds(1))
            }
        }
        finished.append(index)
        return index
    }
    func commit(_ index: Int) { committed.append(index) }
    func empty() -> Bool {
        if growWhenEmpty && !didGrow {
            didGrow = true
            total += 5
            return true
        }
        drainedAfterCommit = committed.count == total && active == 0
        return false
    }
}

private actor TagPhotoRenditionProbe {
    let images: [CGImage]
    let failFirst: Bool
    var requestedSizes: [Int] = []
    var completed: [Int] = []
    var active = 0
    var peakActive = 0

    init(images: [CGImage], failFirst: Bool = false) {
        self.images = images
        self.failFirst = failFirst
    }

    func load(_ url: URL, size: Int) async throws -> CGImage {
        let index = Int(url.lastPathComponent)!
        requestedSizes.append(size)
        active += 1
        peakActive = max(peakActive, active)
        defer { active -= 1 }
        try await Task.sleep(for: .milliseconds(index == 0 ? 40 : (failFirst ? 200 : 5)))
        if failFirst && index == 0 { throw TaggerError.preprocessing }
        completed.append(index)
        return images[index]
    }
}

func registerWDTaggerContractTests() {
    let suite = "WDTagger contract"
    Registry.shared.add(suite: suite, TestCase(name: "Photos tagging requests model-sized pixels with bounded concurrency and preserves image identity and order", knownBug: nil) {
        let images = try (0..<8).map { try ImageFixture.cgImage(width: 448, height: 200 + $0) }
        let probe = TagPhotoRenditionProbe(images: images)
        let sources = (0..<8).map { TaggingImageSource.photo(URL(string: "waifu-photos://test/\($0)")!) }
        let loaded = try await TaggingImageLoader.load(sources, photoRendition: { try await probe.load($0, size: $1) })
        checkEqual(await probe.requestedSizes, Array(repeating: 448, count: 8), "PhotoKit is asked for the model's longest edge")
        checkEqual(await probe.peakActive, 4, "four requests overlap without unbounded album prefetch")
        check(await probe.completed.first != 0, "fixture delivers images out of order")
        checkEqual(loaded.count, images.count, "every image is retained")
        for (actual, expected) in zip(loaded, images) {
            check(actual === expected, "decoded PhotoKit pixels reach inference directly without JPEG conversion or resizing")
        }
        let empty = try await TaggingImageLoader.load([], photoRendition: { _, _ in throw TaggerError.preprocessing })
        check(empty.isEmpty, "an empty queue performs no image work")
    })
    Registry.shared.add(suite: suite, TestCase(name: "tagging image preparation retains video thumbnail support and joins cancelled Photos requests on failure", knownBug: nil) {
        let image = try ImageFixture.cgImage(width: 512, height: 384)
        let jpeg = try ThumbnailCache.encodeJPEG(image)
        let loaded = try await TaggingImageLoader.load([.thumbnail(jpeg)], photoRendition: { _, _ in
            throw TaggerError.preprocessing
        })
        checkEqual(loaded.first?.width, 512, "video thumbnails keep the existing bounded decode")
        checkEqual(loaded.first?.height, 384, "video thumbnail aspect is preserved")
        _ = await checkThrowsAsync("corrupt JPEGs fail image preparation") {
            _ = try await TaggingImageLoader.load([.thumbnail(Data())])
        }
        let probe = TagPhotoRenditionProbe(images: Array(repeating: image, count: 8), failFirst: true)
        let sources = (0..<8).map { TaggingImageSource.photo(URL(string: "waifu-photos://test/\($0)")!) }
        do {
            _ = try await TaggingImageLoader.load(sources, photoRendition: { try await probe.load($0, size: $1) })
            check(false, "PhotoKit failure must propagate")
        } catch TaggerError.preprocessing { }
        checkEqual(await probe.active, 0, "all sibling requests finish cancellation before returning")
        checkEqual(await probe.requestedSizes.count, 4, "a failed request prevents starting the rest of the batch")
        let cancelled = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await TaggingImageLoader.load(sources, photoRendition: { try await probe.load($0, size: $1) })
        }
        do {
            _ = try await cancelled.value
            check(false, "a cancelled batch must fail")
        } catch is CancellationError { }
        checkEqual(await probe.requestedSizes.count, 4, "a pre-cancelled batch starts no PhotoKit requests")
    })
    Registry.shared.add(suite: suite, TestCase(name: "model-sized Photos letterboxing preserves sharp pixels without interpolation", knownBug: nil) {
        let image = taggerContractImage(width: 448, height: 224) { x, y in
            (x + y).isMultiple(of: 2) ? [255, 0, 0, 255] : [0, 0, 255, 255]
        }
        let pixels = try WDTaggerBackend.preprocessRGBA(image)
        var exact = true
        for y in 0..<448 {
            for x in 0..<448 {
                let offset = (y * 448 + x) * 4
                let expected: [UInt8] = (112..<336).contains(y)
                    ? ((x + y - 112).isMultiple(of: 2) ? [255, 0, 0, 255] : [0, 0, 255, 255])
                    : [255, 255, 255, 255]
                if Array(pixels[offset..<offset + 4]) != expected { exact = false }
            }
        }
        check(exact, "448px content and white padding are pixel exact")
    })
    Registry.shared.add(suite: suite, TestCase(name: "remote tagging pipelines a bounded window and commits reordered replies in input order", knownBug: nil) {
        for window in [4, 8] {
            let probe = TagWindowProbe(window: window)
            try await TaggingBatchPipeline.runPipelined(maximumInFlight: window,
                next: { try await probe.next() }, whenEmpty: { await probe.empty() },
                process: { try await probe.process($0) }, commit: { await probe.commit($0) })
            checkEqual(await probe.peakActive, window, "the full request window overlaps")
            checkEqual(await probe.peakRetained, window, "out-of-order results cannot grow the window")
            checkEqual(await probe.committed, Array(0..<25), "every result is committed in source order")
            check(await probe.finished.first != 0, "the test actually exercises reordered replies")
            check(await probe.drainedAfterCommit, "queue completion waits for the final result")
        }
    })
    Registry.shared.add(suite: suite, TestCase(name: "remote tagging rechecks work added after its queued batches drain", knownBug: nil) {
        let probe = TagWindowProbe(window: 4, total: 4, growWhenEmpty: true)
        try await TaggingBatchPipeline.runPipelined(maximumInFlight: 4,
            next: { try await probe.next() }, whenEmpty: { await probe.empty() },
            process: { try await probe.process($0) }, commit: { await probe.commit($0) })
        checkEqual(await probe.committed, Array(0..<9), "new candidates extend the same ordered run")
        check(await probe.drainedAfterCommit, "the extended queue finishes after its final commit")
    })
    Registry.shared.add(suite: suite, TestCase(name: "remote tagging commits submitted results before a later preparation failure", knownBug: nil) {
        let probe = TagWindowProbe(failFetch: 3)
        do {
            try await TaggingBatchPipeline.runPipelined(maximumInFlight: 8,
                next: { try await probe.next() }, whenEmpty: { await probe.empty() },
                process: { try await probe.process($0) }, commit: { await probe.commit($0) })
            check(false, "preparation failure propagates")
        } catch TaggerError.preprocessing {} catch { throw error }
        checkEqual(await probe.committed, [0, 1, 2], "completed submitted work survives the fetch failure")
        checkEqual(await probe.active, 0, "all submitted tasks have joined")
    })
    Registry.shared.add(suite: suite, TestCase(name: "tagging prefetch overlaps one fetch and drains only after committing ordered results", knownBug: nil) {
        let probe = TagBatchPipelineProbe()
        try await TaggingBatchPipeline.run(next: { try await probe.next() },
                                          whenEmpty: { await probe.empty() },
                                          consume: { try await probe.consume($0) })
        let batches = await probe.committed
        checkEqual(batches, [[0, 1], [2, 3], [4]], "ordered full and partial batches")
        check(await probe.sawOverlap, "next fetch overlaps current consumption")
        checkEqual(await probe.maxOutstanding, 2, "only current batch and one following batch are retained")
        check(await probe.drainedAfterCommit, "queue cannot finish before final results are committed")
    })
    Registry.shared.add(suite: suite, TestCase(name: "tagging prefetch preserves completed results when the following fetch fails", knownBug: nil) {
        let probe = TagBatchPipelineProbe(failSecondFetch: true)
        do {
            try await TaggingBatchPipeline.run(next: { try await probe.next() },
                                              whenEmpty: { await probe.empty() },
                                              consume: { try await probe.consume($0) })
            check(false, "fetch failure propagates")
        } catch TaggerError.preprocessing {} catch { throw error }
        checkEqual(await probe.committed, [[0, 1]], "successful inference is committed before next error propagates")
    })
    Registry.shared.add(suite: suite, TestCase(name: "tagging prefetch rechecks a queue that grows while draining", knownBug: nil) {
        let probe = TagBatchPipelineProbe(batches: [[], [5], []], recheckInitialEmpty: true)
        try await TaggingBatchPipeline.run(next: { try await probe.next() },
                                          whenEmpty: { await probe.empty() },
                                          consume: { try await probe.consume($0) })
        checkEqual(await probe.committed, [[5]], "newly queued work is consumed")
        checkEqual(await probe.emptyCommitCounts, [0, 1], "final drain follows the new committed result")
    })
    Registry.shared.add(suite: suite, TestCase(name: "tagging prefetch observes cancellation before consuming a following batch", knownBug: nil) {
        let probe = TagBatchPipelineProbe(cancelConsumer: true)
        let task = Task {
            try await TaggingBatchPipeline.run(next: { try await probe.next() },
                                              whenEmpty: { await probe.empty() },
                                              consume: { try await probe.consume($0) })
        }
        do { try await task.value; check(false, "cancellation propagates") }
        catch is CancellationError {} catch { throw error }
        checkEqual(await probe.committed, [[0, 1]], "only completed current batch is committed")
        check(await probe.fetched <= 2, "cancellation cannot start unbounded fetching")
    })
    Registry.shared.add(suite: suite, TestCase(name: "native probability rows decode exactly from unaligned byte storage", knownBug: nil) {
        let columns = TagVocabulary.headCount
        let values = (0..<(8 * columns)).map { Float($0) / Float(8 * columns) }
        let storage = UnsafeMutableRawPointer.allocate(byteCount: values.count * 4 + 1, alignment: 16)
        defer { storage.deallocate() }
        values.withUnsafeBytes { storage.advanced(by: 1).copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        let bytes = Data(bytesNoCopy: storage.advanced(by: 1), count: values.count * 4, deallocator: .none)
        let decoded = try WDTaggerWorker.decodeProbabilities(bytes, count: 8)
        for row in 0..<8 {
            checkEqual(decoded[row], Array(values[(row * columns)..<((row + 1) * columns)]), "unaligned source preserves every head")
        }
        for count in [0, 9] {
            do {
                _ = try WDTaggerWorker.decodeProbabilities(bytes, count: count)
                check(false, "invalid count is rejected")
            } catch TaggerError.invalidProbabilities {} catch { throw error }
        }
        do {
            _ = try WDTaggerWorker.decodeProbabilities(bytes.dropLast(), count: 8)
            check(false, "truncated row is rejected")
        } catch TaggerError.invalidProbabilities {} catch { throw error }
    })
    Registry.shared.add(suite: suite, TestCase(name: "tagging progress limits redraws and always publishes state transitions", knownBug: nil) {
        await MainActor.run {
            let status = TaggingStatus()
            var published: [TaggingProgress] = []
            let subscription = status.$progress.sink { published.append($0) }
            var progress = TaggingProgress(isRunning: true, origin: .tags, total: 1_000)
            status.update(progress)
            checkEqual(status.progress, progress, "start is visible immediately")
            let publicationsAtStart = published.count
            for batch in 1...19 {
                progress.processed = batch * 8
                progress.elapsed = Double(batch) / 100
                status.update(progress)
            }
            checkEqual(published.count, publicationsAtStart, "fast batches do not cause a render per batch")
            progress.elapsed = 0.25
            status.update(progress)
            checkEqual(status.progress.processed, 152, "throttled progress catches up without losing completed counts")
            progress.total = 2_000
            status.update(progress)
            checkEqual(status.progress.total, 2_000, "expanded queue is visible immediately")
            progress.processed = 160
            progress.isRunning = false
            status.update(progress)
            checkEqual(status.progress, progress, "cancel or finish delivers final counts even inside throttle interval")
            progress = TaggingProgress(isRunning: true, origin: .rating, total: 8)
            status.update(progress)
            checkEqual(status.progress, progress, "new run resets elapsed and panel ownership")
            status.update(TaggingProgress())
            checkEqual(status.progress, TaggingProgress(), "reset is always delivered")
            withExtendedLifetime(subscription) {}
        }
    })
    Registry.shared.add(suite: suite, TestCase(name: "model discovery uses default storage and skips stale configured paths", knownBug: nil) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let storage = root.appendingPathComponent("Models")
        let model = storage.appendingPathComponent("wd-tagger", isDirectory: true)
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        try Data().write(to: model.appendingPathComponent("model.safetensors"))
        try Data().write(to: model.appendingPathComponent("wd-tagger-labels.csv"))
        let stale = root.appendingPathComponent("removed-prototype")
        checkEqual(WDTaggerConfiguration.resolveDirectory(configured: nil, storage: storage, bundled: nil), model, "default installation found")
        checkEqual(WDTaggerConfiguration.resolveDirectory(configured: stale, storage: storage, bundled: nil), model, "stale configuration falls back")
        let custom = root.appendingPathComponent("custom")
        try FileManager.default.copyItem(at: model, to: custom)
        checkEqual(WDTaggerConfiguration.resolveDirectory(configured: custom, storage: storage, bundled: nil), custom, "explicit valid selection wins")
        try FileManager.default.removeItem(at: model.appendingPathComponent("wd-tagger-labels.csv"))
        check(WDTaggerConfiguration.resolveDirectory(configured: nil, storage: storage, bundled: nil) == nil, "incomplete installations are not selected")
    })
    Registry.shared.add(suite: suite, TestCase(name: "probability buffer decoding honors padded rows and strided columns", knownBug: nil) {
        let columns = TagVocabulary.headCount
        for columnStride in [1, 2] {
            let rowStride = columns * columnStride + 16
            let buffer = UnsafeMutablePointer<Float>.allocate(capacity: 8 * rowStride)
            buffer.initialize(repeating: -1, count: 8 * rowStride)
            defer { buffer.deinitialize(count: 8 * rowStride); buffer.deallocate() }
            for row in 0..<8 { for column in 0..<columns {
                buffer[row * rowStride + column * columnStride] = Float(row) / 10 + Float(column) / 100_000
            } }
            let array = try MLMultiArray(dataPointer: buffer, shape: [8, NSNumber(value: columns)], dataType: .float32,
                                         strides: [NSNumber(value: rowStride), NSNumber(value: columnStride)], deallocator: nil)
            for row in 0..<8 {
                let values = try WDTaggerBackend.probabilities(from: array, row: row)
                checkEqual(values, (0..<columns).map { Float(row) / 10 + Float($0) / 100_000 }, "physical strides preserve row \(row)")
            }
        }
    })
    Registry.shared.add(suite: suite, TestCase(name: "older settings and frames retain preferences while new tag fields decode nil", knownBug: nil) {
        let saved = Data(#"{"analysisMode":"quick","analysisWidth":640,"faceWeight":0.2}"#.utf8)
        let settings = try JSONDecoder().decode(WaifuSettings.self, from: saved)
        checkEqual(settings.analysisMode, .quick, "old mode survives migration")
        checkEqual(settings.analysisWidth, 640, "old width survives migration")
        checkEqual(settings.faceWeight, 0.2, "old grouping preference survives")
        check(!settings.taggerEnabled, "tagger remains opt in")
        checkEqual(settings.taggerThreshold, 0.35, "reference threshold is default")
        let original = AnalyzedFrame(sampleIndex: 37, timeSeconds: 2.5, artScore: 0.42,
                                     faceScore: nil, faceCount: 0, faceAreaFraction: nil, featurePrint: nil)
        var json = try JSONSerialization.jsonObject(with: JSONEncoder().encode(original)) as! [String: Any]
        for key in ["tagProbabilities", "tagRating", "topTags", "tagGeneralScore"] { json.removeValue(forKey: key) }
        let decoded = try JSONDecoder().decode(AnalyzedFrame.self, from: JSONSerialization.data(withJSONObject: json))
        checkEqual(decoded, original, "legacy frame metadata survives")
        checkNil(decoded.tagProbabilities, "missing probabilities become nil")
        checkNil(decoded.tagRating, "missing rating becomes nil")
        checkNil(decoded.topTags, "missing traits become nil")
        var new = original
        let probabilities = [Float](repeating: 0.125, count: TagVocabulary.headCount)
        new.tagProbabilities = TagProbabilityCodec.encode(probabilities)
        new.tagRating = .sensitive; new.topTags = ["smile"]; new.tagGeneralScore = 0.125
        checkEqual(try JSONDecoder().decode(AnalyzedFrame.self, from: JSONEncoder().encode(new)), new, "tagged frame roundtrip")
        var invalid = settings; invalid.taggerThreshold = .nan
        checkEqual(invalid.validated().taggerThreshold, 0.35, "nonfinite threshold gets safe default")
    })
    Registry.shared.add(suite: suite, TestCase(name: "RGB preprocessing preserves top-bottom orientation and writes unnormalized RGB planes", knownBug: nil) {
        let image = taggerContractImage(width: 448, height: 448) { _, y in
            y < 224 ? [255, 0, 0, 255] : [0, 0, 255, 255]
        }
        let pixels = try WDTaggerBackend.preprocessRGBA(image)
        checkEqual(Array(pixels.prefix(4)), [255, 0, 0, 255], "top red stays top")
        checkEqual(Array(pixels.suffix(4)), [0, 0, 255, 255], "bottom blue stays bottom")
        let input = try MLMultiArray(shape: [8, 3, 448, 448], dataType: .float32)
        WDTaggerBackend.writeRGB(pixels, to: input, batchIndex: 2)
        let plane = 448 * 448, base = 2 * plane * 3
        checkEqual(input[base].floatValue, 255, "red head first and stays 0...255")
        checkEqual(input[base + plane].floatValue, 0, "green head second")
        checkEqual(input[base + 2 * plane].floatValue, 0, "blue head third")
        checkEqual(input[base + 3 * plane - 1].floatValue, 255, "bottom blue lands at end of blue plane")
    })
    Registry.shared.add(suite: suite, TestCase(name: "letterbox is white, centered, opaque, and downscales wide images", knownBug: nil) {
        for dimensions in [(224, 448), (1600, 800)] {
            let image = taggerContractImage(width: dimensions.0, height: dimensions.1) { _, _ in [255, 0, 0, 255] }
            let pixels = try WDTaggerBackend.preprocessRGBA(image)
            checkEqual(pixels.count, 448 * 448 * 4, "exact model geometry")
            let center = (224 * 448 + 224) * 4
            checkEqual(Array(pixels[center..<center + 4]), [255, 0, 0, 255], "content centered")
            checkEqual(Array(pixels.prefix(4)), [255, 255, 255, 255], "white opaque letterbox corner")
            checkEqual(Array(pixels.suffix(4)), [255, 255, 255, 255], "white opaque opposite corner")
        }
        let transparent = taggerContractImage(width: 448, height: 448) { _, _ in [0, 0, 0, 0] }
        checkEqual(Array(try WDTaggerBackend.preprocessRGBA(transparent).prefix(4)), [255, 255, 255, 255], "alpha composites onto white")
        let odd = taggerContractImage(width: 448, height: 447) { _, _ in [255, 0, 0, 255] }
        let oddPixels = try WDTaggerBackend.preprocessRGBA(odd)
        checkEqual(Array(oddPixels.prefix(4)), [255, 0, 0, 255], "odd-height letterbox has floor padding above")
        checkEqual(Array(oddPixels.suffix(4)), [255, 255, 255, 255], "extra odd-height padding row below")
    })
}
