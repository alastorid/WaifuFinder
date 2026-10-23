import CoreGraphics
import Compression
import Foundation
import ImageIO

private func taggerFixtureDirectory() throws -> URL {
    let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
    let directory = ProcessInfo.processInfo.environment["WAIFU_TEST_TAGGER_DIR"].map { URL(fileURLWithPath: $0) }
        ?? WDTaggerConfiguration.availableDirectory
        ?? root.appendingPathComponent("models/published/WD-SwinV2-Tagger-v3-CoreML")
    guard FileManager.default.fileExists(atPath: directory.appendingPathComponent("wd-tagger-labels.csv").path) else {
        throw SkipTest(reason: "offline WDTagger fixture is absent")
    }
    return directory
}

/// The native worker is a resource of the app bundle, so a bare `./run-tests.sh`
/// binary cannot see it and `WDTaggerBackend` would fail with "requires the Apple
/// Silicon app build". `WAIFU_TEST_TAGGER_WORKER` points at a build tree instead,
/// which is how these run without packaging. Neither being present means the case
/// cannot run, and says so — it is a missing fixture, not a regression.
private func nativeTaggerTestExecutable() throws -> URL {
    let executable = ProcessInfo.processInfo.environment["WAIFU_TEST_TAGGER_WORKER"]
        .map { URL(fileURLWithPath: $0) } ?? WDTaggerWorker.executableURL
    guard let executable, FileManager.default.isExecutableFile(atPath: executable.path) else {
        throw SkipTest(reason: "native WD Tagger worker is absent; run with WAIFU_TEST_TAGGER_INSTALL=1 "
            + "or point WAIFU_TEST_TAGGER_WORKER at a built waifu-wdtagger")
    }
    return executable
}

private struct TaggerTestScorer: FrameScoring {
    func score(_ frame: SampledFrame) async throws -> FrameScoreResult {
        await Task.yield()
        return FrameScoreResult(artScore: Float(frame.sampleIndex), faceScore: nil, faceCount: 0,
                                faceAreaFraction: nil, featurePrint: nil)
    }
}

/// Source-generated asymmetric RGB image; no checked-in media fixture.
private func syntheticTaggerGradient() throws -> CGImage {
    let side = 448
    var pixels = [UInt8](repeating: 255, count: side * side * 4)
    for y in 0..<side {
        for x in 0..<side {
            let offset = (y * side + x) * 4
            pixels[offset] = UInt8(x * 255 / (side - 1))
            pixels[offset + 1] = UInt8(y * 255 / (side - 1))
            pixels[offset + 2] = 37
        }
    }
    guard let provider = CGDataProvider(data: Data(pixels) as CFData),
          let space = CGColorSpace(name: CGColorSpace.sRGB),
          let image = CGImage(width: side, height: side, bitsPerComponent: 8,
                              bitsPerPixel: 32, bytesPerRow: side * 4, space: space,
                              bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                              provider: provider, decode: nil, shouldInterpolate: false,
                              intent: .defaultIntent) else { throw TaggerError.preprocessing }
    return image
}
private struct TaggerTestThumbnails: ThumbnailWriter {
    func write(_ image: CGImage, key: String) async throws {}
}
private actor RecordingTagger: TaggerBackend {
    let result: TagResult
    let fail: Bool
    var batches: [[CGImage]] = []
    init(result: TagResult, fail: Bool = false) { self.result = result; self.fail = fail }
    func tag(batch: [CGImage]) async throws -> [TagResult] {
        batches.append(batch)
        if fail { throw TaggerError.invalidProbabilities }
        return batch.map { _ in result }
    }
}

func registerWDTaggerTests() {
    let suite = "WDTagger"
    Registry.shared.add(suite: suite, TestCase(name: "native model top15 matches independent original BGR Torch reference", knownBug: nil) {
        let modelDirectory = try taggerFixtureDirectory()
        guard FileManager.default.fileExists(atPath: modelDirectory.appendingPathComponent("model.safetensors").path) else {
            throw SkipTest(reason: "offline native weights absent")
        }
        let fixtures = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent(".build/test-fixtures/wd-tagger")
        guard FileManager.default.fileExists(atPath: fixtures.appendingPathComponent("original-bgr-probabilities.bin").path) else {
            throw SkipTest(reason: "generate the synthetic FP32 reference with tools/make-wd-tagger-test-fixture.py")
        }
        let image = try syntheticTaggerGradient()
        let reference = try Data(contentsOf: fixtures.appendingPathComponent("original-bgr-probabilities.bin"))
        let expected = stride(from: 0, to: reference.count, by: 4).map { offset in
            Float(bitPattern: reference.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self).littleEndian })
        }
        let executable = try nativeTaggerTestExecutable()
        let backend = try WDTaggerBackend(directory: modelDirectory, executableURL: executable)
        let tail = try await backend.tag(batch: [image])
        checkEqual(tail.count, 1, "fixed batch tail padding does not emit duplicates")
        let got = try await backend.probabilityRows(batch: [image])[0]
        checkEqual(tail[0].encodedProbabilities, TagProbabilityCodec.encode(got),
                   "worker CPU quantization matches raw Float32 CPU reference exactly")
        let expectedTop = Set(expected.indices.sorted { expected[$0] > expected[$1] }.prefix(15))
        let actualTop = Set(got.indices.sorted { got[$0] > got[$1] }.prefix(15))
        checkEqual(actualTop.intersection(expectedTop).count, 15, "top15 independent reference overlap")
        let maximumError = zip(got, expected).map { abs($0 - $1) }.max()!
        check(maximumError < 0.01, "all heads stay within fp16 tolerance, max error \(maximumError)")
        let dark = try ImageFixture.cgImage(width: 448, height: 448, gray: 0)
        let darkResult = try await backend.tag(batch: [dark])[0]
        let images = (0..<8).map { $0.isMultiple(of: 2) ? image : dark }
        let fullBatch = try await backend.tag(batch: images)
        checkEqual(fullBatch.count, 8, "full native batch emits all eight rows")
        for row in 0..<8 {
            let reference = row.isMultiple(of: 2) ? tail[0].probabilities : darkResult.probabilities
            let error = zip(fullBatch[row].probabilities, reference).map { abs($0 - $1) }.max()!
            check(error < 0.0001, "padded native output row \(row) preserves input identity, error \(error)")
        }
        _ = await checkThrowsAsync("more than eight frames refused") { try await backend.tag(batch: Array(repeating: image, count: 9)) }
        let cancelled = Task { try await backend.tag(batch: [image]) }
        cancelled.cancel()
        _ = await checkThrowsAsync("cancelled inference does not return results") { try await cancelled.value }
    })
    Registry.shared.add(suite: suite, TestCase(name: "independent native model instances preserve mixed batches under concurrent prediction", knownBug: nil) {
        let directory = try taggerFixtureDirectory()
        guard FileManager.default.fileExists(atPath: directory.appendingPathComponent("model.safetensors").path) else {
            throw SkipTest(reason: "offline native weights absent")
        }
        let image = try syntheticTaggerGradient()
        let dark = try ImageFixture.cgImage(width: 448, height: 448, gray: 0)
        let executable = try nativeTaggerTestExecutable()
        let first = try WDTaggerBackend(directory: directory, executableURL: executable)
        let second = try WDTaggerBackend(directory: directory, executableURL: executable)
        let lightReference = try await first.tag(batch: [image])[0].probabilities
        let darkReference = try await first.tag(batch: [dark])[0].probabilities
        try await withThrowingTaskGroup(of: Void.self) { group in
            for backend in [first, second] {
                group.addTask {
                    for round in 0..<6 {
                        let count = round.isMultiple(of: 3) ? 1 : 8
                        let images = (0..<count).map { ($0 + round).isMultiple(of: 2) ? image : dark }
                        let results = try await backend.tag(batch: images)
                        checkEqual(results.count, count, "no rows lost under concurrent inference")
                        for (row, result) in results.enumerated() {
                            let reference = (row + round).isMultiple(of: 2) ? lightReference : darkReference
                            let error = zip(result.probabilities, reference).map { abs($0 - $1) }.max()!
                            check(error < 0.0001, "concurrent model result matches independent input, error \(error)")
                        }
                    }
                }
            }
            try await group.waitForAll()
        }
    })
    Registry.shared.add(suite: suite, TestCase(name: "label order, threshold, rating and character exclusion", knownBug: nil) {
        let vocabulary = try TagVocabulary.load(from: taggerFixtureDirectory().appendingPathComponent("wd-tagger-labels.csv"))
        checkEqual(vocabulary.labels.count, 10_861, "all model heads retained")
        checkEqual(vocabulary.generalTags.count, 8_106, "only general vocabulary exposed")
        checkEqual(Array(vocabulary.labels.prefix(4).map(\.name)), TagRating.allCases.map(\.rawValue), "rating head order")
        var probabilities = [Float](repeating: 0, count: 10_861)
        probabilities[3] = 0.9
        let first = vocabulary.generalIndices[0], second = vocabulary.generalIndices[1]
        probabilities[first] = 0.3501; probabilities[second] = 0.8
        probabilities[vocabulary.labels.firstIndex { $0.category == 4 }!] = 1
        let result = try vocabulary.result(probabilities: probabilities)
        checkEqual(result.rating, .explicit, "highest rating head selected")
        checkEqual(result.topTags, [vocabulary.labels[second].name, vocabulary.labels[first].name], "strict threshold and descending general scores")
        check(abs(result.generalScore - 1.1501) < 0.00001, "sum includes general heads only")
        var boundary = probabilities
        boundary[first] = 0.35
        checkEqual(try vocabulary.result(probabilities: boundary).topTags, [vocabulary.labels[second].name], "threshold equality is excluded like the daemon")
        checkEqual(try vocabulary.result(probabilities: probabilities, threshold: 0.5).topTags,
                   [vocabulary.labels[second].name], "custom threshold respected")
        checkNil(vocabulary.index(for: vocabulary.labels.first { $0.category == 4 }!.name), "character head unavailable to search")
    })
    Registry.shared.add(suite: suite, TestCase(name: "autocomplete ranks exact then prefix then substring and preserves kaomoji", knownBug: nil) {
        let vocabulary = try TagVocabulary.load(from: taggerFixtureDirectory().appendingPathComponent("wd-tagger-labels.csv"))
        checkEqual(vocabulary.autocomplete(query: "SMILE").first, "smile", "case-insensitive exact match first")
        let query = "look"
        let actual = vocabulary.autocomplete(query: query, limit: 10000)
        let expected = vocabulary.generalTags.filter { $0 == query }
            + vocabulary.generalTags.filter { $0 != query && $0.hasPrefix(query) }
            + vocabulary.generalTags.filter { !$0.hasPrefix(query) && $0.contains(query) }
        checkEqual(actual, expected, "stable vocabulary order within match classes")
        checkEqual(TagVocabulary.normalized("looking_at_viewer"), "looking at viewer", "underscore equivalence")
        checkEqual(TagVocabulary.normalized("0_0"), "0_0", "kaomoji keeps underscores")
        checkEqual(vocabulary.autocomplete(query: "", limit: 12), [], "empty query has no suggestions")
    })
    Registry.shared.add(suite: suite, TestCase(name: "live scores use sparse UInt8 LZ4 with bounded quantization and strict cutoff", knownBug: nil) {
        var probabilities = [Float](repeating: 0, count: TagVocabulary.headCount)
        let threshold = TagProbabilityCodec.storageThreshold
        let scores: [Float] = [0, threshold.nextDown, threshold, threshold.nextUp, 0.3501, 0.375, 0.5, 0.75, 0.9, 1]
        for (index, score) in scores.enumerated() { probabilities[index + 4] = score }
        probabilities[TagVocabulary.headCount - 1] = 1
        guard let encoded = TagProbabilityCodec.encode(probabilities),
              let decoded = TagProbabilityCodec.decode(encoded) else {
            check(false, "compact scores round trip"); return
        }
        checkEqual(encoded.prefix(4), Data("WDT4".utf8), "new rows use UInt8 LZ4")
        check(encoded.count < 128, "a sparse row occupies under 128 bytes instead of 43 KB")
        for index in 4..<probabilities.count {
            if probabilities[index] < threshold {
                checkEqual(decoded[index], 0, "below-cutoff traits are discarded")
            } else {
                checkClose(Double(decoded[index]), Double(probabilities[index]), tolerance: 1.0 / 255, "UInt8 error stays within one step")
            }
        }
        checkEqual(decoded[6], threshold, "equality stays exactly at the trait cutoff")
        check(decoded[7] > threshold && decoded[8] > threshold, "just-above-cutoff traits remain admitted")
        checkEqual(decoded[12], Float(229) / 255, "0.9 uses linear UInt8 confidence")
        let indices = [TagVocabulary.headCount - 1, 8, 0, 8, 3]
        checkEqual(TagProbabilityCodec.values(in: encoded, at: indices), indices.map { decoded[$0] },
                   "selective reads preserve order, duplicates, zeros and the final head")
        var unaligned = Data([0xff]); unaligned.append(encoded)
        checkEqual(TagProbabilityCodec.decode(unaligned.dropFirst()), decoded, "sliced storage is supported")
        checkNil(TagProbabilityCodec.values(in: encoded, at: [-1]), "negative head fails")
        checkNil(TagProbabilityCodec.value(in: encoded, at: TagVocabulary.headCount), "past-end head fails")
        checkNil(TagProbabilityCodec.decode(encoded.dropLast()), "truncated LZ4 fails")
        checkNil(TagProbabilityCodec.decode(encoded + Data([0])), "trailing bytes fail")
        var corrupt = encoded
        corrupt.replaceSubrange(4..<8, with: [0xff, 0xff, 0xff, 0xff])
        checkNil(TagProbabilityCodec.decode(corrupt), "unbounded decompression length fails")
        for value: Float in [.nan, .infinity, -0.1, 1.1] {
            probabilities[0] = value
            checkNil(TagProbabilityCodec.encode(probabilities), "invalid source values fail before quantization")
        }
        let zero = TagProbabilityCodec.encode([Float](repeating: 0, count: TagVocabulary.headCount))!
        checkEqual(TagProbabilityCodec.decode(zero), [Float](repeating: 0, count: TagVocabulary.headCount), "empty sparse row is valid")

        // Measure retained payloads for distinct rows, not shared copies of one Data.
        var retained: [Data] = []
        var row = [Float](repeating: 0.01, count: TagVocabulary.headCount)
        for index in 0..<32 { row[index * 17] = 0.75 }
        for index in 0..<1_000 {
            row[TagVocabulary.headCount - 1] = Float(500 + index % 500) / 1_000
            retained.append(TagProbabilityCodec.encode(row)!)
        }
        let compactBytes = retained.reduce(0) { $0 + $1.count }
        let oldBytes = retained.count * (4 + TagVocabulary.headCount * 4)
        check(compactBytes < oldBytes / 100, "representative sparse payloads shrink by more than 99 percent")
        print("      live tag payloads: \(oldBytes) -> \(compactBytes) bytes for 1,000 distinct rows")
    })
    Registry.shared.add(suite: suite, TestCase(name: "quantized worker rows preserve four low rating scores without requantization", knownBug: nil) {
        var codes = [UInt8](repeating: 0, count: TagVocabulary.headCount)
        codes[0] = 25; codes[1] = 51; codes[2] = 51; codes[3] = 38
        codes[4] = 89; codes[5] = 1; codes[6] = 255
        let bytes = Data(codes)
        let rows = try WDTaggerWorker.decodeQuantized(bytes, count: 1)
        let decoded = TagProbabilityCodec.decode(rows[0].scores)!
        checkEqual(Array(decoded.prefix(4)), [Float(25) / 255, Float(51) / 255, Float(51) / 255, Float(38) / 255], "rating probabilities do not use the trait cutoff")
        checkEqual(decoded[4], Float(0.35).nextUp, "worker boundary code retains membership")
        checkEqual(decoded[5], 0.35, "worker equality code is preserved")
        var invalid = bytes; invalid[6] = 2
        _ = checkThrows("invalid worker trait code") { try WDTaggerWorker.decodeQuantized(invalid, count: 1) }
        _ = checkThrows("truncated worker row") { try WDTaggerWorker.decodeQuantized(bytes.dropLast(), count: 1) }
    })
    Registry.shared.add(suite: suite, TestCase(name: "UInt8 codes are stable and obsolete formats are rejected", knownBug: nil) {
        for code in UInt8(89)...UInt8(255) {
            checkEqual(TagProbabilityCodec.quantize(TagProbabilityCodec.probability(code)), code,
                       "linear score codes remain stable through Float32 API boundaries")
        }
        let encoded = TagProbabilityCodec.encode([Float](repeating: 0.8, count: TagVocabulary.headCount))!
        for magic in ["WDT1", "WDT2", "WDT3"] {
            var old = encoded
            old.replaceSubrange(0..<4, with: magic.utf8)
            checkNil(TagProbabilityCodec.decode(old), "obsolete score formats require clearing the cache")
        }
    })
    Registry.shared.add(suite: suite, TestCase(name: "tagging pipeline batches eight and preserves source image order and frame metadata", knownBug: nil) {
        let vocabulary = try TagVocabulary.load(from: taggerFixtureDirectory().appendingPathComponent("wd-tagger-labels.csv"))
        var values = [Float](repeating: 0, count: 10_861)
        values[0] = 0.875; values[vocabulary.generalIndices[0]] = 0.75
        let result = try vocabulary.result(probabilities: values)
        let tagger = RecordingTagger(result: result)
        let samples = try (0..<19).map { index in
            SampledFrame(sampleIndex: index * 30, timeSeconds: Double(index) * 1.01,
                         image: try ImageFixture.cgImage(width: 16, height: 20, gray: CGFloat(index) / 19))
        }
        let input = AsyncThrowingStream<SampledFrame, Error> { continuation in
            samples.forEach { continuation.yield($0) }; continuation.finish()
        }
        let scorer = FrameScorePipeline(scorer: TaggerTestScorer(), thumbnailStore: TaggerTestThumbnails(), analysisWidth: 16, workerCount: 4)
        var output: [AnalyzedFrame] = []
        for try await frame in TaggedFramePipeline(scorer: scorer, tagger: tagger).run(frames: input, minimumFaceAreaFraction: 0.01) { output.append(frame) }
        let batches = await tagger.batches
        checkEqual(batches.map(\.count), [8, 8, 3], "tail remains unpadded until backend")
        checkEqual(output.map(\.id.sampleIndex), samples.map(\.sampleIndex), "original gapped IDs and order")
        checkEqual(output.map(\.timeSeconds), samples.map(\.timeSeconds), "original timestamps")
        for (image, sample) in zip(batches.flatMap { $0 }, samples) { check(image === sample.image, "tagger receives original decoded image") }
        for frame in output {
            checkEqual(frame.topTags, result.topTags, "every frame gets traits")
            checkEqual(frame.tagRating, result.rating, "every frame gets rating")
            checkEqual(TagProbabilityCodec.decode(frame.tagProbabilities!), TagProbabilityCodec.decode(TagProbabilityCodec.encode(values)!), "all heads survive pipeline quantization")
        }
        let failingTagger = RecordingTagger(result: result, fail: true)
        let failingInput = AsyncThrowingStream<SampledFrame, Error> { continuation in
            samples.prefix(1).forEach { continuation.yield($0) }; continuation.finish()
        }
        _ = await checkThrowsAsync("tagger failure propagates instead of returning untagged frames") {
            for try await _ in TaggedFramePipeline(scorer: scorer, tagger: failingTagger).run(frames: failingInput, minimumFaceAreaFraction: 0.01) {}
        }
    })
}
