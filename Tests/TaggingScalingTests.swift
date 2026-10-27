import CoreGraphics
import Foundation

private actor TaggingPreparationGate {
    private var lanes = Set<Int>()
    var count: Int { lanes.count }

    func prepare(on lane: Int) async throws {
        lanes.insert(lane)
        let deadline = Date().addingTimeInterval(5)
        while lanes.count < 2 {
            try Task.checkCancellation()
            guard Date() < deadline else { throw TaggerError.preprocessing }
            try await Task.sleep(for: .milliseconds(2))
        }
    }
}

private struct PreparingTaggingLane: TaggerBackend {
    let id: Int
    let gate: TaggingPreparationGate

    func tag(batch: [CGImage]) async throws -> [TagResult] { throw TaggerError.preprocessing }

    func tag(sources: [TaggingImageSource]) async throws -> [TagResult] {
        try await gate.prepare(on: id)
        return sources.map { _ in TagResult(probabilities: [], rating: .general, topTags: [], generalScore: 0) }
    }
}

func registerTaggingScalingTests() {
    let suite = "Remote tagging scaling"
    Registry.shared.add(suite: suite, TestCase(name: "each server lane prepares its source batch concurrently", knownBug: nil) {
        let gate = TaggingPreparationGate()
        let backend = BalancedTaggerBackend(lanes: (0..<2).map {
            .init(backend: PreparingTaggingLane(id: $0, gate: gate), capacity: 1)
        })
        // These bytes deliberately are not an image. The scheduler must preserve
        // sources and let each selected lane own its preparation strategy.
        try await withThrowingTaskGroup(of: Int.self) { group in
            for _ in 0..<2 {
                group.addTask { try await backend.tag(sources: [.thumbnail(Data([0]))]).count }
            }
            for try await count in group { checkEqual(count, 1, "the lane returns one result per source") }
        }
        checkEqual(await gate.count, 2, "both server preparations overlap before either finishes")
    })

    Registry.shared.add(suite: suite, TestCase(name: "cached JPEGs upload verbatim with the same model input pixels", knownBug: nil) {
        let jpeg = try ThumbnailCache.encodeJPEG(ImageFixture.rgbGradient())
        let payload = try await RemoteTagBatchPayload.encode(sources: [.thumbnail(jpeg)])
        var reader = WireReader(payload)
        checkEqual(try reader.u8(), 1, "one frame")
        checkEqual(try reader.u8(), RemoteTagImageCodec.encoded.rawValue, "cached container codec")
        checkEqual(try reader.u16(), 0, "reserved flags")
        checkEqual(Data(try reader.raw()), jpeg, "the original JPEG travels without a new lossy encoding")
        check(reader.isAtEnd, "no additional image bytes")
        let decoded = try await RemoteTagBatchPayload.decode(payload)
        let local = try WDTaggerBackend.thumbnailImage(data: jpeg)
        checkEqual(try WDTaggerBackend.preprocessRGBA(decoded[0]), try WDTaggerBackend.preprocessRGBA(local),
                   "remote and previous local thumbnail decoding produce identical model pixels")
        check(payload.count < RemoteTagInputCompression.byteCount, "cached JPEGs avoid raw RGBA network expansion")
        print("      fixture upload: \(RemoteTagInputCompression.byteCount) raw RGBA bytes → \(payload.count) cached JPEG bytes")
    })

    Registry.shared.add(suite: suite, TestCase(name: "lossless compressed model inputs preserve every RGBA byte and reject invalid sizes", knownBug: nil) {
        let image = try ImageFixture.rgbGradient()
        let expected = try WDTaggerBackend.preprocessRGBA(image)
        let compressed = try RemoteTagInputCompression.encode(expected)
        checkEqual(try RemoteTagInputCompression.decode(compressed), expected, "LZ4 is byte exact")
        let payload = try await RemoteTagBatchPayload.encode(images: [image], codec: .rgba8LZ4)
        let decoded = try await RemoteTagBatchPayload.decode(payload)
        checkEqual(try WDTaggerBackend.preprocessRGBA(decoded[0]), expected, "the serving decoder preserves model pixels")
        check(compressed.count < expected.count, "a compressible model input uses fewer LAN bytes")
        let invalidInputs: [[UInt8]] = [[], [UInt8(2)], [0, 1], Array(compressed.prefix(compressed.count / 2)),
                        [UInt8](repeating: 0, count: RemoteTagInputCompression.byteCount + 2)]
        for invalid in invalidInputs {
            _ = checkThrows("malformed compressed image") { _ = try RemoteTagInputCompression.decode(invalid) }
        }
        _ = checkThrows("wrong model input size") { _ = try RemoteTagInputCompression.encode([1, 2]) }
        var state: UInt32 = 42
        let noise = (0..<RemoteTagInputCompression.byteCount).map { _ -> UInt8 in
            state = state &* 1_664_525 &+ 1_013_904_223
            return UInt8(truncatingIfNeeded: state >> 24)
        }
        let incompressible = try RemoteTagInputCompression.encode(noise)
        checkEqual(incompressible.first, 0, "incompressible input uses the raw fallback")
        checkEqual(try RemoteTagInputCompression.decode(incompressible), noise, "raw fallback is byte exact")
    })

    Registry.shared.add(suite: suite, TestCase(name: "older tagging capabilities decode without compressed input support", knownBug: nil) {
        let legacy = RemoteCapabilities.TaggerCapability(available: true, imageSize: 448, batchSize: 8,
            headCount: TagVocabulary.headCount, vocabularyFingerprint: 123)
        let data = try JSONEncoder().encode(legacy)
        check(!String(decoding: data, as: UTF8.self).contains("inputCodecs"), "older capability JSON has no codec field")
        let decoded = try JSONDecoder().decode(RemoteCapabilities.TaggerCapability.self, from: data)
        checkNil(decoded.inputCodecs, "old peers retain the existing RGBA8 fallback")
    })
}
