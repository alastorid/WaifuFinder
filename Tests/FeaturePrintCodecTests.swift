import CoreGraphics
import Foundation
import Vision

// FeaturePrint persistence: the codec that turns a `FeaturePrintObservation` into
// bytes and back.
//
// Why this suite exists. A FeaturePrint is 768 floats and Waifu stores one per
// analysed frame, so the bytes on disk are the whole point of the codec: if a
// round trip is not *bit-exact*, every group formed from a restored library is
// subtly different from the one formed in the session that wrote it, and the
// difference is invisible until two frames that ought to match refuse to. The
// reference measured a worst distance error of `0.0` across 400 pairs, so the
// first case pins exactly that number rather than "close enough": there is no
// tolerance because the correctness claim has none.
//
// The other half of the contract is *failure*: a truncated or corrupt payload
// must decode to `nil`, not to plausible-looking nonsense. The zlib container is
// what buys that — it carries its own checksum, so a short read fails to inflate
// instead of yielding a vector with the wrong length. These cases feed the codec
// the three shapes corruption actually takes: a stream cut off after its header,
// random bytes behind the marker, and unmarked text (`"{}"`, which takes the
// "plain JSON from before compression existed" path and must still fail because
// an empty object is not a FeaturePrint).
//
// The observations are real, produced by Vision from a generated `CGImage`, not
// hand-built stand-ins: the codec has no public initialiser that takes bytes, so
// a fake could not exercise the real encode/decode path at all. On a host without
// Vision producing a print, every case that needs one reports SKIPPED rather than
// failing — an environment that cannot build the fixture has not disproved
// anything.

/// Runs `GenerateImageFeaturePrintRequest(.revision2)` over a 64×64 mid-grey
/// image with a coloured rectangle whose position changes with `variant`.
/// This is the same request `VisionFrameScorer.runFeaturePrint` issues in production.
///
/// Grey and tiny on purpose: the request only needs a decodable image, and a
/// solid colour keeps the fixture independent of any file on disk. Anything that
/// stops Vision producing a print — a `nil` context, a `nil` image, a thrown
/// `perform` — becomes `SkipTest`, because no assertion in this suite is about
/// Vision's ability to run.
func makeFeaturePrintObservation(_ variant: Int = 0) async throws -> FeaturePrintObservation {
    guard let context = CGContext(
        data: nil,
        width: 64,
        height: 64,
        bitsPerComponent: 8,
        bytesPerRow: 256,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else {
        throw SkipTest(reason: "could not create a CGContext for the FeaturePrint fixture")
    }
    context.setFillColor(CGColor(red: 0.5, green: 0.5, blue: 0.5, alpha: 1.0))
    context.fill(CGRect(x: 0, y: 0, width: 64, height: 64))
    context.setFillColor(CGColor(red: 0.8, green: 0.2, blue: 0.1, alpha: 1))
    context.fill(CGRect(x: variant * 5, y: variant * 3, width: 16, height: 24))
    guard let image = context.makeImage() else {
        throw SkipTest(reason: "could not rasterise the FeaturePrint fixture")
    }
    do {
        let request = GenerateImageFeaturePrintRequest(.revision2)
        return try await request.perform(on: image)
    } catch {
        throw SkipTest(reason: "Vision could not produce a FeaturePrintObservation: \(error)")
    }
}

func registerFeaturePrintCodecTests() {
    let suite = "featureprint codec"

    Registry.shared.add(suite: suite, TestCase(name: "FeaturePrintCodec round-trip distance bit-exact",
        knownBug: nil) {
        // A round trip through JSON has one job: leave the vector exactly as it
        // was. Two independent observations give a non-zero distance to compare,
        // so a codec that silently perturbed a float would show up as a non-zero
        // error here rather than being masked by comparing a vector with itself.
        let obs1 = try await makeFeaturePrintObservation()
        let obs2 = try await makeFeaturePrintObservation(1)

        let error = checkNotNil(
            FeaturePrintCodec.roundTripDistanceError(obs1, comparedWith: obs2),
            "round-trip encoding/decoding should succeed")
        guard let error else { return }
        checkEqual(error, 0.0, "distance must be bit-exact after round-trip; got error \(error)")
    })

    Registry.shared.add(suite: suite, TestCase(name: "FeaturePrintCodec compressed smaller than raw",
        knownBug: nil) {
        // Compression is not cosmetic: ~4.3 KB of JSON per frame over thousands of
        // frames is tens of megabytes. The marker byte is what makes the stored
        // blob self-describing, so both halves are asserted together — a payload
        // that shrank but lost its marker would decode as garbage JSON.
        let obs = try await makeFeaturePrintObservation()

        let encoded = checkNotNil(FeaturePrintCodec.encode(obs), "encoding a real observation must succeed")
        guard let encoded else { return }

        let rawJSON = try JSONEncoder().encode(obs)
        check(encoded.count < rawJSON.count,
              "compressed (\(encoded.count) B) should be smaller than raw JSON (\(rawJSON.count) B)")
        checkEqual(encoded.first, UInt8(1), "payload must start with marker byte 1")
    })

    Registry.shared.add(suite: suite, TestCase(name: "FeaturePrintCodec corrupt data returns nil",
        knownBug: nil) {
        // Three distinct corruption shapes, one expectation: `nil`.
        //
        //  * Marker + zlib header only: the stream is honest about its format and
        //    missing its tail. The container's checksum is what turns this into a
        //    clean failure instead of a half-length vector.
        //  * Marker + random bytes: the checksum fails.
        //  * `"{}"`: its first byte is not the marker, so it is read as uncompressed
        //    JSON — the pre-compression path — and must still fail, because an
        //    empty object cannot decode into a FeaturePrintObservation.
        let truncated = Data([1, 0x78, 0x9c]) // marker then a zlib header, no body
        checkNil(FeaturePrintCodec.decode(truncated), "truncated payload must decode to nil")

        let random = Data([1] + (0..<100).map { _ in UInt8.random(in: 0...255) })
        checkNil(FeaturePrintCodec.decode(random), "random payload must decode to nil")

        let fakeJSON = Data("{}".utf8)
        checkNil(FeaturePrintCodec.decode(fakeJSON), "invalid JSON must decode to nil")
    })

    Registry.shared.add(suite: suite, TestCase(name: "FeaturePrintCodec empty data returns nil",
        knownBug: nil) {
        // Zero bytes in must not become zero bytes out. A stored empty blob reads
        // back as "no value at all", which would leave a row that looks written
        // and never is; `nil` is the only honest answer at both ends.
        checkNil(FeaturePrintCodec.compress(Data()), "empty input must compress to nil")
        checkNil(FeaturePrintCodec.decompress(Data()), "empty input must decompress to nil")
    })

    Registry.shared.add(suite: suite, TestCase(name: "grouping cache decodes valid and corrupt prints once and releases completed rows",
        knownBug: nil) {
        let observation = try await makeFeaturePrintObservation()
        guard let payload = checkNotNil(FeaturePrintCodec.encode(observation), "real print encodes") else { return }
        var calls = 0
        var window = DecodedFeaturePrintWindow(payloads: [payload, Data([1, 0x78]), nil]) { data in
            calls += 1
            return FeaturePrintCodec.decode(data)
        }
        for _ in 0..<20 {
            checkNotNil(window.observation(at: 0), "valid print remains available")
            checkNil(window.observation(at: 1), "corrupt print is cached as unavailable")
            checkNil(window.observation(at: 2), "missing print remains unavailable")
        }
        checkEqual(calls, 2, "each nonmissing payload decoded exactly once")
        checkEqual(window.retainedCount, 3, "failed entries do not retry")
        for index in 0..<3 { window.discard(index) }
        checkEqual(window.retainedCount, 0, "completed rows release decoded storage")

        var bounded = DecodedFeaturePrintWindow(payloads: Array(repeating: payload, count: 100))
        for row in 0..<100 {
            for next in row..<min(row + 4, 100) { _ = bounded.observation(at: next) }
            check(bounded.retainedCount <= 4, "only active forward window retained")
            bounded.discard(row)
        }
        checkEqual(bounded.retainedCount, 0, "no whole-video decoded cache survives the pass")
    })

    Registry.shared.add(suite: suite, TestCase(name: "cached grouping matches Vision reference at threshold boundaries and across windows",
        knownBug: nil) {
        var prints: [Data] = []
        for variant in 0..<4 {
            let observation = try await makeFeaturePrintObservation(variant)
            guard let payload = checkNotNil(FeaturePrintCodec.encode(observation), "real print encodes") else { return }
            prints.append(payload)
        }
        guard let realDistance = checkNotNil(FrameGrouper.featurePrintDistance(prints[0], prints[1]), "real distance") else { return }
        let boundary = Float(realDistance)
        var frames = (0..<24).map { index in
            AnalyzedFrame(sampleIndex: index, timeSeconds: Double(index / 8) * 121 + Double(index % 8),
                          artScore: 0, faceScore: nil, faceCount: 0, faceAreaFraction: nil,
                          featurePrint: prints[index % prints.count])
        }
        frames.append(AnalyzedFrame(sampleIndex: 24, timeSeconds: 2, artScore: 0, faceScore: nil,
                                    faceCount: 0, faceAreaFraction: nil, featurePrint: Data([1, 0x78])))
        frames.reverse()
        for threshold in [Float(0), boundary.nextDown, boundary, boundary.nextUp, Float(0.5)] {
            var settings = GroupingSettings.default
            settings.similarityThreshold = threshold
            settings.maximumGroupSize = 3
            let cached = try FrameGrouper().groupCancellable(frames, settings: settings)
            let reference = try FrameGrouper(distance: { FrameGrouper.featurePrintDistance($0, $1) })
                .groupCancellable(frames, settings: settings)
            checkEqual(cached, reference, "same exact Vision comparisons, windows, splitting and IDs at \(threshold)")
            let incremental = IncrementalFrameGrouper(settings: settings)
            for frame in frames.sorted(by: { ($0.timeSeconds, $0.id.raw) < ($1.timeSeconds, $1.id.raw) }) {
                try incremental.append(frame)
            }
            checkEqual(try incremental.finish(), reference, "streaming retains Vision threshold decisions at \(threshold)")
        }
    })
}
