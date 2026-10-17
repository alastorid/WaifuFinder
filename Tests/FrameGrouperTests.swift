import Foundation

// `FrameGrouper` grouping — the clustering funnel, pinned on a controlled distance.
//
// The grouper is a funnel, not a matrix: `chronological window → FeaturePrint
// distance → union-find → split → order`. Every one of those stages has a
// detectable failure mode (a merge that should not happen, a split that jumps in
// time, an ordering that flips between runs), so this suite walks each stage with
// a distance it fully controls.
//
// ## Why the distance is injected
//
// Production's default initialiser decodes real Vision FeaturePrint payloads.
// A test cannot synthesise two observations at a chosen distance — Vision decides
// the distance, and it also needs a real image before it will produce anything.
// Testing the *threshold* behaviour against Vision would mean hoping some pair of
// synthesised images happens to straddle 0.5. So the clustering is exercised
// through `FrameGrouper(distance:)`, whose closure decodes a four-byte
// little-endian `Float` "position" and returns `|lhs − rhs|`; the value-level core
// is then exact and repeatable. The production default path — where a payload the
// codec cannot decode yields `nil` and therefore no merge — is asserted once,
// uninjected, at the end. Real FeaturePrint distances are covered separately,
// against real observations.
//
// The encoding is a raw `UInt32` bit pattern rather than JSON: it round-trips a
// `Float` bit-for-bit with no parser in the middle, so a distance assertion can
// never be quietly satisfied by a lossy decode.

// MARK: - Synthetic FeaturePrint payloads

/// Four little-endian bytes holding the `Float`'s bit pattern. Exact round-trip.
///
/// Written byte by byte rather than through JSON: a parser in the middle could
/// round a value and make a distance assertion pass for the wrong reason.
private func encodePosition(_ value: Float) -> Data {
    let bits = value.bitPattern.littleEndian
    return Data([
        UInt8(bits & 0xFF),
        UInt8((bits >> 8) & 0xFF),
        UInt8((bits >> 16) & 0xFF),
        UInt8((bits >> 24) & 0xFF),
    ])
}

/// The inverse of `encodePosition`. Returns `nil` for anything that is not exactly
/// four bytes, so a malformed payload can never be read as a real position.
private func decodePosition(_ data: Data) -> Float? {
    guard data.count == 4 else { return nil }
    let bytes = [UInt8](data)
    let bits = UInt32(bytes[0])
        | (UInt32(bytes[1]) << 8)
        | (UInt32(bytes[2]) << 16)
        | (UInt32(bytes[3]) << 24)
    return Float(bitPattern: bits)
}

/// The injected distance: absolute difference of the two positions, or `nil` when
/// either payload is not one of ours. Mirrors `FeaturePrintDistance`'s contract —
/// an undecodable payload produces no distance and therefore no merge.
private func positionDistance(_ lhs: Data, _ rhs: Data) -> Double? {
    guard let l = decodePosition(lhs), let r = decodePosition(rhs) else { return nil }
    return Double(abs(l - r))
}

private func makeFrame(id: String, time: Double, position: Float) -> AnalyzedFrame {
    FrameFixture.make(id: id, time: time, featurePrint: encodePosition(position))
}

private func settings(threshold: Float = 0.5, maxGroupSize: Int = 60) -> GroupingSettings {
    GroupingSettings(
        similarityThreshold: threshold,
        maximumGroupSize: maxGroupSize,
        faceWeight: 0.5,
        minimumFaceAreaFraction: 0.01
    )
}

/// A grouper whose clustering sees `positionDistance` instead of Vision.
private func grouper() -> FrameGrouper {
    FrameGrouper(distance: positionDistance)
}

func registerFrameGrouperTests() {
    // MARK: - Basic grouping

    Registry.shared.add(suite: "frame grouping", TestCase(
        name: "Frames within threshold form a group",
        knownBug: nil
    ) {
        let frames = [
            makeFrame(id: "f000001", time: 0, position: 0.0),
            makeFrame(id: "f000002", time: 1, position: 0.1),
            makeFrame(id: "f000003", time: 2, position: 0.2),
        ]

        let groups = grouper().group(frames, settings: settings())

        checkEqual(groups.count, 1, "three near-identical positions must be one group")
        checkEqual(groups[0].memberIDs.count, 3, "the group must hold all three frames")
        // Members are ascending regardless of the order clustering visited them.
        checkEqual(groups[0].memberIDs, ["f000001", "f000002", "f000003"],
                   "members must be ascending")
    })

    Registry.shared.add(suite: "frame grouping", TestCase(
        name: "Frames outside threshold do not merge",
        knownBug: nil
    ) {
        let frames = [
            makeFrame(id: "f000001", time: 0, position: 0.0),
            makeFrame(id: "f000002", time: 1, position: 1.0), // distance 1.0 > 0.5
        ]

        let groups = grouper().group(frames, settings: settings())

        // No component of two: a lone frame is not a group.
        check(groups.isEmpty, "a pair beyond the threshold must produce no group")
    })

    Registry.shared.add(suite: "frame grouping", TestCase(
        name: "Single-linkage chain merges across threshold",
        knownBug: nil
    ) {
        // A–B = 0.3 and B–C = 0.3 are within threshold, but A–C = 0.6 is not.
        // Single linkage chains them: A joins B, B joins C, so all three merge.
        let frames = [
            makeFrame(id: "f000001", time: 0, position: 0.0),
            makeFrame(id: "f000002", time: 1, position: 0.3),
            makeFrame(id: "f000003", time: 2, position: 0.6),
        ]

        let groups = grouper().group(frames, settings: settings())

        checkEqual(groups.count, 1, "a chain of within-threshold steps must be one group")
        checkEqual(groups[0].memberIDs.count, 3,
                   "the far ends must merge through the middle frame")
    })

    Registry.shared.add(suite: "frame grouping", TestCase(
        name: "Chronological window excludes distant frames",
        knownBug: nil
    ) {
        // Frames 1 and 3 are close in feature space (0.1) but 300 s apart, well
        // beyond the ±120 s window. Time is filtered first, so they are never
        // compared and the group cannot form.
        let frames = [
            makeFrame(id: "f000001", time: 0, position: 0.0),
            makeFrame(id: "f000002", time: 60, position: 1.0),  // far in feature space
            makeFrame(id: "f000003", time: 300, position: 0.1), // close to 1, far in time
        ]

        let groups = grouper().group(frames, settings: settings())

        check(groups.isEmpty,
              "frames 300 s apart must not merge however similar their vectors")
    })

    Registry.shared.add(suite: "frame grouping", TestCase(
        name: "Frames without featurePrint are excluded",
        knownBug: nil
    ) {
        let frames = [
            makeFrame(id: "f000001", time: 0, position: 0.0),
            FrameFixture.make(id: "f000002", time: 1, featurePrint: nil), // ungroupable
            makeFrame(id: "f000003", time: 2, position: 0.2),
        ]

        let groups = grouper().group(frames, settings: settings())

        // The vectorless frame can join nothing, but it must not stop the two
        // surviving frames from grouping with each other.
        checkEqual(groups.count, 1, "the two frames with vectors must still group")
        checkEqual(groups[0].memberIDs, ["f000001", "f000003"],
                   "the vectorless frame must be absent from the group")
    })

    Registry.shared.add(suite: "frame grouping", TestCase(
        name: "Lone frames are never returned as groups",
        knownBug: nil
    ) {
        let frames = [
            makeFrame(id: "f000001", time: 0, position: 0.0),
            makeFrame(id: "f000002", time: 1000, position: 10.0), // far in time and space
        ]

        let groups = grouper().group(frames, settings: settings())

        check(groups.isEmpty, "a component of one is not a group and must not be returned")
    })

    Registry.shared.add(suite: "frame grouping", TestCase(
        name: "a looser threshold groups what a tighter one keeps apart", knownBug: nil) {
        // The substance of the panel's grouping slider: the threshold *is* how much the
        // app groups, so raising it must merge and lowering it must split. These three
        // frames are 0.4 apart, so at 0.2 none of them is alike and at 0.5 single
        // linkage chains all three together.
        let frames = [
            makeFrame(id: "f000001", time: 0, position: 0.0),
            makeFrame(id: "f000002", time: 1, position: 0.4),
            makeFrame(id: "f000003", time: 2, position: 0.8)
        ]
        check(grouper().group(frames, settings: settings(threshold: 0.2)).isEmpty,
              "tightened, nothing is close enough to pair")
        let loose = grouper().group(frames, settings: settings(threshold: 0.5))
        checkEqual(loose.count, 1, "loosened, the three chain into one group")
        checkEqual(loose.first?.memberIDs, ["f000001", "f000002", "f000003"], "and it holds all of them")
    })

    // MARK: - Maximum group size splitting

    Registry.shared.add(suite: "frame grouping", TestCase(
        name: "Oversized group splits into contiguous chronological runs",
        knownBug: nil
    ) {
        // Ten frames one second apart, all at the same position → one component of
        // ten. `maximumGroupSize` 3 splits it into 3, 3, 3 and a trailing 1. The
        // lone trailing frame is dropped, so exactly three groups of three remain.
        let frames = (0..<10).map { i in
            makeFrame(id: String(format: "f%06d", i + 1), time: Double(i), position: 0.0)
        }

        let groups = grouper().group(frames, settings: settings(maxGroupSize: 3))

        checkEqual(groups.count, 3, "10 frames at max size 3 must leave three whole groups")
        checkEqual(groups[0].memberIDs, ["f000001", "f000002", "f000003"],
                   "first chunk")
        checkEqual(groups[1].memberIDs, ["f000004", "f000005", "f000006"],
                   "second chunk")
        checkEqual(groups[2].memberIDs, ["f000007", "f000008", "f000009"],
                   "third chunk; the trailing lone frame is dropped")
    })

    Registry.shared.add(suite: "frame grouping", TestCase(
        name: "Split preserves chronological order within each chunk",
        knownBug: nil
    ) {
        // Eight frames ten seconds apart at one position, split into 3 + 3 + 2.
        let frames = (0..<8).map { i in
            makeFrame(id: String(format: "f%06d", i + 1), time: Double(i * 10), position: 0.0)
        }
        let times = Dictionary(uniqueKeysWithValues: frames.map { ($0.id.raw, $0.timeSeconds) })

        let groups = grouper().group(frames, settings: settings(maxGroupSize: 3))

        checkEqual(groups.count, 3, "8 frames at max size 3 must leave 3 + 3 + 2")
        for group in groups {
            let memberTimes = group.memberIDs.map { times[$0] ?? -1 }
            checkEqual(memberTimes, memberTimes.sorted(),
                       "chunk \(group.id.raw) must be in chronological order")
        }
    })

    // MARK: - Ordering

    Registry.shared.add(suite: "frame grouping", TestCase(
        name: "Groups sorted by member count descending",
        knownBug: nil
    ) {
        var frames: [AnalyzedFrame] = []
        for i in 0..<4 {
            frames.append(makeFrame(id: String(format: "f%06d", i + 1),
                                    time: Double(i), position: 0.0))
        }
        for i in 0..<2 {
            frames.append(makeFrame(id: String(format: "f%06d", i + 100),
                                    time: 1000 + Double(i), position: 10.0))
        }

        let groups = grouper().group(frames, settings: settings())

        checkEqual(groups.count, 2, "two separate clusters must yield two groups")
        checkEqual(groups[0].memberIDs.count, 4, "the larger group must come first")
        checkEqual(groups[1].memberIDs.count, 2, "the smaller group must come second")
    })

    Registry.shared.add(suite: "frame grouping", TestCase(
        name: "Deterministic tie-break by group ID when counts equal",
        knownBug: nil
    ) {
        // Two clusters of two. Equal counts, so the order is decided by group ID
        // (the lexicographically smallest member), never by dictionary iteration.
        let frames = [
            makeFrame(id: "f000001", time: 0, position: 0.0),
            makeFrame(id: "f000002", time: 1, position: 0.1),
            makeFrame(id: "f000010", time: 100, position: 10.0),
            makeFrame(id: "f000011", time: 101, position: 10.1),
        ]

        let groups = grouper().group(frames, settings: settings())

        checkEqual(groups.count, 2, "two clusters of two must yield two groups")
        checkEqual(groups[0].id.raw, "f000001", "the smaller group ID must sort first")
        checkEqual(groups[1].id.raw, "f000010", "the larger group ID must sort second")
    })

    // MARK: - The injected distance helper

    Registry.shared.add(suite: "frame grouping", TestCase(
        name: "FeaturePrintDistance helper returns correct distance",
        knownBug: nil
    ) {
        // Every other case is only meaningful if the injected distance really is
        // the absolute difference of the encoded positions. Pin the round-trip
        // first.
        checkEqual(decodePosition(encodePosition(0.0)), Float?(0.0),
                   "encoding 0.0 must round-trip exactly")
        checkEqual(decodePosition(encodePosition(0.3)), Float?(0.3),
                   "encoding 0.3 must round-trip exactly")

        let distance = checkNotNil(positionDistance(encodePosition(0.0), encodePosition(0.3)),
                                   "two decodable positions must yield a distance")
        if let distance {
            checkClose(distance, 0.3, tolerance: 0.0001,
                       "the distance must be the absolute difference of the positions")
        }
    })

    Registry.shared.add(suite: "frame grouping", TestCase(
        name: "FeaturePrintDistance helper returns nil for invalid payload",
        knownBug: nil
    ) {
        // Production: a payload the codec cannot decode yields nil, so no merge.
        let production = FrameGrouper.featurePrintDistance(Data("invalid".utf8),
                                                           encodePosition(0.3))
        checkNil(production, "an undecodable FeaturePrint payload must return nil")

        // The injected distance mirrors that contract.
        checkNil(positionDistance(Data("invalid".utf8), encodePosition(0.3)),
                 "the injected distance must return nil for a non-position payload")
    })

    // MARK: - Real production path

    Registry.shared.add(suite: "frame grouping", TestCase(
        name: "Default grouper returns nothing for empty or vectorless input",
        knownBug: nil
    ) {
        // No injection here: the real initialiser and the real codec.
        let production = FrameGrouper()

        check(production.group([], settings: settings()).isEmpty,
              "no frames must yield no groups")

        let vectorless = [
            FrameFixture.make(id: "f000001", time: 0, featurePrint: nil),
            FrameFixture.make(id: "f000002", time: 1, featurePrint: nil),
        ]
        check(production.group(vectorless, settings: settings()).isEmpty,
              "frames without a featurePrint must never form a group")

        // Payloads that are present but not real FeaturePrints decode to nil, which
        // is the same shape of refusal: a payload that cannot be decoded must not
        // merge frames.
        let undecodable = [
            FrameFixture.make(id: "f000001", time: 0, featurePrint: encodePosition(0.0)),
            FrameFixture.make(id: "f000002", time: 1, featurePrint: encodePosition(0.0)),
        ]
        check(production.group(undecodable, settings: settings()).isEmpty,
              "an undecodable payload must not merge frames")
    })
}
