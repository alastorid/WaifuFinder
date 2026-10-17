import Foundation

// Frame ranking: `RankMap` and the `FrameRanker` built on it.
//
// These two functions decide which member of a group of alike frames is offered
// as "the best capture", and they are pure value code — no Vision request, no
// decoded frame, no file on disk. Everything pinned here is arithmetic on ranks,
// so the whole contract can be exercised without an image. That matters because
// the two signals it blends are not commensurable: `artScore` is Apple's
// `overallScore`, measured over roughly −1…+1, and `faceScore` is capture
// quality over 0…1. Blending them directly would silently assert a relationship
// between their units that does not exist. So each is first reduced to its
// **rank within the set being ranked** — 1.0 for the best, 0.0 for the worst —
// and only then blended. Rank is unit-free, so the blend means what it says.
//
// Three properties are load-bearing and each is easy to lose silently:
//
//  * **Ties share a rank.** `RankMap` averages the ranks of a run of equal
//    scores rather than falling back on arrival order, so two equally-scored
//    frames are not separated by an accident of how they were fed in.
//  * **A frame with no significant face ranks on aesthetics alone.** Its
//    `faceRank` stays `nil` and `combined` equals its `artRank`, so a landscape
//    is not dragged around by a portrait's absence. A face below
//    `minimumFaceAreaFraction` is treated exactly like no face at all.
//  * **The final order is deterministic.** Equal `combined` values are broken by
//    frame identifier, so re-running the ranker does not reshuffle frames the
//    user is looking at. That is why the tie cases assert the exact identifier
//    order, not merely that both frames are present.
//
// The ranks produced here are exact binary fractions (0, 0.5, 0.25, 1.0), but
// they are still compared with `checkClose` at a tight tolerance rather than
// `==`, so a future change that introduced drift would fail rather than being
// rounded away by chance.
//
// Two cases below pin the contract that a frame's own face decides whether it
// has a `faceRank` — that a sub-threshold face is treated as no face, and that a
// frame without one must not borrow a rank from a neighbour. They were first
// written as failing `knownBug:` cases, and caught a real off-by-position defect:
// `FrameRanker.rank` built the face rank map from the *filtered* array of face
// scores but looked it up by *frame position*, so the ranks landed on the wrong
// frames whenever some — but not all — had a significant face. The blended
// ordering happened to survive it, which is why only a direct assertion on
// `faceRank` could see it. The defect is fixed; these cases now guard the fix.

/// `RankMap` keys its result by position in the input array, so a missing
/// position is `nil`. Mapping that to `NaN` lets `checkClose` report "no rank at
/// all" as plainly as it reports a wrong value, instead of trapping on `!`.
private func rankAt(_ ranks: [Int: Double], _ position: Int) -> Double {
    ranks[position] ?? .nan
}

/// A stand-in for any scored value. `RankMap` is generic, and this is the proof
/// that it looks only at the score closure and never at `AnalyzedFrame`'s other
/// fields — so it can be reused for whatever projection a caller has in hand.
private struct RankMapProbeItem: Equatable {
    let score: Float
    let label: String
}

func registerFrameRankerTests() {
    let suite = "frame ranking"

    // The 50/50 blend the app ships with: `GroupingSettings.default`'s weight and
    // threshold, spelled out rather than referenced so that a change to the
    // shipped default cannot silently move every expectation below.
    let defaultSettings = GroupingSettings(
        similarityThreshold: 0.35,
        maximumGroupSize: 60,
        faceWeight: 0.5,
        minimumFaceAreaFraction: 0.01
    )

    // MARK: - RankMap

    Registry.shared.add(suite: suite, TestCase(name: "RankMap gives 1.0 to best, 0.0 to worst", knownBug: nil) {
        // The whole ranking scheme in one case: a strictly increasing triple must
        // come back as the two endpoints and an exact midpoint, keyed by input
        // position rather than by value or identifier.
        let values: [Float] = [0.1, 0.5, 0.9]
        let ranks = RankMap(values) { $0 }
        checkClose(rankAt(ranks, 0), 0.0, tolerance: 1e-9, "0.1 is the worst of three, so it ranks 0.0")
        checkClose(rankAt(ranks, 1), 0.5, tolerance: 1e-9, "0.5 sits in the middle, so it ranks 0.5")
        checkClose(rankAt(ranks, 2), 1.0, tolerance: 1e-9, "0.9 is the best, so it ranks 1.0")
    })

    Registry.shared.add(suite: suite, TestCase(name: "RankMap handles ties by sharing average rank", knownBug: nil) {
        // [0.5, 0.5, 1.0]: the two 0.5s tie for worst. Sorted they occupy
        // positions 1 and 2, whose average is 1.5; divided by the span (2) that
        // is 0.75, and 1 - 0.75 = 0.25. Both tied frames must receive it.
        let values: [Float] = [0.5, 0.5, 1.0]
        let ranks = RankMap(values) { $0 }
        checkClose(rankAt(ranks, 2), 1.0, tolerance: 1e-9, "the unique best still ranks 1.0")
        checkClose(rankAt(ranks, 0), 0.25, tolerance: 1e-9, "the first of the tied-worst frames gets the shared midpoint rank")
        checkClose(rankAt(ranks, 1), 0.25, tolerance: 1e-9, "and so does the second")
    })

    Registry.shared.add(suite: suite, TestCase(name: "RankMap single element gets rank 1.0", knownBug: nil) {
        // One element is simultaneously the best and the worst. The `max(count-1, 1)`
        // guard in the source is what stops the only element from dividing by a
        // zero span; the documented answer is that it ranks best.
        let values: [Float] = [0.5]
        let ranks = RankMap(values) { $0 }
        checkEqual(ranks.count, 1, "one input produces one rank")
        checkClose(rankAt(ranks, 0), 1.0, tolerance: 1e-9, "a lone frame is the best it can be")
    })

    Registry.shared.add(suite: suite, TestCase(name: "RankMap empty array returns empty dict", knownBug: nil) {
        // No inputs, no ranks — and in particular not a crash on an empty span.
        let values: [Float] = []
        let ranks = RankMap(values) { $0 }
        check(ranks.isEmpty, "an empty input maps to an empty result, got \(ranks)")
    })

    Registry.shared.add(suite: suite, TestCase(name: "RankMap deterministic tie-break by original index", knownBug: nil) {
        // Two equal values tie for both best and worst. Their average span is
        // 0.5, so each ranks 1 - 0.5 = 0.5. The point of the case is that the
        // sort's index tie-break does not leak into the rank: both positions are
        // present and equal.
        let values: [Float] = [0.5, 0.5]
        let ranks = RankMap(values) { $0 }
        checkEqual(ranks.count, 2, "both tied positions are present")
        checkClose(rankAt(ranks, 0), 0.5, tolerance: 1e-9, "the first of two equal values ranks 0.5")
        checkClose(rankAt(ranks, 1), 0.5, tolerance: 1e-9, "and so does the second — no arbitrary separation")
    })

    Registry.shared.add(suite: suite, TestCase(name: "RankMap helper exposed and works with custom types", knownBug: nil) {
        // `RankMap` is generic over the item and takes a score projection, which
        // is what lets callers rank different views of the same frames. A type
        // that is not `AnalyzedFrame` proves the generality.
        let items = [
            RankMapProbeItem(score: 0.1, label: "a"),
            RankMapProbeItem(score: 0.9, label: "b"),
            RankMapProbeItem(score: 0.5, label: "c"),
        ]
        let ranks = RankMap(items) { $0.score }
        checkClose(rankAt(ranks, 0), 0.0, tolerance: 1e-9, "\"a\" is the worst, so 0.0")
        checkClose(rankAt(ranks, 2), 0.5, tolerance: 1e-9, "\"c\" is in the middle, so 0.5")
        checkClose(rankAt(ranks, 1), 1.0, tolerance: 1e-9, "\"b\" is the best, so 1.0")
    })

    // MARK: - FrameRanker

    Registry.shared.add(suite: suite, TestCase(name: "Frames without faceScore use art rank only", knownBug: nil) {
        // A group with no significant faces must behave exactly as the
        // aesthetics-only ranker did: everyone participates, `faceRank` is nil,
        // and `combined` is the art rank verbatim.
        let frames = [
            FrameFixture.make(id: "f000001", artScore: 0.9), // best art, no face
            FrameFixture.make(id: "f000002", artScore: 0.5), // worst art, no face
        ]
        let ranked = FrameRanker.rank(frames, settings: defaultSettings)
        checkEqual(ranked.count, 2, "both frames are ranked")
        guard ranked.count == 2 else { return }

        checkEqual(ranked[0].id.raw, "f000001", "the better-looking frame is first")
        checkNil(ranked[0].faceRank, "no face, so no face rank at all")
        checkEqual(ranked[0].combined, ranked[0].artRank, "with no face rank the blend is exactly the art rank")
        checkEqual(ranked[1].id.raw, "f000002", "the lesser frame follows")
        checkNil(ranked[1].faceRank, "no face, so no face rank at all")
        checkEqual(ranked[1].combined, ranked[1].artRank, "and its combined value tracks the art rank too")
    })

    Registry.shared.add(suite: suite, TestCase(name: "Face weight shifts order when face ranks differ from art ranks", knownBug: nil) {
        // Frame 1 is the best art and the worst face; frame 2 the reverse. At a
        // 50/50 weight both land on 0.5, so the identifiers must break the tie —
        // f000001 first, even though f000002 has the better face.
        let frames = [
            FrameFixture.make(id: "f000001", artScore: 0.9, faceScore: 0.1, faceAreaFraction: 0.1),
            FrameFixture.make(id: "f000002", artScore: 0.1, faceScore: 0.9, faceAreaFraction: 0.1),
        ]
        let ranked = FrameRanker.rank(frames, settings: defaultSettings)
        checkEqual(ranked.count, 2, "both frames are ranked")
        guard ranked.count == 2 else { return }

        checkClose(ranked[0].combined, 0.5, tolerance: 1e-9, "frame 1: art 1.0, face 0.0 → 0.5")
        checkClose(ranked[1].combined, 0.5, tolerance: 1e-9, "frame 2: art 0.0, face 1.0 → 0.5")
        // Disagreement is a genuine tie, and a tie is broken by identifier.
        checkEqual(ranked[0].id.raw, "f000001", "the lexicographically smaller id wins the tie")
        checkEqual(ranked[1].id.raw, "f000002", "the other follows")
    })

    Registry.shared.add(suite: suite, TestCase(name: "Face weight 0 means pure art ranking", knownBug: nil) {
        // Weight 0 must discount the face signal entirely, even though the frames
        // do have faces and a face rank is still computed for them.
        let frames = [
            FrameFixture.make(id: "f000001", artScore: 0.9, faceScore: 0.1, faceAreaFraction: 0.1),
            FrameFixture.make(id: "f000002", artScore: 0.1, faceScore: 0.9, faceAreaFraction: 0.1),
        ]
        let settings = GroupingSettings(
            similarityThreshold: 0.35,
            maximumGroupSize: 60,
            faceWeight: 0.0, // pure art
            minimumFaceAreaFraction: 0.01
        )
        let ranked = FrameRanker.rank(frames, settings: settings)
        checkEqual(ranked.count, 2, "both frames are ranked")
        guard ranked.count == 2 else { return }

        checkEqual(ranked[0].id.raw, "f000001", "the best art wins regardless of its poor face")
        checkEqual(ranked[1].id.raw, "f000002", "and the poor-art frame follows")
        checkEqual(ranked[0].combined, ranked[0].artRank, "weight 0 leaves combined equal to the art rank")
        checkEqual(ranked[1].combined, ranked[1].artRank, "for both frames")
    })

    Registry.shared.add(suite: suite, TestCase(name: "Face weight 1 means pure face ranking for frames with faces", knownBug: nil) {
        // The mirror of the previous case: weight 1 discards aesthetics and orders
        // by face capture quality alone.
        let frames = [
            FrameFixture.make(id: "f000001", artScore: 0.9, faceScore: 0.1, faceAreaFraction: 0.1),
            FrameFixture.make(id: "f000002", artScore: 0.1, faceScore: 0.9, faceAreaFraction: 0.1),
        ]
        let settings = GroupingSettings(
            similarityThreshold: 0.35,
            maximumGroupSize: 60,
            faceWeight: 1.0, // pure face
            minimumFaceAreaFraction: 0.01
        )
        let ranked = FrameRanker.rank(frames, settings: settings)
        checkEqual(ranked.count, 2, "both frames are ranked")
        guard ranked.count == 2 else { return }

        checkEqual(ranked[0].id.raw, "f000002", "the best face wins regardless of its poor art")
        checkEqual(ranked[1].id.raw, "f000001", "and the poor-face frame follows")
        checkClose(ranked[0].combined, 1.0, tolerance: 1e-9, "weight 1 leaves combined equal to the face rank")
        checkClose(ranked[1].combined, 0.0, tolerance: 1e-9, "for both frames")
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "Frames with faceAreaFraction below threshold use art rank only",
        knownBug: nil) {
        let frames = [
            FrameFixture.make(id: "f000001", artScore: 0.9, faceScore: 0.9, faceAreaFraction: 0.005), // below the 0.01 threshold
            FrameFixture.make(id: "f000002", artScore: 0.5, faceScore: 0.1, faceAreaFraction: 0.1),   // significant
        ]
        let ranked = FrameRanker.rank(frames, settings: defaultSettings)
        checkEqual(ranked.count, 2, "both frames are ranked")
        guard ranked.count == 2 else { return }

        // Frame 1's face is too small to count, so it ranks on art alone (1.0).
        // Frame 2 is the only significant face: best face (1.0) but worst art
        // (0.0), so combined = 0.5 * 0.0 + 0.5 * 1.0 = 0.5.
        checkEqual(ranked[0].id.raw, "f000001", "the art-only frame leads on aesthetics")
        checkNil(ranked[0].faceRank, "a face below the threshold contributes no face rank")
        checkEqual(ranked[1].id.raw, "f000002", "the frame with the significant face follows")
        checkNotNil(ranked[1].faceRank, "the significant face does contribute a face rank")
        checkClose(ranked[1].combined, 0.5, tolerance: 1e-9, "0.5 * artRank(0.0) + 0.5 * faceRank(1.0)")
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "Mixed group: some with faces, some without",
        knownBug: nil) {
        let frames = [
            FrameFixture.make(id: "f000001", artScore: 0.8, faceScore: nil),                        // no face, good art
            FrameFixture.make(id: "f000002", artScore: 0.6, faceScore: 0.9, faceAreaFraction: 0.1), // face, medium art
            FrameFixture.make(id: "f000003", artScore: 0.4, faceScore: 0.1, faceAreaFraction: 0.1), // face, poor art
        ]
        let ranked = FrameRanker.rank(frames, settings: defaultSettings)
        checkEqual(ranked.count, 3, "every frame is ranked, whether or not it has a face")
        guard ranked.count == 3 else { return }

        // Art ranks: f1 1.0, f2 0.5, f3 0.0. Face ranks (only f2, f3): f2 1.0,
        // f3 0.0. f1 combined = 1.0 (art only); f2 = 0.75; f3 = 0.0.
        checkEqual(ranked[0].id.raw, "f000001", "the good-art landscape is first")
        checkNil(ranked[0].faceRank, "a frame with no face must not borrow a face rank")
        checkEqual(ranked[1].id.raw, "f000002", "the blended face frame is second")
        checkNotNil(ranked[1].faceRank, "a frame with a face keeps its face rank")
        checkEqual(ranked[2].id.raw, "f000003", "the poor-art face frame is last")
        checkNotNil(ranked[2].faceRank, "a frame with a face keeps its face rank")
    })

    Registry.shared.add(suite: suite, TestCase(name: "All frames with faces, face weight shifts ranking", knownBug: nil) {
        // Every frame has a significant face, so the face map is complete. Art
        // and face disagree for f1 and f2, which tie at 0.75; f3 is worst on
        // both signals and stays last.
        let frames = [
            FrameFixture.make(id: "f000001", artScore: 0.9, faceScore: 0.5, faceAreaFraction: 0.1),
            FrameFixture.make(id: "f000002", artScore: 0.5, faceScore: 0.9, faceAreaFraction: 0.1),
            FrameFixture.make(id: "f000003", artScore: 0.1, faceScore: 0.1, faceAreaFraction: 0.1),
        ]
        let ranked = FrameRanker.rank(frames, settings: defaultSettings)
        checkEqual(ranked.count, 3, "all three frames are ranked")
        guard ranked.count == 3 else { return }

        // Art ranks: f1 1.0, f2 0.5, f3 0.0. Face ranks: f2 1.0, f1 0.5, f3 0.0.
        checkClose(ranked[0].combined, 0.75, tolerance: 1e-9, "f1: 0.5 * 1.0 + 0.5 * 0.5")
        checkClose(ranked[1].combined, 0.75, tolerance: 1e-9, "f2: 0.5 * 0.5 + 0.5 * 1.0 — a tie with f1")
        checkEqual(ranked[0].id.raw, "f000001", "the tie is broken by identifier")
        checkEqual(ranked[1].id.raw, "f000002", "so the larger id follows")
        checkEqual(ranked[2].id.raw, "f000003", "f3 is last on both signals, decisively")
    })

    Registry.shared.add(suite: suite, TestCase(name: "Ties broken deterministically by identifier", knownBug: nil) {
        // Identical on both signals, and fed in reverse identifier order, so
        // nothing but the id tie-break can produce this ordering. Without it the
        // list would inherit the input order and a group browser would reshuffle
        // between runs.
        let frames = [
            FrameFixture.make(id: "f000002", artScore: 0.5, faceScore: 0.5, faceAreaFraction: 0.1),
            FrameFixture.make(id: "f000001", artScore: 0.5, faceScore: 0.5, faceAreaFraction: 0.1),
        ]
        let ranked = FrameRanker.rank(frames, settings: defaultSettings)
        checkEqual(ranked.count, 2, "both frames are ranked")
        guard ranked.count == 2 else { return }

        checkEqual(ranked[0].combined, ranked[1].combined, "the two frames are genuinely tied")
        checkEqual(ranked[0].id.raw, "f000001", "the lexicographically smaller identifier comes first")
        checkEqual(ranked[1].id.raw, "f000002", "the larger one follows, whatever the input order was")
    })

    Registry.shared.add(suite: suite, TestCase(name: "Empty frames returns empty array", knownBug: nil) {
        // The early `guard` in `rank` must return before any map is built; an
        // empty group is a normal state (nothing alike was found), not an error.
        let ranked = FrameRanker.rank([], settings: defaultSettings)
        check(ranked.isEmpty, "no frames in, no ranks out, got \(ranked.count)")
    })
}
