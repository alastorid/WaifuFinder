import Foundation

// The grid's order, filter and per-group cap, as values.
//
// The panel is independent signal checkboxes and a stepper; everything it decides about
// which frames appear, in what order, and how many of each group happens in
// `FrameArrangement`. So it is pinned here, with no window in the way, in two
// shapes: the *model* cases (one order with time → face → face definition → art → four
// independent rating probabilities) and the *pipeline* cases (filter, then order, then cap — and in
// that order, because each reversal changes the answer).
//
// Three properties are worth more than any individual case:
//
///   * **The precedence is fixed.** Time, face, face definition, art, then the four
///     independent rating probabilities,
///     with the frame identifier last. That is what makes every order total, so
///     two runs cannot produce different lists and a grid position means something.
//   * **A missing face sorts last whichever way the term points.** `faceScore` is
//     `nil` when no significant face was found — an absence, not a low score.
//   * **The cap is applied after the order**, which is what makes it "the best N of
//     each group" rather than "the first N the file happened to list".

/// The shipped grouping knobs, spelled out so a change to the default cannot move
/// an expectation silently.
private let groupingSettings = GroupingSettings(
    similarityThreshold: 0.35,
    maximumGroupSize: 60,
    faceWeight: 0.5,
    minimumFaceAreaFraction: 0.01
)

private func arrange(
    _ frames: [AnalyzedFrame],
    order: FrameOrder,
    faceFilter: FrameFaceFilter = .all,
    limits: FrameLimits = .none,
    perGroupTopCount: Int = 200,
    revealed: Set<FrameGroupID> = []
) -> FrameArrangement.Result {
    FrameArrangement.arrange(
        frames,
        order: order,
        faceFilter: faceFilter,
        limits: limits,
        perGroupTopCount: perGroupTopCount,
        revealedGroups: revealed,
        grouping: groupingSettings
    )
}

private func ids(_ frames: [AnalyzedFrame]) -> [String] { frames.map(\.id.raw) }

private func ratedFrame(id: String, time: Double, scores: [Float]) -> AnalyzedFrame {
    precondition(scores.count == TagRating.allCases.count)
    var frame = FrameFixture.make(id: id, time: time)
    var probabilities = [Float](repeating: 0, count: TagVocabulary.headCount)
    for (index, score) in scores.enumerated() { probabilities[index] = score }
    frame.tagProbabilities = TagProbabilityCodec.encode(probabilities)
    frame.tagRating = TagRating.allCases[scores.indices.max { scores[$0] < scores[$1] }!]
    return frame
}

/// A group identifier with a readable name; the arrangement never parses it.
private func group(_ name: String) -> FrameGroupID { FrameGroupID(raw: name) }

func registerFrameArrangementTests() {
    let suite = "frame arrangement"

    Registry.shared.add(suite: suite, TestCase(
        name: "each rating probability has an independent limit", knownBug: nil) {
        let untagged = FrameFixture.make(id: "f000001", time: 0)
        let general = ratedFrame(id: "f000002", time: 1, scores: [0.9, 0.1, 0.05, 0.05])
        let sensitive = ratedFrame(id: "f000003", time: 2, scores: [0.1, 0.85, 0.2, 0.1])
        let explicit = ratedFrame(id: "f000004", time: 3, scores: [0.05, 0.1, 0.2, 0.9])
        let frames = [untagged, general, sensitive, explicit]

        checkEqual(ids(arrange(frames, order: .chronological).visible),
                   ["f000001", "f000002", "f000003", "f000004"],
                   "an unlimited rating admits every frame, tagged or not")

        var limits = FrameLimits.none
        limits.setLimit(FrameLimit(low: 0.8, high: 1), for: .ratingSensitive)
        checkEqual(ids(arrange(frames, order: .chronological, limits: limits).visible),
                   ["f000003"], "the sensitive range admits the matching probability only")
        limits.setLimit(FrameLimit(low: 0.8, high: 1), for: .ratingExplicit)
        checkEqual(ids(arrange(frames, order: .chronological, limits: limits).visible),
                   [], "separate rating ranges compose independently")
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "traits require all selected tags before the group cap", knownBug: nil) {
        var matching = FrameFixture.make(id: "f000002", time: 2, artScore: 0.3)
        matching.topTags = ["looking at viewer", "smile"]
        var partial = FrameFixture.make(id: "f000001", time: 1, artScore: 0.9)
        partial.topTags = ["smile"]
        let filter = TagFilter(requiredTags: ["LOOKING_AT_VIEWER", "smile"])
        check(filter.admits(matching), "normalization and all-tags match")
        check(!filter.admits(partial), "one trait is insufficient")
        let result = FrameArrangement.arrange([partial, matching], order: .artOnly,
            perGroupTopCount: 1, grouping: groupingSettings, tagFilter: filter)
        checkEqual(ids(result.visible), ["f000002"], "trait filtering precedes ordering and cap")
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "rating probability sorts and limits keep missing data last", knownBug: nil) {
        let missing = FrameFixture.make(id: "f000001", time: 0, artScore: 0.9)
        let general = ratedFrame(id: "f000002", time: 1, scores: [0.9, 0.2, 0.1, 0.1])
        let explicit = ratedFrame(id: "f000003", time: 2, scores: [0.2, 0.1, 0.1, 0.9])
        var generalWithTraits = general
        var explicitWithTraits = explicit
        generalWithTraits.tagGeneralScore = 2
        explicitWithTraits.tagGeneralScore = 5
        let frames = [missing, explicitWithTraits, generalWithTraits]
        checkEqual(ids(arrange(frames, order: .only([.ratingGeneral])).visible),
                   ["f000002", "f000003", "f000001"], "general probability sorts high first, missing last")
        checkEqual(ids(arrange(frames, order: .only([.ratingExplicit])).visible),
                   ["f000003", "f000002", "f000001"], "explicit probability sorts independently")
        checkEqual(ids(arrange(frames, order: .only([.tagGeneralScore])).visible),
                   ["f000003", "f000002", "f000001"], "strongest trait confidence first")
        var ascending = FrameOrder.only([.tagGeneralScore])
        ascending.setAscending(true, for: .tagGeneralScore)
        checkEqual(ids(arrange(frames, order: ascending).visible),
                   ["f000002", "f000003", "f000001"], "missing remains last in ascending order")
        var limits = FrameLimits.none
        limits.setLimit(FrameLimit(low: 0.8, high: 1), for: .ratingExplicit)
        checkEqual(ids(arrange(frames, order: .chronological, limits: limits).visible),
                   ["f000003"], "explicit probability range excludes missing values")
    })

    // MARK: - One order, independent signals

    Registry.shared.add(suite: suite, TestCase(
        name: "the default order uses visual distance alone", knownBug: nil) {
        // Asserted as the projected term list, because that is exactly what the
        // sorter consumes — the precedence is not a convention the UI follows, it is
        // what the type produces.
        checkEqual(FrameOrder.defaultOrder.terms,
                   [SortTerm(key: .minimalVisualDistance, ascending: true)],
                   "visual distance is the only signal on by default")
        checkEqual(FrameSortKey.allCases,
                   [.minimalVisualDistance, .time, .tagRarity, .face, .definition, .art, .photoDistance, .trufor, .ratingGeneral, .ratingSensitive,
                    .ratingQuestionable, .ratingExplicit, .tagGeneralScore],
                   "control order matches sort precedence")
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "switching a signal off drops it and leaves the rest in order", knownBug: nil) {
        var order = FrameOrder.everything
        order.setEnabled(false, for: .minimalVisualDistance)
        order.setEnabled(false, for: .time)
        checkEqual(order.terms,
                   [SortTerm(key: .face, ascending: false), SortTerm(key: .definition, ascending: false), SortTerm(key: .art, ascending: false)],
                   "time off leaves face, definition, then art")

        order.setEnabled(false, for: .face)
        checkEqual(order.terms, [SortTerm(key: .definition, ascending: false), SortTerm(key: .art, ascending: false)], "definition remains independent of face")
        order.setEnabled(false, for: .definition)
        checkEqual(order.terms, [SortTerm(key: .art, ascending: false)], "and then art alone")
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "the last signal cannot be switched off", knownBug: nil) {
        // An order with nothing in it would be identifier order, which is not
        // something a checkbox asks for.
        var order = FrameOrder.only([.art])
        order.setEnabled(false, for: .art)
        checkEqual(order.enabledCount, 1, "art stays on")
        checkEqual(order.terms, [SortTerm(key: .art, ascending: false)], "and the order is unchanged")
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "a switched-off signal keeps its direction", knownBug: nil) {
        var order = FrameOrder.everything
        order.setAscending(true, for: .face)
        order.setEnabled(false, for: .face)
        order.setEnabled(true, for: .face)
        checkEqual(order.term(.face).ascending, true, "the direction survives the round trip")
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "every mutation is addressed by key and cannot corrupt the order", knownBug: nil) {
        // The crash class this replaced: a control's binding captures *how to write*,
        // and SwiftUI may evaluate it long after the value behind it changed shape. So
        // every write has to be total — no position that can go stale, no way to leave
        // the order empty, and no repeated application that drifts.
        var order = FrameOrder.everything
        let writers: [(String, (inout FrameOrder) -> Void)] = [
            ("switch time off", { $0.setEnabled(false, for: .time) }),
            ("switch time back on", { $0.setEnabled(true, for: .time) }),
            ("flip the face direction", { $0.toggleDirection(of: .face) }),
            ("point art descending", { $0.setAscending(false, for: .art) }),
            ("switch off the last remaining signal", { $0.setEnabled(false, for: .art) })
        ]
        for (name, write) in writers {
            for _ in 0..<3 { write(&order) }
            check(order.enabledCount >= 1, "\(name) left at least one signal switched on")
            checkEqual(order.terms.count, order.enabledCount,
                       "\(name) kept the term list and the switches in step")
            check(!order.terms.isEmpty, "\(name) left an order that can actually sort")
        }
    })

    // MARK: - Ordering behaviour

    Registry.shared.add(suite: suite, TestCase(
        name: "a later signal only breaks the earlier signal's ties", knownBug: nil) {
        // Three frames, same timestamp and the same face score: art is the only
        // signal that can decide, because it is the only one whose values differ.
        let frames = [
            FrameFixture.make(id: "f000001", time: 0, artScore: 0.5, faceScore: 0.9, faceAreaFraction: 0.2),
            FrameFixture.make(id: "f000002", time: 0, artScore: 0.9, faceScore: 0.9, faceAreaFraction: 0.2),
            FrameFixture.make(id: "f000003", time: 0, artScore: 0.1, faceScore: 0.9, faceAreaFraction: 0.2)
        ]
        checkEqual(ids(arrange(frames, order: .everything).visible),
                   ["f000002", "f000001", "f000003"],
                   "time and face tie, so art decides — best first")
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "time outranks face, and face outranks art", knownBug: nil) {
        // Deliberately crossed: the newest frame has the worst face and the worst
        // art. Under the full order it must still come last, and with time switched
        // off it must come first.
        let frames = [
            FrameFixture.make(id: "f000001", time: 0, artScore: 0.9, faceScore: 0.9, faceAreaFraction: 0.2),
            FrameFixture.make(id: "f000002", time: 10, artScore: 0.1, faceScore: 0.1, faceAreaFraction: 0.2)
        ]
        checkEqual(ids(arrange(frames, order: .everything).visible),
                   ["f000001", "f000002"], "time first: earliest leads")
        checkEqual(ids(arrange(frames, order: .faceThenArt).visible),
                   ["f000001", "f000002"], "face next: best face leads")
        checkEqual(ids(arrange(frames, order: .chronological).visible),
                   ["f000001", "f000002"], "time alone")

        var descendingTime = FrameOrder.everything
        descendingTime.setAscending(false, for: .time)
        checkEqual(ids(arrange(frames, order: descendingTime).visible),
                   ["f000002", "f000001"], "and the time signal can be reversed on its own")
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "a frame with no face sorts last whichever way the face term points", knownBug: nil) {
        let frames = [
            FrameFixture.make(id: "f000001", time: 0, artScore: 0.99, faceScore: nil),
            FrameFixture.make(id: "f000002", time: 0, artScore: 0.10, faceScore: 0.2, faceAreaFraction: 0.2),
            FrameFixture.make(id: "f000003", time: 0, artScore: 0.20, faceScore: 0.8, faceAreaFraction: 0.2)
        ]
        checkEqual(ids(arrange(frames, order: .faceOnly).visible),
                   ["f000003", "f000002", "f000001"], "best face first, no face last")

        var ascendingFace = FrameOrder.faceOnly
        ascendingFace.setAscending(true, for: .face)
        checkEqual(ids(arrange(frames, order: ascendingFace).visible),
                   ["f000002", "f000003", "f000001"], "worst face first, no face still last")
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "a face below the significance threshold is no face", knownBug: nil) {
        // The ranker's rule, not a second one: an order that ranked a frame on a face
        // the ranking ignores would contradict it.
        let frames = [
            FrameFixture.make(id: "f000001", time: 0, artScore: 0.9, faceScore: 0.95, faceAreaFraction: 0.001),
            FrameFixture.make(id: "f000002", time: 0, artScore: 0.1, faceScore: 0.10, faceAreaFraction: 0.30)
        ]
        checkEqual(ids(arrange(frames, order: .faceOnly).visible),
                   ["f000002", "f000001"],
                   "the sub-threshold face does not outrank the real one, despite the higher number")
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "every order is total, and loses nothing", knownBug: nil) {
        let frames = [
            FrameFixture.make(id: "f000004", time: 1, artScore: 0.5, faceScore: 0.5, faceAreaFraction: 0.2),
            FrameFixture.make(id: "f000001", time: 1, artScore: 0.5, faceScore: 0.5, faceAreaFraction: 0.2),
            FrameFixture.make(id: "f000003", time: 1, artScore: 0.5, faceScore: 0.5, faceAreaFraction: 0.2),
            FrameFixture.make(id: "f000002", time: 1, artScore: 0.5, faceScore: 0.5, faceAreaFraction: 0.2),
            FrameFixture.make(id: "f000005", time: 1, artScore: 0.5, faceScore: nil)
        ]
        let orders: [(String, FrameOrder)] = [
            ("everything", .everything),
            ("face then art", .faceThenArt),
            ("face only", .faceOnly),
            ("art only", .artOnly),
            ("chronological", .chronological)
        ]
        for (name, order) in orders {
            checkEqual(ids(arrange(frames, order: order).visible),
                       ids(arrange(frames, order: order).visible),
                       "\(name) is stable across two runs")
            checkEqual(Set(ids(arrange(frames, order: order).visible)), Set(ids(frames)),
                       "\(name) loses no frame")
            checkEqual(ids(arrange(frames, order: order).visible),
                       ["f000001", "f000002", "f000003", "f000004", "f000005"],
                       "\(name): with every field tied, only the identifier can decide")
        }
    })

    // MARK: - The per-group cap

    Registry.shared.add(suite: suite, TestCase(
        name: "the cap keeps the best N of each group, not the first N", knownBug: nil) {
        // One group whose best frame by art is also its *last* in time. With the cap
        // applied before the order, the wrong frame would survive — which is the only
        // reason the two steps are in the order they are.
        let g = group("g1")
        let frames = [
            FrameFixture.make(id: "f000001", time: 0, artScore: 0.2, groupID: g),
            FrameFixture.make(id: "f000002", time: 1, artScore: 0.9, groupID: g),
            FrameFixture.make(id: "f000003", time: 2, artScore: 0.5, groupID: g),
            FrameFixture.make(id: "f000004", time: 3, artScore: 0.1, groupID: g)
        ]

        let byArt = arrange(frames, order: .artOnly, perGroupTopCount: 1)
        checkEqual(ids(byArt.visible), ["f000002"], "the best art of the group survives")
        checkEqual(byArt.beyondCapByGroup[g], 3, "and the other three are counted, not lost")

        let byTime = arrange(frames, order: .chronological, perGroupTopCount: 1)
        checkEqual(ids(byTime.visible), ["f000001"], "a different order keeps a different frame")
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "the cap counts per group and never touches ungrouped frames", knownBug: nil) {
        let g1 = group("g1")
        let g2 = group("g2")
        let frames = [
            FrameFixture.make(id: "f000001", time: 0, artScore: 0.90, groupID: g1),
            FrameFixture.make(id: "f000002", time: 1, artScore: 0.80, groupID: g1),
            FrameFixture.make(id: "f000003", time: 2, artScore: 0.70, groupID: g1),
            FrameFixture.make(id: "f000004", time: 3, artScore: 0.60, groupID: g1),
            FrameFixture.make(id: "f000005", time: 4, artScore: 0.50, groupID: g2),
            FrameFixture.make(id: "f000006", time: 5, artScore: 0.40, groupID: g2),
            FrameFixture.make(id: "f000007", time: 6, artScore: 0.30, groupID: g2),
            FrameFixture.make(id: "f000008", time: 7, artScore: 0.95)
        ]

        let result = arrange(frames, order: .artOnly, perGroupTopCount: 2)
        checkEqual(ids(result.visible),
                   ["f000008", "f000001", "f000002", "f000005", "f000006"],
                   "the ungrouped frame is kept whole, and each group contributes its best two — in the global order")
        checkEqual(result.beyondCapByGroup[g1], 2, "g1 has two past the cap")
        checkEqual(result.beyondCapByGroup[g2], 1, "g2 has one")
        checkEqual(result.beyondCapByGroup.count, 2, "and nothing else is capped")
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "revealing a group shows all of it and keeps its count", knownBug: nil) {
        // The count has to survive the reveal, or the group would drop out of the
        // list that offers to hide it again.
        let g = group("g1")
        let frames = [
            FrameFixture.make(id: "f000001", time: 0, artScore: 0.9, groupID: g),
            FrameFixture.make(id: "f000002", time: 1, artScore: 0.8, groupID: g),
            FrameFixture.make(id: "f000003", time: 2, artScore: 0.7, groupID: g)
        ]

        let hidden = arrange(frames, order: .artOnly, perGroupTopCount: 1)
        checkEqual(ids(hidden.visible), ["f000001"], "capped to the best one")

        let revealed = arrange(frames, order: .artOnly, perGroupTopCount: 1, revealed: [g])
        checkEqual(ids(revealed.visible), ["f000001", "f000002", "f000003"], "revealed, all three show")
        checkEqual(revealed.beyondCapByGroup[g], 2, "and the count is still reported")
    })

    // MARK: - Limits

    Registry.shared.add(suite: suite, TestCase(
        name: "a limit keeps only the frames inside it", knownBug: nil) {
        let frames = [
            FrameFixture.make(id: "f000001", time: 0, artScore: 0.1),
            FrameFixture.make(id: "f000002", time: 1, artScore: 0.5),
            FrameFixture.make(id: "f000003", time: 2, artScore: 0.9)
        ]
        var limits = FrameLimits.none
        limits.art = FrameLimit(low: 0.4, high: 0.6)
        checkEqual(ids(arrange(frames, order: .everything, limits: limits).visible),
                   ["f000002"], "only the frame inside the art range")

        var oneSided = FrameLimits.none
        oneSided.art = FrameLimit(low: 0.4, high: nil)
        checkEqual(ids(arrange(frames, order: .everything, limits: oneSided).visible),
                   ["f000002", "f000003"], "a one-sided limit only constrains that side")

        var onTime = FrameLimits.none
        onTime.time = FrameLimit(low: nil, high: 1.5)
        checkEqual(ids(arrange(frames, order: .everything, limits: onTime).visible),
                   ["f000001", "f000002"], "and a time limit keeps the early frames")
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "an absent face cannot be inside a face range", knownBug: nil) {
        // `nil` is "no significant face was found" — an absence, not a zero — so it
        // cannot satisfy a face range however wide that range is. With no range set the
        // frame stays and simply sorts last.
        let frames = [
            FrameFixture.make(id: "f000001", time: 0, artScore: 0.5, faceScore: nil),
            FrameFixture.make(id: "f000002", time: 1, artScore: 0.5, faceScore: 0.8, faceAreaFraction: 0.2)
        ]
        var limits = FrameLimits.none
        limits.face = FrameLimit(low: 0, high: 1)
        checkEqual(ids(arrange(frames, order: .everything, limits: limits).visible),
                   ["f000002"], "the landscape is out once a face range is set")
        checkEqual(ids(arrange(frames, order: .everything, limits: .none).visible),
                   ["f000001", "f000002"], "and back in when the range is cleared")
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "the sliders' spans come from the data", knownBug: nil) {
        let frames = [
            FrameFixture.make(id: "f000001", time: 0, artScore: 0.2, faceScore: nil),
            FrameFixture.make(id: "f000002", time: 5, artScore: 0.5, faceScore: 0.30, faceAreaFraction: 0.2),
            FrameFixture.make(id: "f000003", time: 10, artScore: 0.9, faceScore: 0.95, faceAreaFraction: 0.2)
        ]
        let bounds = FrameLimitBounds.bounds(of: frames, grouping: groupingSettings)
        // Compared with a tolerance because the scores are `Float` and the spans are
        // `Double`: 0.2 as a Float is 0.20000000298…, and an exact comparison would be
        // asserting the conversion, not the span.
        checkClose(bounds.art.lowerBound, 0.2, tolerance: 1e-6, "art's span starts where the frames do")
        checkClose(bounds.art.upperBound, 0.9, tolerance: 1e-6, "and ends where they do")
        checkClose(bounds.time.lowerBound, 0.0, tolerance: 1e-9, "time's span is the video's")
        checkClose(bounds.time.upperBound, 10.0, tolerance: 1e-9, "from the first frame to the last")
        if let face = checkNotNil(bounds.face, "there are faces, so there is a face span") {
            checkClose(face.lowerBound, 0.30, tolerance: 1e-6, "measured over the frames with a face")
            checkClose(face.upperBound, 0.95, tolerance: 1e-6, "not over the faceless one")
        }

        // No faces on screen means nothing to put a face range around — the control is
        // absent rather than spanning 0…1 of nothing.
        let faceless = [FrameFixture.make(id: "f000001", artScore: 0.5, faceScore: nil)]
        checkNil(FrameLimitBounds.bounds(of: faceless, grouping: groupingSettings).face,
                 "a face span needs a face")
    })

    // MARK: - Rows

    Registry.shared.add(suite: suite, TestCase(
        name: "rows put each group on its own line, ungrouped last", knownBug: nil) {
        // Order of the lines is the order their best frame appears in, so the grid's
        // "best first" survives the regrouping into lines.
        let g1 = group("g1")
        let g2 = group("g2")
        let frames = [
            FrameFixture.make(id: "f000001", time: 0, artScore: 0.9, groupID: g1),
            FrameFixture.make(id: "f000002", time: 1, artScore: 0.4),
            FrameFixture.make(id: "f000003", time: 2, artScore: 0.8, groupID: g2),
            FrameFixture.make(id: "f000004", time: 3, artScore: 0.7, groupID: g1),
            FrameFixture.make(id: "f000005", time: 4, artScore: 0.2)
        ]
        let visible = arrange(frames, order: .artOnly).visible
        let rows = FrameArrangement.rows(of: visible)

        checkEqual(rows.count, 3, "two groups and one line of leftovers")
        checkEqual(rows[0].group, g1, "the best frame's group leads")
        checkEqual(ids(rows[0].frames), ["f000001", "f000004"], "and holds all of its members, in order")
        checkEqual(rows[1].group, g2, "then the next group")
        checkNil(rows[2].group, "the frames in no group are not a group")
        checkEqual(ids(rows[2].frames), ["f000002", "f000005"], "they get one line of their own, last")
        checkEqual(Set(ids(rows.flatMap(\.frames))), Set(ids(visible)),
                   "and every shown frame lands in exactly one line")
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "rows of an empty arrangement are empty", knownBug: nil) {
        checkEqual(FrameArrangement.rows(of: []).count, 0, "nothing shown, nothing to lay out")
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "face filter partitions frames by significant faces and defaults to all", knownBug: nil) {
        let frames = [
            FrameFixture.make(id: "f000001", time: 0, artScore: 0.99, faceScore: nil),
            FrameFixture.make(id: "f000002", time: 1, artScore: 0.50, faceScore: 0.9, faceAreaFraction: 0.2),
            FrameFixture.make(id: "f000003", time: 2, artScore: 0.40, faceScore: 0.9, faceAreaFraction: 0.001)
        ]
        checkEqual(ids(arrange(frames, order: .everything, faceFilter: .withFaces).visible),
                   ["f000002"],
                   "the landscape and the sub-threshold face are gone; the real face stays")
        checkEqual(ids(arrange(frames, order: .everything, faceFilter: .withoutFaces).visible),
                   ["f000001", "f000003"],
                   "without faces includes absent and sub-threshold faces")
        checkEqual(ids(arrange(frames, order: .everything).visible),
                   ["f000001", "f000002", "f000003"],
                   "and it is a filter, not a reordering: off, they come back")
    })
}
