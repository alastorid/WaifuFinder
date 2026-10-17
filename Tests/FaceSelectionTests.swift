import Foundation

// Face selection: the pure reduction from Vision's per-face results to the one
// face a frame is scored on.
//
// `FaceSelectionRule.select` is the only place a frame's face summary is decided,
// and it makes two claims that a mean would quietly get wrong: an *incidental*
// face must not drag a portrait's score, and among the faces that survive the area
// filter the **largest** one is the subject. Both are pinned here as values — a
// count, an area fraction, a capture-quality score — so a future rewrite that
// "simplifies" the reduction to an average or a maximum is caught immediately.
// The rule reads only `FaceCapture` and returns a `FaceSelection`; no Vision
// request, image, or clip is involved, so these cases run everywhere.
//
// One deliberate divergence from the reference project is worth naming: its
// `FaceCaptureAggregator` takes the *minimum* score over a group, because it asks
// which photo of several people should be kept. This scorer asks how well the
// frame's subject was captured, so it takes the largest significant face's own
// score instead. The tie-break below states the determinism half of that contract:
// faces arrive from Vision unordered, and the same input must reduce the same way
// every run.

func registerFaceSelectionTests() {
    let suite = "face selection"

    // MARK: - no faces

    Registry.shared.add(suite: suite, TestCase(name: "no faces returns an empty selection",
        knownBug: nil) {
        // No face is a real answer about the content, not a failure: count 0 and
        // both reported values nil. `count` exists precisely so "nobody" and "one
        // face, none significant" stay distinguishable.
        let selection = FaceSelectionRule.select(faces: [], minimumFaceAreaFraction: 0.01)
        checkEqual(selection.count, 0, "no faces means a count of zero")
        checkNil(selection.areaFraction, "no face means no area to report")
        checkNil(selection.score, "no face means no capture quality")
    })

    // MARK: - one face

    Registry.shared.add(suite: suite, TestCase(name: "a single face below threshold reports count and area but a nil score",
        knownBug: nil) {
        // 0.005 is below the 0.01 significance threshold, so the face is found and
        // its area reported — how big the biggest face was is a fact independent of
        // whether it counted as a subject — but its score is withheld.
        let faces = [FaceCapture(score: 0.9, areaFraction: 0.005)]
        let selection = FaceSelectionRule.select(faces: faces, minimumFaceAreaFraction: 0.01)
        checkEqual(selection.count, 1, "the face was detected")
        checkEqual(selection.areaFraction, 0.005, "the largest face's area is still reported")
        checkNil(selection.score, "a sub-threshold face must not contribute a score")
    })

    Registry.shared.add(suite: suite, TestCase(name: "a single face above threshold returns its score",
        knownBug: nil) {
        let faces = [FaceCapture(score: 0.85, areaFraction: 0.05)]
        let selection = FaceSelectionRule.select(faces: faces, minimumFaceAreaFraction: 0.01)
        checkEqual(selection.count, 1, "the face was detected")
        checkEqual(selection.areaFraction, 0.05, "the face's area is reported")
        checkEqual(selection.score, 0.85, "the one significant face supplies the score")
    })

    // MARK: - several faces

    Registry.shared.add(suite: suite, TestCase(name: "several faces: the largest significant face wins, not the mean or the best score",
        knownBug: nil) {
        // THE rule. Three faces, all significant: subject 0.30/0.6, incidental
        // background 0.002/0.95, secondary 0.10/0.4. The answer must be the
        // SUBJECT's own score, 0.6. It is deliberately neither the mean of the
        // significant scores (0.65 for subject+secondary, or 0.65 for all three)
        // nor the maximum score (0.95), and neither of those numbers appears here,
        // so an accidental averaging or max reduction cannot pass.
        let faces = [
            FaceCapture(score: 0.6, areaFraction: 0.30),   // subject
            FaceCapture(score: 0.95, areaFraction: 0.002), // incidental background
            FaceCapture(score: 0.4, areaFraction: 0.10)    // secondary
        ]
        let selection = FaceSelectionRule.select(faces: faces, minimumFaceAreaFraction: 0.01)
        checkEqual(selection.count, 3, "every detected face is counted, in front of the filter too")
        checkEqual(selection.areaFraction, 0.30, "the largest face's area is reported")
        checkEqual(selection.score, 0.6,
                   "the largest significant face (area 0.30) supplies the score, not a mean (0.65) and not the max (0.95)")
    })

    Registry.shared.add(suite: suite, TestCase(name: "all faces below threshold: the largest area is reported, score nil",
        knownBug: nil) {
        // Neither face is significant, so no subject was found and `score` is nil —
        // again a fact about the frame, not a failure. The largest area survives so
        // "how big was the biggest face" is not lost to the filter.
        let faces = [
            FaceCapture(score: 0.9, areaFraction: 0.005),
            FaceCapture(score: 0.8, areaFraction: 0.003)
        ]
        let selection = FaceSelectionRule.select(faces: faces, minimumFaceAreaFraction: 0.01)
        checkEqual(selection.count, 2, "both faces were detected")
        checkEqual(selection.areaFraction, 0.005, "the largest area is reported even below threshold")
        checkNil(selection.score, "no significant face means no score")
    })

    // MARK: - determinism

    Registry.shared.add(suite: suite, TestCase(name: "a tie on area is broken by the higher score",
        knownBug: nil) {
        // Two faces of identical area (0.20). Vision gives no order, so the sort
        // must break the tie on score descending; otherwise "the largest face" is
        // ambiguous and the frame's score changes between runs.
        let faces = [
            FaceCapture(score: 0.7, areaFraction: 0.20),
            FaceCapture(score: 0.9, areaFraction: 0.20) // same area, higher score
        ]
        let selection = FaceSelectionRule.select(faces: faces, minimumFaceAreaFraction: 0.01)
        checkEqual(selection.areaFraction, 0.20, "the shared area is reported")
        checkEqual(selection.score, 0.9, "the tie on area is broken by the higher score")
    })

    Registry.shared.add(suite: suite, TestCase(name: "the same input always produces the same selection",
        knownBug: nil) {
        // The pairing of the two claims above: because the reduction sorts on area
        // (score as tie-break) rather than trusting Vision's arrival order, running
        // the identical input twice yields an equal `FaceSelection`. Mid-array here
        // is another area tie, so a stable sort is required, not incidental.
        let faces = [
            FaceCapture(score: 0.5, areaFraction: 0.15),
            FaceCapture(score: 0.8, areaFraction: 0.15), // tie on area
            FaceCapture(score: 0.3, areaFraction: 0.10)
        ]
        let first = FaceSelectionRule.select(faces: faces, minimumFaceAreaFraction: 0.01)
        let second = FaceSelectionRule.select(faces: faces, minimumFaceAreaFraction: 0.01)
        checkEqual(first, second, "the selection must be deterministic")
    })
}
