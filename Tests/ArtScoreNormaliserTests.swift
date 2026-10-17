import Foundation

// Art score normalisation.
//
// `ArtScoreNormaliser` is the one place a raw Vision aesthetics score
// (`overallScore`, roughly −1…+1) is stretched onto 0…1 for display. It is
// display-only — ranking and filtering read the raw score — so the entire
// contract is arithmetic and can be pinned directly, without a Vision request,
// a rendered frame, or Apple-native fixture generation. That matters: the transform exists precisely
// because a UI that assumed 0…1 would squash a real library into the top of a
// slider, and a case that could not see the mapping could not protect against
// that returning.
//
// The awkward input is the degenerate session in which every frame scored the
// same. Dividing by a zero range is undefined; production answers 0.5 for
// `normalise` (a single score has no position on the axis) but the minimum
// itself for `denormalise` (there is only one point to return to). Those are
// separate `guard`s in the source, so they are asserted separately here — a
// future edit that "simplified" one into the other would be caught.
//
// Everything below is checked against the *static* helpers as well as the
// instance methods, because the two are written as parallel copies: a drift
// between them is exactly the kind of silent divergence a value-level suite is
// for.

func registerArtScoreNormaliserTests() {
    let suite = "art score normaliser"

    Registry.shared.add(suite: suite, TestCase(name: "normalise maps min to 0, max to 1, midpoint to 0.5", knownBug: nil) {
        // The static helper takes the range explicitly, so the mapping is
        // checked without constructing an instance.
        let minimum: Float = -0.5
        let maximum: Float = 1.0
        checkClose(ArtScoreNormaliser.normalise(rawScore: minimum, minimum: minimum, maximum: maximum),
                   0.0, tolerance: 1e-9, "the observed minimum maps to 0")
        checkClose(ArtScoreNormaliser.normalise(rawScore: maximum, minimum: minimum, maximum: maximum),
                   1.0, tolerance: 1e-9, "the observed maximum maps to 1")
        // 0.25 is exactly halfway from −0.5 to 1.0.
        checkClose(ArtScoreNormaliser.normalise(rawScore: 0.25, minimum: minimum, maximum: maximum),
                   0.5, tolerance: 1e-9, "the midpoint maps to 0.5")
    })

    Registry.shared.add(suite: suite, TestCase(name: "normalise clamps below the minimum and above the maximum", knownBug: nil) {
        let minimum: Float = -0.5
        let maximum: Float = 1.0
        checkClose(ArtScoreNormaliser.normalise(rawScore: -1.0, minimum: minimum, maximum: maximum),
                   0.0, tolerance: 1e-9, "a score below the range clamps to 0")
        checkClose(ArtScoreNormaliser.normalise(rawScore: 2.0, minimum: minimum, maximum: maximum),
                   1.0, tolerance: 1e-9, "a score above the range clamps to 1")
    })

    Registry.shared.add(suite: suite, TestCase(name: "an equal min/max maps every score to 0.5", knownBug: nil) {
        // Degenerate session: a zero range has no interior, so any input —
        // inside or outside the single-value range — maps to the middle.
        checkClose(ArtScoreNormaliser.normalise(rawScore: 0.5, minimum: 0.5, maximum: 0.5),
                   0.5, tolerance: 1e-9, "the single observed value")
        checkClose(ArtScoreNormaliser.normalise(rawScore: 0.0, minimum: 0.5, maximum: 0.5),
                   0.5, tolerance: 1e-9, "a value away from the single observed value")
        // The instance path shares the same guard and must agree.
        let flat = ArtScoreNormaliser(observedMinimum: 0.5, observedMaximum: 0.5)
        checkClose(flat.normalise(0.0), 0.5, tolerance: 1e-9, "the instance agrees on a degenerate range")
    })

    Registry.shared.add(suite: suite, TestCase(name: "denormalise inverts normalise", knownBug: nil) {
        let minimum: Float = -0.8
        let maximum: Float = 0.9
        let raw: Float = 0.1
        let norm = ArtScoreNormaliser.normalise(rawScore: raw, minimum: minimum, maximum: maximum)
        let back = ArtScoreNormaliser.denormalise(normalised: norm, minimum: minimum, maximum: maximum)
        // Float carries ~7 significant digits; a round trip through Double does
        // not reproduce the raw value exactly, so this is a tolerance case.
        checkClose(Double(back), Double(raw), tolerance: 1e-4, "a round trip returns the raw score")
    })

    Registry.shared.add(suite: suite, TestCase(name: "denormalise clamps to the range and returns the minimum when it is flat", knownBug: nil) {
        let minimum: Float = -0.5
        let maximum: Float = 1.0
        checkEqual(ArtScoreNormaliser.denormalise(normalised: -0.5, minimum: minimum, maximum: maximum),
                   minimum, "a value below 0 clamps to the minimum")
        checkEqual(ArtScoreNormaliser.denormalise(normalised: 1.5, minimum: minimum, maximum: maximum),
                   maximum, "a value above 1 clamps to the maximum")
        // A flat range has exactly one point to return to — the minimum, not
        // 0.5 as `normalise` answers, because `denormalise` cannot invent a
        // second value the session never observed.
        checkEqual(ArtScoreNormaliser.denormalise(normalised: 0.3, minimum: 0.5, maximum: 0.5),
                   0.5, "a flat range returns its single value")
    })

    Registry.shared.add(suite: suite, TestCase(name: "instance methods agree with the static helpers", knownBug: nil) {
        let normaliser = ArtScoreNormaliser(observedMinimum: -0.5, observedMaximum: 1.0)
        let raw: Float = 0.25
        checkEqual(normaliser.normalise(raw),
                   ArtScoreNormaliser.normalise(rawScore: raw, minimum: -0.5, maximum: 1.0),
                   "normalise agrees with the static helper")
        let norm = 0.5
        checkEqual(normaliser.denormalise(norm),
                   ArtScoreNormaliser.denormalise(normalised: norm, minimum: -0.5, maximum: 1.0),
                   "denormalise agrees with the static helper")
    })

    Registry.shared.add(suite: suite, TestCase(name: "fromFrames spans the minimum and maximum artScore", knownBug: nil) {
        let frames = [
            FrameFixture.make(id: "f000000", time: 0, artScore: -0.3),
            FrameFixture.make(id: "f000001", time: 1, artScore: 0.8),
            FrameFixture.make(id: "f000002", time: 2, artScore: 0.1)
        ]
        let normaliser = ArtScoreNormaliser.fromFrames(frames)
        checkEqual(normaliser.observedMinimum, -0.3, "observedMinimum is the lowest artScore in the array")
        checkEqual(normaliser.observedMaximum, 0.8, "observedMaximum is the highest artScore in the array")
    })
}
