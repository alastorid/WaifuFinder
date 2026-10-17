import Foundation

/// Pure helper that maps raw `overallScore` values onto 0…1 for **display only**
/// (e.g. chart axes, slider positions). Never used for ranking, filtering, or
/// any decision logic.
///
/// ## Why this exists
///
/// Apple's `CalculateImageAestheticsScoresRequest` returns `overallScore` over
/// roughly −0.996…+1.000 on real material (the reference observed −0.9399…+1.0000
/// on a 53 k library). It is *not* a probability, not clamped to 0…1, and its
/// distribution is skewed — most photos cluster in the upper half. A UI that
/// assumes 0…1 would put the bulk of a library in the top 20% of a slider.
///
/// `ArtScoreNormaliser` observes the session's actual scores and maps them
/// linearly: `min → 0`, `max → 1`. This is a *display transform only*. The
/// stored `artScore` is never modified; ranking uses raw scores (via
/// group-relative ranks in `FrameRanker`), and filtering uses raw scores.
///
/// ## What this is not
///
/// - Not a normalisation for comparison across sessions. Two sessions have
///   different observed ranges; a score of 0.5 in one is not equivalent to 0.5
///   in another.
/// - Not a "true" 0…1 probability. The raw score is not a probability; linearly
///   stretching it does not make it one.
/// - Not used by `FrameRanker`. The ranker reduces raw scores to ranks *within
///   the group*; it never sees normalised values.
struct ArtScoreNormaliser: Sendable {
    /// The observed minimum raw aesthetics score in the current session.
    let observedMinimum: Float

    /// The observed maximum raw aesthetics score in the current session.
    let observedMaximum: Float

    /// Creates a normaliser from the session's observed score range.
    ///
    /// - Parameters:
    ///   - observedMinimum: The lowest `artScore` seen in this analysis run.
    ///   - observedMaximum: The highest `artScore` seen in this analysis run.
    /// - Precondition: `observedMinimum < observedMaximum`. If equal, the
    ///   normaliser maps everything to 0.5 (a single-value session is degenerate).
    init(observedMinimum: Float, observedMaximum: Float) {
        self.observedMinimum = observedMinimum
        self.observedMaximum = observedMaximum
    }

    /// Creates a normaliser from a collection of analyzed frames.
    ///
    /// Convenience for the engine layer: pass `session.frames` and get a
    /// normaliser that spans exactly what the user sees.
    static func fromFrames(_ frames: [AnalyzedFrame]) -> ArtScoreNormaliser {
        let scores = frames.map(\.artScore)
        let min = scores.min() ?? 0
        let max = scores.max() ?? 1
        return ArtScoreNormaliser(observedMinimum: min, observedMaximum: max)
    }

    /// Maps a raw aesthetics score to 0…1 for display.
    ///
    /// - Parameter rawScore: The verbatim `overallScore` from Vision.
    /// - Returns: `(rawScore - min) / (max - min)`, clamped to 0…1. If
    ///   `min == max`, returns 0.5.
    /// - Note: This is **display only**. Do not use for ranking, thresholds, or
    ///   any logic that affects which frames are selected, grouped, or exported.
    func normalise(_ rawScore: Float) -> Double {
        let range = observedMaximum - observedMinimum
        guard range > 0 else { return 0.5 }
        let normalised = (rawScore - observedMinimum) / range
        return Double(max(0, min(1, normalised)))
    }

    /// Maps a normalised 0…1 value back to the raw score space (for display
    /// of slider handles, etc.). Inverse of `normalise(_:)`.
    func denormalise(_ normalised: Double) -> Float {
        let range = observedMaximum - observedMinimum
        guard range > 0 else { return observedMinimum }
        let clamped = max(0, min(1, normalised))
        return observedMinimum + Float(clamped) * range
    }
}

// MARK: - Testable internals

extension ArtScoreNormaliser {
    /// Internal static helper for tests: normalises a single score given explicit
    /// min/max. Exists so the pure logic can be tested without constructing a
    /// normaliser instance.
    static func normalise(rawScore: Float, minimum: Float, maximum: Float) -> Double {
        let range = maximum - minimum
        guard range > 0 else { return 0.5 }
        let normalised = (rawScore - minimum) / range
        return Double(Swift.max(0, Swift.min(1, normalised)))
    }

    /// Internal static helper for tests: denormalises a 0…1 value given explicit
    /// min/max.
    static func denormalise(normalised: Double, minimum: Float, maximum: Float) -> Float {
        let range = maximum - minimum
        guard range > 0 else { return minimum }
        let clamped = Swift.max(0, Swift.min(1, normalised))
        return minimum + Float(clamped) * range
    }
}