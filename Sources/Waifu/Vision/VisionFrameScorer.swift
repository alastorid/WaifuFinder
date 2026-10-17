import CoreGraphics
import Foundation
import Vision

/// Scores one frame with three concurrent Vision requests:
/// aesthetics (required), face capture quality (optional), FeaturePrint (optional).
/// Native capture-quality detections also supply the boxes for definition;
/// a landmark refinement runs only when a face was found.
///
/// ## Concurrency
///
/// The three models run via `async let` because they are independent passes over
/// pixels that are already decoded. Measured together in the reference they cost
/// ~7 ms wall clock versus ~12 ms sequential — the overlap is real because the
/// Neural Engine schedules them independently.
///
/// ## Failure policy
///
/// - **Aesthetics failure**: propagates. This is the primary observation; a frame
///   that cannot be scored aesthetically is not a frame the user can browse.
/// - **Face detection failure**: does **not** propagate. The frame still scores
///   and browses; `faceScore` and `faceAreaFraction` become `nil`, `faceCount`
///   becomes `0`. A landscape is a valid answer, not an error.
/// - **Landmark refinement failure**: only definition becomes `nil`; detected
///   faces and capture quality remain available.
/// - **FeaturePrint failure**: does **not** propagate. The frame still scores;
///   `featurePrint` becomes `nil`. A frame without a vector simply cannot join a
///   group — `FrameGrouper` treats `nil` as ungroupable.
///
/// This matches the reference's `analyzeWithFeaturePrint` and `faceCaptureQuality`
/// exactly: the primary score survives degradations.
final class VisionFrameScorer: FrameScoring, Sendable {
    /// Minimum face area (as a fraction of the frame) for a face to be considered
    /// the frame's subject. Faces below this are incidental background.
    ///
    /// The default comes from `GroupingSettings.default.minimumFaceAreaFraction`
    /// (0.01 = 1% of frame area). A face covering 1% of a 512×512 analysis frame
    /// is ~2600 px², roughly a 51×51 px face — large enough to be a deliberate
    /// subject, small enough to exclude distant crowd members.
    let minimumFaceAreaFraction: Float

    init(minimumFaceAreaFraction: Float = GroupingSettings.default.minimumFaceAreaFraction) {
        self.minimumFaceAreaFraction = minimumFaceAreaFraction
    }

    func score(_ frame: SampledFrame) async throws -> FrameScoreResult {
        try VisionAvailability.require()

        // Run all three requests concurrently. Cancellation is checked at each
        // `await` point so a cancelled run does not score frames that finished
        // after the signal.
        async let artScoreResult = runAesthetics(frame.image)
        async let faceCaptureResult = runFaceScores(frame.nativeImage ?? frame.image)
        async let featurePrintResult = runFeaturePrint(frame.image)

        // Aesthetics is required — if it throws, the whole frame fails.
        let artScore = try await artScoreResult

        // Face and FeaturePrint are best-effort: catch and log, then continue.
        let faceSelection: FaceSelection
        let definitionScore: Float?
        do {
            let scores = try await faceCaptureResult
            faceSelection = scores.selection
            definitionScore = scores.definition
        } catch {
            // Vision may fail to produce face observations on some frames (e.g.
            // corrupted decode, unusual colour space). This is not a frame failure.
            faceSelection = FaceSelection(count: 0, areaFraction: nil, score: nil)
            definitionScore = nil
        }

        let featurePrint: Data?
        do {
            featurePrint = try await featurePrintResult
        } catch {
            // FeaturePrint is a grouping aid only; its absence is a degradation,
            // not a failure. The reference logs at warn level and continues.
            featurePrint = nil
        }

        // Check cancellation one last time before constructing the result.
        // A frame whose aesthetics succeeded but was cancelled during the other
        // two should not be returned.
        if Task.isCancelled { throw CancellationError() }

        return FrameScoreResult(
            artScore: artScore,
            faceScore: faceSelection.score,
            faceCount: faceSelection.count,
            faceAreaFraction: faceSelection.areaFraction,
            featurePrint: featurePrint,
            faceDefinitionScore: definitionScore
        )
    }

    /// Runs `CalculateImageAestheticsScoresRequest(.revision1)` and returns the
    /// raw `overallScore`.
    ///
    /// **The score is stored verbatim, never rescaled.** On real material it
    /// measures roughly −0.996…+1.000 (the reference observed −0.9399…+1.0000 on
    /// a 53 k library). It is *not* a probability and must not be read as one.
    /// Normalisation for display (e.g. chart axes) happens in `ArtScoreNormaliser`,
    /// which is a pure helper explicitly documented as display-only.
    private func runAesthetics(_ image: CGImage) async throws -> Float {
        let request = CalculateImageAestheticsScoresRequest(.revision1)
        let observation = try await request.perform(on: image)
        return observation.overallScore
    }

    /// Runs `DetectFaceCaptureQualityRequest(.revision3)` and reduces the
    /// per-face results to one `FaceSelection` using `FaceSelectionRule.select`.
    ///
    /// The rule picks the **largest significant face** (area ≥
    /// `minimumFaceAreaFraction`), not a mean. A mean would let an incidental
    /// background face drag a portrait's score — the reference's
    /// `FaceCaptureAggregator` documents this explicitly with measured area
    /// fractions of 0.0017…0.90. The largest significant face is the subject;
    /// the smallest is the person who happens to be in the background.
    ///
    /// Faces are sorted by area descending, then by score descending to break
    /// exact ties, so "the largest" is the same face on every run of the same
    /// frame (determinism).
    private func runFaceScores(_ image: CGImage) async throws -> (selection: FaceSelection, definition: Float?) {
        let request = DetectFaceCaptureQualityRequest(.revision3)
        let observations = try await request.perform(on: image)

        let definition = try await FaceDefinition.evaluate(image, observations: observations)

        let faces = observations.compactMap { observation -> FaceCapture? in
            guard let quality = observation.captureQuality?.score else { return nil }
            let box = observation.boundingBox
            return FaceCapture(score: quality, areaFraction: Float(box.width * box.height))
        }

        let selected = FaceSelectionRule.select(faces: faces, minimumFaceAreaFraction: minimumFaceAreaFraction)
        return (FaceSelection(count: observations.count, areaFraction: selected.areaFraction, score: selected.score), definition)
    }

    /// Runs `GenerateImageFeaturePrintRequest(.revision2)` and encodes the
    /// observation via `FeaturePrintCodec.encode`.
    ///
    /// Returns `nil` on any failure (encoding, observation missing). The caller
    /// treats `nil` as a missing vector, not a failed frame.
    private func runFeaturePrint(_ image: CGImage) async throws -> Data? {
        let request = GenerateImageFeaturePrintRequest(.revision2)
        let observation = try await request.perform(on: image)
        return FeaturePrintCodec.encode(observation)
    }
}
