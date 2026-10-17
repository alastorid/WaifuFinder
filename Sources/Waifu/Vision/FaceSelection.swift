import Foundation

/// One face's capture quality, exactly as Vision reported it.
///
/// **`score` is capture quality, not attractiveness.** Apple's
/// `DetectFaceCaptureQualityRequest` documents the value as reflecting whether
/// a face is better lit, sharper, and more centrally positioned, and describes
/// it as intended for comparing a face against other captures of the *same*
/// face. It is kept per face rather than averaged away at the point of capture
/// because a mean over a group of three people hides the one who was captured
/// badly — the reduction to one number is the ranker's job, not the scorer's.
struct FaceCapture: Sendable, Equatable {
    /// Vision's 0…1 capture-quality value. Never rescaled.
    let score: Float
    /// The face's bounding box as a fraction of the image area, used to tell a
    /// face that carries the frame from one that is incidental background.
    let areaFraction: Float
}

/// The per-frame face summary `FrameScoring` reports.
struct FaceSelection: Sendable, Equatable {
    /// How many faces Vision found, before the area filter. Kept so "one face
    /// in the background" is distinguishable from "a crowd".
    let count: Int
    /// Area of the largest detected face as a fraction of the frame, or `nil`
    /// when no face was detected. Reported even when the largest face is below
    /// the significance threshold, so "how big was the biggest face" and "was
    /// there a subject" stay separable facts.
    let areaFraction: Float?
    /// Capture quality of the largest face that meets the significance
    /// threshold, or `nil` when none does — which is a real answer about the
    /// content, not a failure.
    let score: Float?
}

/// Which face's capture quality represents the frame, and why not an average.
enum FaceSelectionRule {
    /// Reduces Vision's per-face results to the one `FaceSelection` the frame
    /// reports.
    ///
    /// ## Why the largest significant face, and not a mean
    ///
    /// A mean over the faces in a frame asserts something false: that a
    /// portrait's score is the average of its subject and the stranger over
    /// their shoulder. Measured on the reference library, face areas run
    /// 0.0017…0.90 of the frame — wide enough that an unfiltered mean is
    /// dominated by whichever incidental face Vision happened to find. So faces
    /// below `minimumAreaFraction` are discarded first: an incidental
    /// background face must not drag a portrait's score.
    ///
    /// ## Why the largest, and not the worst
    ///
    /// Among the significant faces, the **largest** is the subject — the face
    /// the frame is about. The reference's `FaceCaptureAggregator` takes the
    /// *minimum* instead, because it answers a different question: which photo
    /// of a group should be *kept*, where the person who looks broken in the
    /// survivor is what matters. This scorer answers "how well was this
    /// frame's subject captured", and the subject is the largest significant
    /// face. The two reductions are deliberately different and live in
    /// different layers.
    ///
    /// ## Determinism
    ///
    /// Faces arrive from Vision in no guaranteed order. Areas are sorted
    /// descending, with exact ties broken by capture quality descending, so
    /// "the largest" is the same face on every run of the same frame.
    static func select(faces: [FaceCapture], minimumFaceAreaFraction: Float) -> FaceSelection {
        guard !faces.isEmpty else {
            return FaceSelection(count: 0, areaFraction: nil, score: nil)
        }
        let ordered = faces.sorted {
            $0.areaFraction != $1.areaFraction
                ? $0.areaFraction > $1.areaFraction
                : $0.score > $1.score
        }
        let largest = ordered[0]
        let significant = ordered.first { $0.areaFraction >= minimumFaceAreaFraction }
        return FaceSelection(count: faces.count,
                             areaFraction: largest.areaFraction,
                             score: significant?.score)
    }
}
