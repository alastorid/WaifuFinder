import Foundation

/// Ranks frames within a group on group-relative ranks, blending aesthetics and
/// face capture quality.
///
/// ## Why ranks, not raw scores
///
/// `artScore` is measured over roughly −0.996…+1.000 and `faceScore` over 0…1.
/// They are different instruments measuring different things, and a formula like
/// `artScore * 0.6 + faceScore * 0.4` silently asserts a relationship between
/// their units that does not exist. So each signal is first reduced to its
/// **rank within this group** — 1.0 for the best member, 0.0 for the worst — and
/// the ranks are blended. Rank is unit-free, so the blend means what it says:
/// how far up each signal places this frame *among its own competitors*.
///
/// ## Why the blend is capped at a weight, not a formula
///
/// The weight (`faceWeight`) is a single tunable, and it is applied only to
/// frames that have a significant face (`faceScore != nil`). A group with no
/// faces ranks on aesthetics alone, which is exactly what the aesthetics-only
/// implementation already did. A landscape frame is not dragged around by a
/// portrait's absence.
///
/// ## What it must never do
///
/// Nothing here decides deletion. The highest combined value is a better
/// candidate to *keep*, and the difference between rank 1 and rank 5 is not a
/// licence to remove anything. Deletion remains an explicit, confirmed human act
/// through the UI.
struct FrameRanker: Sendable {
    /// Whether a frame's face is significant enough to take part in ranking.
    ///
    /// **The one place this rule is written.** `rank` and the grid's `.faces`
    /// filter both ask "does this frame have a subject face?", and two answers to
    /// that question would let the filter offer a frame the ranking then ignores.
    ///
    /// For a frame the pipeline produced, `faceScore` is only ever set for a face
    /// that cleared the threshold — `FaceSelectionRule` does that reduction — so
    /// this agrees with `AnalyzedFrame.hasSignificantFace`. The area is re-checked
    /// because frames also arrive from elsewhere (fixtures, a future importer)
    /// where that guarantee does not hold.
    static func faceIsSignificant(_ frame: AnalyzedFrame, settings: GroupingSettings) -> Bool {
        guard frame.faceScore != nil else { return false }
        return (frame.faceAreaFraction ?? 0) >= settings.minimumFaceAreaFraction
    }

    /// Ranks a set of frames best-first, on group-relative ranks.
    ///
    /// - Parameters:
    ///   - frames: The frames to rank, in any order.
    ///   - settings: The `faceWeight` and `minimumFaceAreaFraction` to use.
    /// - Returns: `RankedFrame` array sorted by `combined` descending. Ties are
    ///   broken deterministically by frame identifier.
    static func rank(_ frames: [AnalyzedFrame], settings: GroupingSettings) -> [RankedFrame] {
        let config = settings.validated()
        guard !frames.isEmpty else { return [] }

        // Aesthetics rank: every frame has an artScore, so all participate.
        let aestheticsRanks = RankMap(frames) { $0.artScore }

        // Face quality: only frames with a significant face participate.
        // A frame without a faceScore is absent from this map rather than
        // ranked last, so a landscape group is not dragged around by a
        // portrait's absence.
        let faceValues = frames.map { frame -> Float? in
            FrameRanker.faceIsSignificant(frame, settings: config) ? frame.faceScore : nil
        }
        let facePositions = frames.indices.filter { faceValues[$0] != nil }
        // `RankMap` keys its result by position in the array it is given, and the
        // array here is the *filtered* one. Re-key onto the original frame indices
        // before the lookup below: otherwise a frame without a significant face
        // reads another frame's rank, and the frames that do have one read a
        // neighbour's or nothing. The blended *order* often survives that mistake,
        // which is exactly why it has to be pinned.
        let ranksOverFaces = facePositions.isEmpty
            ? [Int: Double]()
            : RankMap(facePositions.map { faceValues[$0]! }) { $0 }
        var faceRanks: [Int: Double] = [:]
        for (offset, index) in facePositions.enumerated() {
            if let rank = ranksOverFaces[offset] { faceRanks[index] = rank }
        }

        // Blend the two ranks. For frames without a significant face, the
        // combined score is just the aesthetics rank.
        let blended = frames.indices.map { index -> RankedFrame in
            let id = frames[index].id
            let aesthetics = aestheticsRanks[index] ?? 0
            guard let faceRank = faceRanks[index] else {
                return RankedFrame(
                    id: id,
                    artRank: aesthetics,
                    faceRank: nil,
                    combined: aesthetics
                )
            }
            let weight = config.faceWeight
            let combined = (1 - weight) * aesthetics + weight * faceRank
            return RankedFrame(
                id: id,
                artRank: aesthetics,
                faceRank: faceRank,
                combined: combined
            )
        }

        // Sort by combined descending; ties broken by identifier for determinism.
        return blended.sorted {
            $0.combined == $1.combined ? $0.id.raw < $1.id.raw : $0.combined > $1.combined
        }
    }
}