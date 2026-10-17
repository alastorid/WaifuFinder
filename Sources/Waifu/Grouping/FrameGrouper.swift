import CoreGraphics
import Foundation
import Vision

/// Clusters analysed frames into `FrameGroup`s using FeaturePrint distance,
/// single-linkage union-find, and a chronological window.
///
/// ## Why the funnel order matters
///
/// The design is a funnel, not a matrix:
///
///     chronological window → FeaturePrint distance → group
///
/// Restricting by time first is what makes grouping affordable: an all-pairs
/// comparison over 4,000 frames is ~8 million distance computations, while the
/// time-restricted set is a fraction of that (measured). It is also better for
/// correctness — two frames of the same composition sampled minutes apart are not
/// "alternate captures of the same shot", however similar they look.
///
/// ## Union-find, single linkage
///
/// Frames are chained: A merges with B, and B with C, so A and C end up in one
/// group even if A and C are further apart than the threshold. This is the right
/// behaviour for a slow pan or continuous action, where consecutive frames are
/// close and the ends of the shot are not.
///
/// The cost of single linkage is chaining: a dense sequence can merge into one
/// group far larger than any real shot. `maximumGroupSize` bounds that, and the
/// time window already excludes anything seconds apart.
///
/// ## Splitting, not truncating
///
/// When a component exceeds `maximumGroupSize`, it is split into contiguous
/// chronological runs rather than truncated. Each chunk remains a run of
/// neighbouring frames, so the browser never presents a group that jumps in time.
/// The original component is still one finding; splitting only affects presentation.
///
/// ## What is not grouped
///
/// - A frame whose `featurePrint` is `nil` is ungroupable and never placed in a
///   group. A missing vector is a degradation, not a failure: the frame still
///   scores, browses, exports and ranks on aesthetics and face quality — it simply
///   cannot join a group.
/// - A lone frame (a component of size 1) is not a group and is never returned:
///   there is nothing for a group browser to show.
///
/// Groups are returned **descending by member count** so the browser's most useful
/// rows — the ones with the most to choose between — appear first.
final class FrameGrouper: FrameGrouping, @unchecked Sendable {
    /// The distance applied between two encoded FeaturePrint payloads.
    ///
    /// Production uses decoded observations from the active forward window.
    /// The seam exists so the *clustering* — union-find, the
    /// chronological window, the split and the ordering — can be exercised on a
    /// distance a test controls, without asking Vision to synthesise observations
    /// that happen to straddle a threshold. Vision's own distance is covered
    /// separately, against real observations.
    private let distance: (@Sendable (Data, Data) -> Double?)?

    /// Creates a grouper with the default configuration.
    init() {
        self.distance = nil
    }

    /// Creates a grouper that clusters on an injected distance. Test-only; see
    /// `distance` for why the seam exists.
    init(distance: @escaping @Sendable (Data, Data) -> Double?) {
        self.distance = distance
    }

    /// The chronological window half-width, in seconds.
    ///
    /// Frames only compare against neighbours within ±this value. Matches the
    /// reference implementation's default of 120 seconds. This is a constant
    /// rather than a setting because `GroupingSettings` (the contract) does not
    /// expose a window parameter; the engine layer may inject a different value
    /// in the future if needed.
    private let windowSeconds: Double = 120

    /// Clusters frames into groups of two or more.
    ///
    /// - Parameters:
    ///   - frames: The frames to cluster, in any order. Only frames with a
    ///     non-`nil` `featurePrint` and a valid `timeSeconds` are eligible.
    ///   - settings: Threshold, window, and size-limit knobs.
    /// - Returns: Groups sorted by member count descending, then by group ID
    ///   ascending for deterministic tie-breaking.
    func group(_ frames: [AnalyzedFrame], settings: GroupingSettings) -> [FrameGroup] {
        (try? groupCancellable(frames, settings: settings)) ?? []
    }

    func groupCancellable(_ frames: [AnalyzedFrame], settings: GroupingSettings,
                          onProgress: @Sendable (Int, Int) -> Void = { _, _ in }) throws -> [FrameGroup] {
        try Task.checkCancellation()
        let config = settings.validated()
        let eligible = frames.filter { $0.featurePrint != nil && $0.timeSeconds.isFinite }
        guard eligible.count > 1 else { return [] }

        // Chronological, with the frame identifier as a deterministic tie-break:
        // two frames at the same timestamp must not be able to swap places
        // between runs.
        let ordered = eligible.sorted {
            ($0.timeSeconds, $0.id.raw) < ($1.timeSeconds, $1.id.raw)
        }
        let times = ordered.map { $0.timeSeconds }
        // Decode lazily once per frame, not twice per candidate pair. Removing
        // each completed row keeps only future observations in the 120s window,
        // rather than retaining uncompressed prints for the whole video.
        var decoded = DecodedFeaturePrintWindow(payloads: ordered.map(\.featurePrint))

        // Sliding window of candidate indices. `windowEnd` is the first index
        // outside the window, so the inner loop visits exactly the in-window
        // candidates — one pass, no repeated scanning.
        var parent = Array(0..<ordered.count)
        var rank = Array(repeating: 0, count: ordered.count)
        var componentSizes = Array(repeating: 1, count: ordered.count)
        var linkedGroups = 0

        func find(_ node: Int) -> Int {
            var root = node
            while parent[root] != root { root = parent[root] }
            // Path compression, so a chain does not make later merges quadratic.
            var walk = node
            while parent[walk] != root {
                let next = parent[walk]
                parent[walk] = root
                walk = next
            }
            return root
        }

        func union(_ a: Int, _ b: Int) {
            let rootA = find(a), rootB = find(b)
            if rootA == rootB { return }
            if componentSizes[rootA] > 1 { linkedGroups -= 1 }
            if componentSizes[rootB] > 1 { linkedGroups -= 1 }
            linkedGroups += 1
            let joinedSize = componentSizes[rootA] + componentSizes[rootB]
            // Union by rank.
            if rank[rootA] < rank[rootB] {
                parent[rootA] = rootB
                componentSizes[rootB] = joinedSize
            } else if rank[rootA] > rank[rootB] {
                parent[rootB] = rootA
                componentSizes[rootA] = joinedSize
            } else {
                parent[rootB] = rootA
                componentSizes[rootA] = joinedSize
                rank[rootA] += 1
            }
        }

        var windowEnd = 0
        var lastPulse = Date()
        for i in ordered.indices {
            try Task.checkCancellation()
            defer { decoded.discard(i) }
            if Date().timeIntervalSince(lastPulse) >= 0.25 {
                onProgress(i, linkedGroups)
                lastPulse = Date()
            }
            let centre = times[i]
            // Advance the right edge: include all frames within +windowSeconds.
            while windowEnd < ordered.count, times[windowEnd] - centre <= windowSeconds {
                windowEnd += 1
            }
            guard let lhsData = ordered[i].featurePrint else { continue }
            let lhsObservation = distance == nil ? decoded.observation(at: i) : nil
            if distance == nil, lhsObservation == nil { continue }
            // Compare forward only: every pair is visited once, from its earlier
            // member, so a half-matrix walk cannot both miss pairs and double-count.
            for j in (i + 1)..<max(windowEnd, i + 1) {
                if j % 256 == 0 { try Task.checkCancellation() }
                // An edge inside an existing component cannot change single-linkage
                // membership. Avoid repeated Vision distances within a joined shot.
                if find(i) == find(j) { continue }
                guard let rhsData = ordered[j].featurePrint else { continue }
                let separation: Double?
                if let distance {
                    separation = distance(lhsData, rhsData)
                } else if let lhsObservation, let rhsObservation = decoded.observation(at: j) {
                    separation = try? lhsObservation.distance(to: rhsObservation)
                } else {
                    separation = nil
                }
                guard let separation,
                      separation <= Double(config.similarityThreshold) else { continue }
                union(i, j)
            }
        }

        onProgress(ordered.count, linkedGroups)
        try Task.checkCancellation()
        // Collect components. `find` has already compressed the parent array, so
        // grouping by root is one pass.
        var components: [Int: [String]] = [:]
        for index in ordered.indices {
            components[find(index), default: []].append(ordered[index].id.raw)
        }

        return Self.finalize(components: Array(components.values),
                             dates: Dictionary(uniqueKeysWithValues: ordered.map { ($0.id.raw, $0.timeSeconds) }),
                             maximumSize: config.maximumGroupSize)
    }

    /// Shared presentation step for batch regrouping and the incremental pass.
    static func finalize(components: [[String]], dates: [String: Double], maximumSize: Int) -> [FrameGroup] {
        var groups = components
            .filter { $0.count > 1 }
            .map { members in
                FrameGroup(id: FrameGroupID(memberIDs: members), memberIDs: members)
            }

        // Split oversized groups into contiguous chronological runs.
        if maximumSize > 1 {
            groups = groups.flatMap { group in
                split(group, maximumSize: maximumSize, dates: dates)
            }
            // A split can leave a trailing run of one — 10 members cut into runs
            // of 3 leaves 3, 3, 3 and 1. A lone frame is not a group in any other
            // path this type takes (a component of size one is never returned), and
            // a "group" of one has nothing for the browser to choose between, so it
            // is dropped here too rather than presented as the only member.
            groups = groups.filter { $0.memberIDs.count > 1 }
        }

        // Largest groups first; tie-break by group ID for determinism.
        return groups.sorted {
            $0.memberIDs.count == $1.memberIDs.count
                ? $0.id.raw < $1.id.raw
                : $0.memberIDs.count > $1.memberIDs.count
        }
    }

    /// Ranks a set of frames best-first, on group-relative ranks.
    ///
    /// Delegates to `FrameRanker.rank` for the pure ranking logic.
    func rank(_ frames: [AnalyzedFrame], settings: GroupingSettings) -> [RankedFrame] {
        FrameRanker.rank(frames, settings: settings)
    }

    /// Computes the FeaturePrint distance between two encoded observations.
    ///
    /// Exposed as a pure helper so it is testable without Vision availability.
    /// Returns `nil` if either payload cannot be decoded or the distance fails.
    static func featurePrintDistance(_ lhs: Data, _ rhs: Data) -> Double? {
        guard let lhsObs = FeaturePrintCodec.decode(lhs),
              let rhsObs = FeaturePrintCodec.decode(rhs) else { return nil }
        return try? lhsObs.distance(to: rhsObs)
    }

    /// Splits an oversized group into contiguous chronological chunks.
    ///
    /// Applied after union-find so the *whole* event is still one finding — the
    /// caller can report "these 400 frames are one event" — while what is
    /// presented as a group stays browsable. Chunks are contiguous in capture
    /// order rather than arbitrarily truncated, so each chunk is still a run of
    /// neighbouring frames.
    ///
    /// - Parameters:
    ///   - group: The group to split.
    ///   - maximumSize: The maximum size of each chunk. Must be ≥ 2.
    ///   - dates: A map from frame identifier to its timestamp, used to sort
    ///     members chronologically before chunking.
    /// - Returns: An array of groups (chunks). If the group is not oversized,
    ///   returns the original group unchanged.
    private static func split(_ group: FrameGroup, maximumSize: Int,
                       dates: [String: Double]) -> [FrameGroup] {
        guard group.memberIDs.count > maximumSize, maximumSize >= 2 else { return [group] }
        let ordered = group.memberIDs.sorted {
            (dates[$0] ?? 0, $0) < (dates[$1] ?? 0, $1)
        }
        var chunks: [FrameGroup] = []
        var current: [String] = []
        for member in ordered {
            current.append(member)
            if current.count == maximumSize {
                chunks.append(FrameGroup(id: FrameGroupID(memberIDs: current), memberIDs: current))
                current = []
            }
        }
        if !current.isEmpty {
            chunks.append(FrameGroup(id: FrameGroupID(memberIDs: current), memberIDs: current))
        }
        return chunks
    }
}

/// Computes the FeaturePrint distance between two encoded observations.
///
/// Exposed as a pure static function so it is testable without Vision availability.
/// Returns `nil` if either payload cannot be decoded or the distance fails.
@inline(__always)
internal func FeaturePrintDistance(_ lhs: Data, _ rhs: Data) -> Double? {
    guard let lhsObs = FeaturePrintCodec.decode(lhs),
          let rhsObs = FeaturePrintCodec.decode(rhs) else { return nil }
    return try? lhsObs.distance(to: rhsObs)
}

/// Maps values to their group-relative ranks: 1.0 = best, 0.0 = worst.
///
/// Ties share the average of the ranks they span, so equally-scored frames are
/// not separated by an arbitrary ordering of their identifiers. The result is a
/// dictionary keyed by *position* in the input array, not by identifier, because
/// callers rank different projections of the same group.
///
/// Exposed as a pure static function so it is testable without Vision availability.
internal func RankMap<T>(_ items: [T], _ score: (T) -> Float) -> [Int: Double] {
    guard !items.isEmpty else { return [:] }
    let sortedIndices = items.indices.sorted {
        score(items[$0]) == score(items[$1]) ? $0 < $1 : score(items[$0]) > score(items[$1])
    }
    var ranks: [Int: Double] = [:]
    var position = 0
    let count = sortedIndices.count
    while position < count {
        // Collect the whole run of equal scores so they can share a rank.
        var end = position
        let current = score(items[sortedIndices[position]])
        while end + 1 < count, score(items[sortedIndices[end + 1]]) == current {
            end += 1
        }
        // Average rank across the tie, converted to 0-based: the best member
        // ranks 1.0 and the worst 0.0, and everything tied shares the midpoint.
        let average = (Double(position) + Double(end)) / 2 / Double(max(count - 1, 1))
        for offset in position...end {
            ranks[sortedIndices[offset]] = 1 - average
        }
        position = end + 1
    }
    return ranks
}
