import Foundation

/// An immutable snapshot lets all deletion filtering, sorting and geometry work
/// run away from the UI actor. PhotoKit asset lookup is thread-safe.
enum PhotosDeletionLayout {
    struct Input: Sendable {
        let frames: [AnalyzedFrame]
        let source: URL
        let pendingAssets: Set<String>
        let favoriteAssets: Set<String>
        let favoriteFilter: FrameFavoriteFilter
        let rarity: [FrameID: Float]
        let distance: [FrameID: Float]
        let order: FrameOrder
        let faceFilter: FrameFaceFilter
        let limits: FrameLimits
        let settings: WaifuSettings
        let revealedGroups: Set<FrameGroupID>
        let groupingEnabled: Bool
        let tagFilter: TagFilter
        let vocabulary: TagVocabulary?
        let groups: [FrameGroup]
        let defersVisualOrder: Bool
    }
    struct Output: Sendable {
        let base: [AnalyzedFrame]
        let displayed: [AnalyzedFrame]
        let rows: [FrameRow]
        let displayOrder: [AnalyzedFrame]
        let rowPosition: [FrameID: (row: Int, position: Int)]
        let rowHeightRatios: [String: Double]
        let summaryFrames: [AnalyzedFrame]
        let topTags: [FrameID: [String]]
        let beyond: [FrameGroupID: Int]
        let capped: [CappedGroupSummary]
        let visualInput: [VisualOrderInput]?
        let visualIDs: [FrameID]
    }

    static func compute(_ input: Input,
                        assetIdentifier: (@Sendable (URL, Double) -> String?)? = nil) throws -> Output {
        let grouping = input.settings.grouping.validated()
        // Take one album snapshot. Repeatedly resolving through PhotosMedia
        // would contend with visible cells for its lock while a large album sorts.
        let album = assetIdentifier == nil ? PhotosMedia.shared.album(input.source) : nil
        let identifiers = album?.items.map { $0.asset.localIdentifier } ?? []
        let durations = album?.items.map(\.duration) ?? []
        func identifier(for seconds: Double) -> String? {
            if let assetIdentifier { return assetIdentifier(input.source, seconds) }
            guard let album, let index = PhotosMedia.itemIndex(offsets: album.offsets, seconds: seconds),
                  seconds >= album.offsets[index], seconds < album.offsets[index] + durations[index] else { return nil }
            return identifiers[index]
        }
        var normalized = input.tagFilter
        normalized.requiredTags = Set(normalized.requiredTags.map(TagVocabulary.normalized))
        normalized.excludedTags = Set(normalized.excludedTags.map(TagVocabulary.normalized))
        normalized.orGroups = normalized.orGroups.map { TagOrGroup(tags: Set($0.tags.map(TagVocabulary.normalized))) }.filter { !$0.tags.isEmpty }
        normalized.threshold = input.settings.taggerThreshold
        let arrangementFilter = TagFilter(requiredTags: input.tagFilter.requiredTags,
            excludedTags: input.tagFilter.excludedTags, threshold: input.settings.taggerThreshold,
            selectionOrder: input.tagFilter.selectionOrder, exclusionOrder: input.tagFilter.exclusionOrder,
            scoreSortTag: input.tagFilter.scoreSortTag, scoreSortDescending: input.tagFilter.scoreSortDescending,
            orGroups: input.tagFilter.orGroups)
        var seen = Set<FrameID>()
        var enriched: [AnalyzedFrame] = []
        enriched.reserveCapacity(input.frames.count)
        for (index, original) in input.frames.enumerated() {
            if index % 256 == 0 { try Task.checkCancellation() }
            guard seen.insert(original.id).inserted else { continue }
            var frame = original
            frame.photoDistance = input.distance[frame.id]
            frame.tagRarity = input.rarity[frame.id]
            enriched.append(frame)
        }
        func candidates(_ filter: TagFilter) throws -> [AnalyzedFrame] {
            var result: [AnalyzedFrame] = []
            for (index, frame) in enriched.enumerated() {
                if index % 256 == 0 { try Task.checkCancellation() }
                guard input.faceFilter.admits(frame, grouping: grouping),
                      input.limits.admits(frame, grouping: grouping),
                      filter.admits(frame, vocabulary: input.vocabulary) else { continue }
                if !input.pendingAssets.isEmpty || input.favoriteFilter != .all {
                    guard let identifier = identifier(for: frame.timeSeconds),
                          !input.pendingAssets.contains(identifier),
                          input.favoriteFilter.admits(input.favoriteAssets.contains(identifier)) else { continue }
                }
                result.append(frame)
            }
            return result
        }
        func arrange(_ filter: TagFilter, matching: TagFilter) throws -> FrameArrangement.Result {
            FrameArrangement.arrange(enriched, order: input.order, faceFilter: input.faceFilter,
                limits: input.limits, perGroupTopCount: input.settings.revealAllGroups ? Int.max : input.settings.perGroupFrameCount,
                revealedGroups: input.revealedGroups, grouping: grouping, groupingEnabled: input.groupingEnabled,
                tagFilter: filter, vocabulary: input.vocabulary, filteredCandidates: try candidates(matching),
                applyingVisualOrder: false, tagTopCount: input.settings.tagTopCount)
        }
        let arrangement = try arrange(arrangementFilter, matching: normalized)
        var summary = arrangement.visible
        if normalized.orGroups.contains(where: { !$0.tags.isEmpty }) {
            var summaryFilter = arrangementFilter
            summaryFilter.orGroups = []
            var summaryMatching = normalized
            summaryMatching.orGroups = []
            summary = try arrange(summaryFilter, matching: summaryMatching).visible
        }
        let base = arrangement.visible
        let displayed: [AnalyzedFrame]
        var visualIDs: [FrameID] = []
        var visualInput: [VisualOrderInput]?
        if input.order.minimalVisualDistance.enabled && input.defersVisualOrder {
            displayed = base.sorted { $0.id.sampleIndex < $1.id.sampleIndex }
        } else if input.order.minimalVisualDistance.enabled {
            let sorted: [AnalyzedFrame]
            do { sorted = try MinimalVisualDistanceOrder.sorted(base) }
            catch is CancellationError { throw CancellationError() }
            catch { sorted = base }
            visualIDs = sorted.map(\.id)
            visualInput = base.map { VisualOrderInput(id: $0.id, featurePrint: $0.featurePrint) }
            displayed = input.order.minimalVisualDistance.ascending ? sorted : Array(sorted.reversed())
        } else { displayed = base }
        try Task.checkCancellation()
        let rows = FrameArrangement.rows(of: displayed)
        let displayOrder = input.groupingEnabled && !input.order.minimalVisualDistance.enabled ? rows.flatMap(\.frames) : displayed
        var positions: [FrameID: (row: Int, position: Int)] = [:], ratios: [String: Double] = [:]
        for (row, value) in rows.enumerated() {
            if row % 256 == 0 { try Task.checkCancellation() }
            var heightRatio = 0.0
            for (position, frame) in value.frames.enumerated() {
                positions[frame.id] = (row: row, position: position)
                let aspect = frame.thumbnailAspectRatio ?? 16.0 / 9
                heightRatio = max(heightRatio, 1 / (aspect.isFinite && aspect > 0 ? aspect : 16.0 / 9))
            }
            ratios[value.id] = heightRatio
        }
        let byID = Dictionary(uniqueKeysWithValues: enriched.map { ($0.id.raw, $0) })
        let earliest = Dictionary(uniqueKeysWithValues: input.groups.compactMap { group -> (FrameGroupID, Double)? in
            guard arrangement.beyondCapByGroup[group.id] != nil,
                  let time = group.memberIDs.compactMap({ byID[$0]?.timeSeconds }).min() else { return nil }
            return (group.id, time)
        })
        var capped: [CappedGroupSummary] = []
        for (id, count) in arrangement.beyondCapByGroup {
            let label: String
            if let time = earliest[id] { label = timecode(time) } else { label = id.raw }
            capped.append(CappedGroupSummary(id: id, count: count, label: label))
        }
        capped.sort { lhs, rhs in
            lhs.count == rhs.count ? lhs.id.raw < rhs.id.raw : lhs.count > rhs.count
        }
        return Output(base: base, displayed: displayed, rows: rows, displayOrder: displayOrder,
                      rowPosition: positions, rowHeightRatios: ratios, summaryFrames: summary,
                      topTags: arrangement.topTagsByFrame, beyond: arrangement.beyondCapByGroup, capped: capped,
                      visualInput: visualInput, visualIDs: visualIDs)
    }
}
