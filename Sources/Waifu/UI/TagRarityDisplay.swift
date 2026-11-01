import Foundation

/// Pure display rules for the tag browser's rarity bands.
enum TagRarityDisplay {
    /// A rarity band shows this many tags before offering the ellipsis control.
    static let tagLimit = 33

    /// Rank by the strongest matching-photo count. The input arrives in confidence
    /// order, which breaks equal-count ties without an alphabetical reshuffle.
    static func ordered(_ tags: [String], matchCounts: [String: Int]) -> [String] {
        let confidenceRank = Dictionary(
            tags.enumerated().map { ($0.element, $0.offset) },
            uniquingKeysWith: { first, _ in first }
        )
        return tags.sorted {
            let leftMatches = matchCounts[$0, default: 0]
            let rightMatches = matchCounts[$1, default: 0]
            if leftMatches != rightMatches { return leftMatches > rightMatches }
            return confidenceRank[$0, default: .max] < confidenceRank[$1, default: .max]
        }
    }

    /// Apply the display cap after quick filtering, so hidden tags still participate
    /// in matching and only the visible list is shortened.
    static func limited(_ tags: [String], showingAll: Bool,
                        limit: Int = tagLimit) -> [String] {
        showingAll ? tags : Array(tags.prefix(limit))
    }
}
