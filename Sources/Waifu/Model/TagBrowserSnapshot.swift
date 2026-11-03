import Foundation

/// Prepared only when summary/filter data changes. Scrolling reads these arrays;
/// it never unions inverse sets, sorts labels or classifies rarity bands.
struct TagBrowserSnapshot: Equatable, Sendable {
    var common: [String] = []
    var wanted: [String] = []
    var unwanted: [String] = []
    var unselected: [String] = []
    var ungrouped: [String] = []
    var all: [String] = []
    var available: Set<String> = []
    var orGroupTags: [UUID: [String]] = [:]
    var rarityBands: [[String]] = Array(repeating: [], count: TagRarity.allCases.count)

    init() {}

    init(common sourceCommon: [String], facets: [String], counts: [String: Int], total: Int, filter: TagFilter) {
        let hidden = filter.inverseGroupTags
        let grouped = filter.orTags
        func alphabetical(_ tags: Set<String>) -> [String] {
            tags.sorted { $0.localizedStandardCompare($1) == .orderedAscending }
        }
        let commonSet = Set(sourceCommon)
        common = alphabetical(commonSet.subtracting(filter.requiredTags).subtracting(filter.excludedTags)
            .subtracting(grouped).subtracting(hidden))
        wanted = alphabetical(filter.requiredTags.subtracting(hidden))
        unwanted = alphabetical(filter.excludedTags.subtracting(hidden))
        let classified = commonSet.union(filter.requiredTags).union(filter.excludedTags).union(grouped).union(hidden)
        unselected = facets.filter { !classified.contains($0) }
        ungrouped = common + wanted + unwanted + unselected
        all = ungrouped + Array(grouped)
        available = Set(facets).union(commonSet).union(filter.requiredTags).union(filter.excludedTags)
            .union(grouped).subtracting(hidden)
        for group in filter.orGroups { orGroupTags[group.id] = alphabetical(group.tags.subtracting(hidden)) }
        for tag in TagRarityDisplay.ordered(unselected, matchCounts: counts) {
            let rarity = TagRarity.classify(matches: counts[tag, default: 0], total: total)
            rarityBands[rarity.rawValue].append(tag)
        }
    }
}
