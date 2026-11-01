import Foundation

func registerTagBrowserDisplayTests() {
    let suite = "frame arrangement: tag browser"

    Registry.shared.add(suite: suite, TestCase(name: "rarity bands rank by match count and keep confidence order for ties", knownBug: nil) {
        let tags = ["confidence-first", "most-matches", "tie-a", "tie-b"]
        let counts = [
            "confidence-first": 2,
            "most-matches": 9,
            "tie-a": 5,
            "tie-b": 5,
        ]
        checkEqual(TagRarityDisplay.ordered(tags, matchCounts: counts),
                   ["most-matches", "tie-a", "tie-b", "confidence-first"],
                   "higher match counts lead; equal counts retain the incoming confidence order")
    })

    Registry.shared.add(suite: suite, TestCase(name: "rarity bands cap the visible matches without touching the head order", knownBug: nil) {
        let tags = (0..<40).map { "tag-\($0)" }
        checkEqual(TagRarityDisplay.limited(tags, showingAll: false),
                   Array(tags.prefix(TagRarityDisplay.tagLimit)),
                   "the default view stops at the first 33 tags")
        checkEqual(TagRarityDisplay.limited(tags, showingAll: true), tags,
                   "showing all returns every matched tag in the same order")
        checkEqual(TagRarityDisplay.limited(Array(tags.prefix(10)), showingAll: false),
                   Array(tags.prefix(10)),
                   "short bands are unaffected by the cap")
    })
}
