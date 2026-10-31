import Foundation

/// A prepared sidebar row. Tagging changes scores, but not group timestamps or
/// cap counts, so rendering this row never needs to scan the frame collection.
struct CappedGroupSummary: Equatable, Identifiable {
    let id: FrameGroupID
    let count: Int
    let label: String
}

/// The face filter uses the same significance threshold as face ranking.
enum FrameFaceFilter: String, CaseIterable, Identifiable, Hashable, Sendable, Codable {
    case all
    case withFaces
    case withoutFaces

    var id: String { rawValue }

    var label: String {
        switch self {
        case .all: return "All"
        case .withFaces: return "With faces"
        case .withoutFaces: return "Without faces"
        }
    }

    func admits(_ frame: AnalyzedFrame, grouping: GroupingSettings) -> Bool {
        switch self {
        case .all: return true
        case .withFaces: return frame.faceDefinitionScore != nil || FrameRanker.faceIsSignificant(frame, settings: grouping)
        case .withoutFaces: return frame.faceDefinitionScore == nil && !FrameRanker.faceIsSignificant(frame, settings: grouping)
        }
    }
}

/// The Photos-library favorite state filter. Non-Photos sources do not expose
/// favorite state and therefore keep this control hidden.
enum FrameFavoriteFilter: String, CaseIterable, Identifiable, Hashable, Sendable, Codable {
    case all
    case favorite
    case notFavorite

    var id: String { rawValue }

    var label: String {
        switch self {
        case .all: return "All"
        case .favorite: return "Favorite"
        case .notFavorite: return "Not Favorite"
        }
    }

    func admits(_ isFavorite: Bool) -> Bool {
        switch self {
        case .all: return true
        case .favorite: return isFavorite
        case .notFavorite: return !isFavorite
        }
    }
}

/// Required tags compose with face and numeric filters; unwanted tags exclude matches.
struct TagOrGroup: Equatable, Hashable, Sendable, Codable, Identifiable {
    var id: UUID = UUID()
    var tags: Set<String> = []
}

struct TagFilter: Equatable, Sendable, Codable {
    var requiredTags: Set<String> = []
    var excludedTags: Set<String> = []
    var orGroups: [TagOrGroup] = []
    var threshold: Float = 0.35
    /// The order in which tags were selected. The set remains the source of truth
    /// for matching; this order lets the filter panel keep active tags at the front.
    var selectionOrder: [String] = []
    /// The order in which tags were marked unwanted, shown after required tags.
    var exclusionOrder: [String] = []
    /// Optional primary ordering by one selected tag's probability.
    var scoreSortTag: String?
    /// `true` means highest score first.
    var scoreSortDescending: Bool

    init(requiredTags: Set<String> = [], excludedTags: Set<String> = [], threshold: Float = 0.35,
         selectionOrder: [String]? = nil, exclusionOrder: [String]? = nil,
         scoreSortTag: String? = nil, scoreSortDescending: Bool = true, orGroups: [TagOrGroup] = []) {
        let effectiveRequiredTags = requiredTags.subtracting(excludedTags)
        self.requiredTags = effectiveRequiredTags
        self.excludedTags = excludedTags
        self.orGroups = orGroups
        self.threshold = threshold
        let requestedOrder = selectionOrder ?? requiredTags.sorted()
        var seen = Set<String>()
        self.selectionOrder = requestedOrder.filter { effectiveRequiredTags.contains($0) && seen.insert($0).inserted }
            + effectiveRequiredTags.subtracting(seen).sorted()
        let requestedExclusionOrder = exclusionOrder ?? excludedTags.sorted()
        var excludedSeen = Set<String>()
        self.exclusionOrder = requestedExclusionOrder.filter { excludedTags.contains($0) && excludedSeen.insert($0).inserted }
            + excludedTags.subtracting(excludedSeen).sorted()
        self.scoreSortTag = scoreSortTag.flatMap { effectiveRequiredTags.contains($0) ? $0 : nil }
        self.scoreSortDescending = scoreSortDescending
    }

    private enum CodingKeys: String, CodingKey {
        case requiredTags, excludedTags, orGroups, threshold, selectionOrder, exclusionOrder, scoreSortTag, scoreSortDescending
    }

    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        self.init(requiredTags: try values.decodeIfPresent(Set<String>.self, forKey: .requiredTags) ?? [],
                  excludedTags: try values.decodeIfPresent(Set<String>.self, forKey: .excludedTags) ?? [],
                  threshold: try values.decodeIfPresent(Float.self, forKey: .threshold) ?? 0.35,
                  selectionOrder: try values.decodeIfPresent([String].self, forKey: .selectionOrder),
                  exclusionOrder: try values.decodeIfPresent([String].self, forKey: .exclusionOrder),
                  scoreSortTag: try values.decodeIfPresent(String.self, forKey: .scoreSortTag),
                  scoreSortDescending: try values.decodeIfPresent(Bool.self, forKey: .scoreSortDescending) ?? true,
                  orGroups: try values.decodeIfPresent([TagOrGroup].self, forKey: .orGroups) ?? [])
    }

    var hasActiveTags: Bool {
        !requiredTags.isEmpty || !excludedTags.isEmpty || orGroups.contains { !$0.tags.isEmpty }
    }

    var orTags: Set<String> { orGroups.reduce(into: Set<String>()) { $0.formUnion($1.tags) } }

    mutating func addToOrGroup(_ tags: Set<String>, id: UUID) {
        guard let index = orGroups.firstIndex(where: { $0.id == id }) else { return }
        for tag in tags {
            setSelected(false, tag: tag)
            for groupIndex in orGroups.indices { orGroups[groupIndex].tags.remove(tag) }
        }
        orGroups[index].tags.formUnion(tags)
    }

    /// Every multi-selection shortcut creates its own OR clause.
    @discardableResult
    mutating func groupSelectedTags(_ tags: Set<String>) -> UUID? {
        guard tags.count > 1 else { return nil }
        let group = TagOrGroup()
        orGroups.append(group)
        let id = group.id
        addToOrGroup(tags, id: id)
        return id
    }

    /// Replace two OR clauses with their union, keeping the destination's identity.
    @discardableResult
    mutating func mergeOrGroup(_ sourceID: UUID, into destinationID: UUID) -> Bool {
        guard sourceID != destinationID,
              let source = orGroups.first(where: { $0.id == sourceID }),
              let destinationIndex = orGroups.firstIndex(where: { $0.id == destinationID }) else { return false }
        orGroups[destinationIndex].tags.formUnion(source.tags)
        orGroups.removeAll { $0.id == sourceID }
        return true
    }

    mutating func removeFromOrGroups(_ tag: String) {
        for index in orGroups.indices { orGroups[index].tags.remove(tag) }
    }

    var orderedRequiredTags: [String] {
        var seen = Set<String>()
        return selectionOrder.filter { requiredTags.contains($0) && seen.insert($0).inserted }
            + requiredTags.subtracting(seen).sorted()
    }

    var orderedExcludedTags: [String] {
        var seen = Set<String>()
        return exclusionOrder.filter { excludedTags.contains($0) && seen.insert($0).inserted }
            + excludedTags.subtracting(seen).sorted()
    }

    mutating func setSelected(_ selected: Bool, tag: String) {
        if selected {
            removeFromOrGroups(tag)
            excludedTags.remove(tag)
            exclusionOrder.removeAll { $0 == tag }
            requiredTags.insert(tag)
            selectionOrder.removeAll { $0 == tag }
            selectionOrder.append(tag)
        } else {
            excludedTags.remove(tag)
            exclusionOrder.removeAll { $0 == tag }
            requiredTags.remove(tag)
            selectionOrder.removeAll { $0 == tag }
            if scoreSortTag == tag { scoreSortTag = nil }
        }
    }

    mutating func setExcluded(_ excluded: Bool, tag: String) {
        if excluded {
            removeFromOrGroups(tag)
            requiredTags.remove(tag)
            selectionOrder.removeAll { $0 == tag }
            if scoreSortTag == tag { scoreSortTag = nil }
            excludedTags.insert(tag)
            exclusionOrder.removeAll { $0 == tag }
            exclusionOrder.append(tag)
        } else {
            excludedTags.remove(tag)
            exclusionOrder.removeAll { $0 == tag }
        }
    }

    /// Rank within the already-filtered candidate set, before group caps and
    /// display ordering. Read sparse UInt8 scores without mutating their payloads.
    /// With no selection, combine the winners for every general tag. Otherwise
    /// intersect each required tag's winners, preserving the all-required rule.
    struct TopSelection {
        var frames: [AnalyzedFrame]
        var tagsByFrame: [FrameID: [String]] = [:]
    }

    func keepingTopFrames(_ frames: [AnalyzedFrame], count: Int,
                          vocabulary: TagVocabulary?) -> [AnalyzedFrame] {
        topFrameSelection(frames, count: count, vocabulary: vocabulary).frames
    }

    func topFrameSelection(_ frames: [AnalyzedFrame], count: Int,
                           vocabulary: TagVocabulary?) -> TopSelection {
        guard count < 99, let vocabulary else { return TopSelection(frames: frames) }
        if requiredTags.isEmpty {
            let generalIndices = Set(vocabulary.generalIndices)
            let limit = max(count, 1)
            var leaders: [Int: [(id: FrameID, score: Float)]] = [:]
            for frame in frames {
                guard let data = frame.tagProbabilities,
                      let scores = TagProbabilityCodec.sparseValues(in: data) else { continue }
                for entry in scores where generalIndices.contains(entry.index) && entry.value > threshold {
                    var ranked = leaders[entry.index] ?? []
                    let insertion = ranked.firstIndex {
                        entry.value > $0.score || (entry.value == $0.score && frame.id.raw < $0.id.raw)
                    } ?? ranked.count
                    if insertion < limit {
                        ranked.insert((frame.id, entry.value), at: insertion)
                        if ranked.count > limit { ranked.removeLast() }
                        leaders[entry.index] = ranked
                    }
                }
            }
            var tagsByFrame: [FrameID: [String]] = [:]
            for index in leaders.keys.sorted() {
                for leader in leaders[index] ?? [] {
                    tagsByFrame[leader.id, default: []].append(vocabulary.labels[index].name)
                }
            }
            return TopSelection(frames: frames.filter { tagsByFrame[$0.id] != nil }, tagsByFrame: tagsByFrame)
        }
        let tags = Set(requiredTags.map(TagVocabulary.normalized)).sorted()
        let indices = tags.compactMap { vocabulary.index(for: $0) }
        guard indices.count == tags.count else { return TopSelection(frames: []) }
        let scored = frames.compactMap { frame -> (frame: AnalyzedFrame, scores: [Float])? in
            guard let data = frame.tagProbabilities,
                  let scores = TagProbabilityCodec.values(in: data, at: indices) else { return nil }
            return (frame, scores)
        }
        var winners: Set<FrameID>?
        for column in indices.indices {
            let ranked = scored.filter { $0.scores[column] > threshold }.sorted {
                if $0.scores[column] != $1.scores[column] {
                    return $0.scores[column] > $1.scores[column]
                }
                return $0.frame.id.raw < $1.frame.id.raw
            }
            let selected = Set(ranked.prefix(max(count, 1)).map { $0.frame.id })
            winners = winners.map { $0.intersection(selected) } ?? selected
        }
        let selected = winners ?? []
        let visible = frames.filter { selected.contains($0.id) }
        return TopSelection(frames: visible,
                            tagsByFrame: Dictionary(uniqueKeysWithValues: visible.map { ($0.id, tags) }))
    }

    func admits(_ frame: AnalyzedFrame, vocabulary: TagVocabulary? = nil) -> Bool {
        let required = Set(requiredTags.map(TagVocabulary.normalized))
        let excluded = Set(excludedTags.map(TagVocabulary.normalized))
        let alternatives = orGroups.map { Set($0.tags.map(TagVocabulary.normalized)) }.filter { !$0.isEmpty }
        let relevant = required.union(excluded).union(alternatives.reduce(into: Set<String>()) { $0.formUnion($1) })
        guard !relevant.isEmpty else { return true }
        let saved = Set((frame.topTags ?? []).map(TagVocabulary.normalized))
        var matching = saved
        if let vocabulary, let data = frame.tagProbabilities {
            let entries = relevant.compactMap { tag -> (String, Int)? in
                vocabulary.index(for: tag).map { (tag, $0) }
            }
            if let scores = TagProbabilityCodec.values(in: data, at: entries.map(\.1)) {
                matching = Set(zip(entries, scores).compactMap { entry, score in
                    score > threshold ? entry.0 : nil
                })
                // Exclusions historically fall back to saved tags for unknown heads.
                matching.formUnion(excluded.filter { vocabulary.index(for: $0) == nil && saved.contains($0) })
            }
            // Unknown required or alternative heads cannot match a known vocabulary.
            matching.subtract(relevant.filter { vocabulary.index(for: $0) == nil && !excluded.contains($0) })
        }
        return excluded.isDisjoint(with: matching)
            && required.isSubset(of: matching)
            && alternatives.allSatisfy { !$0.isDisjoint(with: matching) }
    }

}

/// A signal the grid can order by. Visual distance has no filtering range.
///
/// Rating probabilities are separate signals. Each of the four model heads can be
/// sorted or limited without changing the other three.
enum FrameSortKey: String, CaseIterable, Identifiable, Sendable {
    case minimalVisualDistance
    case time
    case tagRarity
    case face
    case definition
    case art
    case photoDistance
    case trufor
    case ratingGeneral
    case ratingSensitive
    case ratingQuestionable
    case ratingExplicit
    case tagGeneralScore

    var id: String { rawValue }

    var label: String {
        switch self {
        case .minimalVisualDistance: return "Minimal Visual Distance"
        case .time: return "Time"
        case .definition: return "Face definition"
        case .tagRarity: return "Tag rarity"
        case .face: return "Face"
        case .trufor: return "TruFor"
        case .photoDistance: return "Distance"
        case .art: return "Art"
        case .ratingGeneral: return "General"
        case .ratingSensitive: return "Sensitive"
        case .ratingQuestionable: return "Questionable"
        case .ratingExplicit: return "Explicit"
        case .tagGeneralScore: return "Traits confidence"
        }
    }

    var detail: String {
        switch self {
        case .minimalVisualDistance: return "Arrange shown photos so consecutive images look similar. Reverse follows the same sequence backward. Other enabled signals can affect group representatives and the route's starting order."
        case .time: return "Position in the video. Ties fall through the other enabled signals."
        case .definition: return "Detail in native-resolution face crops, without size or position weighting."
        case .tagRarity: return "Mean inverse tag frequency across the tagged source, normalized from the most common photo (0) to the rarest (1). Empty tag sets score 0. Filters do not change these scores."
        case .face: return "Vision face capture quality. Frames with no significant face sort last."
        case .trufor: return "TruFor’s official forgery detection score. Lower values suggest less manipulation; unscored photos remain visible and sort last."
        case .photoDistance: return "Visual distance to the nearest other source photo. Scaled from least isolated (0) to most isolated (1); similar pairs and series score low. Filtering does not change scores. Missing or unsupported features, or a photo without a valid neighbor, have no score."
        case .art: return "Apple's image aesthetics score."
        case .ratingGeneral: return "WD Tagger's probability for the general rating class."
        case .ratingSensitive: return "WD Tagger's probability for the sensitive rating class."
        case .ratingQuestionable: return "WD Tagger's probability for the questionable rating class."
        case .ratingExplicit: return "WD Tagger's probability for the explicit rating class."
        case .tagGeneralScore: return "Sum of the strongest general trait probabilities; untagged frames sort last."
        }
    }

    var ratingClass: TagRating? {
        switch self {
        case .ratingGeneral: return .general
        case .ratingSensitive: return .sensitive
        case .ratingQuestionable: return .questionable
        case .ratingExplicit: return .explicit
        default: return nil
        }
    }

    var supportsFiltering: Bool { self != .minimalVisualDistance }
}

/// Whether one signal takes part, and which way it points.
struct OrderTerm: Hashable, Sendable, Codable {
    var enabled: Bool
    var ascending: Bool

    /// Best-first — what "order by score" means unless someone says otherwise.
    static let best = OrderTerm(enabled: true, ascending: false)

    /// Earliest-first — what "order by time" means.
    static let chronological = OrderTerm(enabled: true, ascending: true)

    /// Present in the order but switched off, keeping its direction for when it is
    /// switched back on.
    static func off(ascending: Bool) -> OrderTerm { OrderTerm(enabled: false, ascending: ascending) }

    var directionLabel: String { ascending ? "Ascending" : "Descending" }
}

/// Visual distance arranges the shown set after the other signals sort candidates
/// and determine group representatives. Those signals use time, face, face definition,
/// art, then each rating probability in fixed precedence.
///
/// One term always decides the tie before the next is consulted, and the frame
/// identifier is the last resort — which is what makes every order **total**, so two
/// runs over the same frames cannot produce different lists and a grid position means
/// something.
///
/// At least one signal is always enabled: an order with none would fall back to
/// identifier order, which is not something a user asks for by unticking a box.
struct FrameOrder: Equatable, Sendable, Codable {
    var minimalVisualDistance: OrderTerm
    var time: OrderTerm
    var face: OrderTerm
    var art: OrderTerm
    var definition: OrderTerm
    var ratingGeneral: OrderTerm
    var ratingSensitive: OrderTerm
    var ratingQuestionable: OrderTerm
    var ratingExplicit: OrderTerm
    var photoDistance: OrderTerm = .off(ascending: false)
    var tagRarity: OrderTerm = .off(ascending: false)
    var tagGeneralScore: OrderTerm
    var trufor: OrderTerm

    /// By default, only visual distance is enabled. Other signals remain available
    /// as independent options; rating probabilities are opt-in in all presets.
    init(minimalVisualDistance: OrderTerm = .chronological,
         time: OrderTerm = .off(ascending: true), face: OrderTerm = .off(ascending: false),
         art: OrderTerm = .off(ascending: false), definition: OrderTerm = .off(ascending: false),
         ratingGeneral: OrderTerm = .off(ascending: false), ratingSensitive: OrderTerm = .off(ascending: false),
         ratingQuestionable: OrderTerm = .off(ascending: false), ratingExplicit: OrderTerm = .off(ascending: false),
         photoDistance: OrderTerm = .off(ascending: false), tagRarity: OrderTerm = .off(ascending: false), tagGeneralScore: OrderTerm = .off(ascending: false), trufor: OrderTerm = .off(ascending:true)) {
        self.minimalVisualDistance = minimalVisualDistance
        self.time = time
        self.face = face
        self.art = art
        self.definition = definition
        self.ratingGeneral = ratingGeneral
        self.ratingSensitive = ratingSensitive
        self.ratingQuestionable = ratingQuestionable
        self.ratingExplicit = ratingExplicit
        self.photoDistance = photoDistance
        self.tagRarity = tagRarity
        self.tagGeneralScore = tagGeneralScore
        self.trufor=trufor
    }

    private enum CodingKeys:String,CodingKey { case minimalVisualDistance, time, face, art, definition, ratingGeneral, ratingSensitive, ratingQuestionable, ratingExplicit, tagGeneralScore, tagRarity, photoDistance, trufor }
    init(from decoder:Decoder) throws {
        self.init()
        let values=try decoder.container(keyedBy:CodingKeys.self)
        minimalVisualDistance=try values.decodeIfPresent(OrderTerm.self,forKey:.minimalVisualDistance) ?? minimalVisualDistance
        time=try values.decodeIfPresent(OrderTerm.self,forKey:.time) ?? time
        face=try values.decodeIfPresent(OrderTerm.self,forKey:.face) ?? face
        art=try values.decodeIfPresent(OrderTerm.self,forKey:.art) ?? art
        definition=try values.decodeIfPresent(OrderTerm.self,forKey:.definition) ?? definition
        ratingGeneral=try values.decodeIfPresent(OrderTerm.self,forKey:.ratingGeneral) ?? ratingGeneral
        ratingSensitive=try values.decodeIfPresent(OrderTerm.self,forKey:.ratingSensitive) ?? ratingSensitive
        ratingQuestionable=try values.decodeIfPresent(OrderTerm.self,forKey:.ratingQuestionable) ?? ratingQuestionable
        ratingExplicit=try values.decodeIfPresent(OrderTerm.self,forKey:.ratingExplicit) ?? ratingExplicit
        photoDistance=try values.decodeIfPresent(OrderTerm.self,forKey:.photoDistance) ?? photoDistance
        tagRarity=try values.decodeIfPresent(OrderTerm.self,forKey:.tagRarity) ?? tagRarity
        tagGeneralScore=try values.decodeIfPresent(OrderTerm.self,forKey:.tagGeneralScore) ?? tagGeneralScore
        trufor=try values.decodeIfPresent(OrderTerm.self,forKey:.trufor) ?? trufor
    }

    var withoutForensic:FrameOrder {
        var value=self;value.trufor.enabled=false
        if value.enabledCount==0 { value.time = .chronological }
        return value
    }

    // MARK: - The terms in force

    /// Terms in fixed precedence, skipping the signals that are switched off.
    var terms: [SortTerm] {
        var terms: [SortTerm] = []
        if minimalVisualDistance.enabled { terms.append(SortTerm(key: .minimalVisualDistance, ascending: minimalVisualDistance.ascending)) }
        if time.enabled { terms.append(SortTerm(key: .time, ascending: time.ascending)) }
        if tagRarity.enabled { terms.append(SortTerm(key: .tagRarity, ascending: tagRarity.ascending)) }
        if face.enabled { terms.append(SortTerm(key: .face, ascending: face.ascending)) }
        if definition.enabled { terms.append(SortTerm(key: .definition, ascending: definition.ascending)) }
        if art.enabled { terms.append(SortTerm(key: .art, ascending: art.ascending)) }
        if photoDistance.enabled { terms.append(SortTerm(key: .photoDistance, ascending: photoDistance.ascending)) }
        if ratingGeneral.enabled { terms.append(SortTerm(key: .ratingGeneral, ascending: ratingGeneral.ascending)) }
        if ratingSensitive.enabled { terms.append(SortTerm(key: .ratingSensitive, ascending: ratingSensitive.ascending)) }
        if ratingQuestionable.enabled { terms.append(SortTerm(key: .ratingQuestionable, ascending: ratingQuestionable.ascending)) }
        if ratingExplicit.enabled { terms.append(SortTerm(key: .ratingExplicit, ascending: ratingExplicit.ascending)) }
        if trufor.enabled { terms.append(SortTerm(key:.trufor,ascending:trufor.ascending)) }
        if tagGeneralScore.enabled { terms.append(SortTerm(key: .tagGeneralScore, ascending: tagGeneralScore.ascending)) }
        return terms
    }

    var enabledCount: Int {
        [minimalVisualDistance.enabled, time.enabled, definition.enabled, face.enabled, art.enabled, ratingGeneral.enabled, ratingSensitive.enabled,
         ratingQuestionable.enabled, ratingExplicit.enabled, photoDistance.enabled, tagRarity.enabled, tagGeneralScore.enabled, trufor.enabled].filter { $0 }.count
    }

    /// The term for `key`, always present (a switched-off signal is a term with
    /// `enabled == false`).
    func term(_ key: FrameSortKey) -> OrderTerm {
        switch key {
        case .minimalVisualDistance: return minimalVisualDistance
        case .time: return time
        case .definition: return definition
        case .face: return face
        case .ratingGeneral: return ratingGeneral
        case .ratingSensitive: return ratingSensitive
        case .ratingQuestionable: return ratingQuestionable
        case .ratingExplicit: return ratingExplicit
        case .trufor: return trufor
        case .photoDistance: return photoDistance
        case .tagRarity: return tagRarity
        case .tagGeneralScore: return tagGeneralScore
        case .art: return art
        }
    }

    // MARK: - Mutation, addressed by key

    /// Switches a signal on or off.
    ///
    /// Refuses to switch off the last one. **Addressed by key, never by position**,
    /// and that is load-bearing: a control's `Binding` outlives the body pass that
    /// built it — SwiftUI evaluates it again on later updates and on layout — so a
    /// mutation that captured an index into a list can read an index that no longer
    /// exists. That is how this panel crashed before, in a binding's getter, from
    /// `NSHostingView.layout()`.
    mutating func setEnabled(_ enabled: Bool, for key: FrameSortKey) {
        guard enabled || enabledCount > 1 else { return }
        switch key {
        case .minimalVisualDistance: minimalVisualDistance.enabled = enabled
        case .time: time.enabled = enabled
        case .definition: definition.enabled = enabled
        case .face: face.enabled = enabled
        case .ratingGeneral: ratingGeneral.enabled = enabled
        case .ratingSensitive: ratingSensitive.enabled = enabled
        case .ratingQuestionable: ratingQuestionable.enabled = enabled
        case .ratingExplicit: ratingExplicit.enabled = enabled
        case .trufor: trufor.enabled = enabled
        case .photoDistance: photoDistance.enabled = enabled
        case .tagRarity: tagRarity.enabled = enabled
        case .tagGeneralScore: tagGeneralScore.enabled = enabled
        case .art: art.enabled = enabled
        }
    }

    /// Points a signal's term one way or the other. A signal that is switched off
    /// keeps the direction for when it is switched back on.
    mutating func setAscending(_ ascending: Bool, for key: FrameSortKey) {
        switch key {
        case .minimalVisualDistance: minimalVisualDistance.ascending = ascending
        case .time: time.ascending = ascending
        case .definition: definition.ascending = ascending
        case .face: face.ascending = ascending
        case .ratingGeneral: ratingGeneral.ascending = ascending
        case .ratingSensitive: ratingSensitive.ascending = ascending
        case .ratingQuestionable: ratingQuestionable.ascending = ascending
        case .ratingExplicit: ratingExplicit.ascending = ascending
        case .trufor: trufor.ascending = ascending
        case .photoDistance: photoDistance.ascending = ascending
        case .tagRarity: tagRarity.ascending = ascending
        case .tagGeneralScore: tagGeneralScore.ascending = ascending
        case .art: art.ascending = ascending
        }
    }

    mutating func toggleDirection(of key: FrameSortKey) {
        setAscending(!term(key).ascending, for: key)
    }

    // MARK: - Presets

    /// Each signal points its natural way; the signals in `keys` are
    /// the ones switched on.
    static func only(_ keys: Set<FrameSortKey>) -> FrameOrder {
        FrameOrder(
            minimalVisualDistance: OrderTerm(enabled: keys.contains(.minimalVisualDistance), ascending: true),
            time: OrderTerm(enabled: keys.contains(.time), ascending: true),
            face: OrderTerm(enabled: keys.contains(.face), ascending: false),
            art: OrderTerm(enabled: keys.contains(.art), ascending: false),
            definition: OrderTerm(enabled: keys.contains(.definition), ascending: false),
            ratingGeneral: OrderTerm(enabled: keys.contains(.ratingGeneral), ascending: false),
            ratingSensitive: OrderTerm(enabled: keys.contains(.ratingSensitive), ascending: false),
            ratingQuestionable: OrderTerm(enabled: keys.contains(.ratingQuestionable), ascending: false),
            ratingExplicit: OrderTerm(enabled: keys.contains(.ratingExplicit), ascending: false),
            photoDistance: OrderTerm(enabled: keys.contains(.photoDistance), ascending: false),
            tagRarity: OrderTerm(enabled: keys.contains(.tagRarity), ascending: false),
            tagGeneralScore: OrderTerm(enabled: keys.contains(.tagGeneralScore), ascending: false),
            trufor: OrderTerm(enabled:keys.contains(.trufor),ascending:true)
        )
    }

    static let defaultOrder = FrameOrder()
    /// All ordinary sorting signals enabled, retained for explicit presets and fixtures.
    static let everything = FrameOrder.only([.minimalVisualDistance, .time, .face, .definition, .art])
    static let faceThenArt = FrameOrder.only([.face, .art])
    static let definitionOnly = FrameOrder.only([.definition])
    static let faceOnly = FrameOrder.only([.face])
    static let artOnly = FrameOrder.only([.art])
    static let chronological = FrameOrder.only([.time])
}

/// One key and the way it points, as the sorter consumes it.
struct SortTerm: Hashable, Sendable {
    var key: FrameSortKey
    var ascending: Bool
}

/// The four rating-head probabilities stored at the start of the tagger vector.
/// Live vectors need only four scalar reads, even when several controls are active.
private struct RatingProbabilityScores {
    private let values: [Float]?

    init(frame: AnalyzedFrame) {
        self.init(probabilities: frame.tagProbabilities.flatMap {
            TagProbabilityCodec.values(in: $0, at: Array(0..<TagRating.allCases.count))
        })
    }

    init(probabilities: [Float]?) {
        if let probabilities, probabilities.count >= TagRating.allCases.count {
            values = Array(probabilities.prefix(TagRating.allCases.count))
        } else {
            values = nil
        }
    }

    func value(for rating: TagRating) -> Double? {
        guard let values, values.indices.contains(rating.index) else { return nil }
        return Double(values[rating.index])
    }
}

/// A low and/or high limit on one signal, which a user drags on the fly.
///
/// `nil` on either end means **no limit**, which is not the same as zero: with no
/// limit the end is not consulted, and a frame that has no value for the signal at
/// all — the face score of a landscape — passes. Once a face limit is set, though,
/// a missing face cannot be inside the range, so those frames drop out.
struct FrameLimit: Hashable, Sendable, Codable {
    var low: Double?
    var high: Double?

    static let unlimited = FrameLimit()

    var isUnlimited: Bool { low == nil && high == nil }

    func contains(_ value: Double?) -> Bool {
        guard !isUnlimited else { return true }
        guard let value else { return false }
        if let low, value < low { return false }
        if let high, value > high { return false }
        return true
    }
}

/// One independent limit per ordering and rating signal.
struct FrameLimits: Hashable, Sendable, Codable {
    var time: FrameLimit = .unlimited
    var face: FrameLimit = .unlimited
    var art: FrameLimit = .unlimited
    var definition: FrameLimit = .unlimited
    var ratingGeneral: FrameLimit = .unlimited
    var ratingSensitive: FrameLimit = .unlimited
    var ratingQuestionable: FrameLimit = .unlimited
    var ratingExplicit: FrameLimit = .unlimited
    var photoDistance: FrameLimit = .unlimited
    var tagRarity: FrameLimit = .unlimited
    var tagGeneralScore: FrameLimit = .unlimited
    var trufor: FrameLimit = .unlimited

    static let none = FrameLimits()

    var isEmpty: Bool {
        time.isUnlimited && face.isUnlimited && art.isUnlimited && definition.isUnlimited
            && ratingGeneral.isUnlimited && ratingSensitive.isUnlimited
            && ratingQuestionable.isUnlimited && ratingExplicit.isUnlimited
            && photoDistance.isUnlimited && tagRarity.isUnlimited && tagGeneralScore.isUnlimited && trufor.isUnlimited
    }

    private var hasRatingLimits: Bool {
        !ratingGeneral.isUnlimited || !ratingSensitive.isUnlimited
            || !ratingQuestionable.isUnlimited || !ratingExplicit.isUnlimited
    }

    func limit(for key: FrameSortKey) -> FrameLimit {
        switch key {
        case .minimalVisualDistance: return .unlimited
        case .time: return time
        case .definition: return definition
        case .face: return face
        case .ratingGeneral: return ratingGeneral
        case .ratingSensitive: return ratingSensitive
        case .ratingQuestionable: return ratingQuestionable
        case .ratingExplicit: return ratingExplicit
        case .trufor: return trufor
        case .photoDistance: return photoDistance
        case .tagRarity: return tagRarity
        case .tagGeneralScore: return tagGeneralScore
        case .art: return art
        }
    }

    mutating func setLimit(_ limit: FrameLimit, for key: FrameSortKey) {
        switch key {
        case .minimalVisualDistance: break
        case .time: time = limit
        case .definition: definition = limit
        case .face: face = limit
        case .ratingGeneral: ratingGeneral = limit
        case .ratingSensitive: ratingSensitive = limit
        case .ratingQuestionable: ratingQuestionable = limit
        case .ratingExplicit: ratingExplicit = limit
        case .trufor: trufor = limit
        case .photoDistance: photoDistance = limit
        case .tagRarity: tagRarity = limit
        case .tagGeneralScore: tagGeneralScore = limit
        case .art: art = limit
        }
    }

    mutating func clear() { self = .none }

    /// Whether `frame` is inside every limit that is set.
    ///
    /// A face limit is measured against the *significant* face score — the ranker's
    /// rule — so the limits and the ordering cannot disagree about which frames have
    /// a face at all.
    func admits(_ frame: AnalyzedFrame, grouping: GroupingSettings) -> Bool {
        let faceScore = FrameRanker.faceIsSignificant(frame, settings: grouping)
            ? frame.faceScore.map(Double.init)
            : nil
        guard time.contains(frame.timeSeconds)
            && art.contains(Double(frame.artScore))
            && face.contains(faceScore)
            && definition.contains(frame.faceDefinitionScore.map(Double.init))
            && (frame.validTruForScore == nil || trufor.contains(frame.validTruForScore.map(Double.init)))
            && photoDistance.contains(frame.photoDistance.map(Double.init))
            && tagRarity.contains(frame.tagRarity.map(Double.init))
            && tagGeneralScore.contains(frame.tagGeneralScore.map(Double.init)) else { return false }
        guard hasRatingLimits else { return true }
        let ratings = RatingProbabilityScores(frame: frame)
        return ratingGeneral.contains(ratings.value(for: .general))
            && ratingSensitive.contains(ratings.value(for: .sensitive))
            && ratingQuestionable.contains(ratings.value(for: .questionable))
            && ratingExplicit.contains(ratings.value(for: .explicit))
    }

    /// Applies only pre-tagger limits when choosing frames for an on-demand run.
    /// Rating and trait-confidence ranges depend on the tag results being produced.
    func admitsForTagging(_ frame: AnalyzedFrame, grouping: GroupingSettings) -> Bool {
        let faceScore = FrameRanker.faceIsSignificant(frame, settings: grouping)
            ? frame.faceScore.map(Double.init)
            : nil
        return time.contains(frame.timeSeconds)
            && art.contains(Double(frame.artScore))
            && face.contains(faceScore)
            && definition.contains(frame.faceDefinitionScore.map(Double.init))
            && (frame.validTruForScore == nil || trufor.contains(frame.validTruForScore.map(Double.init)))
    }
}

/// The span each signal's slider covers. Rating probabilities use the model's full
/// `0...1` probability range; other score signals use their observed span.
///
/// Measured from the frames the face filter admits — not from the *displayed* frames.
/// Using the displayed set would be circular: dragging the art thumb would narrow
/// art's own bounds, which would move the thumb again. The bounds are the data's
/// span, so they change when a different video or a different face filter changes
/// what is on screen, and never while a thumb is being dragged.
struct FrameLimitBounds: Equatable, Sendable {
    var time: ClosedRange<Double>
    var art: ClosedRange<Double>
    /// `nil` when no frame on screen has a significant face — there is nothing to
    /// put a face range around.
    var face: ClosedRange<Double>?
    var definition: ClosedRange<Double>? = nil
    var ratingGeneral: ClosedRange<Double>? = nil
    var ratingSensitive: ClosedRange<Double>? = nil
    var ratingQuestionable: ClosedRange<Double>? = nil
    var ratingExplicit: ClosedRange<Double>? = nil
    var photoDistance: ClosedRange<Double>? = 0...1
    var tagRarity: ClosedRange<Double>? = 0...1
    var tagGeneralScore: ClosedRange<Double>? = nil
    var trufor: ClosedRange<Double>? = nil

    func range(for key: FrameSortKey) -> ClosedRange<Double>? {
        switch key {
        case .minimalVisualDistance: return nil
        case .time: return time
        case .definition: return definition
        case .face: return face
        case .ratingGeneral: return ratingGeneral
        case .ratingSensitive: return ratingSensitive
        case .ratingQuestionable: return ratingQuestionable
        case .ratingExplicit: return ratingExplicit
        case .trufor: return trufor
        case .photoDistance: return photoDistance
        case .tagRarity: return tagRarity
        case .tagGeneralScore: return tagGeneralScore
        case .art: return art
        }
    }

    static func bounds(of frames: [AnalyzedFrame], grouping: GroupingSettings) -> FrameLimitBounds {
        let times = frames.map(\.timeSeconds)
        let arts = frames.map { Double($0.artScore) }
        let faces = frames.compactMap {
            FrameRanker.faceIsSignificant($0, settings: grouping) ? $0.faceScore.map(Double.init) : nil
        }
        let hasRatingScores = frames.contains { $0.tagProbabilities != nil }
        return FrameLimitBounds(
            time: span(times) ?? 0...1,
            art: span(arts) ?? -1...1,
            face: span(faces),
            definition: span(frames.compactMap { $0.faceDefinitionScore.map(Double.init) }),
            ratingGeneral: hasRatingScores ? 0...1 : nil,
            ratingSensitive: hasRatingScores ? 0...1 : nil,
            ratingQuestionable: hasRatingScores ? 0...1 : nil,
            ratingExplicit: hasRatingScores ? 0...1 : nil,
            tagGeneralScore: span(frames.compactMap { $0.tagGeneralScore.map(Double.init) }),
            trufor: span(frames.compactMap { $0.validTruForScore.map(Double.init) }).map { max(0,$0.lowerBound)...min(1,$0.upperBound) }
        )
    }

    /// `min...max` of the values, widened when they are all equal so a slider over it
    /// has somewhere to move.
    private static func span(_ values: [Double]) -> ClosedRange<Double>? {
        guard let low = values.min(), let high = values.max() else { return nil }
        guard high - low < 1e-9 else { return low...high }
        return (low - 0.5)...(high + 0.5)
    }
}

/// One line of the grid: a group's frames, or the frames that are in no group.
struct FrameRow: Identifiable, Equatable, Sendable {
    /// `nil` for the row of frames that belong to no group.
    let group: FrameGroupID?
    var frames: [AnalyzedFrame]

    /// Stable across refreshes so the grid does not rebuild rows it already has.
    var id: String { group?.raw ?? "\u{0}ungrouped" }
}

/// Turns the analysed frames into the list the grid renders, plus a note of what
/// the per-group cap left out.
///
/// Pure on purpose: which frames, in what order, and how many of each group is
/// decided here, so it can be pinned by value with no window in the way and the view
/// has nothing left to decide.
///
/// The steps run in a fixed order, and that order is the behaviour worth naming:
///
///   1. the face filter and the value limits;
///   2. the ordering;
///   3. the per-group cap.
///   4. optional visual path ordering across the remaining photos.
///
/// Ordering before the cap is what makes the cap mean "the *best* N of each group".
/// Capping first would keep whichever N the file happened to list first.
enum FrameArrangement {
    /// What the grid shows, and what the per-group cap left behind.
    struct Result: Sendable, Equatable {
        /// The frames to render, already ordered.
        var visible: [AnalyzedFrame] = []
        /// The actual tag rankings that admitted each visible frame.
        var topTagsByFrame: [FrameID: [String]] = [:]
        /// Frames past the per-group cap, per group — counted whether or not the
        /// group is revealed. A revealed group still needs its count, or revealing it
        /// would remove it from the list and there would be no way to put it back.
        var beyondCapByGroup: [FrameGroupID: Int] = [:]
    }

    /// - Parameters:
    ///   - frames: the session's frames, in time order.
    ///   - order: which signals to order by, in the app's fixed precedence.
    ///   - faceFilter: all frames, significant faces, or their complement.
    ///   - limits: low/high limits per signal, from the sliders.
    ///   - perGroupTopCount: how many frames of each group to keep. Frames in no
    ///     group are never capped — they are distinct moments, not alternates.
    ///   - revealedGroups: groups whose cap is lifted because the user asked to see
    ///     the rest of them.
    ///   - grouping: the ranking knobs (which decide whether a face counts).
    static func arrange(
        _ frames: [AnalyzedFrame],
        order: FrameOrder,
        faceFilter: FrameFaceFilter = .all,
        limits: FrameLimits = .none,
        perGroupTopCount: Int,
        revealedGroups: Set<FrameGroupID> = [],
        grouping: GroupingSettings,
        groupingEnabled: Bool = true,
        tagFilter: TagFilter = TagFilter(),
        vocabulary: TagVocabulary? = nil,
        filteredCandidates: [AnalyzedFrame]? = nil,
        applyingVisualOrder: Bool = true,
        tagTopCount: Int = 99
    ) -> Result {
        let candidates = filteredCandidates ?? frames.filter { frame in
            faceFilter.admits(frame, grouping: grouping)
                && limits.admits(frame, grouping: grouping)
                && tagFilter.admits(frame, vocabulary: vocabulary)
        }
        let scoreSortIndex = tagFilter.scoreSortTag.flatMap { vocabulary?.index(for: $0) }
        let topSelection = tagFilter.topFrameSelection(candidates, count: tagTopCount, vocabulary: vocabulary)
        let ordered = sorted(topSelection.frames, by: order, grouping: grouping,
                             tagScoreIndex: scoreSortIndex,
                             tagScoreDescending: tagFilter.scoreSortDescending,
                             applyingVisualOrder: false)

        var result = Result()
        result.visible.reserveCapacity(ordered.count)

        var groupCounts: [FrameGroupID: Int] = [:]
        for frame in ordered {
            guard groupingEnabled else {
                var ungrouped = frame
                ungrouped.groupID = nil
                result.visible.append(ungrouped)
                continue
            }
            guard let groupID = frame.groupID else {
                result.visible.append(frame)
                continue
            }
            let position = (groupCounts[groupID] ?? 0) + 1
            groupCounts[groupID] = position
            let beyondCap = position > perGroupTopCount
            if beyondCap {
                result.beyondCapByGroup[groupID, default: 0] += 1
            }
            if !beyondCap || revealedGroups.contains(groupID) {
                result.visible.append(frame)
            }
        }
        if applyingVisualOrder, order.minimalVisualDistance.enabled {
            result.visible = (try? MinimalVisualDistanceOrder.sorted(
                result.visible, reversed: !order.minimalVisualDistance.ascending
            )) ?? result.visible
        }
        for frame in result.visible {
            if let tags = topSelection.tagsByFrame[frame.id] { result.topTagsByFrame[frame.id] = tags }
        }
        return result
    }

    /// Splits the ordered frames into grid lines: **one per group**, in the order the
    /// frames appear, with the frames that belong to no group collected into a final
    /// line.
    ///
    /// A row's position in the result is its best frame's position in `visible`, so
    /// the "best first" ordering survives the regrouping into lines. Frames with no
    /// group are not a group — they are moments the grouper found nothing to pair —
    /// so they get one line of their own at the end rather than being scattered
    /// through the groups.
    static func rows(of visible: [AnalyzedFrame]) -> [FrameRow] {
        var rows: [FrameRow] = []
        var indexByGroup: [FrameGroupID: Int] = [:]
        var ungrouped: [AnalyzedFrame] = []

        for frame in visible {
            guard let group = frame.groupID else {
                ungrouped.append(frame)
                continue
            }
            if let index = indexByGroup[group] {
                rows[index].frames.append(frame)
            } else {
                indexByGroup[group] = rows.count
                rows.append(FrameRow(group: group, frames: [frame]))
            }
        }
        if !ungrouped.isEmpty {
            rows.append(FrameRow(group: nil, frames: ungrouped))
        }
        return rows
    }

    /// Orders `frames` by `order`. Exposed separately so a caller with its set
    /// already chosen does not have to go through the cap.
    static func sorted(
        _ frames: [AnalyzedFrame],
        by order: FrameOrder,
        grouping: GroupingSettings,
        tagScoreIndex: Int? = nil,
        tagScoreDescending: Bool = true,
        applyingVisualOrder: Bool = true
    ) -> [AnalyzedFrame] {
        // The fields an order can read, resolved once per frame rather than on every
        // comparison: a sort makes O(n log n) comparisons and this is n predicate
        // evaluations.
        let terms = order.terms
        let includeRatingProbabilities = terms.contains { $0.key.ratingClass != nil }
        let keys = frames.map {
            FrameKey(frame: $0, grouping: grouping,
                     includeRatingProbabilities: includeRatingProbabilities,
                     tagScoreIndex: tagScoreIndex)
        }
        let ordered = keys.sorted {
            precedes($0, $1, under: terms,
                     tagScoreDescending: tagScoreIndex == nil ? nil : tagScoreDescending)
        }.map(\.frame)
        guard applyingVisualOrder, order.minimalVisualDistance.enabled else { return ordered }
        return (try? MinimalVisualDistanceOrder.sorted(
            ordered, reversed: !order.minimalVisualDistance.ascending
        )) ?? ordered
    }

    // MARK: - Comparison

    /// A frame with what the order is allowed to look at.
    private struct FrameKey {
        let frame: AnalyzedFrame
        let art: Float
        /// `nil` when the frame has no *significant* face — decided by the ranker's
        /// own rule, so the order and the ranking agree about what a face is.
        let face: Float?
        let definition: Float?
        let ratingGeneral: Float?
        let ratingSensitive: Float?
        let ratingQuestionable: Float?
        let ratingExplicit: Float?
        let photoDistance: Float?
        let tagRarity: Float?
        let tagGeneralScore: Float?
        let trufor: Float?
        let tagScore: Float?
        let time: Double
        let identifier: String

        init(frame: AnalyzedFrame, grouping: GroupingSettings, includeRatingProbabilities: Bool,
             tagScoreIndex: Int?) {
            self.frame = frame
            self.art = frame.artScore
            self.definition = frame.faceDefinitionScore
            var indices = includeRatingProbabilities ? Array(0..<TagRating.allCases.count) : []
            if let tagScoreIndex { indices.append(tagScoreIndex) }
            let probabilities = indices.isEmpty ? nil : frame.tagProbabilities.flatMap {
                TagProbabilityCodec.values(in: $0, at: indices)
            }
            let ratings = includeRatingProbabilities ? RatingProbabilityScores(probabilities: probabilities) : nil
            self.ratingGeneral = ratings?.value(for: .general).map(Float.init)
            self.ratingSensitive = ratings?.value(for: .sensitive).map(Float.init)
            self.ratingQuestionable = ratings?.value(for: .questionable).map(Float.init)
            self.ratingExplicit = ratings?.value(for: .explicit).map(Float.init)
            self.trufor=frame.validTruForScore
            self.photoDistance = frame.photoDistance
            self.tagRarity = frame.tagRarity
            self.tagGeneralScore = frame.tagGeneralScore
            self.tagScore = tagScoreIndex == nil ? nil : probabilities?.last
            self.face = FrameRanker.faceIsSignificant(frame, settings: grouping) ? frame.faceScore : nil
            self.time = frame.timeSeconds
            self.identifier = frame.id.raw
        }

        func compareTagScore(to other: FrameKey, descending: Bool) -> ComparisonResult {
            switch (tagScore, other.tagScore) {
            case let (left?, right?):
                return Self.oriented(Self.compare(left, right), ascending: !descending)
            case (nil, nil): return .orderedSame
            case (nil, _?): return .orderedDescending
            case (_?, nil): return .orderedAscending
            }
        }

        func compare(_ other: FrameKey, by term: SortTerm) -> ComparisonResult {
            switch term.key {
            case .minimalVisualDistance:
                // A route is a property of the whole shown set, applied after
                // representative selection rather than in a pairwise comparator.
                return .orderedSame
            case .art:
                return Self.oriented(Self.compare(art, other.art), ascending: term.ascending)
            case .time:
                return Self.oriented(Self.compare(time, other.time), ascending: term.ascending)
            case .photoDistance, .tagRarity, .face, .definition, .ratingGeneral, .ratingSensitive, .ratingQuestionable, .ratingExplicit, .tagGeneralScore, .trufor:
                // A missing face is not a low score — `nil` means no significant face
                // was found — and a missing rating head has no probability — so it
                // sorts after every frame with a value whichever way the term points.
                let leftValue: Float?
                let rightValue: Float?
                switch term.key {
                case .face: (leftValue, rightValue) = (face, other.face)
                case .definition: (leftValue, rightValue) = (definition, other.definition)
                case .ratingGeneral: (leftValue, rightValue) = (ratingGeneral, other.ratingGeneral)
                case .ratingSensitive: (leftValue, rightValue) = (ratingSensitive, other.ratingSensitive)
                case .ratingQuestionable: (leftValue, rightValue) = (ratingQuestionable, other.ratingQuestionable)
                case .ratingExplicit: (leftValue, rightValue) = (ratingExplicit, other.ratingExplicit)
                case .trufor: (leftValue,rightValue)=(trufor,other.trufor)
                case .photoDistance: (leftValue, rightValue) = (photoDistance, other.photoDistance)
                case .tagRarity: (leftValue, rightValue) = (tagRarity, other.tagRarity)
                case .tagGeneralScore: (leftValue, rightValue) = (tagGeneralScore, other.tagGeneralScore)
                case .minimalVisualDistance, .time, .art: return .orderedSame
                }
                switch (leftValue, rightValue) {
                case let (left?, right?):
                    return Self.oriented(Self.compare(left, right), ascending: term.ascending)
                case (nil, nil):
                    return .orderedSame
                case (nil, _?):
                    return .orderedDescending
                case (_?, nil):
                    return .orderedAscending
                }
            }
        }

        private static func oriented(_ result: ComparisonResult, ascending: Bool) -> ComparisonResult {
            guard !ascending else { return result }
            switch result {
            case .orderedAscending: return .orderedDescending
            case .orderedDescending: return .orderedAscending
            case .orderedSame: return .orderedSame
            }
        }

        private static func compare<T: Comparable>(_ lhs: T, _ rhs: T) -> ComparisonResult {
            if lhs < rhs { return .orderedAscending }
            if lhs > rhs { return .orderedDescending }
            return .orderedSame
        }
    }

    private static func precedes(_ lhs: FrameKey, _ rhs: FrameKey, under terms: [SortTerm],
                                 tagScoreDescending: Bool?) -> Bool {
        if let tagScoreDescending {
            switch lhs.compareTagScore(to: rhs, descending: tagScoreDescending) {
            case .orderedAscending: return true
            case .orderedDescending: return false
            case .orderedSame: break
            }
        }
        for term in terms {
            switch lhs.compare(rhs, by: term) {
            case .orderedAscending: return true
            case .orderedDescending: return false
            case .orderedSame: continue
            }
        }
        // Total order: the identifier decides, so the list cannot reshuffle.
        return lhs.identifier < rhs.identifier
    }
}

extension FrameLimits {
    private enum CodingKeys:String,CodingKey { case time,face,art,definition,ratingGeneral,ratingSensitive,ratingQuestionable,ratingExplicit,tagGeneralScore,tagRarity,photoDistance,trufor }
    init(from decoder:Decoder) throws {
        self.init()
        let values=try decoder.container(keyedBy:CodingKeys.self)
        time=try values.decodeIfPresent(FrameLimit.self,forKey:.time) ?? .unlimited
        face=try values.decodeIfPresent(FrameLimit.self,forKey:.face) ?? .unlimited
        art=try values.decodeIfPresent(FrameLimit.self,forKey:.art) ?? .unlimited
        definition=try values.decodeIfPresent(FrameLimit.self,forKey:.definition) ?? .unlimited
        ratingGeneral=try values.decodeIfPresent(FrameLimit.self,forKey:.ratingGeneral) ?? .unlimited
        ratingSensitive=try values.decodeIfPresent(FrameLimit.self,forKey:.ratingSensitive) ?? .unlimited
        ratingQuestionable=try values.decodeIfPresent(FrameLimit.self,forKey:.ratingQuestionable) ?? .unlimited
        ratingExplicit=try values.decodeIfPresent(FrameLimit.self,forKey:.ratingExplicit) ?? .unlimited
        photoDistance=try values.decodeIfPresent(FrameLimit.self,forKey:.photoDistance) ?? .unlimited
        tagRarity=try values.decodeIfPresent(FrameLimit.self,forKey:.tagRarity) ?? .unlimited
        tagGeneralScore=try values.decodeIfPresent(FrameLimit.self,forKey:.tagGeneralScore) ?? .unlimited
        trufor=try values.decodeIfPresent(FrameLimit.self,forKey:.trufor) ?? .unlimited
    }
    var withoutForensic:FrameLimits { var value=self;value.trufor = .unlimited;return value }
}
