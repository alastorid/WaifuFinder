import Foundation

/// A prepared sidebar row. Tagging changes scores, but not group timestamps or
/// cap counts, so rendering this row never needs to scan the frame collection.
struct CappedGroupSummary: Equatable, Identifiable {
    let id: FrameGroupID
    let count: Int
    let label: String
}

/// The face filter uses the same significance threshold as face ranking.
enum FrameFaceFilter: String, CaseIterable, Identifiable, Hashable, Sendable {
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

/// Required tags compose with face and numeric filters; unwanted tags exclude matches.
struct TagFilter: Equatable, Sendable {
    var requiredTags: Set<String> = []
    var excludedTags: Set<String> = []
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
         scoreSortTag: String? = nil, scoreSortDescending: Bool = true) {
        let effectiveRequiredTags = requiredTags.subtracting(excludedTags)
        self.requiredTags = effectiveRequiredTags
        self.excludedTags = excludedTags
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

    func admits(_ frame: AnalyzedFrame, vocabulary: TagVocabulary? = nil) -> Bool {
        if excludedTags.contains(where: { excluded in
            let normalized = TagVocabulary.normalized(excluded)
            if let vocabulary, let data = frame.tagProbabilities,
               let probabilities = TagProbabilityCodec.decode(data),
               let index = vocabulary.index(for: normalized), probabilities.indices.contains(index) {
                return probabilities[index] > threshold
            }
            return (frame.topTags ?? []).contains { TagVocabulary.normalized($0) == normalized }
        }) { return false }
        guard !requiredTags.isEmpty else { return true }
        let required = Set(requiredTags.map(TagVocabulary.normalized))
        if let vocabulary, let data = frame.tagProbabilities,
           let probabilities = TagProbabilityCodec.decode(data) {
            return required.allSatisfy { tag in
                guard let index = vocabulary.index(for: tag), probabilities.indices.contains(index) else { return false }
                return probabilities[index] > threshold
            }
        }
        return required.isSubset(of: Set((frame.topTags ?? []).map(TagVocabulary.normalized)))
    }
}

/// A signal the grid can order by. Visual distance has no filtering range.
///
/// Rating probabilities are separate signals. Each of the four model heads can be
/// sorted or limited without changing the other three.
enum FrameSortKey: String, CaseIterable, Identifiable, Sendable {
    case minimalVisualDistance
    case time
    case face
    case definition
    case art
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
        case .face: return "Face"
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
        case .minimalVisualDistance: return "Arrange shown photos so consecutive images look similar. Reverse follows the same sequence backward. Other signals choose group representatives and break ties."
        case .time: return "Position in the video. Ties fall through the other enabled signals."
        case .definition: return "Detail in native-resolution face crops, without size or position weighting."
        case .face: return "Vision face capture quality. Frames with no significant face sort last."
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
struct OrderTerm: Hashable, Sendable {
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

/// Visual distance orders the complete shown set after the other signals choose
/// representatives. Those signals use time, face, face definition, art, then each
/// rating probability in fixed precedence.
///
/// One term always decides the tie before the next is consulted, and the frame
/// identifier is the last resort — which is what makes every order **total**, so two
/// runs over the same frames cannot produce different lists and a grid position means
/// something.
///
/// At least one signal is always enabled: an order with none would fall back to
/// identifier order, which is not something a user asks for by unticking a box.
struct FrameOrder: Equatable, Sendable {
    var minimalVisualDistance: OrderTerm
    var time: OrderTerm
    var face: OrderTerm
    var art: OrderTerm
    var definition: OrderTerm
    var ratingGeneral: OrderTerm
    var ratingSensitive: OrderTerm
    var ratingQuestionable: OrderTerm
    var ratingExplicit: OrderTerm
    var tagGeneralScore: OrderTerm

    /// The ordinary visual factors in force — rating probabilities remain opt-in.
    init(minimalVisualDistance: OrderTerm = .chronological,
         time: OrderTerm = .chronological, face: OrderTerm = .best, art: OrderTerm = .best, definition: OrderTerm = .best,
         ratingGeneral: OrderTerm = .off(ascending: false), ratingSensitive: OrderTerm = .off(ascending: false),
         ratingQuestionable: OrderTerm = .off(ascending: false), ratingExplicit: OrderTerm = .off(ascending: false),
         tagGeneralScore: OrderTerm = .off(ascending: false)) {
        self.minimalVisualDistance = minimalVisualDistance
        self.time = time
        self.face = face
        self.art = art
        self.definition = definition
        self.ratingGeneral = ratingGeneral
        self.ratingSensitive = ratingSensitive
        self.ratingQuestionable = ratingQuestionable
        self.ratingExplicit = ratingExplicit
        self.tagGeneralScore = tagGeneralScore
    }

    // MARK: - The terms in force

    /// Terms in fixed precedence, skipping the signals that are switched off.
    var terms: [SortTerm] {
        var terms: [SortTerm] = []
        if minimalVisualDistance.enabled { terms.append(SortTerm(key: .minimalVisualDistance, ascending: minimalVisualDistance.ascending)) }
        if time.enabled { terms.append(SortTerm(key: .time, ascending: time.ascending)) }
        if face.enabled { terms.append(SortTerm(key: .face, ascending: face.ascending)) }
        if definition.enabled { terms.append(SortTerm(key: .definition, ascending: definition.ascending)) }
        if art.enabled { terms.append(SortTerm(key: .art, ascending: art.ascending)) }
        if ratingGeneral.enabled { terms.append(SortTerm(key: .ratingGeneral, ascending: ratingGeneral.ascending)) }
        if ratingSensitive.enabled { terms.append(SortTerm(key: .ratingSensitive, ascending: ratingSensitive.ascending)) }
        if ratingQuestionable.enabled { terms.append(SortTerm(key: .ratingQuestionable, ascending: ratingQuestionable.ascending)) }
        if ratingExplicit.enabled { terms.append(SortTerm(key: .ratingExplicit, ascending: ratingExplicit.ascending)) }
        if tagGeneralScore.enabled { terms.append(SortTerm(key: .tagGeneralScore, ascending: tagGeneralScore.ascending)) }
        return terms
    }

    var enabledCount: Int {
        [minimalVisualDistance.enabled, time.enabled, definition.enabled, face.enabled, art.enabled, ratingGeneral.enabled, ratingSensitive.enabled,
         ratingQuestionable.enabled, ratingExplicit.enabled, tagGeneralScore.enabled].filter { $0 }.count
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
            tagGeneralScore: OrderTerm(enabled: keys.contains(.tagGeneralScore), ascending: false)
        )
    }

    static let everything = FrameOrder()
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
/// Keeping this extraction in one value means each arrangement decodes a frame's
/// compressed vector once, even when several rating controls are active.
private struct RatingProbabilityScores {
    private let values: [Float]?

    init(frame: AnalyzedFrame) {
        self.init(probabilities: frame.tagProbabilities.flatMap(TagProbabilityCodec.decode))
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
struct FrameLimit: Hashable, Sendable {
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
struct FrameLimits: Hashable, Sendable {
    var time: FrameLimit = .unlimited
    var face: FrameLimit = .unlimited
    var art: FrameLimit = .unlimited
    var definition: FrameLimit = .unlimited
    var ratingGeneral: FrameLimit = .unlimited
    var ratingSensitive: FrameLimit = .unlimited
    var ratingQuestionable: FrameLimit = .unlimited
    var ratingExplicit: FrameLimit = .unlimited
    var tagGeneralScore: FrameLimit = .unlimited

    static let none = FrameLimits()

    var isEmpty: Bool {
        time.isUnlimited && face.isUnlimited && art.isUnlimited && definition.isUnlimited
            && ratingGeneral.isUnlimited && ratingSensitive.isUnlimited
            && ratingQuestionable.isUnlimited && ratingExplicit.isUnlimited
            && tagGeneralScore.isUnlimited
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
    var tagGeneralScore: ClosedRange<Double>? = nil

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
            tagGeneralScore: span(frames.compactMap { $0.tagGeneralScore.map(Double.init) })
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
        applyingVisualOrder: Bool = true
    ) -> Result {
        let candidates = filteredCandidates ?? frames.filter { frame in
            faceFilter.admits(frame, grouping: grouping)
                && limits.admits(frame, grouping: grouping)
                && tagFilter.admits(frame, vocabulary: vocabulary)
        }
        let scoreSortIndex = tagFilter.scoreSortTag.flatMap { vocabulary?.index(for: $0) }
        let ordered = sorted(candidates, by: order, grouping: grouping,
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
        let tagGeneralScore: Float?
        let tagScore: Float?
        let time: Double
        let identifier: String

        init(frame: AnalyzedFrame, grouping: GroupingSettings, includeRatingProbabilities: Bool,
             tagScoreIndex: Int?) {
            self.frame = frame
            self.art = frame.artScore
            self.definition = frame.faceDefinitionScore
            let probabilities: [Float]?
            if includeRatingProbabilities || tagScoreIndex != nil {
                probabilities = frame.tagProbabilities.flatMap(TagProbabilityCodec.decode)
            } else {
                probabilities = nil
            }
            let ratings = includeRatingProbabilities ? RatingProbabilityScores(probabilities: probabilities) : nil
            self.ratingGeneral = ratings?.value(for: .general).map(Float.init)
            self.ratingSensitive = ratings?.value(for: .sensitive).map(Float.init)
            self.ratingQuestionable = ratings?.value(for: .questionable).map(Float.init)
            self.ratingExplicit = ratings?.value(for: .explicit).map(Float.init)
            self.tagGeneralScore = frame.tagGeneralScore
            self.tagScore = tagScoreIndex.flatMap { index in
                probabilities.flatMap { $0.indices.contains(index) ? $0[index] : nil }
            }
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
            case .face, .definition, .ratingGeneral, .ratingSensitive, .ratingQuestionable, .ratingExplicit, .tagGeneralScore:
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
