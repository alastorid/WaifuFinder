import AppKit
import SwiftUI

/// A faceted tag filter. A Common chip leads the wrapping chips and discloses the
/// tags every frame shown carries; opening it expands the chip in place, those tags
/// filling a group directly beneath it while the rest of the line waits below. What
/// remains is wanted, unwanted, and the rest by mean score, each in the order the
/// section documents. A tag the filter has taken a position on is always shown
/// outside the Common group, so the chip's styling matches the section holding it.
struct TagFilterView: View {
    @ObservedObject var session: WaifuSession
    @State private var quickFilterText = ""
    @State private var commonExpanded = false

    private var enabled: Bool {
        session.previewFrame == nil && session.uncover == nil && !session.progress.isRunning
    }

    private var hasActiveTags: Bool {
        !session.tagFilter.requiredTags.isEmpty || !session.tagFilter.excludedTags.isEmpty
    }

    private var showsTaggingStatus: Bool {
        session.taggingProgress.isRunning || session.taggedFrameCount == 0
    }

    private func alphabetical(_ tags: Set<String>) -> [String] {
        tags.sorted { $0.localizedStandardCompare($1) == .orderedAscending }
    }

    /// Tags carried by enough frames to be worth offering, minus those the filter
    /// has classified. A common tag that is also wanted or unwanted leads that
    /// group instead of appearing here.
    private var commonTags: [String] {
        alphabetical(Set(session.tagCommonTags)
            .subtracting(session.tagFilter.requiredTags)
            .subtracting(session.tagFilter.excludedTags))
    }

    private var wantedTags: [String] {
        alphabetical(session.tagFilter.requiredTags)
    }

    private var unwantedTags: [String] {
        alphabetical(session.tagFilter.excludedTags)
    }

    /// Tags offered by the frames shown that are neither common nor classified.
    private var unselectedTags: [String] {
        session.tagFacetTags.filter {
            !session.tagFilter.requiredTags.contains($0)
                && !session.tagFilter.excludedTags.contains($0)
                && !session.tagCommonTags.contains($0)
        }
    }

    private var tags: [String] {
        commonTags + standaloneTags
    }

    /// Everything the Common group does not hold, in the same lead order.
    private var standaloneTags: [String] {
        wantedTags + unwantedTags + unselectedTags
    }

    private func matching(_ tags: [String]) -> [String] {
        guard !quickFilterText.isEmpty else { return tags }
        return tags.filter { $0.localizedCaseInsensitiveContains(quickFilterText) }
    }

    /// Active filters must remain visible and removable, even when the quick
    /// search or the current results no longer contain their tags.
    private var visibleStandaloneTags: [String] {
        wantedTags + unwantedTags + matching(unselectedTags)
    }

    private var visibleCommonTags: [String] { matching(commonTags) }

    /// What the group discloses, and so how many subviews the flow should hold
    /// inside it. The chip and the group box are both drawn from this, so a group
    /// with nothing to show contributes neither.
    private var commonGroupTagCount: Int {
        commonExpanded ? visibleCommonTags.count : 0
    }

    private var hasVisibleTags: Bool {
        !visibleStandaloneTags.isEmpty || !visibleCommonTags.isEmpty
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            if session.isLoadingPhotosAlbum {
                Text("Loading...")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                if session.settings.tagAdvancedMode {
                    advancedControls
                }
                if showsTaggingStatus {
                    TaggingPanelStatus(session: session, panel: .tags)
                }
                if hasActiveTags || !showsTaggingStatus {
                    if session.tagVocabulary == nil && !hasActiveTags {
                        Text("Install WD Tagger in Settings to filter by tags.")
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                            .fixedSize(horizontal: false, vertical: true)
                    } else if tags.isEmpty {
                        Text(session.isSummarizingTags ? "Updating tags…" : session.frames.isEmpty
                             ? "Tags will appear after a video is analyzed."
                             : "No tags among the frames currently shown.")
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                            .fixedSize(horizontal: false, vertical: true)
                    } else {
                        HStack(spacing: 6) {
                            Image(systemName: "magnifyingglass")
                                .font(.system(size: 11))
                                .foregroundStyle(.secondary)
                            TextField("Quick filter tags", text: $quickFilterText)
                                .textFieldStyle(.plain)
                                .font(.caption)
                                .accessibilityLabel("Quick filter tags")
                                .help("Search available tags. Wanted and unwanted tags stay visible.")
                            if !quickFilterText.isEmpty {
                                Button {
                                    quickFilterText = ""
                                } label: {
                                    Image(systemName: "xmark")
                                        .font(.system(size: 10, weight: .semibold))
                                        .foregroundStyle(.secondary)
                                }
                                .buttonStyle(.plain)
                                .help("Clear quick filter")
                                .accessibilityLabel("Clear quick filter")
                            }
                        }
                        .padding(.horizontal, 7)
                        .padding(.vertical, 5)
                        .background(Color.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 6))

                        TagChipFlow(spacing: 5, groupTagCount: commonGroupTagCount) {
                            if commonGroupTagCount > 0 {
                                commonTagGroupBackground
                            }
                            if !commonTags.isEmpty {
                                commonGroupChip
                            }
                            // The group's tags are inside the group while it is open and
                            // absent while it is not, which is what the flow's group size
                            // counts, so the two cannot drift apart.
                            ForEach(visibleCommonTags.prefix(commonGroupTagCount), id: \.self) { tag in
                                tagChip(for: tag)
                            }
                            ForEach(visibleStandaloneTags, id: \.self) { tag in
                                tagChip(for: tag)
                            }
                        }
                        .overlayPreferenceValue(TagChipBoundsPreferenceKey.self) { anchors in
                            GeometryReader { geometry in
                                let selectedFrames: [String: CGRect] = Dictionary(uniqueKeysWithValues: anchors.compactMap { tag, anchor -> (String, CGRect)? in
                                    guard enabled else { return nil }
                                    return (tag, geometry[anchor])
                                })
                                TagRightClickMonitor(tagFrames: selectedFrames) { tag in
                                    rightClickTag(tag)
                                }
                                .frame(width: geometry.size.width, height: geometry.size.height)
                                .allowsHitTesting(false)
                            }
                        }

                        if !hasVisibleTags {
                            Text("No tags match this quick filter.")
                                .font(.caption)
                                .foregroundStyle(.tertiary)
                        }

                        Text("Required tags must match; unwanted tags remove matching frames. Available tags update with the frames shown and rank by average confidence above the threshold.")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                            .fixedSize(horizontal: false, vertical: true)
                    }

                    if !showsTaggingStatus, session.untaggedFrameCountForTagging > 0 {
                        TaggingStartButton(session: session, panel: .tags)
                    }
                }
            }
        }
        .disabled(!enabled)
        .onChange(of: quickFilterText) { _, text in
            // A quick filter that only matches common tags would otherwise land
            // inside a collapsed group, leaving the panel looking empty-handed.
            if !text.isEmpty, !matching(commonTags).isEmpty, !commonExpanded {
                withAnimation(.easeInOut(duration: 0.16)) { commonExpanded = true }
            }
        }
        .onChange(of: visibleCommonTags.isEmpty) { _, isEmpty in
            // A filter that matches none of the common tags empties the group, and
            // a group with nothing in it has nothing to expand to.
            if isEmpty, commonExpanded {
                commonExpanded = false
            }
        }
    }

    private var advancedControls: some View {
        VStack(spacing: 8) {
            HStack(spacing: 8) {
                Text("Top").frame(width: 58, alignment: .leading)
                ValueSlider(bounds: 1...99, value: Double(session.settings.tagTopCount), step: 1,
                            onChange: setTopCount, onCommit: setTopCount,
                            format: { $0.rounded() == 99 ? "all" : "\(Int($0.rounded()))" })
                    .accessibilityLabel("Top frames per tag")
                    .accessibilityValue(topCountText)
            }
            .help("Keep the top N frames for every tag, or for each selected tag. Set to all to remove the limit.")
            HStack(spacing: 8) {
                Text("Threshold").frame(width: 58, alignment: .leading)
                ValueSlider(bounds: 0.35...1, value: Double(session.settings.taggerThreshold), step: 0.01,
                            onChange: setThreshold, onCommit: setThreshold,
                            format: { String(format: "%.2f", $0) })
                    .accessibilityLabel("Tag threshold")
                    .accessibilityValue(String(format: "%.2f", session.settings.taggerThreshold))
            }
            .help("Filter using stored FP8 tag scores. Lowering the threshold restores matches without retagging.")
        }
        .font(.caption)
        .monospacedDigit()
        .foregroundStyle(.secondary)
    }

    private var topCountText: String {
        session.settings.tagTopCount == 99 ? "all" : "\(session.settings.tagTopCount)"
    }

    private func setTopCount(_ value: Double) {
        var settings = session.settings
        settings.tagTopCount = min(max(Int(value.rounded()), 1), 99)
        session.settings = settings
    }

    private func setThreshold(_ value: Double) {
        var settings = session.settings
        settings.taggerThreshold = Float(min(max((value * 100).rounded() / 100, 0.35), 1))
        session.settings = settings
    }

    /// The chip that stands in for the common tags themselves. It borrows the
    /// normal tag capsule so the group reads as one of the tags, and only the
    /// chevron and the count mark it as a control. Opening it hands the flow the
    /// tags it stands for, which the flow lays out inside the group it draws
    /// directly beneath the chip.
    private var commonGroupChip: some View {
        Button {
            withAnimation(.easeInOut(duration: 0.16)) { commonExpanded.toggle() }
        } label: {
            HStack(spacing: 4) {
                Image(systemName: "chevron.right")
                    .rotationEffect(.degrees(commonExpanded ? 90 : 0))
                    .font(.system(size: 9, weight: .semibold))
                Text("Common")
                    .font(.system(size: 14))
                    .lineLimit(1)
                Text("\(visibleCommonTags.count)")
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(.tertiary)
            }
            .tagChipBody(isSelected: false, isExcluded: false)
        }
        .buttonStyle(.plain)
        .help(commonExpanded
            ? "Collapse the common tags"
            : "Show the \(commonTags.count) tags every frame shown carries")
        .accessibilityLabel("Common tags")
        .accessibilityValue(commonExpanded ? "Expanded" : "Collapsed")
    }

    /// The inset box the expanded group is drawn into. The flow measures it
    /// against the group it contains, so the border wraps the chip and the tags
    /// it discloses without either of them having to know the group's size.
    private var commonTagGroupBackground: some View {
        RoundedRectangle(cornerRadius: 7)
            .fill(Color.secondary.opacity(0.06))
            .overlay {
                RoundedRectangle(cornerRadius: 7).strokeBorder(Color.secondary.opacity(0.14), lineWidth: 1)
            }
            .layoutValue(key: TagChipGroupBackgroundKey.self, value: true)
            .allowsHitTesting(false)
            .accessibilityHidden(true)
    }

    private func tagChip(for tag: String) -> some View {
        let isSelected = session.tagFilter.requiredTags.contains(tag)
        let isExcluded = session.tagFilter.excludedTags.contains(tag)
        let sortsByTag = session.tagFilter.scoreSortTag == tag
        return Button {
            var filter = session.tagFilter
            if isExcluded {
                filter.setSelected(true, tag: tag)
                filter.scoreSortTag = tag
                filter.scoreSortDescending = true
            } else if isSelected {
                if sortsByTag {
                    filter.scoreSortDescending.toggle()
                } else {
                    filter.scoreSortTag = tag
                    filter.scoreSortDescending = true
                }
            } else {
                filter.setSelected(true, tag: tag)
                filter.scoreSortTag = tag
                filter.scoreSortDescending = true
            }
            session.tagFilter = filter
        } label: {
            HStack(spacing: 4) {
                if sortsByTag {
                    Image(systemName: session.tagFilter.scoreSortDescending ? "arrow.down" : "arrow.up")
                        .font(.system(size: 9, weight: .semibold))
                }
                highlightedTagText(tag)
                    .font(.system(size: 14))
                    .lineLimit(1)
                    .strikethrough(isExcluded)
            }
            .tagChipBody(isSelected: isSelected, isExcluded: isExcluded)
        }
        .buttonStyle(.plain)
        .anchorPreference(key: TagChipBoundsPreferenceKey.self, value: .bounds) {
            [tag: $0]
        }
        .accessibilityLabel(tag)
        .accessibilityValue(accessibilityValue(isSelected: isSelected, isExcluded: isExcluded, sortsByTag: sortsByTag))
        .help(helpText(isSelected: isSelected, isExcluded: isExcluded))
    }

    private func highlightedTagText(_ tag: String) -> Text {
        guard !quickFilterText.isEmpty,
              let range = tag.range(of: quickFilterText, options: [.caseInsensitive, .diacriticInsensitive]) else {
            return Text(tag)
        }
        return Text(String(tag[..<range.lowerBound]))
            + Text(String(tag[range])).foregroundColor(.red)
            + Text(String(tag[range.upperBound...]))
    }

    private func accessibilityValue(isSelected: Bool, isExcluded: Bool, sortsByTag: Bool) -> String {
        if isExcluded { return "Unwanted; matching frames are excluded" }
        guard isSelected else { return "Not selected" }
        guard sortsByTag else { return "Selected" }
        return session.tagFilter.scoreSortDescending
            ? "Selected, highest score first"
            : "Selected, lowest score first"
    }

    /// Only the right-click effect is worth a tooltip: the click behaviour is
    /// visible in the chip's own styling and its sort arrow. Right-click clears a
    /// selected or unwanted tag, and marks an unselected one unwanted.
    private func helpText(isSelected: Bool, isExcluded: Bool) -> String {
        if isSelected || isExcluded { return "Right-click to remove this tag" }
        return "Right-click to exclude frames with this tag"
    }

    private func rightClickTag(_ tag: String) {
        guard enabled else { return }
        var filter = session.tagFilter
        if filter.requiredTags.contains(tag) || filter.excludedTags.contains(tag) {
            filter.setSelected(false, tag: tag)
        } else {
            filter.setExcluded(true, tag: tag)
        }
        session.tagFilter = filter
    }
}

private extension View {
    func tagChipBody(isSelected: Bool, isExcluded: Bool) -> some View {
        modifier(TagChipBody(isSelected: isSelected, isExcluded: isExcluded))
    }
}

/// The capsule every tag shares, so the chips inside the Common group read as tags
/// rather than as a nested control of their own.
private struct TagChipBody: ViewModifier {
    let isSelected: Bool
    let isExcluded: Bool

    func body(content: Content) -> some View {
        content
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(
                isSelected ? Color.accentColor.opacity(0.2) :
                    (isExcluded ? Color.orange.opacity(0.16) : Color.secondary.opacity(0.09)),
                in: Capsule()
            )
            .overlay {
                Capsule().strokeBorder(
                    isSelected ? Color.accentColor.opacity(0.55) :
                        (isExcluded ? Color.orange.opacity(0.5) : Color.secondary.opacity(0.18)),
                    lineWidth: 1
                )
            }
            .contentShape(Capsule())
    }
}

private struct TagChipBoundsPreferenceKey: PreferenceKey {
    nonisolated(unsafe) static var defaultValue: [String: Anchor<CGRect>] = [:]

    static func reduce(value: inout [String: Anchor<CGRect>], nextValue: () -> [String: Anchor<CGRect>]) {
        value.merge(nextValue(), uniquingKeysWith: { _, latest in latest })
    }
}

/// Observes secondary mouse clicks in the chip layout and consumes a click on a
/// selected chip so it removes the tag immediately, without opening a context menu.
private struct TagRightClickMonitor: NSViewRepresentable {
    let tagFrames: [String: CGRect]
    let onRightClick: (String) -> Void

    func makeNSView(context: Context) -> MonitorView {
        let view = MonitorView()
        view.tagFrames = tagFrames
        view.onRightClick = onRightClick
        return view
    }

    func updateNSView(_ view: MonitorView, context: Context) {
        view.tagFrames = tagFrames
        view.onRightClick = onRightClick
    }

    static func dismantleNSView(_ view: MonitorView, coordinator: ()) {
        view.removeMonitor()
    }

    final class MonitorView: NSView {
        var tagFrames: [String: CGRect] = [:]
        var onRightClick: ((String) -> Void)?
        private var monitor: Any?

        override var isFlipped: Bool { true }

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            removeMonitor()
            installMonitor()
        }

        func removeMonitor() {
            if let monitor { NSEvent.removeMonitor(monitor) }
            monitor = nil
        }

        private func installMonitor() {
            guard monitor == nil, window != nil else { return }
            monitor = NSEvent.addLocalMonitorForEvents(matching: [.rightMouseDown]) { [weak self] event in
                let eventWindowNumber = event.window?.windowNumber
                let locationInWindow = event.locationInWindow
                let handled = MainActor.assumeIsolated {
                    guard let self, let window = self.window,
                          eventWindowNumber == window.windowNumber else { return false }
                    let point = self.convert(locationInWindow, from: nil)
                    guard let (tag, _) = self.tagFrames.first(where: { $0.value.contains(point) }) else {
                        return false
                    }
                    self.onRightClick?(tag)
                    return true
                }
                return handled ? nil : event
            }
        }
    }
}

/// Places tags at their natural width and starts a new line when the next chip
/// would exceed the filter panel's available width. An expanded Common group is
/// laid out in place: its background is sized to the block holding the group chip
/// and the tags that chip discloses, and the tags after the group carry on below
/// that block as if it were the chips it stands for.
private struct TagChipFlow: Layout {
    /// Gap between chips, and between wrapped lines.
    var spacing: CGFloat
    /// How many tags the group discloses. Zero leaves the group chip collapsed and
    /// nothing else in the flow knows the group exists.
    var groupTagCount: Int
    /// Padding between the group's border and the chips inside it.
    var groupInset: CGFloat = 7

    private struct Placement {
        let x: CGFloat
        let y: CGFloat
        let size: CGSize
    }

    private struct Arrangement {
        var placements: [Placement]
        var width: CGFloat
        var height: CGFloat
    }

    /// The block an expanded group occupies: its background, the chip that stands
    /// for it, and the tags inside it.
    private struct Group {
        var placements: [Placement]
        var width: CGFloat
        var height: CGFloat
    }

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let availableWidth = proposal.width ?? .infinity
        let arrangement = arrange(subviews, width: availableWidth)
        return CGSize(width: proposal.width ?? arrangement.width, height: arrangement.height)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let arrangement = arrange(subviews, width: bounds.width)
        for (subview, placement) in zip(subviews, arrangement.placements) {
            subview.place(
                at: CGPoint(x: bounds.minX + placement.x, y: bounds.minY + placement.y),
                proposal: ProposedViewSize(width: placement.size.width, height: placement.size.height)
            )
        }
    }

    private func arrange(_ subviews: Subviews, width: CGFloat) -> Arrangement {
        let group = groupRange(in: subviews)
        var placements: [Placement] = []
        var usedWidth: CGFloat = 0
        var y: CGFloat = 0

        if !group.isEmpty {
            let block = arrangeGroup(subviews, indices: group, width: width)
            placements.append(contentsOf: block.placements)
            usedWidth = max(usedWidth, block.width)
            y = block.height + spacing
        }

        // The tags the group does not stand for, wrapping as they did before it
        // existed — only their starting line has moved.
        var trailing = Band(spacing: spacing, width: width)
        for index in group.upperBound..<subviews.endIndex {
            trailing.add(subviews[index].sizeThatFits(.unspecified))
        }
        placements.append(contentsOf: trailing.placements.map {
            Placement(x: $0.x, y: $0.y + y, size: $0.size)
        })
        usedWidth = max(usedWidth, trailing.usedWidth)

        // The gap that opened the line below the group is not part of the flow's
        // height unless a chip actually moved onto that line.
        return Arrangement(
            placements: placements,
            width: width.isFinite ? min(width, usedWidth) : usedWidth,
            height: trailing.isEmpty ? y - spacing : y + trailing.height
        )
    }

    /// The leading subviews an expanded group occupies. The group's chip is the
    /// first tag of the flow by construction — the filter puts it there — so the
    /// group is taken from the front, and the background the caller drew in front
    /// of the chip is part of it.
    private func groupRange(in subviews: Subviews) -> Range<Int> {
        guard groupTagCount > 0, subviews.count >= 1 + groupTagCount else { return 0..<0 }
        let hasBackground = subviews.first?[TagChipGroupBackgroundKey.self] == true
        return 0..<(1 + groupTagCount + (hasBackground ? 1 : 0))
    }

    /// The group leads the flow, so it is always measured against the top of it.
    private func arrangeGroup(_ subviews: Subviews, indices: Range<Int>, width: CGFloat) -> Group {
        let hasBackground = subviews[indices.lowerBound][TagChipGroupBackgroundKey.self]
        let chipIndex = indices.lowerBound + (hasBackground ? 1 : 0)
        let chipSize = subviews[chipIndex].sizeThatFits(.unspecified)

        // The tags wrap inside the border, so they are laid out against the width
        // the border leaves rather than the panel's own.
        var content = Band(spacing: spacing, width: width.isFinite ? max(0, width - groupInset * 2) : .infinity)
        for index in (chipIndex + 1)..<indices.upperBound {
            content.add(subviews[index].sizeThatFits(.unspecified))
        }

        // The tags start under the chip, which the group chip is itself the label
        // of, so the group's height is its top inset, the chip, the tags, and its
        // bottom inset.
        let tagsTop = groupInset + chipSize.height + spacing
        let height = (content.isEmpty ? groupInset + chipSize.height : tagsTop + content.height) + groupInset
        let groupWidth = width.isFinite ? width : groupInset * 2 + content.usedWidth

        var placements: [Placement] = []
        if hasBackground {
            placements.append(Placement(x: 0, y: 0, size: CGSize(width: groupWidth, height: height)))
        }
        placements.append(Placement(x: groupInset, y: groupInset, size: chipSize))
        for placement in content.placements {
            placements.append(Placement(
                x: groupInset + placement.x,
                y: tagsTop + placement.y,
                size: placement.size
            ))
        }
        return Group(placements: placements, width: groupWidth, height: height)
    }

    /// Fills lines with chips at their natural width, breaking before the first
    /// chip that would not fit on the line being filled.
    private struct Band {
        let spacing: CGFloat
        let width: CGFloat
        private(set) var placements: [Placement] = []
        private var x: CGFloat = 0
        private var y: CGFloat = 0
        private var lineHeight: CGFloat = 0

        init(spacing: CGFloat, width: CGFloat) {
            self.spacing = spacing
            self.width = width
        }

        var isEmpty: Bool { placements.isEmpty }
        /// The widest line, and so the width the band asks for when nothing
        /// proposes one.
        var usedWidth: CGFloat { placements.map { $0.x + $0.size.width }.max() ?? 0 }
        var height: CGFloat { placements.map { $0.y + $0.size.height }.max() ?? 0 }

        mutating func add(_ size: CGSize) {
            if x > 0 && x + size.width > width {
                y += lineHeight + spacing
                x = 0
                lineHeight = 0
            }
            placements.append(Placement(x: x, y: y, size: size))
            x += size.width + spacing
            lineHeight = max(lineHeight, size.height)
        }
    }
}

private struct TagChipGroupBackgroundKey: LayoutValueKey {
    static var defaultValue: Bool { false }
}
