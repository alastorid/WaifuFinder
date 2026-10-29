import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// Selectable tags with optional rarity bands.
struct TagFilterView: View {
    @ObservedObject var session: WaifuSession
    @State private var quickFilterText = ""
    @State private var collapsedGroups: Set<String> = [TagRarity.normal.title]
    @AppStorage("tagFilterGroupByRarity") private var groupingEnabled = false
    @State private var selectedTags: Set<String> = []
    @State private var selectionRect: CGRect?

    private var enabled: Bool {
        session.previewFrame == nil && session.uncover == nil && !session.progress.isRunning
    }

    private var hasActiveTags: Bool {
        session.tagFilter.hasActiveTags
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
            .subtracting(session.tagFilter.excludedTags)
            .subtracting(session.tagFilter.orTags))
    }

    private var wantedTags: [String] {
        alphabetical(session.tagFilter.requiredTags)
    }

    private var unwantedTags: [String] {
        alphabetical(session.tagFilter.excludedTags)
    }

    /// Tags offered by the frames shown that are neither common nor classified.
    private var unselectedTags: [String] {
        let common = Set(session.tagCommonTags)
        let grouped = session.tagFilter.orTags
        return session.tagFacetTags.filter {
            !session.tagFilter.requiredTags.contains($0)
                && !session.tagFilter.excludedTags.contains($0)
                && !common.contains($0)
                && !grouped.contains($0)
        }
    }

    private var tags: [String] {
        commonTags + standaloneTags + Array(session.tagFilter.orTags)
    }

    /// Everything the Common group does not hold, in the same lead order.
    private var standaloneTags: [String] {
        wantedTags + unwantedTags + unselectedTags
    }

    private func matching(_ tags: [String]) -> [String] {
        guard !quickFilterText.isEmpty else { return tags }
        return tags.filter { $0.localizedCaseInsensitiveContains(quickFilterText) }
    }

    private var ungroupedTags: [String] { commonTags + standaloneTags }

    private func ordered(_ tags: [String]) -> [String] { tags }

    private func rarityTags(_ rarity: TagRarity) -> [String] {
        ungroupedTags.filter {
            TagRarity.classify(matches: session.tagMatchCounts[$0, default: 0], total: session.tagCandidateCount) == rarity
        }
    }

    private var hasVisibleTags: Bool {
        !matching(ungroupedTags).isEmpty
            || session.tagFilter.orGroups.contains { !matching(Array($0.tags)).isEmpty }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            if session.isLoadingPhotosAlbum {
                Text("Loading...")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                advancedControls
                if showsTaggingStatus {
                    TaggingPanelStatus(session: session, panel: .tags)
                }
                if hasActiveTags || !session.tagFilter.orGroups.isEmpty || !showsTaggingStatus {
                    if session.isSummarizingTags, !tags.isEmpty {
                        HStack(spacing: 5) {
                            ProgressView()
                                .controlSize(.small)
                            Text("Updating tags…")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                    if session.tagVocabulary == nil && !hasActiveTags && session.tagFilter.orGroups.isEmpty {
                        Text("Install WD Tagger in Settings to filter by tags.")
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                            .fixedSize(horizontal: false, vertical: true)
                    } else if tags.isEmpty && session.tagFilter.orGroups.isEmpty {
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
                            if !quickFilterText.isEmpty {
                                Button {
                                    quickFilterText = ""
                                } label: {
                                    Image(systemName: "xmark")
                                        .font(.system(size: 10, weight: .semibold))
                                        .foregroundStyle(.secondary)
                                }
                                .buttonStyle(.plain)
                                .accessibilityLabel("Clear quick filter")
                            }
                        }
                        .padding(.horizontal, 7)
                        .padding(.vertical, 5)
                        .background(Color.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 6))

                        VStack(alignment: .leading, spacing: 8) {
                            ForEach(session.tagFilter.orGroups) { group in
                                orGroup(group)
                            }
                            if groupingEnabled {
                                ForEach(TagRarity.allCases) { rarity in
                                    let allTags = rarityTags(rarity)
                                    let visible = matching(allTags)
                                    if !visible.isEmpty {
                                        rarityGroup(rarity, tags: visible, allTags: allTags)
                                    }
                                }
                            } else {
                                chipFlow(matching(ungroupedTags))
                            }
                        }
                        .overlayPreferenceValue(TagChipBoundsPreferenceKey.self) { anchors in
                            GeometryReader { geometry in
                                let frames: [String: CGRect] = Dictionary(uniqueKeysWithValues: anchors.compactMap { tag, anchor -> (String, CGRect)? in
                                    guard enabled else { return nil }
                                    return (tag, geometry[anchor])
                                })
                                TagRightClickMonitor(tagFrames: frames, selectionRect: $selectionRect,
                                                     onRightClick: rightClickTag, onSelect: { tags, extending in
                                    if extending { selectedTags.formUnion(tags) } else { selectedTags = tags }
                                }, onGroupShortcut: groupSelectedTags)
                                .frame(width: geometry.size.width, height: geometry.size.height)
                                .allowsHitTesting(false)
                                if let rect = selectionRect {
                                    Rectangle().fill(Color.accentColor.opacity(0.12))
                                        .overlay(Rectangle().stroke(Color.accentColor.opacity(0.7)))
                                        .frame(width: rect.width, height: rect.height)
                                        .position(x: rect.midX, y: rect.midY)
                                        .allowsHitTesting(false)
                                }
                            }
                        }

                        if !hasVisibleTags && !quickFilterText.isEmpty {
                            Text("No tags match this quick filter.")
                                .font(.caption)
                                .foregroundStyle(.tertiary)
                        }

                        Text("Required tags must match; each OR group matches any of its tags. Unwanted tags remove matching frames. Candidate tags ignore OR groups.")
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
            if !text.isEmpty { collapsedGroups.removeAll() }
        }
    }

    private func chipFlow(_ tags: [String]) -> some View {
        TagChipFlow(spacing: 5, groupTagCount: 0) {
            ForEach(tags, id: \.self) { tag in tagChip(for: tag) }
        }
    }

    private func rarityGroup(_ rarity: TagRarity, tags: [String], allTags: [String]) -> some View {
        let title = rarity.title
        let color: Color = switch rarity {
        case .legendary: .orange
        case .exquisite: .yellow
        case .magic: .blue
        case .normal: .white
        }
        return DisclosureGroup(isExpanded: Binding(
            get: { !collapsedGroups.contains(title) },
            set: { expanded in
                if expanded { collapsedGroups.remove(title) } else { collapsedGroups.insert(title) }
            }
        )) {
            chipFlow(tags).padding(.top, 4)
        } label: {
            Text("\(title) (\(tags.count))")
                .font(.caption)
                .contextMenu {
                    Button("Exclude \(title) tags") {
                        var filter = session.tagFilter
                        for tag in allTags { filter.setExcluded(true, tag: tag) }
                        session.tagFilter = filter
                    }
                }
        }
        .padding(7)
        .background(color.opacity(0.14), in: RoundedRectangle(cornerRadius: 7))
        .overlay(RoundedRectangle(cornerRadius: 7).stroke(color.opacity(0.6)))
    }

    private func orGroup(_ group: TagOrGroup) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text("OR (\(group.tags.count))")
                .font(.caption.weight(.semibold))
                .help("Drag this label onto another OR group to merge them")
                .onDrag { TagGroupDragPayload.provider(for: group.id) } preview: {
                    Text("OR (\(group.tags.count))")
                        .tagChipBody(isSelected: true, isExcluded: false)
                }
                .contextMenu {
                    Button("Remove") {
                        var filter = session.tagFilter
                        filter.orGroups.removeAll { $0.id == group.id }
                        session.tagFilter = filter
                    }
                }
            if group.tags.isEmpty {
                Text("Drop tags here").font(.caption).foregroundStyle(.tertiary)
            } else {
                chipFlow(matching(ordered(alphabetical(group.tags))))
            }
        }
        .padding(7)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.accentColor.opacity(0.06), in: RoundedRectangle(cornerRadius: 7))
        .overlay(RoundedRectangle(cornerRadius: 7).stroke(Color.accentColor.opacity(0.35)))
        .contentShape(Rectangle())
        .onDrop(of: [UTType.utf8PlainText], isTargeted: nil) { providers in
            guard let provider = providers.first(where: {
                $0.hasItemConformingToTypeIdentifier(UTType.utf8PlainText.identifier)
            }) else { return false }
            provider.loadDataRepresentation(forTypeIdentifier: UTType.utf8PlainText.identifier) { data, _ in
                guard let data else { return }
                if let sourceID = TagGroupDragPayload.decode(data) {
                    Task { @MainActor in
                        var filter = session.tagFilter
                        guard filter.mergeOrGroup(sourceID, into: group.id) else { return }
                        session.tagFilter = filter
                    }
                } else if let tags = TagDragPayload.decode(data) {
                    Task { @MainActor in
                        guard session.tagFilter.orGroups.contains(where: { $0.id == group.id }) else { return }
                        let validTags = tags.intersection(availableTags)
                        guard !validTags.isEmpty else { return }
                        var filter = session.tagFilter
                        filter.addToOrGroup(validTags, id: group.id)
                        session.tagFilter = filter
                        selectedTags.subtract(validTags)
                    }
                }
            }
            return true
        }
    }

    private var availableTags: Set<String> {
        Set(session.tagFacetTags).union(session.tagCommonTags)
            .union(session.tagFilter.requiredTags).union(session.tagFilter.excludedTags)
            .union(session.tagFilter.orTags)
    }

    private func groupSelectedTags() -> Bool {
        guard enabled, !session.crystalOrbIsOpen, !session.showExportSheet,
              !session.showPreviewExportSheet, session.errorMessage == nil else { return false }
        let validTags = selectedTags.intersection(availableTags)
        guard validTags.count > 1 else { return false }
        var filter = session.tagFilter
        guard filter.groupSelectedTags(validTags) != nil else { return false }
        session.tagFilter = filter
        selectedTags.subtract(validTags)
        return true
    }

    private var advancedControls: some View {
        VStack(spacing: 8) {
            HStack(spacing: 8) {
                Text("Top").frame(width: 58, alignment: .leading)
                ValueSlider(bounds: 1...99, value: Double(session.settings.tagTopCount), step: 1,
                            onChange: setTopCount, onCommit: setTopCount,
                            resetValue: Double(WaifuSettings.default.tagTopCount),
                            format: { $0.rounded() == 99 ? "all" : "\(Int($0.rounded()))" })
                    .accessibilityLabel("Top frames per tag")
                    .accessibilityValue(topCountText)
            }
            HStack(spacing: 8) {
                Text("Threshold").frame(width: 58, alignment: .leading)
                ValueSlider(bounds: 0.35...1, value: Double(session.settings.taggerThreshold), step: 0.01,
                            onChange: setThreshold, onCommit: setThreshold,
                            resetValue: Double(WaifuSettings.default.taggerThreshold),
                            format: { String(format: "%.2f", $0) })
                    .accessibilityLabel("Tag threshold")
                    .accessibilityValue(String(format: "%.2f", session.settings.taggerThreshold))
            }
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

    private func tagChip(for tag: String) -> some View {
        let isSelected = session.tagFilter.requiredTags.contains(tag)
        let isExcluded = session.tagFilter.excludedTags.contains(tag)
        let sortsByTag = session.tagFilter.scoreSortTag == tag
        return HStack(spacing: 4) {
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
        .overlay(Capsule().strokeBorder(Color.accentColor, lineWidth: selectedTags.contains(tag) ? 2 : 0))
        .onTapGesture { activateTag(tag) }
        .focusable()
        .onKeyPress(keys: [.space, .return]) { _ in
            activateTag(tag)
            return .handled
        }
        .accessibilityAction { activateTag(tag) }
        .onDrag {
            let tags = selectedTags.contains(tag) ? selectedTags : [tag]
            return TagDragPayload.provider(for: tags)
        } preview: {
            Text(tag)
                .tagChipBody(isSelected: true, isExcluded: false)
                .overlay(alignment: .topTrailing) {
                    if selectedTags.contains(tag) && selectedTags.count > 1 {
                        Text("\(selectedTags.count)").font(.system(size: 10, weight: .bold))
                            .foregroundStyle(.white).padding(4)
                            .background(Color.red, in: Circle()).offset(x: 8, y: -8)
                    }
                }
        }
        .accessibilityAddTraits(.isButton)
        .anchorPreference(key: TagChipBoundsPreferenceKey.self, value: .bounds) {
            [tag: $0]
        }
        .accessibilityLabel(tag)
        .accessibilityValue(accessibilityValue(isSelected: isSelected, isExcluded: isExcluded, sortsByTag: sortsByTag))
    }

    private func activateTag(_ tag: String) {
        if NSEvent.modifierFlags.contains(.command) {
            if selectedTags.contains(tag) { selectedTags.remove(tag) } else { selectedTags.insert(tag) }
            return
        }
        selectedTags.removeAll()
        var filter = session.tagFilter
        let isExcluded = filter.excludedTags.contains(tag)
        let isSelected = filter.requiredTags.contains(tag)
        let sortsByTag = filter.scoreSortTag == tag
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

    /// Right-click clears a selected or unwanted tag, and marks an unselected
    /// one unwanted.
    private func rightClickTag(_ tag: String) {
        guard enabled else { return }
        var filter = session.tagFilter
        if filter.orTags.contains(tag) {
            filter.removeFromOrGroups(tag)
        } else if filter.requiredTags.contains(tag) || filter.excludedTags.contains(tag) {
            filter.setSelected(false, tag: tag)
        } else {
            filter.setExcluded(true, tag: tag)
        }
        session.tagFilter = filter
    }
}

/// A drag carries its complete selection, independent of the source view's lifetime.
enum TagDragPayload {
    private static let prefix = "waifu-tag-drag-v1:"

    static func provider(for tags: Set<String>) -> NSItemProvider {
        let provider = NSItemProvider()
        let data = encode(tags)
        provider.registerDataRepresentation(forTypeIdentifier: UTType.utf8PlainText.identifier,
                                            visibility: .ownProcess) { completion in
            completion(data, nil)
            return nil
        }
        return provider
    }

    static func encode(_ tags: Set<String>) -> Data {
        let data = (try? JSONEncoder().encode(tags.sorted())) ?? Data("[]".utf8)
        return Data(prefix.utf8) + data
    }

    static func decode(_ data: Data) -> Set<String>? {
        let prefixData = Data(prefix.utf8)
        guard data.starts(with: prefixData),
              let values = try? JSONDecoder().decode([String].self, from: data.dropFirst(prefixData.count)),
              !values.isEmpty, values.allSatisfy({ !$0.isEmpty }) else { return nil }
        return Set(values)
    }
}

/// A group drag names its source so the drop can remove that clause after merging.
enum TagGroupDragPayload {
    private static let prefix = "waifu-or-group-v1:"

    static func provider(for id: UUID) -> NSItemProvider {
        let provider = NSItemProvider()
        let data = encode(id)
        provider.registerDataRepresentation(forTypeIdentifier: UTType.utf8PlainText.identifier,
                                            visibility: .ownProcess) { completion in
            completion(data, nil)
            return nil
        }
        return provider
    }

    static func encode(_ id: UUID) -> Data { Data((prefix + id.uuidString).utf8) }

    static func decode(_ data: Data) -> UUID? {
        guard let text = String(data: data, encoding: .utf8), text.hasPrefix(prefix) else { return nil }
        return UUID(uuidString: String(text.dropFirst(prefix.count)))
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
    @Binding var selectionRect: CGRect?
    let onRightClick: (String) -> Void
    let onSelect: (Set<String>, Bool) -> Void
    let onGroupShortcut: () -> Bool

    func makeNSView(context: Context) -> MonitorView {
        let view = MonitorView()
        view.tagFrames = tagFrames
        view.onRightClick = onRightClick
        view.onSelectionRect = { selectionRect = $0 }
        view.onSelect = onSelect
        view.onGroupShortcut = onGroupShortcut
        return view
    }

    func updateNSView(_ view: MonitorView, context: Context) {
        view.tagFrames = tagFrames
        view.onRightClick = onRightClick
        view.onSelectionRect = { selectionRect = $0 }
        view.onSelect = onSelect
        view.onGroupShortcut = onGroupShortcut
    }

    static func dismantleNSView(_ view: MonitorView, coordinator: ()) {
        view.removeMonitor()
    }

    final class MonitorView: NSView {
        var tagFrames: [String: CGRect] = [:]
        var onRightClick: ((String) -> Void)?
        var onSelectionRect: ((CGRect?) -> Void)?
        var onSelect: ((Set<String>, Bool) -> Void)?
        var onGroupShortcut: (() -> Bool)?
        private var selectionStart: CGPoint?
        private var extending = false
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
            monitor = NSEvent.addLocalMonitorForEvents(matching: [.rightMouseDown, .leftMouseDown, .leftMouseDragged, .leftMouseUp, .keyDown]) { [weak self] event in
                let eventWindowNumber = event.window?.windowNumber
                let locationInWindow = event.locationInWindow
                let handled = MainActor.assumeIsolated {
                    guard let self, let window = self.window,
                          eventWindowNumber == window.windowNumber else { return false }
                    if event.type == .keyDown {
                        let modifiers = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
                        guard window.isKeyWindow, window.attachedSheet == nil, !event.isARepeat,
                              !(window.firstResponder is NSTextView),
                              (window.firstResponder as? NSTextField)?.isEditable != true,
                              modifiers.intersection([.command, .control, .option]).isEmpty,
                              event.charactersIgnoringModifiers?.lowercased() == "f" else { return false }
                        return self.onGroupShortcut?() ?? false
                    }
                    let point = self.convert(locationInWindow, from: nil)
                    if event.type == .rightMouseDown {
                        guard let (tag, _) = self.tagFrames.first(where: { $0.value.contains(point) }) else { return false }
                        self.onRightClick?(tag)
                        return true
                    }
                    if event.type == .leftMouseDown {
                        guard self.bounds.contains(point), !self.tagFrames.values.contains(where: { $0.contains(point) }) else {
                            self.selectionStart = nil
                            return false
                        }
                        self.selectionStart = point
                        self.extending = event.modifierFlags.contains(.command)
                    } else if event.type == .leftMouseDragged, let start = self.selectionStart {
                        let rect = CGRect(x: min(start.x, point.x), y: min(start.y, point.y),
                                          width: abs(point.x - start.x), height: abs(point.y - start.y))
                        if rect.width > 3 || rect.height > 3 { self.onSelectionRect?(rect) }
                    } else if event.type == .leftMouseUp, let start = self.selectionStart {
                        self.selectionStart = nil
                        let rect = CGRect(x: min(start.x, point.x), y: min(start.y, point.y),
                                          width: abs(point.x - start.x), height: abs(point.y - start.y))
                        if rect.width > 3 || rect.height > 3 {
                            self.onSelect?(Set(self.tagFrames.filter { $0.value.intersects(rect) }.map(\.key)), self.extending)
                        }
                        self.onSelectionRect?(nil)
                    }
                    return false
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
