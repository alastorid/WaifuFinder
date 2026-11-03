import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// Selectable tags with optional rarity bands.
struct TagFilterView: View {
    @ObservedObject var session: WaifuSession
    @State private var quickFilterText = ""
    @State private var collapsedGroups: Set<String> = defaultCollapsedGroups
    @AppStorage("tagFilterRarityGrouping") private var groupingEnabled = true
    @AppStorage("tagRarityAlwaysShowAllGroups") private var alwaysShowAllRarityGroups = ""
    @State private var expandedRarityGroups: Set<Int> = []
    @State private var expandedTagGroups: Set<String> = []
    @AppStorage("tagAlwaysShowAllGroups") private var alwaysShowAllTagGroups = ""
    @AppStorage("tagShowAllGroups") private var showAllGroups = false
    @State private var selectedTags: Set<String> = []
    @State private var selectionRect: CGRect?


    private static var defaultCollapsedGroups: Set<String> {
        Set(TagRarity.allCases.filter { $0 != .legendary }.map(\.title))
    }

    private var enabled: Bool {
        session.previewFrame == nil && session.uncover == nil && !session.progress.isRunning
    }

    private var hasActiveTags: Bool {
        session.tagFilter.hasActiveTags
    }

    private var showsTaggingStatus: Bool {
        session.taggingProgress.isRunning || session.taggedFrameCount == 0
    }

    private var commonTags: [String] { session.tagBrowserSnapshot.common }
    private var wantedTags: [String] { session.tagBrowserSnapshot.wanted }
    private var unwantedTags: [String] { session.tagBrowserSnapshot.unwanted }
    private var unselectedTags: [String] { session.tagBrowserSnapshot.unselected }
    private var tags: [String] { session.tagBrowserSnapshot.all }

    private func matching(_ tags: [String]) -> [String] {
        guard !quickFilterText.isEmpty else { return tags }
        return tags.filter { $0.localizedCaseInsensitiveContains(quickFilterText) }
    }

    private var ungroupedTags: [String] { session.tagBrowserSnapshot.ungrouped }

    private func alwaysShowsAll(_ rarity: TagRarity) -> Bool {
        alwaysShowAllRarityGroups.split(separator: ",").contains { Int($0) == rarity.rawValue }
    }

    private func setAlwaysShowsAll(_ show: Bool, for rarity: TagRarity) {
        var values = Set(alwaysShowAllRarityGroups.split(separator: ",").compactMap { Int($0) })
        if show { values.insert(rarity.rawValue) } else { values.remove(rarity.rawValue) }
        alwaysShowAllRarityGroups = values.sorted().map(String.init).joined(separator: ",")
        if !show { expandedRarityGroups.remove(rarity.rawValue) }
    }

    private func alwaysShowsAllGroup(_ key: String) -> Bool {
        alwaysShowAllTagGroups.split(separator: ",").contains { $0 == key }
    }

    private func showsAllGroup(_ key: String) -> Bool {
        showAllGroups || expandedTagGroups.contains(key) || alwaysShowsAllGroup(key)
    }

    private func setAlwaysShowsAllGroup(_ show: Bool, key: String) {
        var values = Set(alwaysShowAllTagGroups.split(separator: ",").map(String.init))
        if show { values.insert(key) } else { values.remove(key) }
        alwaysShowAllTagGroups = values.sorted().joined(separator: ",")
        if !show { expandedTagGroups.remove(key) }
    }

    private func alwaysShowAllGroupToggle(_ key: String) -> some View {
        Toggle("Always show all tags in this group", isOn: Binding(
            get: { alwaysShowsAllGroup(key) },
            set: { setAlwaysShowsAllGroup($0, key: key) }
        ))
    }

    private var hasVisibleOrGroups: Bool {
        !session.tagFilter.unwantedGroups.isEmpty
            || session.tagFilter.orGroups.contains { !matching(Array($0.tags)).isEmpty }
    }

    private var hasVisibleTags: Bool {
        if groupingEnabled {
            return !matching(commonTags).isEmpty
                || !wantedTags.isEmpty
                || !unwantedTags.isEmpty
                || !matching(unselectedTags).isEmpty
                || hasVisibleOrGroups
        }
        return !matching(ungroupedTags).isEmpty || hasVisibleOrGroups
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
                if hasActiveTags || !session.tagFilter.orGroups.isEmpty || !session.tagFilter.unwantedGroups.isEmpty || !showsTaggingStatus {
                    if session.isSummarizingTags, !tags.isEmpty {
                        HStack(spacing: 5) {
                            ProgressView()
                                .controlSize(.small)
                            Text("Updating tags…")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                    if session.tagVocabulary == nil && !hasActiveTags && session.tagFilter.orGroups.isEmpty && session.tagFilter.unwantedGroups.isEmpty {
                        Text("Install WD Tagger in Settings to filter by tags.")
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                            .fixedSize(horizontal: false, vertical: true)
                    } else if tags.isEmpty && session.tagFilter.orGroups.isEmpty && session.tagFilter.unwantedGroups.isEmpty {
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
                            if groupingEnabled {
                                groupedTagSections
                            } else {
                                ForEach(session.tagFilter.unwantedGroups) { group in
                                    unwantedGroup(group)
                                }
                                ForEach(session.tagFilter.orGroups) { group in
                                    orGroup(group)
                                }
                                chipFlow(matching(ungroupedTags), groupKey: "ungrouped")
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

                        Text("Wanted tags must match; each OR group matches any of its tags. Unwanted tags remove matching frames. Candidate tags ignore OR groups. Select tags with Command-click or a rectangle, then press F1 to group them.")
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
        .onChange(of: session.tagControlsResetRevision) { _, _ in
            quickFilterText = ""
            selectedTags.removeAll()
            expandedTagGroups.removeAll()
            expandedRarityGroups.removeAll()
            showAllGroups = false
            selectionRect = nil
            groupingEnabled = true
            collapsedGroups = Self.defaultCollapsedGroups
        }
        .onChange(of: groupingEnabled) { _, enabled in
            if enabled { collapsedGroups = Self.defaultCollapsedGroups }
        }
        .onChange(of: quickFilterText) { _, text in
            if !text.isEmpty { collapsedGroups.removeAll() }
        }
    }

    @ViewBuilder
    private var groupedTagSections: some View {
        let visibleCommon = matching(commonTags)
        if !visibleCommon.isEmpty {
            tagSection("Common", tags: visibleCommon, color: .secondary, isLite: true)
        }
        let wantedGroups = session.tagFilter.unwantedGroups.filter(\.isWanted)
        let unwantedGroups = session.tagFilter.unwantedGroups.filter { !$0.isWanted }
        if !wantedTags.isEmpty || !wantedGroups.isEmpty {
            tagSection("Wanted", tags: wantedTags, color: .accentColor, inverseGroups: wantedGroups)
        }
        if !unwantedTags.isEmpty || !unwantedGroups.isEmpty {
            tagSection("Unwanted", tags: unwantedTags, color: .orange, inverseGroups: unwantedGroups)
        }
        ForEach(session.tagFilter.orGroups) { group in
            orGroup(group)
        }
        ForEach(TagRarity.allCases) { rarity in
            let allTags = session.tagBrowserSnapshot.rarityBands[rarity.rawValue]
            let visible = matching(allTags)
            if !visible.isEmpty {
                rarityGroup(rarity, tags: visible, allTags: allTags)
            }
        }
    }

    private func chipFlow(_ tags: [String], groupKey: String) -> some View {
        let displayed = TagRarityDisplay.limited(tags, showingAll: showsAllGroup(groupKey))
        return TagChipFlow(spacing: 5) {
            ForEach(displayed, id: \.self) { tag in tagChip(for: tag) }
            if displayed.count < tags.count {
                showAllTagsButton(total: tags.count) { expandedTagGroups.insert(groupKey) }
            }
        }
    }

    /// Common tags share one quiet container. Wanted and unwanted tags use the
    /// same shape with enough tint to stay visible above the rarity bands.
    private func tagSection(_ title: String, tags: [String], color: Color,
                            isLite: Bool = false, inverseGroups: [TagInverseGroup] = []) -> some View {
        let key = "section:\(title)"
        let showsAll = showsAllGroup(key)
        let total = tags.count + inverseGroups.count
        let displayedGroups = showsAll ? inverseGroups : Array(inverseGroups.prefix(TagRarityDisplay.tagLimit))
        let displayedTags = TagRarityDisplay.limited(tags, showingAll: showsAll,
            limit: max(0, TagRarityDisplay.tagLimit - displayedGroups.count))
        return VStack(alignment: .leading, spacing: 5) {
            Text("\(title) (\(tags.count + inverseGroups.count))")
                .font(.caption.weight(isLite ? .regular : .semibold))
                .foregroundStyle(color)
                .contextMenu {
                    groupActions(title, tags: Set(tags))
                        .disabled(tags.isEmpty)
                    Divider()
                    alwaysShowAllGroupToggle(key)
                }
            TagChipFlow(spacing: 5) {
                ForEach(displayedGroups) { group in unwantedGroup(group) }
                ForEach(displayedTags, id: \.self) { tag in tagChip(for: tag) }
                if displayedGroups.count + displayedTags.count < total {
                    showAllTagsButton(total: total) { expandedTagGroups.insert(key) }
                }
            }
        }
        .padding(7)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(color.opacity(isLite ? 0.04 : 0.08), in: RoundedRectangle(cornerRadius: 7))
        .overlay {
            RoundedRectangle(cornerRadius: 7)
                .strokeBorder(color.opacity(isLite ? 0.14 : 0.35))
        }
    }

    @ViewBuilder
    private func groupActions(_ title: String, tags: Set<String>) -> some View {
        Button("Include \(title) tags") {
            var filter = session.tagFilter
            filter.includeTags(tags)
            session.tagFilter = filter
        }
        Button("Exclude \(title) tags") {
            var filter = session.tagFilter
            for tag in tags { filter.setExcluded(true, tag: tag) }
            session.tagFilter = filter
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
        let showsAll = showAllGroups || alwaysShowsAll(rarity) || expandedRarityGroups.contains(rarity.rawValue)
        let displayedTags = TagRarityDisplay.limited(tags, showingAll: showsAll)
        let hiddenCount = tags.count - displayedTags.count
        let isExpanded = !collapsedGroups.contains(title)
        return VStack(alignment: .leading, spacing: 4) {
            Button {
                if isExpanded { collapsedGroups.insert(title) } else { collapsedGroups.remove(title) }
            } label: {
                HStack(spacing: 5) {
                    Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                        .font(.caption2.weight(.semibold))
                    Text("\(title) (\(tags.count))")
                        .font(.caption)
                    Spacer(minLength: 0)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel("\(title), \(tags.count) tags")
            .accessibilityValue(isExpanded ? "Expanded" : "Collapsed")
            .contextMenu {
                groupActions(title, tags: Set(allTags))
                Divider()
                Toggle("Always show all tags in this group", isOn: Binding(
                    get: { alwaysShowsAll(rarity) },
                    set: { setAlwaysShowsAll($0, for: rarity) }
                ))
            }
            if isExpanded {
                TagChipFlow(spacing: 5) {
                    ForEach(displayedTags, id: \.self) { tag in tagChip(for: tag) }
                    if hiddenCount > 0 {
                        showAllTagsButton(total: tags.count) { expandedRarityGroups.insert(rarity.rawValue) }
                    }
                }
                .padding(.leading, 15)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(7)
        .background(color.opacity(0.14), in: RoundedRectangle(cornerRadius: 7))
        .overlay(RoundedRectangle(cornerRadius: 7).stroke(color.opacity(0.6)))
    }

    private func showAllTagsButton(total: Int, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text("…")
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(.secondary)
                .padding(.horizontal, 9)
                .padding(.vertical, 4)
                .background(Color.secondary.opacity(0.09), in: Capsule())
                .overlay {
                    Capsule().strokeBorder(Color.secondary.opacity(0.18))
                }
                .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .help("Show all \(total) tags")
        .accessibilityLabel("Show all \(total) tags")
    }

    private func orGroup(_ group: TagOrGroup) -> some View {
        let key = "or:\(group.id.uuidString)"
        return VStack(alignment: .leading, spacing: 5) {
            Text("OR (\(group.tags.count))")
                .font(.caption.weight(.semibold))
                .contextMenu {
                    groupActions("OR", tags: group.tags)
                    Divider()
                    alwaysShowAllGroupToggle(key)
                    Divider()
                    Button("Remove") {
                        var filter = session.tagFilter
                        filter.orGroups.removeAll { $0.id == group.id }
                        session.tagFilter = filter
                    }
                }
            if group.tags.isEmpty {
                Text("Select tags and press F1").font(.caption).foregroundStyle(.tertiary)
            } else {
                chipFlow(matching(session.tagBrowserSnapshot.orGroupTags[group.id, default: []]), groupKey: key)
            }
        }
        .padding(7)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.accentColor.opacity(0.06), in: RoundedRectangle(cornerRadius: 7))
        .overlay(RoundedRectangle(cornerRadius: 7).stroke(Color.accentColor.opacity(0.35)))
        .contentShape(Rectangle())

    }

    private func unwantedGroup(_ group: TagInverseGroup) -> some View {
        let color: Color = group.isWanted ? .teal : .purple
        return Text("\(group.tags.count) tags")
            .font(.system(size: 14))
            .strikethrough(!group.isWanted)
            .foregroundStyle(color)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(color.opacity(0.16), in: Capsule())
            .overlay(Capsule().strokeBorder(color.opacity(0.55)))
            .contentShape(Capsule())
            .onTapGesture { setInverseGroupWanted(true, id: group.id) }
            .focusable()
            .onKeyPress(keys: [.space, .return]) { _ in
                setInverseGroupWanted(true, id: group.id)
                return .handled
            }
            .accessibilityAction { setInverseGroupWanted(true, id: group.id) }
            .accessibilityAddTraits(.isButton)
            .contextMenu {
                Button("Unwant") { setInverseGroupWanted(false, id: group.id) }
                    .disabled(!group.isWanted)
                Button("Want") { setInverseGroupWanted(true, id: group.id) }
                    .disabled(group.isWanted)
                Divider()
                Button("Remove") {
                    var filter = session.tagFilter
                    filter.unwantedGroups.removeAll { $0.id == group.id }
                    session.tagFilter = filter
                }
            }
            .help("Inverse tag group. Right-click to choose Unwant, Want, or Remove.")
            .accessibilityLabel("\(group.isWanted ? "Wanted" : "Unwanted") group, \(group.tags.count) tags")
    }

    private func setInverseGroupWanted(_ wanted: Bool, id: UUID) {
        var filter = session.tagFilter
        guard let index = filter.unwantedGroups.firstIndex(where: { $0.id == id }) else { return }
        filter.unwantedGroups[index].isWanted = wanted
        session.tagFilter = filter
    }

    private var availableTags: Set<String> { session.tagBrowserSnapshot.available }

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
                              modifiers.intersection([.command, .control, .option, .shift]).isEmpty,
                              event.keyCode == 122 else { return false } // F1, independent of keyboard layout
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

/// Intrinsic chip sizes and wrapping are cached independently of scroll position.
/// SwiftUI can ask for layout repeatedly without measuring every label again.
private struct TagChipFlow: Layout {
    var spacing: CGFloat

    struct Cache {
        var sizes: [CGSize]
        var arrangements: [CGFloat: TagFlowGeometry] = [:]
    }

    func makeCache(subviews: Subviews) -> Cache {
        Cache(sizes: subviews.map { $0.sizeThatFits(.unspecified) })
    }

    func updateCache(_ cache: inout Cache, subviews: Subviews) {
        cache = makeCache(subviews: subviews)
    }

    private func arrangement(width: CGFloat, cache: inout Cache) -> TagFlowGeometry {
        if let existing = cache.arrangements[width] { return existing }
        let result = TagFlowGeometry(sizes: cache.sizes, width: width, spacing: spacing)
        if cache.arrangements.count >= 4 { cache.arrangements.removeAll(keepingCapacity: true) }
        cache.arrangements[width] = result
        return result
    }

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout Cache) -> CGSize {
        let result = arrangement(width: proposal.width ?? .infinity, cache: &cache)
        return CGSize(width: proposal.width ?? result.size.width, height: result.size.height)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout Cache) {
        let result = arrangement(width: bounds.width, cache: &cache)
        for (subview, placement) in zip(subviews, result.placements) {
            subview.place(at: CGPoint(x: bounds.minX + placement.origin.x, y: bounds.minY + placement.origin.y),
                          proposal: ProposedViewSize(placement.size))
        }
    }
}
