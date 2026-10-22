import SwiftUI

private struct VisualSortSpinner: View {
    @State private var rotating = false

    var body: some View {
        ZStack {
            Circle().stroke(.secondary.opacity(0.2), lineWidth: 2)
            Circle()
                .trim(from: 0, to: 0.7)
                .stroke(Color.accentColor, style: StrokeStyle(lineWidth: 2, lineCap: .round))
                .rotationEffect(.degrees(rotating ? 360 : 0))
                .animation(.linear(duration: 0.8).repeatForever(autoreverses: false), value: rotating)
        }
        .frame(width: 14, height: 14)
        .accessibilityLabel("Sorting by visual distance")
        .help("Sorting by visual distance")
        .onAppear { rotating = true }
    }
}

private struct RegroupProgressRing: View {
    @ObservedObject var progress: RegroupProgress

    var body: some View {
        ZStack {
            Circle().stroke(.secondary.opacity(0.2), lineWidth: 2)
            Circle()
                .trim(from: 0, to: progress.fraction)
                .stroke(Color.accentColor, style: StrokeStyle(lineWidth: 2, lineCap: .round))
                .rotationEffect(.degrees(-90))
        }
        .frame(width: 14, height: 14)
        .accessibilityElement()
        .accessibilityLabel("Updating groups")
        .accessibilityValue("\(Int(progress.fraction * 100)) percent")
        .help("Updating groups · \(Int(progress.fraction * 100))%")
    }
}

/// The right-hand panel: what the grid shows, in what order, how much of each group,
/// and how hard the app groups.
///
/// When enabled, visual distance arranges the shown set after the other enabled
/// signals rank candidates and choose group representatives. Otherwise the independent
/// switches order by time, face, face definition, art, then rating probabilities, and carry
/// a **range**, which filters:
/// drag the two ends to keep only frames between them. The range's ends are the span
/// the data actually covers, printed on the two handles themselves, so the control
/// says what it is filtering by instead of hiding it behind a 0…1 bar — and says it
/// without spending a second line per signal on it.
///
/// Every control writes through a keyed mutator on `FrameOrder` / `FrameLimits`
/// rather than through a captured position, because a `Binding(get:set:)` outlives
/// the body pass that built it — SwiftUI evaluates it again on later updates and on
/// layout. That is how this panel crashed once, in a getter, from
/// `NSHostingView.layout()`.
struct FrameOrderPanel: View {
    @ObservedObject var session: WaifuSession
    /// Shared with the cloud in the title bar; see `ChiaroscuroLink`.
    var link: ChiaroscuroLink
    @State private var orderExpanded = true
    @State private var groupingExpanded = true
    @State private var tagsExpanded = true

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 10.5) {
                PanelPresetManager(session: session)
                orderPanel
                tagsPanel
                groupingPanel
                showSection
                Divider()
                summary
                Divider()
                actions
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .background(Color(nsColor: .controlBackgroundColor))
    }

    // MARK: - Order and limits

    private var orderPanel: some View {
        collapsiblePanel(title: "Sort and Filter", isExpanded: $orderExpanded) {
            HStack(spacing: 4) {
                shownFraction
                Text("shown")
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            .monospacedDigit()
        } content: {
            orderSection
        }
    }

    /// The fraction on its own, because it is the fraction — not the word beside it
    /// — that starts the east egg: three seconds of stillness on a fraction that is
    /// hiding frames sends a wave of light to the switched-off cloud in the title
    /// bar. The padding is only there to give a caption-sized target a resting
    /// place; it changes nothing about how the panel reads.
    private var shownFraction: some View {
        let shown = session.displayedFrames.count
        let total = session.frames.count
        return Text("\(formattedCount(shown))/\(formattedCount(total))")
            .padding(.vertical, 2)
            .contentShape(Rectangle())
            .chiaroscuroSource(link, isLinkable: Chiaroscuro.canLink(shown: shown, total: total))
    }

    private var orderSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            signalRow(.minimalVisualDistance)
            signalRow(.time)
            signalRow(.face)
            signalRow(.definition)
            signalRow(.art)
            ratingSection
        }
    }

    private var ratingSection: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 10) {
                if session.isLoadingPhotosAlbum {
                    Text("Loading...")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                } else if session.taggingProgress.isRunning || session.taggedFrameCount == 0 {
                    TaggingPanelStatus(session: session, panel: .rating)
                } else {
                    signalRow(.ratingGeneral)
                    signalRow(.ratingSensitive)
                    signalRow(.ratingQuestionable)
                    signalRow(.ratingExplicit)

                    if session.untaggedFrameCountForTagging > 0 {
                        TaggingStartButton(session: session, panel: .rating)
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        } label: {
            Text("Rating")
                .font(.headline)
        }
    }

    /// One signal: enablement and direction, plus a range for filterable scores.
    private func signalRow(_ key: FrameSortKey) -> some View {
        let term = session.frameOrder.term(key)
        let limit = session.frameLimits.limit(for: key)
        let isVisualOrder = key == .minimalVisualDistance
        let direction = isVisualOrder
            ? (term.ascending ? "forward" : "reverse")
            : (term.ascending ? "asc" : "desc")
        // The last signal standing cannot be switched off — an order with nothing in
        // it would be identifier order, which no checkbox asks for. `FrameOrder`
        // refuses it too; disabling the control says so before the click.
        let lockedOn = term.enabled && session.frameOrder.enabledCount <= 1

        return HStack(spacing: 8) {
            Toggle(isOn: enabledBinding(key)) {
                HStack(spacing: 6) {
                    Text(key.label).font(.callout)
                    if isVisualOrder, session.isSortingVisualDistance {
                        VisualSortSpinner()
                    }
                }
            }
            .toggleStyle(.checkbox)
            .fixedSize()
            .disabled(lockedOn)
            .help(key.detail)

            if key.supportsFiltering {
                signalRange(key, limit: limit)
                    .frame(maxWidth: .infinity)
            } else {
                Spacer(minLength: 0)
            }

            Button {
                session.frameOrder.toggleDirection(of: key)
            } label: {
                HStack(spacing: 3) {
                    Image(systemName: isVisualOrder
                          ? (term.ascending ? "arrow.right" : "arrow.left")
                          : (term.ascending ? "arrow.up" : "arrow.down"))
                    Text(direction)
                }
                .font(.system(size: 10))
            }
            .buttonStyle(.borderless)
            .fixedSize()
            .disabled(!term.enabled)
            .help(isVisualOrder
                  ? "\(key.label): \(direction). Click to reverse the sequence."
                  : (term.ascending
                     ? "\(key.label) ascending — click for descending"
                     : "\(key.label) descending — click for ascending"))
        }
    }

    @ViewBuilder
    private func signalRange(_ key: FrameSortKey, limit: FrameLimit) -> some View {
        if let bounds = session.limitBounds.range(for: key) {
            let low = clamp(limit.low ?? bounds.lowerBound, to: bounds)
            let high = clamp(limit.high ?? bounds.upperBound, to: bounds)
            RangeSlider(bounds: bounds, low: low, high: high,
                        format: { valueText(key, $0) }) { newLow, newHigh in
                session.frameLimits.setLimit(FrameLimit(low: newLow, high: newHigh), for: key)
            } onReset: {
                session.frameLimits.setLimit(.unlimited, for: key)
            }
            .padding(.horizontal, 2)
            .help("\(key.label) — \(valueText(key, low)) to \(valueText(key, high)). Drag either end of the bar; double-click to clear this range.")
        } else {
            Text("no \(key.label.lowercased()) values on screen")
                .font(.caption)
                .foregroundStyle(.tertiary)
                .lineLimit(1)
        }
    }

    /// Keeps a slider showing a thumb inside its track even if the data's span shrank
    /// under a limit that was set earlier.
    private func clamp(_ value: Double, to bounds: ClosedRange<Double>) -> Double {
        min(max(value, bounds.lowerBound), bounds.upperBound)
    }

    private func valueText(_ key: FrameSortKey, _ value: Double) -> String {
        key == .time ? timecode(value) : String(format: "%.2f", value)
    }

    private func enabledBinding(_ key: FrameSortKey) -> Binding<Bool> {
        Binding(
            get: { session.frameOrder.term(key).enabled },
            set: { session.frameOrder.setEnabled($0, for: key) }
        )
    }

    // MARK: - Grouping

    /// The checkbox reports the layout currently in use. `groupRows` remains the
    /// user's saved preference while grouping is paused or visual-distance ordering
    /// requires a continuous grid.
    private var groupRowsBinding: Binding<Bool> {
        Binding(
            get: { session.usesGroupRows },
            set: { session.groupRows = $0 }
        )
    }

    private var groupingPanel: some View {
        collapsiblePanel(title: "Grouping", isExpanded: $groupingExpanded,
                         titleAccessory: session.isRegrouping
                            ? AnyView(RegroupProgressRing(progress: session.regroupProgress))
                            : nil) {
            HStack(spacing: 8) {
                if session.groupingEnabled {
                    Text("\(session.groups.count) group\(session.groups.count == 1 ? "" : "s")")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Toggle(isOn: $session.groupingEnabled) { EmptyView() }
                    .labelsHidden()
                    .toggleStyle(.checkbox)
                    .accessibilityLabel("Enable grouping")
                    .help("Enable or pause grouping similar frames.")
            }
        } content: {
            VStack(alignment: .leading, spacing: 10) {
                if session.groupingEnabled {
                    HStack(spacing: 8) {
                        Text("Similarity")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .frame(width: 58, alignment: .leading)
                        ValueSlider(
                            bounds: groupingBounds,
                            value: Double(session.groupingThreshold),
                            step: Double(groupingStep),
                            // Update the preview continuously, then regroup on release.
                            onChange: { session.groupingThreshold = Float($0) },
                            onCommit: { value in
                                session.groupingThreshold = Float(value)
                                session.regroup()
                            },
                            resetValue: Double(WaifuSettings.default.similarityThreshold),
                            format: { String(format: "%.2f", $0) }
                        )
                        .accessibilityLabel("Similarity")
                        .accessibilityValue(String(format: "%.2f", session.groupingThreshold))
                        .help("Lower values make more, smaller groups; higher values make fewer, larger groups. Double-click to restore the default. ←/→ step by \(String(format: "%.2f", groupingStep)).")
                    }
                } else {
                    Text("Frames are shown together. Per-group limits and group rows are paused.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }

                perGroupSection

                Toggle(isOn: groupRowsBinding) {
                    Text("Group view")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                    .toggleStyle(.switch)
                    .controlSize(.small)
                    .disabled(!session.groupingEnabled || session.frameOrder.minimalVisualDistance.enabled)
                    .help(!session.groupingEnabled
                          ? "Turn grouping on to lay out one group per line."
                          : session.frameOrder.minimalVisualDistance.enabled
                          ? "Turn off Minimal Visual Distance to lay out one group per line."
                          : "Keep each group together, either on its own line or in one continuous flow grid.")

            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var tagsPanel: some View {
        collapsiblePanel(title: "Tags", isExpanded: $tagsExpanded) {
            if !session.tagFilter.requiredTags.isEmpty || !session.tagFilter.excludedTags.isEmpty {
                Button("Clear") {
                    session.tagFilter = TagFilter()
                }
                .buttonStyle(.borderless)
                .font(.caption)
                .help("Turn off all tag filters")
            }
        } content: {
            TagFilterView(session: session)
        }
    }

    private var tagsAdvancedBinding: Binding<Bool> {
        Binding(get: { session.settings.tagAdvancedMode }, set: { advanced in
            var settings = session.settings
            settings.tagAdvancedMode = advanced
            session.settings = settings
        })
    }

    /// Bordered panel with an independently controlled disclosure header.
    private func collapsiblePanel<Accessory: View, Content: View>(
        title: String,
        isExpanded: Binding<Bool>,
        titleAccessory: AnyView? = nil,
        @ViewBuilder accessory: () -> Accessory,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: isExpanded.wrappedValue ? 8 : 0) {
            HStack(spacing: 8) {
                Button {
                    withAnimation(.easeInOut(duration: 0.16)) {
                        isExpanded.wrappedValue.toggle()
                    }
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: "chevron.right")
                            .rotationEffect(.degrees(isExpanded.wrappedValue ? 90 : 0))
                            .font(.system(size: 10, weight: .semibold))
                        Text(title)
                            .font(.headline)
                        if let titleAccessory {
                            titleAccessory
                        }
                        Spacer(minLength: 0)
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .contextMenu {
                    if title == "Tags" {
                        Picker("Tags mode", selection: tagsAdvancedBinding) {
                            Text("Basic").tag(false)
                            Text("Advanced").tag(true)
                        }
                        .pickerStyle(.inline)
                    }
                }
                .accessibilityLabel("\(title) panel")
                .accessibilityValue(isExpanded.wrappedValue ? "Expanded" : "Collapsed")

                accessory()
            }
            if isExpanded.wrappedValue {
                content()
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// One step of the grouping knob, for the arrow keys. The stepping itself lives in
    /// `ValueSlider`, which owns the focus those keys need.
    private let groupingStep: Float = 0.01
    private var groupingBounds: ClosedRange<Double> { 0.05...1.2 }

    // MARK: - Per-group cap
    private var perGroupSection: some View {
        HStack(spacing: 8) {
            Text("Top")
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(width: 58, alignment: .leading)
            ValueSlider(
                bounds: 1...34,
                value: session.settings.revealAllGroups ? 34 : Double(session.settings.perGroupFrameCount),
                step: 1,
                onChange: setGroupTop,
                onCommit: setGroupTop,
                resetValue: Double(WaifuSettings.default.perGroupFrameCount),
                format: { $0.rounded() == 34 ? "all" : "\(Int($0.rounded()))" }
            )
            .disabled(!session.groupingEnabled)
            .accessibilityLabel("Top items per group")
            .accessibilityValue(session.settings.revealAllGroups ? "all" : "\(session.settings.perGroupFrameCount)")
            .help("Keep the top 1–33 items in each group. Move fully right to reveal all. Double-click to restore the default. Use +N more in the grid to reveal an individual group.")
        }
    }

    /// One settings write keeps the count and reveal mode in sync. The extra
    /// endpoint preserves 33 as a finite cap and existing saved preferences.
    private func setGroupTop(_ value: Double) {
        let count = min(max(Int(value.rounded()), 1), 34)
        var settings = session.settings
        settings.revealAllGroups = count == 34
        if count < 34 { settings.perGroupFrameCount = count }
        session.settings = settings
    }

    // MARK: - Show
    private var showSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Show")
                .font(.headline)

            VStack(alignment: .leading, spacing: 4) {
                Text("Faces")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                Picker("Faces", selection: $session.faceFilter) {
                    ForEach(FrameFaceFilter.allCases) { filter in
                        Text(filter.label).tag(filter)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .controlSize(.small)
                .help("Filter by significant faces, including small faces scored in Face definition mode.")
            }
        }
    }

    // MARK: - Summary and actions

    private var summary: some View {
        VStack(alignment: .leading, spacing: 4) {
            stat("Selected", "\(session.selectedFrameIDs.count)")
            stat("Groups", session.groupingEnabled ? "\(session.groups.count)" : "Off")
            if let elapsed = elapsedDescription {
                stat("Analysed in", elapsed)
            }
        }
    }

    private var actions: some View {
        VStack(alignment: .leading, spacing: 6) {
            Button("Select all shown") {
                for frame in session.displayedFrames { session.select(frame.id) }
            }
            .disabled(session.displayedFrames.isEmpty)

            Button("Clear selection") {
                session.clearSelection()
            }
            .disabled(session.selectedFrameIDs.isEmpty)
        }
        .buttonStyle(.link)
    }

    private func formattedCount(_ count: Int) -> String {
        count.formatted(IntegerFormatStyle<Int>(locale: Locale(identifier: "en_US")))
    }

    private func stat(_ label: String, _ value: String) -> some View {
        HStack(spacing: 6) {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer(minLength: 8)
            Text(value)
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(.primary)
        }
    }

    /// The wall-clock the analysis took, once it has finished.
    private var elapsedDescription: String? {
        guard session.progress.phase == .complete, session.progress.elapsed > 0 else { return nil }
        let elapsed = session.progress.elapsed
        return elapsed < 60
            ? String(format: "%.1fs", elapsed)
            : String(format: "%dm %02ds", Int(elapsed) / 60, Int(elapsed) % 60)
    }
}
