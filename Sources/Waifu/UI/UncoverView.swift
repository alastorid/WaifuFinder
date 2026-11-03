import AppKit
import SwiftUI
import UniformTypeIdentifiers

struct UncoverView: View {
    @ObservedObject var session: UncoverSession
    let close: () -> Void
    @Environment(\.openSettings) private var openSettings
    @State private var keyMonitor: Any?
    @State private var modelsChangedWhileBusy = false
    @State private var arrows = PreviewArrowNavigation()
    @State private var showsBusyProgress = false
    @State private var showsFrameProgress = false
    /// Observed so the Reconstruct button's ᴿ mark follows the selected
    /// algorithm and the connection live.
    @ObservedObject private var remote = RemoteClient.shared

    var body: some View {
        GeometryReader { geometry in
            VStack(spacing: 0) {
                header
                Divider()
                if session.busy && showsBusyProgress {
                    VStack(spacing: 16) {
                        ProgressView()
                        Text(session.status)
                            .font(.callout.monospacedDigit())
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    UncoverCanvas(session: session)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .overlay(alignment: .bottomTrailing) {
                            if !session.busy && session.navigatingFrame && showsFrameProgress {
                                ProgressView().controlSize(.small)
                                    .padding(8)
                                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
                                    .padding(10)
                                    .accessibilityLabel(session.readingSource ? "Loading source frame" : "Loading result frame")
                                    .allowsHitTesting(false)
                            }
                        }
                }
                Divider()
                ScrollView(.vertical) {
                    controls(width: geometry.size.width)
                        .padding(12)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(height: min(geometry.size.height * 0.5, geometry.size.width < 760 ? 390 : 320))
                Divider()
                actions
                    .padding(.horizontal, 12)
                    .padding(.vertical, 10)
            }
        }
        .background(Color(nsColor: .windowBackgroundColor))
        .task(id: session.busy) {
            showsBusyProgress = false
            guard session.busy else { return }
            do { try await Task.sleep(for: .milliseconds(330)) } catch { return }
            guard !Task.isCancelled, session.busy else { return }
            showsBusyProgress = true
        }
        .task(id: session.navigatingFrame) {
            showsFrameProgress = false
            guard session.navigatingFrame else { return }
            do { try await Task.sleep(for: .milliseconds(330)) } catch { return }
            guard !Task.isCancelled, session.navigatingFrame else { return }
            showsFrameProgress = true
        }
        .onAppear { installKeys() }
        .onDisappear { if let keyMonitor { NSEvent.removeMonitor(keyMonitor) }; keyMonitor = nil; arrows.reset() }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didResignActiveNotification)) { _ in arrows.reset() }
        .onChange(of: session.navigatingFrame) { _, reading in
            if !reading, session.error == nil, !session.busy, let delta = arrows.loaded() { session.step(delta) }
        }
        .onReceive(NotificationCenter.default.publisher(for: .waifuModelsChanged)) { _ in
            if !session.busy { session.refreshBackend() }
            else { modelsChangedWhileBusy = true }
        }
        .onChange(of: session.busy) { _, busy in
            if busy { arrows.reset() }
            else if modelsChangedWhileBusy { modelsChangedWhileBusy = false; session.refreshBackend() }
        }
    }

    private var header: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 12) {
                title
                Text(session.geometry).font(.caption.monospaced()).foregroundStyle(.secondary)
                Spacer(minLength: 0)
                zoomControls
                Button("Close", action: close)
            }
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    title
                    Spacer(minLength: 0)
                    Button("Close", action: close)
                }
                HStack(spacing: 8) {
                    Text(session.geometry)
                        .font(.caption.monospaced()).foregroundStyle(.secondary)
                        .lineLimit(1).truncationMode(.middle)
                    Spacer(minLength: 0)
                    zoomControls
                }
            }
        }
        .buttonStyle(.borderless)
        .padding(12)
    }

    private var title: some View {
        Text("Uncover Waifu").font(.headline).fixedSize()
    }

    private var zoomControls: some View {
        HStack(spacing: 8) {
            Button("−") { session.zoom = max(0.25, session.zoom / 1.5) }
            Text(String(format: "%.0f%%", session.zoom * 100))
                .font(.caption.monospacedDigit()).frame(minWidth: 40)
            Button("+") { session.zoom = min(16, session.zoom * 1.5) }
            HStack(spacing: 2) {
                scaleModeButton(.original, symbol: nil, label: "Original pixel size")
                scaleModeButton(.fitWidth, symbol: "arrow.left.and.right", label: "Fit video to width")
                scaleModeButton(.fitHeight, symbol: "arrow.up.and.down", label: "Fit video to height")
            }
            .padding(3)
            .background(.quaternary, in: RoundedRectangle(cornerRadius: 6))
        }
        .fixedSize()
    }

    private func scaleModeButton(_ mode: UncoverViewScaleMode, symbol: String?, label: String) -> some View {
        Button { session.setViewScaleMode(mode) } label: {
            Group {
                if let symbol { Image(systemName: symbol) } else { Text("Orig") }
            }
            .padding(.horizontal, 6).padding(.vertical, 4)
            .background(session.viewScaleMode == mode ? Color.accentColor.opacity(0.2) : .clear,
                        in: RoundedRectangle(cornerRadius: 4))
        }
        .accessibilityLabel(label)
        .help(label)
    }

    private func controls(width: CGFloat) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            if session.editing || !session.hasCurrentResult { sourceTimeline }
            if session.hasCurrentResult {
                if !session.editing { resultTimeline }
                controlRow("Compare") {
                    UncoverControlFlow(spacing: 6) {
                        ForEach(UncoverComparison.allCases, id: \.self) { comparison in
                            choice(comparison.rawValue, selected: session.comparison == comparison) { session.comparison = comparison }
                        }
                        Button(session.showB ? "Show Original" : "Show AI") {
                            session.comparison = .toggle
                            session.showB.toggle()
                        }
                        Button(session.editing ? "Compare" : "Edit Region") { session.toggleEditing() }
                    }
                }
                .disabled(session.busy)
            }
            controlRow("Algorithm") {
                UncoverControlFlow(spacing: 6) {
                    // Keep all licensed choices visible independently of the remote connection.
                    ForEach(session.availableAlgorithms, id: \.self) { algorithm in
                        choice(algorithm.label, selected: session.algorithm == algorithm) { session.algorithm = algorithm }
                    }
                    if session.capabilities?.available != true {
                        Button("Manage Models…") {
                            UserDefaults.standard.set("models", forKey: "settingsTab")
                            openSettings()
                        }
                    }
                }
            }
            .disabled(session.busy)
            controlRow("Downsample") {
                UncoverControlFlow(spacing: 6) {
                    ForEach(UncoverScale.allCases, id: \.self) { scale in
                        choice(scale == .one ? "1× Original" : scale.label, selected: session.scale == scale) { session.scale = scale }
                            .disabled(session.capabilities?.scales.contains(scale) != true)
                    }
                }
            }
            .disabled(session.busy)
            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 16, alignment: .topLeading),
                                     count: width >= 560 ? 2 : 1),
                      alignment: .leading, spacing: 12) {
                VStack(alignment: .leading, spacing: 6) {
                    fieldLabel("Start")
                    HStack(spacing: 6) {
                        TextField("Seconds", value: $session.start, format: .number.precision(.fractionLength(3)))
                            .frame(width: 90).textFieldStyle(.roundedBorder)
                            .accessibilityLabel("Start seconds")
                        Text("seconds").foregroundStyle(.secondary)
                        Text("to \(timecode(session.end))")
                            .monospacedDigit().foregroundStyle(.secondary)
                    }
                }
                VStack(alignment: .leading, spacing: 6) {
                    fieldLabel("Duration")
                    UncoverControlFlow(spacing: 6) {
                        ForEach([1.0, 2.0, 5.0], id: \.self) { duration in
                            choice("\(Int(duration))s", selected: session.duration == duration) { session.duration = duration }
                        }
                        HStack(spacing: 5) {
                            TextField("Seconds", value: $session.duration, format: .number.precision(.fractionLength(3)))
                                .frame(width: 80).textFieldStyle(.roundedBorder)
                                .accessibilityLabel("Duration seconds")
                            Text("s").foregroundStyle(.secondary)
                        }
                    }
                }
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        fieldLabel("Edge Feather")
                        Spacer(minLength: 8)
                        Text("\(Int(session.feather)) px").monospacedDigit().foregroundStyle(.secondary)
                    }
                    Slider(value: $session.feather, in: 0...32, step: 1)
                        .accessibilityLabel("Edge Feather")
                }
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        fieldLabel("Blend Strength")
                        Spacer(minLength: 8)
                        Text("\(Int((session.strength * 100).rounded()))%").monospacedDigit().foregroundStyle(.secondary)
                    }
                    Slider(value: $session.strength, in: 0...1)
                        .accessibilityLabel("Blend Strength")
                }
            }
            .disabled(session.busy)
            VStack(alignment: .leading, spacing: 4) {
                Text(session.status).foregroundStyle(.secondary)
                if let limitation = session.limitation { Text(limitation).foregroundStyle(.secondary) }
                if session.algorithm == .flashVSR11 {
                    Text("FlashVSR outputs 4× source size and is tuned for 4× super-resolution; other input reductions change the effective ratio.")
                        .foregroundStyle(.secondary)
                }
                if let error = session.error { Text(error).foregroundStyle(.red).textSelection(.enabled) }
                Text("Scroll to pan · Pinch to zoom · Option-drag to pan").foregroundStyle(.tertiary)
            }
            .fixedSize(horizontal: false, vertical: true)
        }
        .font(.caption)
        .controlSize(.small)
    }

    private func fieldLabel(_ title: String) -> some View {
        Text(title).font(.caption.weight(.medium)).foregroundStyle(.secondary)
    }

    private func controlRow<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        HStack(alignment: .top, spacing: 12) {
            fieldLabel(title).frame(width: 84, alignment: .leading).padding(.top, 5)
            content().frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func choice(_ title: String, selected: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title).fixedSize()
                .padding(.horizontal, 8).padding(.vertical, 5)
                .foregroundStyle(selected ? Color.accentColor : Color.primary)
                .background(selected ? Color.accentColor.opacity(0.16) : Color.secondary.opacity(0.07),
                            in: RoundedRectangle(cornerRadius: 5))
                .overlay(RoundedRectangle(cornerRadius: 5)
                    .strokeBorder(selected ? Color.accentColor.opacity(0.65) : Color.secondary.opacity(0.22)))
        }
        .buttonStyle(.plain)
        .accessibilityValue(selected ? "Selected" : "Not selected")
    }

    private var sourceTimeline: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                fieldLabel("Source")
                Spacer(minLength: 8)
                Text("\(timecode(session.sourceTime)) · \(session.sourceDisplayedIndex + 1)/\(session.sourceFrameCount)")
                    .monospacedDigit().foregroundStyle(.secondary)
            }
            HStack(spacing: 10) {
                Button { session.step(-1) } label: { Image(systemName: "backward.end.fill") }
                Slider(value: Binding(get: { Double(session.sourceRequestedIndex) },
                                      set: { session.seekSource(index: Int($0.rounded())) }),
                       in: 0...Double(max(1, session.sourceFrameCount - 1)))
                    .accessibilityLabel("Source frame")
                Button { session.step(1) } label: { Image(systemName: "forward.end.fill") }
            }
        }
        .disabled(session.busy || session.sourceFrameCount == 0)
    }

    private var resultTimeline: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                fieldLabel("Result")
                Spacer(minLength: 8)
                Text("\(timecode(session.displayedTime)) · \(session.displayedIndex + 1)/\(session.frameCount)").monospacedDigit()
            }
            HStack(spacing: 10) {
                Button { session.togglePlayback() } label: { Image(systemName: session.playing ? "pause.fill" : "play.fill") }
                Button { session.step(-1) } label: { Image(systemName: "backward.end.fill") }
                Slider(value: Binding(get: { Double(session.displayedIndex) }, set: { session.seek(index: Int($0.rounded())) }),
                       in: 0...Double(max(1, session.frameCount - 1)))
                    .accessibilityLabel("Result frame")
                Button { session.step(1) } label: { Image(systemName: "forward.end.fill") }
                Toggle("Loop", isOn: $session.loop).toggleStyle(.checkbox).fixedSize()
            }
        }
        .disabled(session.busy)
    }

    private var actions: some View {
        HStack(spacing: 10) {
            Spacer(minLength: 0)
            if session.busy {
                if showsBusyProgress { ProgressView().controlSize(.small) }
                Button("Cancel") { session.cancel() }
            } else {
                Button { session.reconstruct() } label: {
                    Text("Reconstruct").remoteMark(remote.offersRemote(session.algorithm, scale: session.scale))
                }
                .disabled(!session.canReconstruct)
                Button("Export…") { save() }.disabled(!session.hasCurrentResult)
            }
        }
    }

    private func save() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = session.isStillPhoto ? [.png] : [.quickTimeMovie]
        panel.nameFieldStringValue = session.source.deletingPathExtension().lastPathComponent + (session.isStillPhoto ? "-uncovered.png" : "-uncovered.mov")
        panel.begin { response in
            guard response == .OK, let url = panel.url else { return }
            session.export(to: url)
        }
    }
    private func installKeys() {
        guard keyMonitor == nil else { return }
        keyMonitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown, .keyUp],
                handler: Diag.tracedKeyHandler("uncover.monitor", categories: [.keys, .uncover]) { event in
                MainActor.assumeIsolated {
                    let isUp = event.type == .keyUp
                    if isUp, event.keyCode == 123 || event.keyCode == 124 {
                        arrows.up(event.keyCode == 123 ? -1 : 1)
                        return false
                    }
                    guard !isUp else { return false }
                    if event.keyCode == 53 { close(); return true }
                    if event.window?.firstResponder is NSTextView { return false }
                    guard event.modifierFlags.intersection([.command, .control, .option]).isEmpty else { return false }
                    guard !session.busy else { return false }
                    switch event.keyCode {
                    case 49: session.togglePlayback(); return true
                    case 123, 124:
                        if let delta = arrows.down(event.keyCode == 123 ? -1 : 1,
                                                   isRepeat: event.isARepeat, loading: session.navigatingFrame) {
                            session.step(delta)
                        }
                        return true
                    case 11: session.comparison = .toggle; session.showB.toggle(); return true
                    default: return false
                    }
                }
            })
    }
}

/// Wrap whole controls rather than compressing their labels into neighbouring
/// fields. Width and content changes invalidate cached measurements; scrolling
/// does not change the arrangement.
private struct UncoverControlFlow: Layout {
    var spacing: CGFloat = 6
    struct Cache {
        var sizes: [CGSize]
        var arrangements: [CGFloat: TagFlowGeometry] = [:]
    }
    func makeCache(subviews: Subviews) -> Cache {
        Cache(sizes: subviews.map { $0.sizeThatFits(.unspecified) })
    }
    func updateCache(_ cache: inout Cache, subviews: Subviews) { cache = makeCache(subviews: subviews) }
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
        for (subview, frame) in zip(subviews, result.placements) {
            subview.place(at: CGPoint(x: bounds.minX + frame.minX, y: bounds.minY + frame.minY),
                          proposal: ProposedViewSize(frame.size))
        }
    }
}
