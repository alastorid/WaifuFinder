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

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 14) {
                Text("Uncover Waifu").font(.headline)
                Text(session.geometry).font(.system(size: 11, design: .monospaced)).foregroundStyle(.secondary)
                Spacer()
                Button("−") { session.zoom = max(0.25, session.zoom / 1.5) }
                Text(String(format: "%.0f%%", session.zoom * 100)).font(.caption.monospacedDigit())
                Button("+") { session.zoom = min(16, session.zoom * 1.5) }
                HStack(spacing: 2) {
                    Button { session.setViewScaleMode(.original) } label: { Text("Orig") }
                        .help("Show at original pixel size")
                        .accessibilityLabel("Original pixel size")
                        .padding(.horizontal, 6).padding(.vertical, 4)
                        .background(session.viewScaleMode == .original ? Color.accentColor.opacity(0.2) : .clear,
                                    in: RoundedRectangle(cornerRadius: 4))
                    Button { session.setViewScaleMode(.fitWidth) } label: { Image(systemName: "arrow.left.and.right") }
                        .help("Fit video to width")
                        .accessibilityLabel("Fit video to width")
                        .padding(.horizontal, 6).padding(.vertical, 4)
                        .background(session.viewScaleMode == .fitWidth ? Color.accentColor.opacity(0.2) : .clear,
                                    in: RoundedRectangle(cornerRadius: 4))
                    Button { session.setViewScaleMode(.fitHeight) } label: { Image(systemName: "arrow.up.and.down") }
                        .help("Fit video to height")
                        .accessibilityLabel("Fit video to height")
                        .padding(.horizontal, 6).padding(.vertical, 4)
                        .background(session.viewScaleMode == .fitHeight ? Color.accentColor.opacity(0.2) : .clear,
                                    in: RoundedRectangle(cornerRadius: 4))
                }.buttonStyle(.borderless).padding(3).background(.quaternary, in: RoundedRectangle(cornerRadius: 6))
                Button("Close", action: close)
            }.buttonStyle(.borderless).padding(12)
            Divider()
            if session.busy {
                VStack(spacing: 16) {
                    ProgressView()
                    Text(session.status).font(.callout.monospacedDigit()).foregroundStyle(.secondary)
                }.frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                UncoverCanvas(session: session)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .overlay(alignment: .bottomTrailing) {
                        if session.readingSource {
                            ProgressView()
                                .controlSize(.small)
                                .padding(8)
                                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
                                .padding(10)
                                .accessibilityLabel("Loading source frame")
                                .allowsHitTesting(false)
                        }
                    }
            }
            Divider()
            VStack(alignment: .leading, spacing: 10) {
                if session.editing || !session.hasCurrentResult {
                    HStack(spacing: 10) {
                        Text("Source").font(.caption).foregroundStyle(.secondary)
                        Button { session.step(-1) } label: { Image(systemName: "backward.end.fill") }
                            .help("Previous source frame")
                        Slider(value: Binding(get: { Double(session.sourceRequestedIndex) },
                                              set: { session.seekSource(index: Int($0.rounded())) }),
                               in: 0...Double(max(1, session.sourceFrameCount - 1)))
                            .accessibilityLabel("Source frame")
                        Button { session.step(1) } label: { Image(systemName: "forward.end.fill") }
                            .help("Next source frame")
                        Text("\(timecode(session.sourceTime)) · \(session.sourceDisplayedIndex + 1)/\(session.sourceFrameCount)")
                            .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    }.disabled(session.busy || session.sourceFrameCount == 0)
                }
                if session.hasCurrentResult {
                    if !session.editing {
                        HStack {
                            Button { session.togglePlayback() } label: { Image(systemName: session.playing ? "pause.fill" : "play.fill") }
                            Button { session.step(-1) } label: { Image(systemName: "backward.end.fill") }
                            Slider(value: Binding(get: { Double(session.displayedIndex) }, set: { session.seek(index: Int($0.rounded())) }),
                                   in: 0...Double(max(1, session.frameCount - 1)))
                            Button { session.step(1) } label: { Image(systemName: "forward.end.fill") }
                            Text("\(timecode(session.displayedTime)) · \(session.displayedIndex + 1)/\(session.frameCount)")
                                .font(.system(size: 11, design: .monospaced))
                            Toggle("Loop", isOn: $session.loop).toggleStyle(.checkbox)
                        }.disabled(session.busy)
                    }
                    HStack {
                        Picker("Compare", selection: $session.comparison) {
                            ForEach(UncoverComparison.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                        }.pickerStyle(.segmented).frame(maxWidth: 360)
                        Button(session.showB ? "Show Original" : "Show AI") { session.comparison = .toggle; session.showB.toggle() }
                        Button(session.editing ? "Compare" : "Edit Region") { session.toggleEditing() }
                        Spacer()
                    }.disabled(session.busy)
                }
                HStack(spacing: 12) {
                    Text("Algorithm").font(.caption)
                    Picker("Algorithm", selection: $session.algorithm) {
                        ForEach(session.availableAlgorithms, id: \.self) { Text($0.label).tag($0) }
                    }.labelsHidden().pickerStyle(.segmented).controlSize(.small).frame(width: 390)
                    if session.capabilities?.available != true {
                        Button("Manage Models…") {
                            UserDefaults.standard.set("models", forKey: "settingsTab")
                            openSettings()
                        }
                    }
                    Spacer()
                }.disabled(session.busy)
                HStack(spacing: 16) {
                    Text("Downsample").font(.caption)
                    Picker("Downsample", selection: $session.scale) {
                        ForEach(UncoverScale.allCases, id: \.self) { scale in
                            Text(scale == .one ? "1× Original" : scale.label).tag(scale).disabled(session.capabilities?.scales.contains(scale) != true)
                        }
                    }.labelsHidden().pickerStyle(.segmented).frame(width: 380)
                        .help("Sets source downsampling before reconstruction. 1× keeps the original pixels.")
                    Text("Start").font(.caption)
                    TextField("Seconds", value: $session.start, format: .number.precision(.fractionLength(3)))
                        .frame(width: 85).textFieldStyle(.roundedBorder)
                    Text("Duration").font(.caption)
                    Picker("Duration", selection: $session.duration) {
                        Text("1s").tag(1.0); Text("2s").tag(2.0); Text("5s").tag(5.0)
                    }.labelsHidden().frame(width: 90)
                    TextField("Custom seconds", value: $session.duration, format: .number.precision(.fractionLength(3)))
                        .frame(width: 65).textFieldStyle(.roundedBorder)
                    Text("to \(timecode(session.end))").font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    Spacer()
                }.disabled(session.busy)
                HStack(spacing: 12) {
                    Text("Edge Feather").font(.caption)
                    Slider(value: $session.feather, in: 0...32, step: 1).frame(width: 130)
                    Text("\(Int(session.feather)) px").font(.caption.monospacedDigit()).frame(width: 40)
                    Text("Blend Strength").font(.caption)
                    Slider(value: $session.strength, in: 0...1).frame(width: 130)
                    Text("\(Int((session.strength * 100).rounded()))%").font(.caption.monospacedDigit()).frame(width: 40)
                    Spacer()
                    if session.busy {
                        ProgressView().controlSize(.small)
                        Button("Cancel") { session.cancel() }
                    } else {
                        Button("Reconstruct") { session.reconstruct() }.disabled(!session.canReconstruct)
                        Button("Export…") { save() }.disabled(!session.hasCurrentResult)
                    }
                }
                HStack {
                    Text(session.status).font(.caption).foregroundStyle(.secondary)
                    if let limitation = session.limitation { Text(limitation).font(.caption).foregroundStyle(.secondary) }
                    if session.algorithm == .flashVSR11 {
                        Text("FlashVSR outputs 4× source size and is tuned for 4× super-resolution; other input reductions change the effective ratio.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Text("Scroll to pan · Pinch to zoom · Option-drag to pan").font(.caption).foregroundStyle(.tertiary)
                }
                if let error = session.error {
                    Text(error).font(.caption).foregroundStyle(.red).lineLimit(3).textSelection(.enabled)
                }
            }.padding(12)
        }
        .background(Color(nsColor: .windowBackgroundColor))
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
