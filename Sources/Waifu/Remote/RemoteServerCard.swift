import AppKit
import SwiftUI

/// Added servers keep their setup controls in the separate add-server form.
struct RemoteServerCard: View {
    @ObservedObject var client: RemoteClient
    let vocabularyFingerprint: UInt64?
    let remove: () -> Void
    @State private var removalRevealed = false

    private var capabilities: RemoteCapabilities? { client.capabilities ?? client.lastKnownCapabilities }

    var body: some View {
        ZStack(alignment: .trailing) {
            if removalRevealed {
                Button(role: .destructive, action: remove) {
                    VStack(spacing: 5) {
                        Image(systemName: "trash")
                        Text("Disconnect")
                    }
                    .font(.caption)
                    .frame(width: 104)
                    .frame(maxHeight: .infinity)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .foregroundStyle(.red)
                .accessibilityLabel("Disconnect and remove server")
            }
            GroupBox {
                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 12) {
                        VStack(alignment: .leading, spacing: 3) {
                            Text(capabilities?.serverName ?? client.manualAddress)
                                .font(.callout.weight(.semibold))
                            Text(client.manualAddress).font(.caption.monospaced()).foregroundStyle(.secondary)
                        }
                        Spacer()
                        Toggle("Use this server", isOn: Binding(get: { client.isEnabled }, set: {
                            client.setEnabled($0)
                            removalRevealed = false
                        }))
                        .labelsHidden()
                        .toggleStyle(.switch)
                    }
                    HStack(spacing: 6) {
                        Circle().fill(client.status.isConnected ? Color.green : Color.secondary)
                            .frame(width: 6, height: 6)
                        Text(client.isEnabled ? client.status.label : "Off")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    if let capabilities {
                        capability("Tagging", available: capabilities.offersRemoteTagging)
                        ForEach(capabilities.uncover, id: \.algorithm) { entry in
                            capability(UncoverAlgorithm(rawValue: entry.algorithm)?.label ?? entry.algorithm,
                                       available: entry.available)
                        }
                        if !client.status.isConnected {
                            Text("Last known capabilities").font(.caption2).foregroundStyle(.tertiary)
                        }
                    } else {
                        Text("Capabilities will appear when connected.").font(.caption).foregroundStyle(.secondary)
                    }
                    if client.offersRemoteTagging,
                       let fingerprint = client.remoteVocabulary?.cacheFingerprint,
                       let vocabularyFingerprint, fingerprint != vocabularyFingerprint {
                        Text("Different tag labels · excluded from this tagging pool")
                            .font(.caption).foregroundStyle(.orange)
                    }
                }
                .padding(4)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .background(Color(nsColor: .windowBackgroundColor))
            .offset(x: removalRevealed ? -112 : 0)
        }
        .clipped()
        .background(RemoteCardSwipeMonitor { revealed in
            withAnimation(.easeOut(duration: 0.18)) { removalRevealed = revealed }
        })
        .simultaneousGesture(DragGesture(minimumDistance: 24).onEnded { value in
            guard abs(value.translation.width) > abs(value.translation.height) else { return }
            withAnimation(.easeOut(duration: 0.18)) { removalRevealed = value.translation.width < 0 }
        })
        .accessibilityAction(named: Text("Disconnect and remove server"), remove)
    }

    private func capability(_ title: String, available: Bool) -> some View {
        HStack(spacing: 6) {
            Image(systemName: available ? "checkmark.circle.fill" : "circle.dashed")
                .foregroundStyle(available ? Color.green : Color.secondary)
            Text(title)
            Spacer()
            Text(available ? "Ready" : "Unavailable").foregroundStyle(.secondary)
        }
        .font(.caption)
    }
}

/// Observe horizontal trackpad scrolling inside this card without taking vertical
/// scrolling or clicks away from the settings pane and its controls.
private struct RemoteCardSwipeMonitor: NSViewRepresentable {
    let changed: (Bool) -> Void

    func makeNSView(context: Context) -> MonitorView {
        let view = MonitorView()
        view.changed = changed
        return view
    }

    func updateNSView(_ view: MonitorView, context: Context) { view.changed = changed }
    static func dismantleNSView(_ view: MonitorView, coordinator: ()) { view.removeMonitor() }

    final class MonitorView: NSView {
        var changed: ((Bool) -> Void)?
        private var monitor: Any?
        private var distance: CGFloat = 0
        private var tracking = false

        override func hitTest(_ point: NSPoint) -> NSView? { nil }

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            removeMonitor()
            guard window != nil else { return }
            monitor = NSEvent.addLocalMonitorForEvents(matching: .scrollWheel) { [weak self] event in
                let windowNumber = event.window?.windowNumber
                let location = event.locationInWindow
                let precise = event.hasPreciseScrollingDeltas
                let phase = event.phase
                let horizontal = event.scrollingDeltaX
                let vertical = event.scrollingDeltaY
                let inverted = event.isDirectionInvertedFromDevice
                let handled = MainActor.assumeIsolated {
                    guard let self, let window = self.window, windowNumber == window.windowNumber,
                          self.bounds.contains(self.convert(location, from: nil)), precise else { return false }
                    if phase.contains(.began) || phase.contains(.mayBegin) {
                        self.distance = 0
                        self.tracking = false
                    }
                    if abs(horizontal) > abs(vertical), abs(horizontal) > 0 { self.tracking = true }
                    guard self.tracking else { return false }
                    self.distance += inverted ? horizontal : -horizontal
                    if abs(self.distance) > 32 {
                        self.changed?(self.distance < 0)
                    }
                    if phase.contains(.ended) || phase.contains(.cancelled) {
                        self.distance = 0
                        self.tracking = false
                    }
                    return true
                }
                return handled ? nil : event
            }
        }

        func removeMonitor() {
            if let monitor { NSEvent.removeMonitor(monitor) }
            monitor = nil
            tracking = false
            distance = 0
        }
    }
}
