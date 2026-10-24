import SwiftUI

/// Three fixed memories, selected and recalled through the same segmented control.
struct PanelPresetManager: View {
    @ObservedObject var session: WaifuSession
    @State private var selectedSlot: Int?

    var body: some View {
        HStack(spacing: 8) {
            Text("Preset")
                .font(.headline)

            HStack(spacing: 0) {
                ForEach(0..<3) { slot in
                    if slot > 0 {
                        Rectangle().fill(Color.secondary.opacity(0.25))
                            .frame(width: 1, height: 12)
                    }
                    Button {
                        session.loadPanelPreset(session.panelPresets[slot].id)
                        selectedSlot = slot
                    } label: {
                        Text("\(slot + 1)")
                            .monospacedDigit()
                            .frame(maxWidth: .infinity, minHeight: 21)
                            .background(selectedSlot == slot ? Color.accentColor.opacity(0.2) : Color.clear)
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("Load preset \(slot + 1)")
                    .accessibilityAddTraits(selectedSlot == slot ? .isSelected : [])
                }
            }
            .background(Color.secondary.opacity(0.06))
            .clipShape(RoundedRectangle(cornerRadius: 4))
            .overlay {
                RoundedRectangle(cornerRadius: 4)
                    .strokeBorder(Color.secondary.opacity(0.25), lineWidth: 1)
            }
            .disabled(!session.canLoadPanelPreset)

            Button("Memory") {
                if let selectedSlot { session.savePanelPreset(at: selectedSlot) }
            }
            .disabled(selectedSlot == nil)
            .fixedSize()

            Button("Clear") {
                session.clearPanelSettings(selectedSlot: selectedSlot)
            }
            .disabled(!session.canLoadPanelPreset)
            .fixedSize()
        }
        .controlSize(.small)
    }
}
