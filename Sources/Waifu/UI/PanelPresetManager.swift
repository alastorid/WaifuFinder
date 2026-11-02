import SwiftUI

/// Three fixed memories, selected and recalled through the same segmented control.
struct PanelPresetManager: View {
    @ObservedObject var session: WaifuSession
    @State private var selectedSlot: Int?
    @State private var memoryArmed = false

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
                        if memoryArmed {
                            if session.savePanelPreset(at: slot) {
                                memoryArmed = false
                                selectedSlot = slot
                            }
                        } else {
                            session.loadPanelPreset(session.panelPresets[slot].id)
                            selectedSlot = slot
                        }
                    } label: {
                        Text("\(slot + 1)")
                            .monospacedDigit()
                            .frame(maxWidth: .infinity, minHeight: 21)
                            .background(selectedSlot == slot ? Color.accentColor.opacity(0.2) : Color.clear)
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("\(memoryArmed ? "Save" : "Load") preset \(slot + 1)")
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

            Button {
                memoryArmed.toggle()
            } label: {
                Text("Memory")
                    .padding(.horizontal, 8)
                    .frame(minHeight: 21)
                    .foregroundStyle(memoryArmed ? Color.white : Color.primary)
                    .background(memoryArmed ? Color.red : Color.secondary.opacity(0.12))
                    .clipShape(RoundedRectangle(cornerRadius: 4))
            }
            .buttonStyle(.plain)
            .accessibilityAddTraits(memoryArmed ? .isSelected : [])
            .help(memoryArmed ? "Click a slot to save the current preferences. Click Memory again to cancel." : "Click Memory, then a slot to save the current preferences.")
            .disabled(!session.canLoadPanelPreset)
            .fixedSize()

            Button("Clear") {
                session.clearPanelSettings()
            }
            .disabled(!session.canLoadPanelPreset)
            .fixedSize()
        }
        .controlSize(.small)
    }
}
