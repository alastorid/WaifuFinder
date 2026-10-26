import SwiftUI

/// Always-visible, inline choices styled like the panel preset segments.
struct TruForDisplayButtons: View {
    @Binding var selection: TruForDisplay
    var body: some View {
        HStack(spacing: 0) {
            ForEach(TruForDisplay.allCases) { mode in
                if mode != TruForDisplay.allCases.first {
                    Rectangle().fill(Color.secondary.opacity(0.25)).frame(width: 1, height: 12)
                }
                Button { selection = mode } label: {
                    Text(mode.label)
                        .font(.caption)
                        .fixedSize()
                        .padding(.horizontal, 6)
                        .frame(minHeight: 23)
                        .background(selection == mode ? Color.accentColor.opacity(0.2) : Color.secondary.opacity(0.06))
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityAddTraits(selection == mode ? .isSelected : [])
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 4))
        .overlay {
            RoundedRectangle(cornerRadius: 4).strokeBorder(Color.secondary.opacity(0.25), lineWidth: 1)
        }
        .fixedSize(horizontal: true, vertical: false)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("TruFor display")
    }
}
