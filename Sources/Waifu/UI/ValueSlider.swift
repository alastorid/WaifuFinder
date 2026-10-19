import SwiftUI

/// A single-value slider drawn the same way as `RangeSlider`.
///
/// Two reasons this is not the system `Slider`:
///
///   * **It looks like the rest of the panel.** The range sliders beside it are a slim
///     ruled track with a small round knob; the system control is a different weight and
///     a different knob, and next to them it reads as a bolted-on widget. Same track
///     height, same knob, same fill as its neighbours.
///   * **It can take the arrow keys.** The system slider does not become first
///     responder on a click in this panel, so `←`/`→` never reached it — the stepping
///     was unreachable, not merely unimplemented. A focusable custom view with an
///     explicit tap-to-focus takes the keys, and a hairline halo says so.
///
/// Dragging anywhere on the track jumps and scrubs; a drag reports continuously through
/// `onChange` and once at the end through `onCommit`, which is where an expensive
/// follow-up (re-grouping) belongs.
struct ValueSlider: View {
    let bounds: ClosedRange<Double>
    let value: Double
    let step: Double
    let onChange: (Double) -> Void
    let onCommit: (Double) -> Void
    /// Optional value pill, matching the labelled handles in RangeSlider.
    var format: ((Double) -> String)? = nil

    @Environment(\.isEnabled) private var isEnabled
    @State private var dragging = false
    @FocusState private var isFocused: Bool

    private var knob: CGFloat { format == nil ? 12 : 14 }
    private var handleWidth: CGFloat {
        guard let format else { return knob }
        let digits = max(format(bounds.lowerBound).count, format(bounds.upperBound).count, format(value).count)
        return max(knob, CGFloat(digits) * 0.62 * 9 + 8)
    }
    private let track: CGFloat = 4
    private let space = "waifu.value-slider"

    private var span: Double { max(bounds.upperBound - bounds.lowerBound, 1e-9) }

    var body: some View {
        GeometryReader { geometry in
            let width = max(geometry.size.width, handleWidth)
            let travel = max(width - handleWidth, 1)
            let position = handleWidth / 2 + CGFloat((value - bounds.lowerBound) / span) * travel

            ZStack(alignment: .leading) {
                Capsule()
                    .fill(Color.secondary.opacity(0.18))
                    .frame(height: track)
                    .padding(.horizontal, handleWidth / 2)

                Capsule()
                    .fill((isEnabled ? Color.accentColor : Color.secondary).opacity(0.55))
                    .frame(width: max(0, position - handleWidth / 2), height: track)
                    .offset(x: handleWidth / 2)

                handle
                    .offset(x: position - handleWidth / 2)
            }
            .frame(width: width, height: knob)
            .coordinateSpace(name: space)
            // The halo is the affordance for the arrow keys: without it there is nothing
            // on screen to say this control can be stepped from the keyboard.
            .overlay {
                if isFocused {
                    Capsule()
                        .strokeBorder(Color.accentColor.opacity(0.45), lineWidth: 1)
                        .padding(.horizontal, handleWidth / 2)
                        .padding(.vertical, -3)
                }
            }
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0, coordinateSpace: .named(space))
                    .onChanged { gesture in
                        // Focus here rather than in a tap gesture: a zero-distance drag
                        // begins on touch-down and consumes the click, so a simultaneous
                        // `onTapGesture` never ran — which is exactly why the arrow keys
                        // had nothing to reach. Touching the slider is using it.
                        isFocused = true
                        dragging = true
                        onChange(value(at: gesture.location.x, travel: travel))
                    }
                    .onEnded { gesture in
                        dragging = false
                        onCommit(value(at: gesture.location.x, travel: travel))
                    }
            )
        }
        .frame(height: knob)
        .opacity(isEnabled ? 1 : 0.45)
        .allowsHitTesting(isEnabled)
        .focusable(isEnabled)
        .focusEffectDisabled()
        .focused($isFocused)
        .onKeyPress(.leftArrow) { commitStep(-1); return .handled }
        .onKeyPress(.rightArrow) { commitStep(1); return .handled }
    }

    @ViewBuilder
    private var handle: some View {
        if let format {
            Text(format(value))
                .font(.system(size: 9, weight: .medium, design: .monospaced))
                .lineLimit(1)
                .minimumScaleFactor(0.85)
                .foregroundStyle(isFocused || dragging ? Color.accentColor : Color.secondary)
                .frame(width: handleWidth, height: knob)
                .background {
                    RoundedRectangle(cornerRadius: knob / 2, style: .continuous)
                        .fill(dragging ? Color.accentColor.opacity(0.18)
                                      : Color(nsColor: .controlBackgroundColor))
                        .shadow(radius: dragging ? 2 : 0)
                }
                .overlay {
                    RoundedRectangle(cornerRadius: knob / 2, style: .continuous)
                        .strokeBorder(isFocused || dragging ? Color.accentColor : Color.secondary.opacity(0.55),
                                      lineWidth: 1)
                }
        } else {
            Circle()
                .fill(Color(nsColor: .controlBackgroundColor))
                .overlay {
                    Circle().strokeBorder(isFocused || dragging ? Color.accentColor : Color.secondary.opacity(0.5),
                                          lineWidth: 1)
                }
                .frame(width: knob, height: knob)
                .shadow(radius: dragging ? 3 : 1)
        }
    }

    /// The value under an x position, clamped to the bounds.
    private func value(at x: CGFloat, travel: CGFloat) -> Double {
        let ratio = Double((x - handleWidth / 2) / travel)
        return bounds.lowerBound + min(max(ratio, 0), 1) * span
    }

    /// One step, snapped to the step grid so the readout stays tidy, then committed —
    /// a key press is discrete, so there is no release to wait for.
    private func commitStep(_ direction: Int) {
        guard isEnabled else { return }
        let stepped = min(max(value + Double(direction) * step, bounds.lowerBound), bounds.upperBound)
        let snapped = (stepped / step).rounded() * step
        guard snapped != value else { return }
        onCommit(snapped)
    }
}
