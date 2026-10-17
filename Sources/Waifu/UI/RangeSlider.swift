import SwiftUI

/// A two-thumb range slider over one signal's span.
///
/// SwiftUI's `Slider` is single-thumb and macOS has no range slider, so this is a
/// small one: a track, the selected span filled between two knobs, and a knob you
/// drag at each end. It edits **values in the signal's own units**, not percentages,
/// because the exact limits are the point — each handle *is* the label, so the control
/// says what it is filtering by instead of hiding it behind a 0…1 bar.
///
/// The bounds come from the data (`FrameLimitBounds`), so the control always spans
/// what is on screen and both ends are reachable values rather than an arbitrary
/// 0…1.
struct RangeSlider: View {
    let bounds: ClosedRange<Double>
    let low: Double
    let high: Double
    /// The signal's own units: a timecode for time, two decimals for a score. A
    /// handle draws its value through this, which is why there is no readout row
    /// under the track.
    let format: (Double) -> String
    let onChanged: (Double, Double) -> Void

    @State private var dragging: End?

    private enum End { case low, high }

    /// A handle is a pill around its own number rather than a bare circle, so it is
    /// as wide as that number and taller than the old knob to seat 9 pt text. The
    /// trade is height *within* one row: the readout line this replaces cost more
    /// than the extra pixels the pill takes.
    private let knob: CGFloat = 14
    private let font = Font.system(size: 9, weight: .medium, design: .monospaced)
    /// SF Mono's advance is 0.6 em and `.monospaced` disables kerning, so a monospaced
    /// string is exactly this wide — no measurement pass, and no dependence on a font
    /// having finished loading (which would resize the handles under the drag).
    private let advance: CGFloat = 0.62 * 9
    private let textPadding: CGFloat = 4
    private let track: CGFloat = 3
    private let space = "waifu.range-slider"

    private var span: Double { max(bounds.upperBound - bounds.lowerBound, 1e-9) }

    /// One width for both handles — the wider of the two numbers — so a value sits at
    /// the same x at either end and the track's scale cannot shift under a drag when
    /// one handle's text gains or loses a character.
    private var handleWidth: CGFloat {
        let digits = max(format(low).count, format(high).count)
        return max(knob, CGFloat(digits) * advance + textPadding * 2)
    }

    var body: some View {
        GeometryReader { geometry in
            let width = max(geometry.size.width, handleWidth)
            let travel = max(width - handleWidth, 1)

            ZStack(alignment: .leading) {
                Capsule()
                    .fill(Color.secondary.opacity(0.22))
                    .frame(height: track)
                    .padding(.horizontal, handleWidth / 2)

                Capsule()
                    .fill(Color.accentColor.opacity(0.5))
                    .frame(width: max(0, position(high, travel: travel) - position(low, travel: travel)),
                           height: track)
                    .offset(x: position(low, travel: travel))

                knobView(format(low), active: dragging == .low)
                    .offset(x: position(low, travel: travel) - handleWidth / 2)

                knobView(format(high), active: dragging == .high)
                    .offset(x: position(high, travel: travel) - handleWidth / 2)
            }
            .frame(width: width, height: knob)
            .coordinateSpace(name: space)
            .contentShape(Rectangle())
            .gesture(drag(travel: travel))
        }
        .frame(height: knob)
        // The number is legible but small, and nothing else on screen states the span
        // in words; this is where VoiceOver gets it.
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Range")
        .accessibilityValue("\(format(low)) to \(format(high))")
    }

    /// Places a value along the track: its share of the span, offset by half a handle
    /// because a handle is centred on its value rather than left-aligned to it.
    private func position(_ value: Double, travel: CGFloat) -> CGFloat {
        handleWidth / 2 + CGFloat((value - bounds.lowerBound) / span) * travel
    }

    private func drag(travel: CGFloat) -> some Gesture {
        // The gesture is read in the track's own space, so a thumb can be dragged by
        // its absolute position rather than a translation measured from wherever the
        // drag happened to start — the thumb follows the cursor exactly.
        DragGesture(minimumDistance: 0, coordinateSpace: .named(space))
            .onChanged { value in
                // Pick once on mouse-down, then keep that handle captured even
                // when the pointer crosses the midpoint or reaches the other end.
                let end = dragging ?? nearestEnd(at: value.startLocation.x, travel: travel)
                dragging = end
                move(end, to: value.location.x, travel: travel)
            }
            .onEnded { value in
                if let dragging { move(dragging, to: value.location.x, travel: travel) }
                dragging = nil
            }
    }

    private func nearestEnd(at x: CGFloat, travel: CGFloat) -> End {
        // When the handles overlap, either side can still expand the range.
        if low == high { return x < position(low, travel: travel) ? .low : .high }
        return abs(x - position(low, travel: travel)) < abs(x - position(high, travel: travel)) ? .low : .high
    }

    private func move(_ end: End, to x: CGFloat, travel: CGFloat) {
        let ratio = Double((x - handleWidth / 2) / travel)
        let value = bounds.lowerBound + min(max(ratio, 0), 1) * span
        switch end {
        case .low: onChanged(min(value, high), high)
        case .high: onChanged(low, max(value, low))
        }
    }

    /// One draggable end: its own value, printed on it, held at `handleWidth` so both
    /// ends read as the same control.
    private func knobView(_ text: String, active: Bool) -> some View {
        Text(text)
            .font(font)
            .lineLimit(1)
            // Belt and braces: `handleWidth` is sized from the text, so this only
            // bites if a signal ever formats wider than its character count implies.
            .minimumScaleFactor(0.85)
            .foregroundStyle(active ? Color.accentColor : Color.secondary)
            .frame(width: handleWidth, height: knob)
            .background {
                RoundedRectangle(cornerRadius: knob / 2, style: .continuous)
                    .fill(active ? Color.accentColor.opacity(0.18)
                                 : Color(nsColor: .controlBackgroundColor))
                    // On the shape, not the row: a shadow on the whole view would
                    // silhouette the text sitting on top of it as well.
                    .shadow(radius: active ? 2 : 0)
            }
            .overlay {
                RoundedRectangle(cornerRadius: knob / 2, style: .continuous)
                    .strokeBorder(active ? Color.accentColor : Color.secondary.opacity(0.55),
                                  lineWidth: 1)
            }
    }
}
