import AppKit
import SwiftUI

//
//  ChiaroscuroLink.swift
//  WaifuFinder — the east egg in the Sort and Filter fraction.
//
//  Rest the pointer on the "shown" fraction for three seconds while the filter is
//  actually hiding frames, and a wave of light runs from that fraction to the
//  switched-off iCloud cloud in the title bar.
//
//  It links two things the window otherwise keeps apart: how many frames the filter
//  let through, and the cloud that is not fetching the ones it left out. Nothing is
//  announced, nothing is written, and the light leaves no trace, because it is only
//  ever visible to a pointer that stays put.
//
//  Three conditions, all of them required, and none of them negotiable:
//
//  **The filter is hiding frames.** `shown == total` is a finished filter, not a
//  link — there is nothing to connect to anything.
//
//  **The cloud download is switched off.** A cloud that is downloading has no
//  business being the far end of this, so the destination only exists while iCloud
//  downloads are off. Switching them on takes the destination away and the egg with
//  it; there is no other destination, anywhere.
//
//  **The pointer stays for three seconds.** A pointer crossing the panel on its way
//  somewhere else never triggers it.
//
//  Chiaroscuro, not neon. One warm source, a soft halo, a shadow under the line,
//  and the cloud lit briefly when the light lands: a candle in a dark room, not a
//  glow bright enough to read as a defect.
//
//  Two things about how it is put together are worth knowing before changing them:
//
//  **Every position is measured by AppKit, in window coordinates.** The title bar is
//  not laid out with the window content — SwiftUI hosts a toolbar item in the
//  window's own title-bar view, so a coordinate space declared on the content never
//  reaches it — and the light has to arrive *on* the cloud, in the title bar. So
//  each end measures itself with `convert(_:to: nil)`, which walks up to the window
//  whatever hosted it, and the light is drawn in those same coordinates.
//
//  **The light is an AppKit view over the window frame, not a SwiftUI canvas in the
//  content.** A canvas is laid out with the content and cannot paint over a title
//  bar it does not cover, so the last stretch of the wave — the part that matters —
//  would be missing. The overlay is above the whole window and answers `nil` to
//  hit-testing, so the window keeps behaving as if it were not there.
//

// MARK: - The rules

/// The one predicate and the two timings, all of them pure.
enum Chiaroscuro {
    /// How long the pointer has to rest on the fraction. Long enough that crossing
    /// the panel on the way somewhere else never counts as an invitation.
    static let hoverDelay: TimeInterval = 3

    /// The whole flight: leaving the fraction, arriving at the cloud, and the slow
    /// fade that follows.
    static let flight: TimeInterval = 2.1

    /// Only a fraction that is hiding frames is worth linking. `shown == total` is
    /// a filter that kept everything, and there is nothing to connect to anything.
    static func canLink(shown: Int, total: Int) -> Bool { shown < total }
}

/// The stillness timer, as a value.
///
/// The controller owns the sleeping; this owns the decision, and time arrives as a
/// parameter, so "three seconds" can be checked without a window, a run loop or a
/// wait.
struct ChiaroscuroArm: Equatable {
    private(set) var beganAt: TimeInterval?
    private(set) var firedAt: TimeInterval?

    /// Whether the pointer is on the fraction right now.
    var isHovering: Bool { beganAt != nil }

    /// Whether the light has already been sent for this hover.
    var hasFired: Bool { firedAt != nil }

    /// How long the pointer has rested, or `nil` when it is not there at all.
    func rested(for time: TimeInterval) -> TimeInterval? {
        guard let beganAt else { return nil }
        return max(0, time - beganAt)
    }

    mutating func hoverBegan(at time: TimeInterval) {
        // Re-entering without leaving is not a new hold, and a second start time
        // would restart the three seconds on a pointer that never moved.
        guard beganAt == nil else { return }
        beganAt = time
        firedAt = nil
    }

    /// Leaving forgets the whole hold. A pointer on its way past the panel must not
    /// inherit the seconds it spent passing over.
    mutating func hoverEnded() {
        beganAt = nil
        firedAt = nil
    }

    /// True exactly once per hover, at the moment the hold becomes long enough.
    mutating func fire(at time: TimeInterval) -> Bool {
        guard firedAt == nil, let rested = rested(for: time),
              rested >= Chiaroscuro.hoverDelay else { return false }
        firedAt = time
        return true
    }
}

// MARK: - The light

/// One soft blob of the wave, trailing behind the head.
struct ChiaroscuroSpark: Equatable {
    let point: CGPoint
    let radius: Double
    /// 0…1, before the egg's restraint is applied to it.
    let brightness: Double
}

/// The arc the light runs along, and how lit each part of it is.
///
/// Coordinates are window coordinates — origin at the window's bottom left, y
/// upwards — because that is what AppKit reports and what the overlay draws in.
/// All of it is arithmetic on two points, so the shape of the wave is checkable
/// without a window: where the head is at a given moment, that the wake only ever
/// reaches backwards from it, that the blobs fade with age, and that nothing at all
/// is drawn before the light leaves or after it has gone.
struct ChiaroscuroBeam: Equatable {
    let source: CGPoint
    let destination: CGPoint
    /// 0 as the light leaves the fraction, 1 when it has reached the cloud, and on
    /// to `1 + fadeOut` while it goes out again.
    let travel: Double

    /// How long the light stays on the cloud before it is gone. In travel units,
    /// not seconds: the flight's pace is the controller's business.
    static let fadeOut: Double = 0.5
    /// The wake's length along the arc, as a fraction of the whole flight.
    static let trail: Double = 0.42
    /// How finely the arc is sampled into a path. A quadratic Bézier a window wide
    /// needs very few to read as curved, and this is stroked sixty times a second.
    static let samples = 28

    init(source: CGPoint, destination: CGPoint, travel: Double) {
        self.source = source
        self.destination = destination
        self.travel = travel
    }

    /// A quadratic arc bowed towards the top of the window, because the cloud is up
    /// there in the title bar, and light that ran in a straight line would read as a
    /// progress bar.
    var control: CGPoint {
        let dx = destination.x - source.x
        let dy = destination.y - source.y
        let distance = max(1, (dx * dx + dy * dy).squareRoot())
        // Bow upwards, whichever way the two ends are ordered.
        let perpendicular = CGPoint(x: -dy / distance, y: dx / distance)
        let sign = perpendicular.y < 0 ? -1.0 : 1.0
        let bow = min(60, distance * 0.22)
        return CGPoint(x: (source.x + destination.x) / 2 + perpendicular.x * bow * sign,
                       y: (source.y + destination.y) / 2 + perpendicular.y * bow * sign)
    }

    /// A point on the arc. `t` is clamped, so nothing can ask for a position off
    /// the end of the light's own path.
    func point(at t: Double) -> CGPoint {
        let t = min(max(t, 0), 1)
        let tail = 1 - t
        return CGPoint(x: tail * tail * source.x + 2 * tail * t * control.x + t * t * destination.x,
                       y: tail * tail * source.y + 2 * tail * t * control.y + t * t * destination.y)
    }

    /// Whether the light is anywhere near the window at all.
    var isLit: Bool { travel > 0 && travel < 1 + Self.fadeOut }

    /// How lit the light is in total. It must not arrive at full strength on its
    /// first frame, and it leaves slowly rather than blinking out.
    var presence: Double {
        smoothStep(0, 0.07, travel) * (1 - smoothStep(1, 1 + Self.fadeOut, travel))
    }

    /// Where the head is on the arc, 0…1, eased so the light arrives instead of
    /// stopping dead on the cloud.
    var head: Double {
        0.5 - 0.5 * cos(.pi * min(max(travel, 0), 1))
    }

    /// The lit stretch of arc: from the oldest of the wake to the head.
    var litRange: ClosedRange<Double> {
        let head = head
        return max(0, head - Self.trail)...head
    }

    /// The wave's crests, brightest and largest at the head and diminishing behind
    /// it, each on the arc at its own age. The ages are spaced so the crests read as
    /// separate swells rolling away from the light rather than as one long blur.
    var wake: [ChiaroscuroSpark] {
        let head = head
        let swell = 0.78 + 0.22 * sin(2 * .pi * min(max(travel, 0), 1 + Self.fadeOut) * 2)
        return ChiaroscuroBeam.crestAges.map { crest in
            ChiaroscuroSpark(point: point(at: head - Self.trail * crest),
                             radius: 14 * (1 - 0.6 * crest) * presence,
                             brightness: pow(1 - crest, 1.15) * 0.5 * swell * presence)
        }
    }

    /// How much of the cloud the light has reached. Nothing at all until the light
    /// is nearly there, so the cloud lights up rather than being lit from the start.
    var bloom: Double {
        smoothStep(0.68, 1, travel) * presence
    }

    /// Where the crests sit along the wake: on the head, and three rolling away
    /// behind it. Four rather than two or three, because spaced too far apart they
    /// read as beads on a string instead of one wave.
    ///
    /// Internal rather than private because the suite checks each crest against its
    /// own age, and a copy of this list in the test could drift from the one the
    /// light is actually drawn from.
    static let crestAges: [Double] = [0, 0.17, 0.4, 0.68]

    /// The lit arc as a path to stroke.
    var path: CGPath {
        let range = litRange
        let path = CGMutablePath()
        path.move(to: point(at: range.lowerBound))
        for step in 1...Self.samples {
            path.addLine(to: point(at: range.lowerBound
                                    + (range.upperBound - range.lowerBound) * Double(step) / Double(Self.samples)))
        }
        return path
    }

    private func smoothStep(_ edge0: Double, _ edge1: Double, _ value: Double) -> Double {
        guard edge1 > edge0 else { return value >= edge1 ? 1 : 0 }
        let t = min(max((value - edge0) / (edge1 - edge0), 0), 1)
        return t * t * (3 - 2 * t)
    }
}

// MARK: - The link

/// Shared by the fraction, the cloud and the light drawn between them: one per
/// window, owned by the root view.
///
/// Deliberately not `@Published`: the light moves sixty times a second, and a root
/// view observing this would re-render the whole window — grid, panel and all — for
/// the sake of an easter egg. The light is told about its own beam instead.
@MainActor
final class ChiaroscuroLink: ObservableObject {
    /// The fraction, in window coordinates, for as long as it is on screen.
    private(set) var source: CGRect?
    /// The cloud, in window coordinates, and only while it is a switched-off cloud
    /// in this window's title bar. Nothing else is ever a destination, so a cloud
    /// that is downloading leaves this `nil` and the egg cannot start.
    private(set) var destination: CGRect?

    /// How far along the arc the light is, `nil` while the window shows nothing.
    private(set) var travel: Double?

    /// Set by the light's window layer; called whenever the beam changes.
    var onBeamChanged: ((ChiaroscuroBeam?) -> Void)?

    private var arm = ChiaroscuroArm()
    private var hold: Task<Void, Never>?
    private var flight: Task<Void, Never>?

    /// The light to draw, or `nil` when there is nothing on the window.
    var beam: ChiaroscuroBeam? {
        guard let travel, let source, let destination else { return nil }
        let beam = ChiaroscuroBeam(source: centre(of: source),
                                   destination: centre(of: destination),
                                   travel: travel)
        return beam.isLit ? beam : nil
    }

    func setSource(_ rect: CGRect?) {
        guard rect != source else { return }
        source = rect
        onBeamChanged?(beam)
    }

    func setDestination(_ rect: CGRect?) {
        guard rect != destination else { return }
        destination = rect
        onBeamChanged?(beam)
    }

    /// The pointer arrived on the fraction. Nothing moves yet: the light is for a
    /// pointer that stays.
    func hoverBegan() {
        // Nothing to link to means no egg at all, however long the pointer rests.
        guard destination != nil else { return }
        arm.hoverBegan(at: ProcessInfo.processInfo.systemUptime)
        hold?.cancel()
        hold = Task { [weak self] in
            try? await Task.sleep(for: .seconds(Chiaroscuro.hoverDelay))
            guard !Task.isCancelled else { return }
            self?.send()
        }
    }

    /// The pointer left. The hold is forgotten; a light already on its way is left to
    /// arrive, because it was asked for while the pointer was still there.
    func hoverEnded() {
        hold?.cancel()
        hold = nil
        arm.hoverEnded()
    }

    private func send() {
        guard arm.fire(at: ProcessInfo.processInfo.systemUptime),
              source != nil, destination != nil else { return }
        let end = 1 + ChiaroscuroBeam.fadeOut
        flight?.cancel()
        // Progress comes from the clock, not from a frame count, so a late wake-up
        // makes the flight shorter rather than slower.
        flight = Task { [weak self] in
            let began = ProcessInfo.processInfo.systemUptime
            while !Task.isCancelled {
                let elapsed = ProcessInfo.processInfo.systemUptime - began
                guard elapsed < Chiaroscuro.flight else { break }
                self?.setTravel(end * elapsed / Chiaroscuro.flight)
                try? await Task.sleep(for: .milliseconds(16))
            }
            self?.setTravel(nil)
        }
    }

    private func setTravel(_ value: Double?) {
        guard travel != value else { return }
        travel = value
        onBeamChanged?(beam)
    }

    private func centre(of rect: CGRect) -> CGPoint {
        CGPoint(x: rect.midX, y: rect.midY)
    }
}

// MARK: - Measuring the two ends

/// A view that reports its own rectangle in window coordinates, on every layout that
/// changes it and never otherwise.
///
/// AppKit is asked rather than SwiftUI because the title bar is not laid out with
/// the content: `convert(_:to: nil)` walks up to the window from wherever the view
/// was hosted, which is the only measurement that is true for both ends of the light.
private final class ChiaroscuroProbeView: NSView {
    var report: ((CGRect) -> Void)?
    /// While false the view measures nothing. That is how the cloud stops being a
    /// destination the moment iCloud downloads are switched on.
    var measures = true
    private var reported = CGRect.null

    override func layout() {
        super.layout()
        guard measures, window != nil, let report else { return }
        let rect = convert(bounds, to: nil)
        guard rect != reported, rect.width > 0, rect.height > 0 else { return }
        reported = rect
        // Reported after this layout pass: the panel and the toolbar are being laid
        // out right now, and a state change in the middle of that is how a panel ends
        // up re-entering its own body pass.
        DispatchQueue.main.async { [weak self] in
            guard let self, self.window != nil else { return }
            report(rect)
        }
    }
}

/// The SwiftUI half of a probe. Its NSView has no content: it needs a place in the
/// hierarchy to measure from, and takes the size it is given.
private struct ChiaroscuroProbe: NSViewRepresentable {
    var isActive = true
    let report: (CGRect) -> Void

    func makeNSView(context: Context) -> ChiaroscuroProbeView {
        let view = ChiaroscuroProbeView()
        view.report = report
        view.measures = isActive
        return view
    }

    func updateNSView(_ view: ChiaroscuroProbeView, context: Context) {
        view.report = report
        view.measures = isActive
    }
}

extension View {
    /// The near end of the light: where it leaves from, and the surface whose
    /// stillness starts the flight. `isLinkable` is the fraction's own condition — a
    /// filter that hid nothing has nothing to link.
    func chiaroscuroSource(_ link: ChiaroscuroLink, isLinkable: Bool) -> some View {
        background {
            ChiaroscuroProbe { link.setSource($0) }
        }
        .onHover { inside in
            if inside, isLinkable { link.hoverBegan() } else { link.hoverEnded() }
        }
        // Closing the panel leaves no fraction to start anything from.
        .onDisappear { link.setSource(nil) }
    }

    /// The far end of the light, and the only destination there is. `isEligible` is
    /// true exactly while this is the switched-off cloud in the title bar: switching
    /// downloads on takes the destination away, and the egg with it.
    func chiaroscuroDestination(_ link: ChiaroscuroLink, isEligible: Bool) -> some View {
        background {
            ChiaroscuroProbe(isActive: isEligible) { link.setDestination($0) }
        }
        // Closing the Photos workspace leaves no cloud to aim at, so the light stops
        // being available rather than flying at a rectangle in a title bar that has
        // none.
        .onDisappear { link.setDestination(nil) }
    }
}

// MARK: - The light on the window

/// The wave: a transparent view over the whole window frame — above the content
/// *and* the title bar — drawing in window coordinates.
///
/// Not private because `draw(_:in:windowOrigin:)` is the seam that lets the look of
/// the light be rendered offscreen into a bitmap: the wave that gets judged there is
/// this drawing, not a copy of it that can drift.
final class ChiaroscuroOverlayView: NSView {
    var beam: ChiaroscuroBeam? {
        didSet {
            guard oldValue != beam else { return }
            needsDisplay = true
        }
    }

    /// Unflipped, like every view that works in window coordinates: y runs up, so a
    /// rectangle measured with `convert(_:to: nil)` is one that can be drawn here.
    override var isFlipped: Bool { false }
    override var isOpaque: Bool { false }

    /// Never in the hit-test path. The window behaves exactly as if this view were
    /// not installed: the buttons underneath still take their clicks and the title bar
    /// still drags the window.
    override func hitTest(_ point: NSPoint) -> NSView? { nil }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        guard let window else {
            removeFromSuperview()
            NotificationCenter.default.removeObserver(self)
            return
        }
        cover(window)
        NotificationCenter.default.addObserver(
            self, selector: #selector(windowChanged(_:)),
            name: NSWindow.didResizeNotification, object: window)
        NotificationCenter.default.addObserver(
            self, selector: #selector(windowChanged(_:)),
            name: NSWindow.didBecomeKeyNotification, object: window)
    }

    deinit { NotificationCenter.default.removeObserver(self) }

    @objc private func windowChanged(_ notification: Notification) {
        guard let window = notification.object as? NSWindow else { return }
        cover(window)
    }

    private func cover(_ window: NSWindow) {
        // The theme frame is the only view above both the content and the title bar,
        // and it is not the one SwiftUI lays out — which is the point.
        guard let frame = window.contentView?.superview else { return }
        let wholeWindow = frame.convert(window.frame, from: nil)
        self.frame = frame.bounds.union(wholeWindow)
        // SwiftUI rebuilds the theme frame's subviews when the toolbar changes, which
        // can drop this view behind them.
        frame.addSubview(self)
        needsDisplay = true
    }

    override func draw(_ dirtyRect: NSRect) {
        guard let beam, beam.isLit,
              let context = NSGraphicsContext.current?.cgContext else { return }
        // Window coordinates are not this view's coordinates — it sits wherever the
        // theme frame put it — so the difference is measured with AppKit and handed
        // over, which leaves every path, gradient and radius below expressed in the
        // coordinates the beam is already in.
        draw(beam, in: context,
             windowOrigin: convert(CGPoint(x: bounds.minX, y: bounds.minY), to: nil))
    }

    /// The light, in window coordinates. Separate from `draw(_:)` and not private so
    /// the look of the wave can be rendered offscreen, into a bitmap, without a
    /// window: what is judged there is this drawing and not a copy of it.
    func draw(_ beam: ChiaroscuroBeam, in context: CGContext, windowOrigin: CGPoint) {
        guard beam.isLit else { return }
        context.saveGState()
        defer { context.restoreGState() }
        context.setShouldAntialias(true)
        context.translateBy(x: -windowOrigin.x, y: -windowOrigin.y)

        drawDark(context, beam)
        drawShadow(context, beam)
        drawHalo(context, beam)
        drawCrests(context, beam)
        drawCore(context, beam)
        drawBloom(context, beam)
    }

    // MARK: The light, in passes

    /// Chiaroscuro is contrast: a light only reads as light because there is
    /// something dark for it to be brighter than. This is that something — a wide,
    /// very soft settling of shadow under the whole line, so the light looks like it
    /// is passing over the window rather than painted on it.
    private func drawDark(_ context: CGContext, _ beam: ChiaroscuroBeam) {
        guard beam.presence > 0 else { return }
        let path = beam.path.shifted(dx: 0, dy: -5)
        stroke(context, path, passes: [(30, 0.024), (24, 0.026), (18, 0.028)],
               colour: ChiaroscuroOverlayView.shadow, strength: beam.presence)
    }

    /// And the shadow itself: tighter, closer under the line, which is what gives the
    /// light a side to come from.
    private func drawShadow(_ context: CGContext, _ beam: ChiaroscuroBeam) {
        guard beam.presence > 0 else { return }
        let path = beam.path.shifted(dx: 0, dy: -2.5)
        stroke(context, path, passes: [(10, 0.04), (6, 0.05)],
               colour: ChiaroscuroOverlayView.shadow, strength: beam.presence)
    }

    /// The warm spill around the line, gone by the tail and lit at the head.
    private func drawHalo(_ context: CGContext, _ beam: ChiaroscuroBeam) {
        guard beam.presence > 0 else { return }
        let passes: [(CGFloat, CGFloat)] = [(24, 0.038), (18, 0.040), (13, 0.042), (8, 0.044)]
        for (width, alpha) in passes {
            fadingStroke(context, beam.path, width: width,
                         colour: ChiaroscuroOverlayView.halo,
                         alpha: alpha * beam.presence, beam: beam)
        }
    }

    /// The wave's crests, rolling away behind the light.
    private func drawCrests(_ context: CGContext, _ beam: ChiaroscuroBeam) {
        for spark in beam.wake where spark.brightness > 0.002 {
            radial(context, at: spark.point, radius: spark.radius, stops: [
                (0, ChiaroscuroOverlayView.core, 0.95 * spark.brightness),
                (0.45, ChiaroscuroOverlayView.halo, 0.45 * spark.brightness),
                (1, ChiaroscuroOverlayView.halo, 0),
            ])
        }
    }

    /// The thin bright line inside the spill. This is the restrained part: the
    /// strength of a small light, never enough to be mistaken for a control.
    private func drawCore(_ context: CGContext, _ beam: ChiaroscuroBeam) {
        guard beam.presence > 0 else { return }
        stroke(context, beam.path, passes: [(3.6, 0.16)],
               colour: ChiaroscuroOverlayView.core, strength: beam.presence)
        fadingStroke(context, beam.path, width: 1.5,
                     colour: ChiaroscuroOverlayView.core,
                     alpha: 0.5 * beam.presence, beam: beam)
    }

    /// The cloud, lit while the light is on it, and dimmer than the light itself:
    /// the destination is lit, not replaced.
    private func drawBloom(_ context: CGContext, _ beam: ChiaroscuroBeam) {
        guard beam.bloom > 0.002 else { return }
        radial(context, at: beam.destination, radius: 46, stops: [
            (0, ChiaroscuroOverlayView.core, 0.30 * beam.bloom),
            (0.4, ChiaroscuroOverlayView.halo, 0.16 * beam.bloom),
            (1, ChiaroscuroOverlayView.halo, 0),
        ])
    }

    // MARK: Drawing, without a blur filter

    /// Softness here is several flat strokes of decreasing width and increasing
    /// strength. It is a fifth of the cost of blurring the window sixty times a
    /// second, allocates nothing per pass, and at these strengths the banding
    /// between passes is not visible.
    private func stroke(_ context: CGContext, _ path: CGPath, passes: [(CGFloat, CGFloat)],
                        colour: NSColor, strength: Double) {
        for (width, alpha) in passes {
            context.saveGState()
            context.addPath(path)
            context.setStrokeColor(colour.withAlphaComponent(alpha * strength).cgColor)
            context.setLineWidth(width)
            context.setLineCap(.round)
            context.setLineJoin(.round)
            context.strokePath()
            context.restoreGState()
        }
    }

    /// The same, but fading along the arc: dark at the tail, lit at the head.
    private func fadingStroke(_ context: CGContext, _ path: CGPath, width: CGFloat,
                              colour: NSColor, alpha: CGFloat, beam: ChiaroscuroBeam) {
        guard let gradient = CGGradient(
            colorsSpace: CGColorSpaceCreateDeviceRGB(),
            colors: [colour.withAlphaComponent(0).cgColor,
                     colour.withAlphaComponent(alpha).cgColor] as CFArray,
            locations: [0, 1]) else { return }
        context.saveGState()
        context.addPath(path)
        context.setLineWidth(width)
        context.setLineCap(.round)
        context.setLineJoin(.round)
        context.replacePathWithStrokedPath()
        context.clip()
        context.drawLinearGradient(gradient,
                                   start: beam.point(at: beam.litRange.lowerBound),
                                   end: beam.point(at: beam.head),
                                   options: [])
        context.restoreGState()
    }

    private func radial(_ context: CGContext, at centre: CGPoint, radius: CGFloat,
                        stops: [(CGFloat, NSColor, CGFloat)]) {
        let colours = stops.map { $0.1.withAlphaComponent($0.2).cgColor } as CFArray
        guard let gradient = CGGradient(colorsSpace: CGColorSpaceCreateDeviceRGB(),
                                        colors: colours,
                                        locations: stops.map { $0.0 }) else { return }
        context.saveGState()
        context.drawRadialGradient(gradient, startCenter: centre, startRadius: 0,
                                   endCenter: centre, endRadius: radius,
                                   options: [.drawsBeforeStartLocation, .drawsAfterEndLocation])
        context.restoreGState()
    }

    /// The light's own colours. Deliberately *not* white: a near-white light on a
    /// pale panel has nothing to be brighter than, so the warm end of the spectrum is
    /// what makes it read as light at all in the light appearance.
    private static let shadow = NSColor.black
    private static let core = NSColor(calibratedRed: 1.0, green: 0.90, blue: 0.68, alpha: 1)
    private static let halo = NSColor(calibratedRed: 1.0, green: 0.68, blue: 0.30, alpha: 1)
}

extension CGPath {
    /// A copy moved by a window-space offset. Core Graphics' own two-argument
    /// `copyShifting` shifts in place and returns nothing.
    func shifted(dx: CGFloat, dy: CGFloat) -> CGPath {
        var transform = CGAffineTransform(translationX: dx, y: dy)
        return copy(using: &transform) ?? self
    }
}

/// Installs the light over the window, and takes it away again when the layer goes.
///
/// This view is mounted for the lifetime of the window and is not `@State`-driven:
/// it costs one empty NSView and no redraw until there is a beam, and it keeps a
/// light that moves sixty times a second from invalidating the SwiftUI hierarchy.
struct ChiaroscuroWindowLayer: NSViewRepresentable {
    let link: ChiaroscuroLink

    func makeCoordinator() -> Coordinator { Coordinator(link: link) }

    func makeNSView(context: Context) -> NSView {
        let anchor = NSView(frame: .zero)
        // The hosting view has no window until it is in one, so the install waits
        // for the next pass of the run loop.
        DispatchQueue.main.async { context.coordinator.install(over: anchor) }
        return anchor
    }

    func updateNSView(_ view: NSView, context: Context) {
        context.coordinator.link = link
        context.coordinator.install(over: view)
        context.coordinator.publish()
    }

    static func dismantleNSView(_ view: NSView, coordinator: Coordinator) {
        coordinator.remove()
    }

    @MainActor
    final class Coordinator {
        var link: ChiaroscuroLink
        private var overlay: ChiaroscuroOverlayView?

        init(link: ChiaroscuroLink) {
            self.link = link
        }

        /// Puts the light over the window the host view lives in, once per window.
        func install(over view: NSView) {
            guard overlay?.superview == nil, let window = view.window else { return }
            let overlay = ChiaroscuroOverlayView(frame: .zero)
            overlay.beam = link.beam
            window.contentView?.superview?.addSubview(overlay)
            self.overlay = overlay
            link.onBeamChanged = { [weak self] beam in
                self?.overlay?.beam = beam
            }
        }

        func publish() {
            overlay?.beam = link.beam
        }

        func remove() {
            link.onBeamChanged = nil
            overlay?.removeFromSuperview()
            overlay = nil
        }
    }
}