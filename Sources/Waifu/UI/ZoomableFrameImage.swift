import AppKit
import SwiftUI

/// A native scroll view gives the preview cursor-anchored magnification and lets
/// the same zoom level and image point survive as the displayed frame changes.
struct ZoomableFrameImage: NSViewRepresentable {
    let image: NSImage
    let frameID: FrameID
    @Binding var zoomRatio: CGFloat
    @Binding var center: CGPoint
    @Binding var displayedScale: CGFloat

    func makeCoordinator() -> Coordinator {
        Coordinator(zoomRatio: $zoomRatio, center: $center, displayedScale: $displayedScale)
    }

    func makeNSView(context: Context) -> FrameZoomScrollView {
        let view = FrameZoomScrollView()
        view.onStateChange = { [weak coordinator = context.coordinator] ratio, center in
            coordinator?.store(ratio: ratio, center: center)
        }
        view.onScaleChange = { [weak coordinator = context.coordinator] scale in
            coordinator?.store(scale: scale)
        }
        return view
    }

    func updateNSView(_ view: FrameZoomScrollView, context: Context) {
        context.coordinator.zoomRatio = $zoomRatio
        context.coordinator.center = $center
        context.coordinator.displayedScale = $displayedScale
        view.onStateChange = { [weak coordinator = context.coordinator] ratio, center in
            coordinator?.store(ratio: ratio, center: center)
        }
        view.onScaleChange = { [weak coordinator = context.coordinator] scale in
            coordinator?.store(scale: scale)
        }
        view.show(image, id: frameID, zoomRatio: zoomRatio, center: center)
    }

    final class Coordinator {
        var zoomRatio: Binding<CGFloat>
        var center: Binding<CGPoint>
        var displayedScale: Binding<CGFloat>

        init(zoomRatio: Binding<CGFloat>, center: Binding<CGPoint>, displayedScale: Binding<CGFloat>) {
            self.zoomRatio = zoomRatio
            self.center = center
            self.displayedScale = displayedScale
        }

        func store(ratio: CGFloat, center point: CGPoint) {
            if abs(zoomRatio.wrappedValue - ratio) > 0.000001 {
                zoomRatio.wrappedValue = ratio
            }
            if abs(center.wrappedValue.x - point.x) > 0.000001
                || abs(center.wrappedValue.y - point.y) > 0.000001 {
                center.wrappedValue = point
            }
        }

        func store(scale: CGFloat) {
            guard scale.isFinite, scale > 0,
                  abs(displayedScale.wrappedValue - scale) > 0.000001 else { return }
            displayedScale.wrappedValue = scale
        }
    }
}

// MARK: - Geometry

/// Where a frame sits in the preview's viewport, and how large it is drawn there.
///
/// **The geometry is a value, not a scroll view**, so the fit can be pinned by a test
/// that never opens a window — see `Tests/FrameZoomGeometryTests.swift`. The two rules
/// it states are the ones the preview got wrong:
///
///   * At ratio 1 the frame **touches the viewport on its limiting axis** and passes
///     through the middle of the other. Not smaller than it ("a frame floating in the
///     middle"), not larger than it ("a frame cropped at an arbitrary offset").
///   * The frame is **centred** on the axis with room to spare. AppKit will not do this
///     for us: a document view smaller than the clip view is pinned to the scroll
///     origin rather than centred, so the padding has to live in the document view.
///
/// `viewport` is in real points. It is the clip view's bounds size *only because this
/// app never uses `NSScrollView`'s magnification* — see `FrameZoomScrollView` for why
/// that is not a detail.
struct FrameZoomGeometry: Equatable {
    /// Image scale limits in source pixels per point. The fit-relative ratio limits
    /// adjust with the viewport so the source's native 100% size remains reachable.
    static let ratioLimits: ClosedRange<CGFloat> = 0.25...16

    /// The visible area, in points.
    let viewport: CGSize
    /// The frame's own size in points — the size it has at ratio 1.
    let imageSize: CGSize
    /// The zoom as a multiple of the fit. Clamped on the way in, so every consumer —
    /// buttons, a pinch, a stale binding — reads a ratio it can actually lay out.
    let ratio: CGFloat

    init(viewport: CGSize, imageSize: CGSize, ratio: CGFloat) {
        self.viewport = viewport
        self.imageSize = imageSize
        self.ratio = Self.clampedRatio(ratio, fitScale: Self.fitScale(viewport: viewport, imageSize: imageSize))
    }

    /// `1` for a ratio that is not a number at all: a NaN from a division by a
    /// viewport that has not been laid out yet is not a zoom level to honour. An
    /// infinity is a zoom level that ran away, and stops at the nearest limit.
    static func clampedRatio(_ ratio: CGFloat) -> CGFloat {
        clampedRatio(ratio, fitScale: 1)
    }

    /// Keep zoom limits in image scale (source pixels per point), while `ratio`
    /// remains a multiple of fit for stable window-resize behavior.
    static func clampedRatio(_ ratio: CGFloat, fitScale: CGFloat) -> CGFloat {
        guard fitScale.isFinite, fitScale > 0 else {
            guard !ratio.isNaN else { return 1 }
            return min(ratioLimits.upperBound, max(ratioLimits.lowerBound, ratio))
        }
        let lower = min(1, ratioLimits.lowerBound / fitScale)
        let upper = max(1, ratioLimits.upperBound / fitScale)
        guard !ratio.isNaN else { return 1 }
        return min(upper, max(lower, ratio))
    }

    /// The limiting axis, so the frame fills the viewport exactly on one axis and
    /// exactly all of the other. A degenerate size yields `1` — a scale to draw at,
    /// not a collapse to draw at zero.
    var fitScale: CGFloat {
        Self.fitScale(viewport: viewport, imageSize: imageSize)
    }

    static func fitScale(viewport: CGSize, imageSize: CGSize) -> CGFloat {
        guard imageSize.width > 0, imageSize.height > 0,
              viewport.width > 0, viewport.height > 0 else { return 1 }
        return min(viewport.width / imageSize.width, viewport.height / imageSize.height)
    }

    var scale: CGFloat { fitScale * ratio }

    var scaledImageSize: CGSize {
        CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
    }

    /// The document view is never smaller than the viewport. That padding is the whole
    /// reason a fitted frame can be centred: a canvas the size of the window has a
    /// scroll origin for every position in it, while a document view exactly the size
    /// of the frame has one position and AppKit pins it to the origin.
    var canvasSize: CGSize {
        CGSize(width: max(viewport.width, scaledImageSize.width),
               height: max(viewport.height, scaledImageSize.height))
    }

    /// The frame inside the canvas, centred on both axes.
    var imageRect: NSRect {
        NSRect(x: (canvasSize.width - scaledImageSize.width) / 2,
               y: (canvasSize.height - scaledImageSize.height) / 2,
               width: scaledImageSize.width,
               height: scaledImageSize.height)
    }

    /// Whether the frame is larger than the viewport on either axis, and so has a
    /// position to keep when the frames change underneath it.
    var isScrollable: Bool {
        scaledImageSize.width > viewport.width || scaledImageSize.height > viewport.height
    }

    /// How far to scroll so `focus` — a normalised point in the image — sits at the
    /// centre of the viewport. Clamped to what the canvas can scroll, which is what
    /// centres a frame that is smaller than the viewport: the origin is pinned and the
    /// frame sits in the middle of the window rather than in its top-left corner.
    func scrollOrigin(focus: CGPoint) -> CGPoint {
        let centre = CGPoint(x: viewport.width / 2, y: viewport.height / 2)
        return scrollOrigin(keeping: centre, under: focus)
    }

    /// How far to scroll so the viewport position `point` — normally the pointer — stays
    /// over the image point `focus` it is over now. This is what a pinch needs, and it
    /// is *not* `scrollOrigin(focus:)`: re-centring the focus would move the picture
    /// out from under the pointer the gesture is being made with.
    func scrollOrigin(keeping point: CGPoint, under focus: CGPoint) -> CGPoint {
        let wanted = CGPoint(x: focus.x * scaledImageSize.width + imageRect.minX - point.x,
                             y: focus.y * scaledImageSize.height + imageRect.minY - point.y)
        let limit = CGSize(width: max(0, canvasSize.width - viewport.width),
                           height: max(0, canvasSize.height - viewport.height))
        return CGPoint(x: min(max(wanted.x, 0), limit.width), y: min(max(wanted.y, 0), limit.height))
    }

    /// The image point at a canvas position — the inverse of `scrollOrigin(focus:)`,
    /// and how a pinch reads the point it is anchored on.
    func focus(atCanvasPoint point: CGPoint) -> CGPoint {
        guard scaledImageSize.width > 0, scaledImageSize.height > 0 else { return CGPoint(x: 0.5, y: 0.5) }
        let raw = CGPoint(x: (point.x - imageRect.minX) / scaledImageSize.width,
                          y: (point.y - imageRect.minY) / scaledImageSize.height)
        return CGPoint(x: min(1, max(0, raw.x)), y: min(1, max(0, raw.y)))
    }
}

// MARK: - The scroll view

/// The preview's zoom surface: a frame you can magnify, pan and step through.
///
/// **The canvas holds the scale, and AppKit's magnification is deliberately unused.**
/// `NSScrollView.magnification` does not scale the document view's frame — it scales the
/// *clip view's bounds*, which is how much document the viewport shows. Fitting by
/// setting it therefore measures the fit against a viewport that its own magnification
/// has already changed: the next layout pass computed the "viewport" as
/// `physical ÷ magnification`, refitted to that, and the frame arrived at its own
/// pixel size, cropped, centred on whatever point the last step had stored. The two
/// symptoms were one bug: an offset frame at the wrong scale.
///
/// So this view never calls `setMagnification`, and overrides it to keep a stray call
/// from reintroducing that feedback: `magnification` stays 1, the clip view's bounds
/// size stays the real viewport, and `FrameZoomGeometry` does the arithmetic. The
/// document view is a plain canvas — the image view inside it is the only thing that
/// draws, and its frame is the fitted rectangle.
final class FrameZoomScrollView: NSScrollView {
    var onStateChange: ((CGFloat, CGPoint) -> Void)?
    var onScaleChange: ((CGFloat) -> Void)?

    /// Never smaller than the viewport, so a frame that fits has room to be centred.
    private let canvas = NSView()
    private let frameImageView = NSImageView()
    private var displayedID: FrameID?
    /// The frame's own size in points — what the geometry scales, and therefore what the
    /// ratio is a multiple of.
    private var imageSize: CGSize = .zero
    /// The zoom as a multiple of the fit, and the image point kept at the middle of
    /// the viewport.
    private var requestedRatio: CGFloat = 1
    private var requestedCenter = CGPoint(x: 0.5, y: 0.5)
    private var lastViewportSize = CGSize.zero
    private var applyingLayout = false
    nonisolated(unsafe) private var boundsObserver: NSObjectProtocol?

    override init(frame: NSRect) {
        super.init(frame: frame)
        configure()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        configure()
    }

    deinit {
        if let boundsObserver { NotificationCenter.default.removeObserver(boundsObserver) }
    }

    private func configure() {
        drawsBackground = false
        borderType = .noBorder
        hasHorizontalScroller = true
        hasVerticalScroller = true
        autohidesScrollers = true
        scrollerStyle = .overlay
        // Left on so a trackpad pinch still reaches `magnify(with:)`; the magnification
        // itself is this class's business, not AppKit's.
        allowsMagnification = true

        documentView = canvas
        canvas.addSubview(frameImageView)
        // `.scaleNone` draws the image at *its own* size whatever this view's frame
        // says — the frame overflowed the viewport and AppKit cropped it. The frame
        // given to the image view is the fitted rectangle, so drawing it proportionally
        // lands on exactly that rectangle.
        frameImageView.imageScaling = .scaleProportionallyUpOrDown
        frameImageView.imageAlignment = .alignCenter

        let doubleClick = NSClickGestureRecognizer(target: self, action: #selector(toggleZoom(_:)))
        doubleClick.numberOfClicksRequired = 2
        addGestureRecognizer(doubleClick)

        contentView.postsBoundsChangedNotifications = true
        boundsObserver = NotificationCenter.default.addObserver(
            forName: NSView.boundsDidChangeNotification,
            object: contentView,
            queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated { self?.publishState() }
        }
    }

    /// Nothing here may set the magnification: the clip view's bounds size is the
    /// viewport `FrameZoomGeometry` fits against, and a magnification makes that a
    /// number derived from the zoom rather than the window.
    override func setMagnification(_ magnification: CGFloat, centeredAt point: NSPoint) {
        super.setMagnification(1, centeredAt: point)
    }

    func show(_ image: NSImage, id: FrameID, zoomRatio: CGFloat, center: CGPoint) {
        let isNewFrame = displayedID != id
        let isNewImage = frameImageView.image !== image
        let targetImageSize = isNewImage ? image.size : imageSize
        let fitScale = FrameZoomGeometry.fitScale(viewport: contentView.bounds.size, imageSize: targetImageSize)
        let ratio = FrameZoomGeometry.clampedRatio(zoomRatio, fitScale: fitScale)
        let focus = CGPoint(x: min(1, max(0, center.x)), y: min(1, max(0, center.y)))
        // A completed source read can replace the image already on screen, and
        // SwiftUI calls this on every state change, so an
        // unchanged request is otherwise left alone: re-applying on every update would
        // re-centre under the user's scroll.
        guard isNewFrame || isNewImage || ratio != requestedRatio || focus != requestedCenter else { return }
        if !isNewFrame, !isNewImage, ratio != requestedRatio {
            let pointer = window.map { contentView.convert($0.mouseLocationOutsideOfEventStream, from: nil) }
            let anchor = pointer.flatMap { contentView.bounds.contains($0) ? $0 : nil }
                ?? contentView.bounds.center
            zoom(to: ratio, atClipPoint: anchor)
            return
        }
        if isNewImage {
            frameImageView.image = image
            imageSize = image.size
        }
        displayedID = id
        requestedRatio = ratio
        requestedCenter = focus
        applyLayout()
    }

    override func layout() {
        super.layout()
        guard !applyingLayout else { return }
        let viewport = contentView.bounds.size
        guard viewport.width > 1, viewport.height > 1, viewport != lastViewportSize else { return }
        lastViewportSize = viewport
        applyLayout()
    }

    /// A window resize refits: the frame follows the new viewport at the zoom level
    /// and the image point the user was looking at. `origin` overrides the scroll, which
    /// is how a pinch keeps the pointer where the user is holding it.
    private func applyLayout(origin: CGPoint? = nil) {
        let geometry = currentGeometry
        guard geometry.viewport.width > 1, geometry.viewport.height > 1,
              geometry.imageSize.width > 1, geometry.imageSize.height > 1 else { return }
        applyingLayout = true
        canvas.frame = NSRect(origin: .zero, size: geometry.canvasSize)
        frameImageView.frame = geometry.imageRect
        frameImageView.needsDisplay = true
        contentView.scroll(to: origin ?? geometry.scrollOrigin(focus: requestedCenter))
        reflectScrolledClipView(contentView)
        applyingLayout = false
        publishState()
    }

    /// The layout for what is on screen now: the real viewport, the frame's own size,
    /// and the zoom that is being asked for.
    private var currentGeometry: FrameZoomGeometry {
        FrameZoomGeometry(viewport: contentView.bounds.size, imageSize: imageSize, ratio: requestedRatio)
    }

    /// AppKit reports magnification deltas with the pointer location. Keeping that
    /// point fixed on screen makes the mouse position the zoom anchor: the image point
    /// under the pointer before the zoom is the focus afterwards, and the scroll keeps
    /// the pointer over it. Re-centring instead would slide the picture sideways under
    /// a hand that has not moved.
    override func magnify(with event: NSEvent) {
        let anchor = contentView.convert(event.locationInWindow, from: nil)
        // Converted clip coordinates include the existing scroll origin. The
        // anchor passed to layout must be relative to the visible viewport.
        zoom(to: requestedRatio * (1 + event.magnification), atClipPoint: anchor)
    }

    /// Use one pointer-preserving path for pinch, double-click and zoom controls.
    func zoom(to ratio: CGFloat, atClipPoint anchor: CGPoint) {
        let point = CGPoint(x: anchor.x - contentView.bounds.minX,
                            y: anchor.y - contentView.bounds.minY)
        let focus = currentGeometry.focus(atCanvasPoint: anchor)
        requestedRatio = FrameZoomGeometry.clampedRatio(ratio, fitScale: currentGeometry.fitScale)
        applyLayout(origin: currentGeometry.scrollOrigin(keeping: point, under: focus))
    }

    @objc private func toggleZoom(_ gesture: NSClickGestureRecognizer) {
        let anchor = gesture.location(in: contentView)
        let isFit = abs(requestedRatio - 1) <= 0.01
        let targetRatio = isFit ? 1 / max(currentGeometry.fitScale, .leastNonzeroMagnitude) : 1
        zoom(to: targetRatio, atClipPoint: anchor)
    }

    override func scrollWheel(with event: NSEvent) {
        super.scrollWheel(with: event)
        publishState()
    }

    /// The zoom and the image point at the centre of the viewport, read back off the
    /// clip view: the user moved them, so the bindings holding them have to be told.
    private func publishState() {
        guard !applyingLayout, imageSize.width > 1, imageSize.height > 1 else { return }
        let geometry = currentGeometry
        onScaleChange?(geometry.scale)
        // With nothing to scroll there is no position to report: the focus point stays
        // where the user left it, so magnifying back in returns there rather than to
        // the middle of a frame that is smaller than the window.
        guard geometry.isScrollable else {
            onStateChange?(geometry.ratio, requestedCenter)
            return
        }
        requestedCenter = geometry.focus(atCanvasPoint: contentView.bounds.center)
        onStateChange?(geometry.ratio, requestedCenter)
    }
}

private extension NSRect {
    var center: NSPoint { NSPoint(x: midX, y: midY) }
}
