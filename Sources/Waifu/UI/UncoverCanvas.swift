import AppKit
import CoreImage
import SwiftUI

/// One viewport owns both alternatives, so comparison changes cannot move pixels.
struct UncoverCanvas: NSViewRepresentable {
    @ObservedObject var session: UncoverSession
    func makeNSView(context: Context) -> UncoverCanvasView { UncoverCanvasView() }
    func updateNSView(_ view: UncoverCanvasView, context: Context) {
        view.session = session
        view.updateFit()
        view.needsDisplay = true
    }
}

@MainActor
final class UncoverCanvasView: NSView {
    weak var session: UncoverSession?
    private var pan = CGPoint.zero
    private var wipe: CGFloat = 0.5
    private var dragStart = CGPoint.zero
    private var originalRect = CGRect.zero
    private var moving = false
    private var handle: Int?
    private var draggingWipe = false
    private var panning = false
    private var selecting = false
    private var oldPan = CGPoint.zero
    private var fitRequest = 0
    private let previewContext = CIContext(options: [.cacheIntermediates: false, .useSoftwareRenderer: true])
    private var previewSource: NSImage?
    private var previewROI = CGRect.zero
    private var previewScale: UncoverScale?
    private var previewImage: NSImage?
    func updateFit() {
        guard let session, fitRequest != session.fitRequest else { return }
        fitRequest = session.fitRequest; pan = .zero
    }
    override var isFlipped: Bool { true }
    override var acceptsFirstResponder: Bool { true }

    private var viewport: CGRect {
        var rect = bounds.insetBy(dx: 10, dy: 10)
        if let session, !session.editing, session.comparison == .sideBySide { rect.size.width = (rect.width - 12) / 2 }
        return rect
    }
    private var imageRect: CGRect {
        guard let session else { return .zero }
        let imageSize = session.displaySize
        guard imageSize.width > 0, imageSize.height > 0 else { return .zero }
        let fitScale: CGFloat
        switch session.viewScaleMode {
        case .original: fitScale = 1
        case .fitWidth: fitScale = viewport.width / imageSize.width
        case .fitHeight: fitScale = viewport.height / imageSize.height
        }
        let size = CGSize(width: imageSize.width * fitScale * session.zoom,
                          height: imageSize.height * fitScale * session.zoom)
        return CGRect(x: viewport.midX - size.width / 2 + pan.x,
                      y: viewport.midY - size.height / 2 + pan.y, width: size.width, height: size.height)
    }
    private func displayRect(_ pixels: CGRect) -> CGRect {
        guard let session, session.videoSize.width > 0 else { return .zero }
        let rect = imageRect, size = session.videoSize
        return CGRect(x: rect.minX + pixels.minX / size.width * rect.width,
                      y: rect.minY + pixels.minY / size.height * rect.height,
                      width: pixels.width / size.width * rect.width, height: pixels.height / size.height * rect.height)
    }
    private func nativePoint(_ point: CGPoint) -> CGPoint {
        guard let session else { return .zero }
        let rect = imageRect
        return CGPoint(x: max(0, min(session.videoSize.width, (point.x - rect.minX) / rect.width * session.videoSize.width)),
                       y: max(0, min(session.videoSize.height, (point.y - rect.minY) / rect.height * session.videoSize.height)))
    }

    override func draw(_ dirtyRect: NSRect) {
        NSColor.black.setFill(); bounds.fill()
        guard let session, let original = session.original else { return }
        let rect = imageRect
        if session.editing || session.showOriginal || !session.hasCurrentResult || session.imageB == nil {
            draw(original, rect: rect, clip: bounds)
        } else {
            switch session.comparison {
            case .sideBySide:
                draw(session.imageA ?? original, rect: rect, clip: viewport)
                let shift = viewport.width + 12
                draw(session.imageB ?? original, rect: rect.offsetBy(dx: shift, dy: 0), clip: viewport.offsetBy(dx: shift, dy: 0))
                label("Original", at: CGPoint(x: viewport.minX + 8, y: 14))
                label("AI · \(session.algorithm.label)", at: CGPoint(x: viewport.minX + shift + 8, y: 14))
            case .wipe:
                draw(session.imageA ?? original, rect: rect, clip: bounds)
                let divider = bounds.width * wipe
                draw(session.imageB ?? original, rect: rect, clip: CGRect(x: divider, y: 0, width: bounds.width - divider, height: bounds.height))
                NSColor.white.setStroke()
                let line = NSBezierPath(); line.move(to: CGPoint(x: divider, y: 0)); line.line(to: CGPoint(x: divider, y: bounds.height)); line.stroke()
                label("Original", at: CGPoint(x: 18, y: 14))
                label("AI · \(session.algorithm.label)", at: CGPoint(x: divider + 10, y: 14))
            case .toggle:
                draw((session.showB ? session.imageB : session.imageA) ?? original, rect: rect, clip: bounds)
                label(session.showB ? "AI · \(session.algorithm.label)" : "Original", at: CGPoint(x: 18, y: 14))
            }
        }
        if session.editing, let roi = try? session.roi.pixels(in: session.videoSize) {
            let outline = displayRect(roi)
            if !session.showOriginal, let input = downsamplePreview(original, roi: roi, scale: session.scale) {
                // Show actual reduced pixels at the selection's size, without smoothing them away.
                draw(input, rect: outline, clip: bounds, interpolation: .none)
                label("\(session.scale.rawValue)× input · \(Int(input.size.width))×\(Int(input.size.height))",
                      at: CGPoint(x: outline.minX + 6, y: outline.minY + 6))
            }
            NSColor.white.setStroke()
            let path = NSBezierPath(rect: outline); path.lineWidth = 1.5; path.stroke()
            for point in handles(outline) {
                NSColor.white.setFill()
                CGRect(x: point.x - 3, y: point.y - 3, width: 6, height: 6).fill()
            }
        }
    }
    private func downsamplePreview(_ original: NSImage, roi: CGRect, scale: UncoverScale) -> NSImage? {
        if previewSource === original, previewROI == roi, previewScale == scale { return previewImage }
        previewSource = original; previewROI = roi; previewScale = scale; previewImage = nil
        guard let pixels = original.cgImage(forProposedRect: nil, context: nil, hints: nil),
              let cropped = pixels.cropping(to: roi),
              let degraded = try? RegionImages.resize(cropped, to: scale.degradedSize(roi.size), context: previewContext) else { return nil }
        previewImage = NSImage(cgImage: degraded, size: CGSize(width: degraded.width, height: degraded.height))
        return previewImage
    }
    private func draw(_ image: NSImage, rect: CGRect, clip: CGRect, interpolation: NSImageInterpolation = .high) {
        NSGraphicsContext.saveGraphicsState()
        NSBezierPath(rect: clip).addClip()
        NSGraphicsContext.current?.imageInterpolation = interpolation
        image.draw(in: rect, from: .zero, operation: .copy, fraction: 1, respectFlipped: true, hints: nil)
        NSGraphicsContext.restoreGraphicsState()
    }
    private func label(_ text: String, at point: CGPoint) {
        (text as NSString).draw(at: point, withAttributes: [.font: NSFont.systemFont(ofSize: 12), .foregroundColor: NSColor.white,
                                                           .backgroundColor: NSColor.black.withAlphaComponent(0.6)])
    }
    private func handles(_ r: CGRect) -> [CGPoint] {
        [CGPoint(x: r.minX, y: r.minY), CGPoint(x: r.midX, y: r.minY), CGPoint(x: r.maxX, y: r.minY),
         CGPoint(x: r.maxX, y: r.midY), CGPoint(x: r.maxX, y: r.maxY), CGPoint(x: r.midX, y: r.maxY),
         CGPoint(x: r.minX, y: r.maxY), CGPoint(x: r.minX, y: r.midY)]
    }
    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
        selecting = false
        guard let session, !session.busy, session.videoSize.width > 0 else { return }
        let point = convert(event.locationInWindow, from: nil)
        panning = event.modifierFlags.contains(.option)
        if panning { dragStart = point; oldPan = pan; return }
        draggingWipe = !session.editing && session.comparison == .wipe
        if draggingWipe { wipe = max(0, min(1, point.x / bounds.width)); needsDisplay = true; return }
        guard session.editing, imageRect.contains(point) else { return }
        selecting = true
        dragStart = nativePoint(point)
        originalRect = (try? session.roi.pixels(in: session.videoSize)) ?? .zero
        handle = handles(displayRect(originalRect)).firstIndex(where: { hypot($0.x - point.x, $0.y - point.y) <= 9 })
        moving = handle == nil && originalRect.contains(dragStart)
        if !moving && handle == nil { originalRect = CGRect(origin: dragStart, size: .zero) }
    }
    override func mouseDragged(with event: NSEvent) {
        guard let session, !session.busy else { return }
        let point = convert(event.locationInWindow, from: nil)
        if panning { pan = CGPoint(x: oldPan.x + point.x - dragStart.x, y: oldPan.y + point.y - dragStart.y); needsDisplay = true; return }
        if draggingWipe { wipe = max(0, min(1, point.x / bounds.width)); needsDisplay = true; return }
        guard session.editing, selecting else { return }
        let native = nativePoint(point)
        var rect = originalRect
        if moving {
            rect.origin.x = max(0, min(session.videoSize.width - rect.width, rect.minX + native.x - dragStart.x))
            rect.origin.y = max(0, min(session.videoSize.height - rect.height, rect.minY + native.y - dragStart.y))
        } else if let handle {
            var left = rect.minX, right = rect.maxX, top = rect.minY, bottom = rect.maxY
            if [0, 6, 7].contains(handle) { left = native.x }
            if [2, 3, 4].contains(handle) { right = native.x }
            if [0, 1, 2].contains(handle) { top = native.y }
            if [4, 5, 6].contains(handle) { bottom = native.y }
            rect = CGRect(x: min(left, right), y: min(top, bottom), width: max(1, abs(right - left)), height: max(1, abs(bottom - top)))
        } else {
            rect = CGRect(x: min(dragStart.x, native.x), y: min(dragStart.y, native.y),
                          width: max(1, abs(native.x - dragStart.x)), height: max(1, abs(native.y - dragStart.y)))
        }
        if let snapped = try? UncoverROI.from(pixels: rect, in: session.videoSize).pixels(in: session.videoSize) {
            session.roi = .from(pixels: snapped, in: session.videoSize)
        }
        needsDisplay = true
    }
    override func mouseUp(with event: NSEvent) { panning = false; draggingWipe = false; selecting = false; handle = nil }
    override func scrollWheel(with event: NSEvent) {
        pan.x += event.scrollingDeltaX; pan.y += event.scrollingDeltaY; needsDisplay = true
    }
    override func magnify(with event: NSEvent) {
        guard let session else { return }
        session.zoom = max(0.25, min(16, session.zoom * (1 + event.magnification)))
    }
    override var mouseDownCanMoveWindow: Bool { false }
}
