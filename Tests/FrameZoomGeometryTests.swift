import AppKit
import Foundation

/// `CGPoint` has no `+`, and a scroll origin is added to a viewport position often
/// enough here to be worth naming.
private func offset(_ point: CGPoint, _ delta: CGPoint) -> CGPoint {
    CGPoint(x: point.x + delta.x, y: point.y + delta.y)
}

/// The preview's fit, as a value.
///
/// These cases pin the defect from a trace and an offscreen render: the frame arrived
/// at its own pixel size, cropped and centred on a stale point, because the fit was
/// measured against a viewport that the scroll view's own magnification had already
/// changed. The geometry is pure, so the whole fit — scale, position, and what happens
/// when the window resizes or the frames change underneath it — is checkable here
/// without a window, a focus engine or a key press.
func registerFrameZoomGeometryTests() {
    let viewport = CGSize(width: 800, height: 600)
    let landscape = CGSize(width: 1920, height: 1080)
    let portrait = CGSize(width: 1080, height: 1920)

    Registry.shared.add(suite: "Frame zoom", TestCase(name: "the actual scrolled preview keeps the same image point under the mouse through repeated zooms", knownBug: nil) {
        try await Task { @MainActor in
            for size in [landscape, portrait] {
                let pixels = try ImageFixture.cgImage(width: Int(size.width), height: Int(size.height), gray: 0.5)
                let image = NSImage(cgImage: pixels, size: size)
                let scroll = FrameZoomScrollView(frame: NSRect(origin: .zero, size: viewport))
                let id = FrameID(sampleIndex: 0)
                var reportedRatio: CGFloat = 4
                var reportedCenter = CGPoint(x: 0.5, y: 0.5)
                scroll.onStateChange = { ratio, center in reportedRatio = ratio; reportedCenter = center }
                scroll.show(image, id: id, zoomRatio: 4, center: reportedCenter)
                scroll.layoutSubtreeIfNeeded()
                // Start panned on both axes. Clip coordinates now contain an
                // offset and cannot be used as viewport coordinates directly.
                scroll.contentView.scroll(to: CGPoint(x: 90, y: 110))
                scroll.reflectScrolledClipView(scroll.contentView)
                let pointer = CGPoint(x: 260, y: 270)
                let before = FrameZoomGeometry(viewport: scroll.contentView.bounds.size, imageSize: size, ratio: 4)
                let focus = before.focus(atCanvasPoint: offset(scroll.contentView.bounds.origin, pointer))
                for ratio: CGFloat in [4.25, 5, 6, 5.25, 4.5] {
                    scroll.zoom(to: ratio, atClipPoint: offset(scroll.contentView.bounds.origin, pointer))
                    scroll.layoutSubtreeIfNeeded()
                    let geometry = FrameZoomGeometry(viewport: scroll.contentView.bounds.size, imageSize: size, ratio: ratio)
                    let actual = geometry.focus(atCanvasPoint: offset(scroll.contentView.bounds.origin, pointer))
                    checkClose(Double(actual.x), Double(focus.x), tolerance: 0.001,
                               "horizontal anchor survives a panned zoom to \(ratio) for \(size)")
                    checkClose(Double(actual.y), Double(focus.y), tolerance: 0.001,
                               "vertical anchor survives a panned zoom to \(ratio) for \(size)")
                    let origin = scroll.contentView.bounds.origin
                    scroll.show(image, id: id, zoomRatio: reportedRatio, center: reportedCenter)
                    checkEqual(scroll.contentView.bounds.origin, origin, "binding feedback does not recenter the zoom")
                    let imageView = scroll.documentView?.subviews.first as? NSImageView
                    check(imageView?.image === image, "zoom keeps the native image as its drawing source")
                }
            }
        }.value
    })

    Registry.shared.add(suite: "Frame zoom", TestCase(name: "a fitted frame touches the viewport and is centred on the other axis", knownBug: nil) {
        // The bug, in one line: the frame has to reach the border on the axis it is
        // limited by, and sit in the middle on the other. Both halves were wrong at
        // once — a frame at the wrong scale is wrong *because* it was placed wrongly.
        for image in [landscape, portrait, CGSize(width: 1000, height: 1000),
                      CGSize(width: 640, height: 480), CGSize(width: 4000, height: 300)] {
            let fit = FrameZoomGeometry(viewport: viewport, imageSize: image, ratio: 1)
            // Exactly one axis is limited by the viewport; the other has slack, and the
            // slack has to be split evenly or the frame hangs off a corner.
            let slack = CGSize(width: viewport.width - fit.scaledImageSize.width,
                               height: viewport.height - fit.scaledImageSize.height)
            check(abs(slack.width) < 0.001 || abs(slack.height) < 0.001,
                  "the fit touches at least one axis exactly (\(image))")
            check(slack.width >= -0.001 && slack.height >= -0.001,
                  "the fit never overflows the viewport (\(image))")
            checkClose(Double(fit.imageRect.minX), Double(slack.width / 2), tolerance: 0.001,
                       "the frame is centred horizontally (\(image))")
            checkClose(Double(fit.imageRect.minY), Double(slack.height / 2), tolerance: 0.001,
                       "the frame is centred vertically (\(image))")
            checkClose(Double(fit.imageRect.maxX), Double(viewport.width - slack.width / 2), tolerance: 0.001,
                       "the frame's right edge is centred too (\(image))")
            checkClose(Double(fit.imageRect.maxY), Double(viewport.height - slack.height / 2), tolerance: 0.001,
                       "the frame's bottom edge is centred too (\(image))")
        }

        // The limiting axis is chosen by the geometry, not by which dimension happens
        // to be first in the struct.
        let wide = FrameZoomGeometry(viewport: viewport, imageSize: landscape, ratio: 1)
        let tall = FrameZoomGeometry(viewport: viewport, imageSize: portrait, ratio: 1)
        checkClose(Double(wide.fitScale), Double(viewport.width / landscape.width), tolerance: 0.0001,
                   "16:9 in a 4:3 window is limited by width")
        checkClose(Double(tall.fitScale), Double(viewport.height / portrait.height), tolerance: 0.0001,
                   "9:16 in a 4:3 window is limited by height")
        check(wide.isScrollable == false, "a frame at the fit is inside the window, so there is nothing to scroll")
    })

    Registry.shared.add(suite: "Frame zoom", TestCase(name: "the canvas is never smaller than the viewport, which is what centres a fitted frame", knownBug: nil) {
        // AppKit pins a document view smaller than the clip view to the scroll origin
        // rather than centring it, so a fitted frame needs somewhere to be centred.
        for ratio in [FrameZoomGeometry.ratioLimits.lowerBound, 0.5, 1, 4,
                      FrameZoomGeometry.ratioLimits.upperBound] as [CGFloat] {
            for image in [landscape, portrait] {
                let geometry = FrameZoomGeometry(viewport: viewport, imageSize: image, ratio: ratio)
                check(geometry.canvasSize.width >= viewport.width - 0.001,
                      "the canvas is at least as wide as the window (ratio \(ratio), \(image))")
                check(geometry.canvasSize.height >= viewport.height - 0.001,
                      "the canvas is at least as tall as the window (ratio \(ratio), \(image))")
                check(geometry.scrollOrigin(focus: CGPoint(x: 0.5, y: 0.5)).x >= -0.001,
                      "a fitted frame can be scrolled to a position, not left at the origin")
                // Whatever the zoom, the frame stays inside the canvas it is centred in.
                check(geometry.imageRect.minX >= -0.001 && geometry.imageRect.maxX <= geometry.canvasSize.width + 0.001,
                      "the frame never hangs outside the canvas (ratio \(ratio), \(image))")
            }
        }
    })

    Registry.shared.add(suite: "Frame zoom", TestCase(name: "the scroll origin puts the stored image point at the centre of the viewport", knownBug: nil) {
        // Stepping between frames keeps the point the user was looking at: this is the
        // whole reason the preview survives a step, and it only holds if the origin and
        // the point agree.
        for ratio in [0.5, 1, 2, 4] as [CGFloat] {
            for image in [landscape, portrait] {
                let geometry = FrameZoomGeometry(viewport: viewport, imageSize: image, ratio: ratio)
                let limit = CGSize(width: max(0, geometry.canvasSize.width - viewport.width),
                                   height: max(0, geometry.canvasSize.height - viewport.height))
                for focus in [CGPoint(x: 0.5, y: 0.5), CGPoint(x: 0.25, y: 0.75),
                              CGPoint(x: 0, y: 1), CGPoint(x: 1, y: 0)] {
                    let origin = geometry.scrollOrigin(focus: focus)
                    check(origin.x >= -0.001 && origin.x <= limit.width + 0.001,
                          "the origin stays inside the canvas horizontally (ratio \(ratio), \(image), \(focus))")
                    check(origin.y >= -0.001 && origin.y <= limit.height + 0.001,
                          "the origin stays inside the canvas vertically (ratio \(ratio), \(image), \(focus))")
                    let readBack = geometry.focus(atCanvasPoint: CGPoint(x: origin.x + viewport.width / 2,
                                                                          y: origin.y + viewport.height / 2))
                    // What is on screen is what stays on screen: reading the point at the
                    // centre back out and scrolling to it must not move the frame.
                    let again = geometry.scrollOrigin(focus: readBack)
                    checkClose(Double(again.x), Double(origin.x), tolerance: 0.001,
                               "the origin is a fixed point of reading it back (ratio \(ratio), \(image), \(focus))")
                    checkClose(Double(again.y), Double(origin.y), tolerance: 0.001,
                               "the origin is a fixed point of reading it back (ratio \(ratio), \(image), \(focus))")
                    // And where the canvas is long enough to reach the request, the
                    // request is what gets honoured. Where it is not, the frame sits at
                    // the limit instead of being placed off the canvas.
                    let atLimit = origin.x < 0.001 || origin.y < 0.001
                        || origin.x >= limit.width - 0.001 || origin.y >= limit.height - 0.001
                    if !atLimit {
                        checkClose(Double(readBack.x), Double(focus.x), tolerance: 0.001,
                                   "a reachable focus point is honoured (ratio \(ratio), \(image), \(focus))")
                        checkClose(Double(readBack.y), Double(focus.y), tolerance: 0.001,
                                   "a reachable focus point is honoured (ratio \(ratio), \(image), \(focus))")
                    }
                }
            }
        }
    })

    Registry.shared.add(suite: "Frame zoom", TestCase(name: "a resized window refits without moving the point being looked at", knownBug: nil) {
        // Resizing the window must not throw the user's place away: the ratio is a
        // multiple of the fit, so it survives a refit by definition, and the focus point
        // is a fraction of the frame rather than a pixel. The check is across a resize —
        // the image point at the middle of the window before is the one at the middle of
        // it after, provided both windows are big enough to hold the point.
        let sizes = [CGSize(width: 800, height: 600), CGSize(width: 2400, height: 1600),
                     CGSize(width: 1400, height: 500), CGSize(width: 900, height: 1400)]
        for (before, after) in zip(sizes, sizes.dropFirst() + sizes.prefix(1)) {
            // The refit still touches an axis and centres on the other, at every shape.
            let fit = FrameZoomGeometry(viewport: after, imageSize: landscape, ratio: 1)
            let slack = CGSize(width: after.width - fit.scaledImageSize.width,
                               height: after.height - fit.scaledImageSize.height)
            check(abs(slack.width) < 0.001 || abs(slack.height) < 0.001,
                  "the refit touches an axis exactly (\(after))")
            checkClose(Double(fit.imageRect.minX), Double(slack.width / 2), tolerance: 0.001,
                       "the refit stays centred horizontally (\(after))")
            checkClose(Double(fit.imageRect.minY), Double(slack.height / 2), tolerance: 0.001,
                       "the refit stays centred vertically (\(after))")

            let zoomedBefore = FrameZoomGeometry(viewport: before, imageSize: landscape, ratio: 3)
            let zoomedAfter = FrameZoomGeometry(viewport: after, imageSize: landscape, ratio: 3)
            checkClose(Double(zoomedAfter.scale / zoomedAfter.fitScale), 3, tolerance: 0.0001,
                       "the zoom level is still three times the fit after a resize (\(before) → \(after))")
            // A window narrower than the magnified frame cannot hold the middle of it at
            // a reachable position, so the comparison is only made where it can.
            let holds = { (geometry: FrameZoomGeometry) in
                geometry.scaledImageSize.width >= geometry.viewport.width
                    && geometry.scaledImageSize.height >= geometry.viewport.height
            }
            guard holds(zoomedBefore), holds(zoomedAfter) else { continue }
            let centreBefore = CGPoint(x: before.width / 2, y: before.height / 2)
            let focus = zoomedBefore.focus(atCanvasPoint: offset(zoomedBefore.scrollOrigin(focus: CGPoint(x: 0.5, y: 0.5)),
                                                                  centreBefore))
            let centreAfter = CGPoint(x: after.width / 2, y: after.height / 2)
            let readBack = zoomedAfter.focus(atCanvasPoint: offset(zoomedAfter.scrollOrigin(focus: focus),
                                                                    centreAfter))
            checkClose(Double(readBack.x), Double(focus.x), tolerance: 0.001,
                       "the point at the middle of the window survives \(before) → \(after)")
            checkClose(Double(readBack.y), Double(focus.y), tolerance: 0.001,
                       "the point at the middle of the window survives \(before) → \(after)")
        }
    })

    Registry.shared.add(suite: "Frame zoom", TestCase(name: "a pinch anchors on the point under the pointer", knownBug: nil) {
        // Magnification arrives as a delta with the pointer's position. The image point
        // under the pointer is read *before* the scale changes and the scroll keeps the
        // pointer over it — which is a different rule from putting that point in the
        // middle of the window, and the difference is the whole gesture.
        let pointer = CGPoint(x: 260, y: 270)
        let before = FrameZoomGeometry(viewport: viewport, imageSize: landscape, ratio: 2)
        let focus = before.focus(atCanvasPoint: offset(before.scrollOrigin(focus: CGPoint(x: 0.5, y: 0.5)), pointer))
        check(before.isScrollable, "a frame at 2× has room to move in the window")
        for delta in [0.1, 0.5, 1] as [CGFloat] {
            let after = FrameZoomGeometry(viewport: viewport, imageSize: landscape,
                                          ratio: FrameZoomGeometry.clampedRatio(before.ratio * (1 + delta)))
            let origin = after.scrollOrigin(keeping: pointer, under: focus)
            let readBack = after.focus(atCanvasPoint: offset(origin, pointer))
            checkClose(Double(readBack.x), Double(focus.x), tolerance: 0.001,
                       "the frame point under the pointer does not move (delta \(delta))")
            checkClose(Double(readBack.y), Double(focus.y), tolerance: 0.001,
                       "the frame point under the pointer does not move (delta \(delta))")
            // Anchoring is not centring: the point stays where the hand is, which for
            // an off-centre pointer is measurably not the middle of the window.
            let centred = after.scrollOrigin(focus: focus)
            check(abs(centred.x - origin.x) > 0.5 || abs(centred.y - origin.y) > 0.5,
                  "anchoring the pointer is not the same as centring the point (delta \(delta))")
        }
        // Zooming far enough out that the frame no longer fills the window cannot keep
        // a pointer anchored — it is centred instead — but the frame must never be
        // placed off the canvas, and a smaller frame must land in the middle.
        for delta in [-0.25, -0.5, -0.9] as [CGFloat] {
            let after = FrameZoomGeometry(viewport: viewport, imageSize: landscape,
                                          ratio: FrameZoomGeometry.clampedRatio(before.ratio * (1 + delta)))
            let origin = after.scrollOrigin(keeping: pointer, under: focus)
            let limit = CGSize(width: max(0, after.canvasSize.width - viewport.width),
                               height: max(0, after.canvasSize.height - viewport.height))
            check(origin.x >= -0.001 && origin.x <= limit.width + 0.001 && origin.y >= -0.001
                  && origin.y <= limit.height + 0.001,
                  "zooming out keeps the frame inside the canvas (delta \(delta))")
            // Once the frame is smaller than the window on an axis there is nothing to
            // anchor: it lands in the middle of that axis instead.
            if after.scaledImageSize.width < viewport.width {
                checkClose(Double(after.imageRect.midX), Double(viewport.width / 2), tolerance: 0.001,
                           "a frame narrower than the window ends up centred (delta \(delta))")
            }
            if after.scaledImageSize.height < viewport.height {
                checkClose(Double(after.imageRect.midY), Double(viewport.height / 2), tolerance: 0.001,
                           "a frame shorter than the window ends up centred (delta \(delta))")
            }
        }
    })

    Registry.shared.add(suite: "Frame zoom", TestCase(name: "the zoom is clamped to limits a layout can draw", knownBug: nil) {
        checkEqual(FrameZoomGeometry.clampedRatio(0), FrameZoomGeometry.ratioLimits.lowerBound,
                   "zooming out stops at a quarter of the fit rather than collapsing")
        checkEqual(FrameZoomGeometry.clampedRatio(1e9), FrameZoomGeometry.ratioLimits.upperBound,
                   "zooming in stops at 16×")
        checkEqual(FrameZoomGeometry.clampedRatio(1), 1, "a fit request is left alone")
        checkEqual(FrameZoomGeometry.clampedRatio(.nan), 1, "a NaN ratio falls back to the fit")
        checkEqual(FrameZoomGeometry.clampedRatio(.infinity), FrameZoomGeometry.ratioLimits.upperBound,
                   "an infinite ratio stops at the top rather than drawing nothing")
        // The stored ratio cannot escape the limits through the initialiser either, so
        // no consumer has to clamp a second time.
        for ratio in [-4, 0, 0.3, 2, 16, 900] as [CGFloat] {
            let geometry = FrameZoomGeometry(viewport: viewport, imageSize: landscape, ratio: ratio)
            checkEqual(geometry.ratio, FrameZoomGeometry.clampedRatio(ratio, fitScale: geometry.fitScale),
                       "a ratio of \(ratio) is clamped for this fit scale")
            check(FrameZoomGeometry.ratioLimits.contains(geometry.scale),
                  "the displayed source scale remains within the image limits")
        }
    })

    Registry.shared.add(suite: "Frame zoom", TestCase(name: "native 100 percent remains reachable from fit at every viewport size", knownBug: nil) {
        for (viewport, image) in [
            (CGSize(width: 800, height: 600), CGSize(width: 6000, height: 4000)),
            (CGSize(width: 1200, height: 800), CGSize(width: 24000, height: 16000)),
            (CGSize(width: 300, height: 200), CGSize(width: 64, height: 48))
        ] {
            let fitScale = FrameZoomGeometry.fitScale(viewport: viewport, imageSize: image)
            let fit = FrameZoomGeometry(viewport: viewport, imageSize: image, ratio: 1)
            let native = FrameZoomGeometry(viewport: viewport, imageSize: image, ratio: 1 / fitScale)
            checkClose(Double(fit.scale), Double(fitScale), tolerance: 0.000001,
                       "fit keeps the viewport-sized scale for \(image)")
            checkClose(Double(native.scale), 1, tolerance: 0.000001,
                       "the native ratio reaches one source pixel per point for \(image)")
            checkClose(Double(FrameZoomGeometry.clampedRatio(1 / fitScale, fitScale: fitScale)),
                       Double(1 / fitScale), tolerance: 0.000001,
                       "the native ratio is not clipped by a fit-relative zoom limit")
        }
    })

    Registry.shared.add(suite: "Frame zoom", TestCase(name: "a viewport or frame that is not ready yet draws at a scale, not at nothing", knownBug: nil) {
        // SwiftUI lays out after the first image arrives, and a preview can be opened on
        // a frame that has not been decoded. Neither may divide by zero.
        for viewport in [CGSize.zero, CGSize(width: 800, height: 0), CGSize(width: 0, height: 600)] {
            let geometry = FrameZoomGeometry(viewport: viewport, imageSize: landscape, ratio: 1)
            check(geometry.fitScale.isFinite && geometry.fitScale > 0,
                  "an unlaid-out viewport still yields a usable scale (\(viewport))")
            check(geometry.canvasSize.width.isFinite && geometry.canvasSize.height.isFinite,
                  "the canvas stays a finite size (\(viewport))")
            check(geometry.scrollOrigin(focus: CGPoint(x: 0.5, y: 0.5)).x.isFinite,
                  "the scroll origin stays finite (\(viewport))")
        }
        for image in [CGSize.zero, CGSize(width: 1920, height: 0), CGSize(width: 0, height: 1080)] {
            let geometry = FrameZoomGeometry(viewport: viewport, imageSize: image, ratio: 1)
            check(geometry.fitScale.isFinite && geometry.fitScale > 0,
                  "a frame with no pixels yet still yields a usable scale (\(image))")
            check(geometry.focus(atCanvasPoint: CGPoint(x: 10, y: 10)).x.isFinite,
                  "reading a focus point from a frameless canvas stays finite (\(image))")
        }
    })

    Registry.shared.add(suite: "Frame zoom", TestCase(name: "the surface never lets AppKit's magnification move the viewport out from under the fit", knownBug: nil) {
        // The mechanism of the regression, pinned without a window: a scroll view that
        // magnifies reports a clip view whose bounds size is the viewport *divided by*
        // the magnification, so a fit derived from it feeds back on itself. This view
        // keeps magnification at 1 and does the scaling in the canvas instead.
        await Task { @MainActor in
            let scroll = FrameZoomScrollView(frame: NSRect(x: 0, y: 0, width: 800, height: 600))
            scroll.setMagnification(0.5, centeredAt: NSPoint(x: 400, y: 300))
            checkEqual(scroll.magnification, 1, "a stray magnification is pinned at 1")
            scroll.setMagnification(4, centeredAt: NSPoint(x: 100, y: 100))
            checkEqual(scroll.magnification, 1, "the next stray magnification is pinned at 1 too")
            checkEqual(scroll.contentView.bounds.size, CGSize(width: 800, height: 600),
                       "the clip view's bounds stay the real viewport, so the fit is not measured against the zoom")
            check(scroll.documentView != nil, "the canvas is the document view")
            check(scroll.documentView?.subviews.count == 1,
                  "the canvas holds the image view that does the drawing")
        }.value
    })
}
