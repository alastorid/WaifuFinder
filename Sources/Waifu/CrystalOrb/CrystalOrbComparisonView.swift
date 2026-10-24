import SwiftUI
import AppKit

struct CrystalOrbComparisonView: View {
    @ObservedObject var model: WaifuCrystalOrbModel
    var session: WaifuSession? = nil
    @ObservedObject private var navigation: CrystalOrbNavigation
    private let forensic: LineageOverlayConfiguration?
    init(model: WaifuCrystalOrbModel, session: WaifuSession? = nil, forensic: LineageOverlayConfiguration? = nil) {
        self.model = model; self.session = session; self.navigation = model.navigation; self.forensic = forensic
    }
    @Environment(\.displayScale) private var backing
    var body: some View {
        GeometryReader { outer in
            if let pair = model.pair {
                let divided = model.mode == .sideBySide
                let reference = CGSize(width: divided ? outer.size.width / 2 : outer.size.width,
                    height: outer.size.height)
                let fit = model.fillViewport ? max(reference.width / pair.alignment.size.width, reference.height / pair.alignment.size.height) : min(reference.width / pair.alignment.size.width, reference.height / pair.alignment.size.height)
                let scale = max(0.00001, (model.actualPixels ? 1 / backing : fit) * model.zoom)
                Group {
                    if let pane = model.maximizedPane {
                        paneView(pane, pair: pair, scale: scale)
                    } else if model.mode == .sideBySide {
                        HSplitView { paneView("A", pair: pair, scale: scale); paneView("B", pair: pair, scale: scale) }
                    } else { paneView(model.mode == .difference ? "Δ" : "Mixed", pair: pair, scale: scale) }
                }
            } else if let query = model.query {
                Image(decorative: query.image, scale: 1).resizable().scaledToFit().frame(maxWidth: .infinity, maxHeight: .infinity)
                    .googleLensContextMenu(isAvailable: session != nil) {
                        if let session {
                            Task { await GoogleLensSearch.search(image: query.image, in: session) }
                        }
                    }
            }
        }.background(Color.black.opacity(0.94))
    }
    private func paneView(_ kind: String, pair: WaifuCrystalOrbModel.Pair, scale: CGFloat) -> some View {
        CrystalOrbInspectionPane(model: model, pair: pair, kind: kind, scale: scale, session: session, forensic:forensic)
    }
}

private struct CrystalOrbInspectionPane: View {
    @ObservedObject var model: WaifuCrystalOrbModel
    let pair: WaifuCrystalOrbModel.Pair
    let kind: String
    let scale: CGFloat
    let session: WaifuSession?
    @ObservedObject private var navigation: CrystalOrbNavigation
    private let forensic: LineageOverlayConfiguration?
    init(model: WaifuCrystalOrbModel, pair: WaifuCrystalOrbModel.Pair, kind: String, scale: CGFloat,
         session: WaifuSession? = nil, forensic: LineageOverlayConfiguration? = nil) {
        self.model = model; self.pair = pair; self.kind = kind; self.scale = scale; self.session = session
        self.navigation = model.navigation; self.forensic = forensic
    }
    @State private var blinkCandidate = true
    @State private var dragCenter: CGPoint?
    @State private var dragRegistration: CrystalOrbRegistration?
    @State private var dragStart: CGPoint?
    private var canvas: CGSize { pair.alignment.size }
    private var center: CGPoint { model.navigationCenter ?? CGPoint(x: canvas.width / 2, y: canvas.height / 2) }
    private var title: String {
        if kind == "Δ" { return "DIFFERENCE · " + model.differenceSettings.display.rawValue }
        if kind == "A" { return "TARGET A" }
        if kind == "B" { return "CANDIDATE B" }
        return model.mode == .swipe ? "TARGET A ← Swipe → CANDIDATE B" : (model.showCandidate != model.holdingToggle ? "CANDIDATE B" : "TARGET A")
    }
    private var lensSearchUsesCandidate: Bool {
        if kind == "A" { return false }
        if kind != "Mixed" { return true }
        if model.mode == .toggle { return model.showCandidate != model.holdingToggle }
        if model.mode == .blink { return blinkCandidate }
        return true
    }
    var body: some View {
        GeometryReader { geometry in
            let viewport = CrystalOrbViewport(size: geometry.size, center: center, scale: scale)
            ZStack(alignment: .topLeading) {
                pixels(viewport)
                    .allowsHitTesting(false)
                annotation(viewport).allowsHitTesting(false)
                CrystalOrbPointerSurface(onMove: { model.hover(at: viewport.comparison($0)) },
                    onExit: { model.hover(at: nil) },
                    onDrag: { start, current, ended in
                        model.hover(at: viewport.comparison(current))
                        drag(start, current, ended, viewport: viewport)
                    },
                    onPan: { delta in model.center(on: CGPoint(x: center.x - delta.width / scale, y: center.y + delta.height / scale)) },
                    onZoom: { factor, location in
                        let anchor = viewport.comparison(location)
                        let zoom = min(128, max(0.05, model.zoom * factor))
                        let ratio = zoom / model.zoom
                        model.zoom = zoom
                        model.center(on: CGPoint(x: anchor.x - (anchor.x - center.x) / ratio, y: anchor.y - (anchor.y - center.y) / ratio))
                    })
                if let forensic, kind == "B" || (kind == "Mixed" && model.mode == .toggle && model.showCandidate != model.holdingToggle) || (kind == "Mixed" && model.mode == .swipe) || (kind == "Mixed" && model.mode == .alpha) || (kind == "Mixed" && model.mode == .blink && blinkCandidate) {
                    LineageEvidenceOverlay(configuration:forensic,viewport:viewport,alignment:pair.alignment)
                        .mask(alignment:.trailing) {
                            Rectangle().frame(width:kind == "Mixed" && model.mode == .swipe ? viewport.size.width*(1-model.swipe) : viewport.size.width)
                        }
                }
                HStack(spacing: 6) {
                    if model.mode == .toggle {
                        Button(title) { model.showCandidate.toggle() }.buttonStyle(.plain).font(.caption.weight(.medium))
                    } else { Text(title).font(.caption.weight(.medium)).lineLimit(1) }
                    if !(kind == "Δ" ? pair.differenceIsOriginal : (kind == "A" ? pair.targetIsOriginal : pair.isOriginal)) { Text("Preview").font(.caption2).foregroundStyle(.yellow) }
                    if !model.pairIsCurrent { Text("Previous pair").font(.caption2) }
                    else if kind == "Δ", let diff = pair.difference,
                            diff.alignment != pair.alignment || diff.settings != model.differenceSettings {
                        Text("Previous analysis · Refining").font(.caption2)
                    }
                    Spacer(minLength: 0)
                    Button { model.maximizedPane = model.maximizedPane == kind ? nil : kind } label: {
                        Image(systemName: model.maximizedPane == kind ? "arrow.down.right.and.arrow.up.left" : "arrow.up.left.and.arrow.down.right")
                    }.buttonStyle(.plain).accessibilityLabel("Maximize or restore " + title)
                }.padding(7).background(.regularMaterial)
                if kind == "Mixed", model.mode == .swipe { divider(geometry.size) }
                if model.showOverview { overview(viewport).frame(width: 120, height: 85).padding(10).frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomTrailing) }
            }.coordinateSpace(name: "orb-pane-" + kind).clipped()
        }.frame(minWidth: 80, minHeight: 70)
        .googleLensContextMenu(isAvailable: session != nil) {
            if let session {
                let image = lensSearchUsesCandidate ? pair.candidate : pair.target
                Task { await GoogleLensSearch.search(image: image, in: session) }
            }
        }
        .task(id:model.mode) {
            blinkCandidate = true
            guard model.mode == .blink else { return }
            while !Task.isCancelled {
                do { try await Task.sleep(for:.milliseconds(650)) } catch { return }
                blinkCandidate.toggle()
            }
        }
    }
    @ViewBuilder private func pixels(_ viewport: CrystalOrbViewport) -> some View {
        if kind == "Δ" {
            if let difference = pair.difference {
                if model.differenceSettings.display == .red {
                    source(candidate: false, viewport: viewport)
                    raster(difference, mask: false, viewport: viewport).opacity(model.opacity)
                } else { raster(difference, mask: false, viewport: viewport) }
            }
            else { source(candidate: true, viewport: viewport) }
        } else if kind == "Mixed", model.mode == .alpha {
            source(candidate:false,viewport:viewport)
            source(candidate:true,viewport:viewport).opacity(model.opacity)
        } else if kind == "Mixed", model.mode == .blink {
            source(candidate:blinkCandidate,viewport:viewport)
        } else if kind == "Mixed", model.mode == .swipe {
            source(candidate: true, viewport: viewport)
            source(candidate: false, viewport: viewport).mask(alignment: .leading) { Rectangle().frame(width: viewport.size.width * model.swipe) }
        } else {
            source(candidate: kind == "B" || (kind == "Mixed" && model.showCandidate != model.holdingToggle), viewport: viewport)
        }
        if model.redOverlay, kind != "Δ", let difference = pair.difference, difference.alignment == pair.alignment, difference.settings == model.differenceSettings { raster(difference, mask: true, viewport: viewport).opacity(model.opacity) }
    }
    private func source(candidate: Bool, viewport: CrystalOrbViewport) -> some View {
        let image = candidate ? pair.candidate : pair.target
        let transform = CGAffineTransform(a: 1, b: 0, c: 0, d: -1, tx: 0, ty: CGFloat(image.height))
            .concatenating(pair.alignment.transform(candidate: candidate, decodedSize: CGSize(width: image.width, height: image.height)))
            .concatenating(CGAffineTransform(a: 1, b: 0, c: 0, d: -1, tx: 0, ty: canvas.height))
        let origin = viewport.screen(CGPoint(x: 0, y: canvas.height))
        return Image(decorative: image, scale: 1).resizable().interpolation(scale >= 8 ? .none : .high)
            .frame(width: CGFloat(image.width), height: CGFloat(image.height))
            .transformEffect(transform)
            .frame(width: canvas.width, height: canvas.height, alignment: .topLeading)
            .scaleEffect(scale, anchor: .topLeading)
            .offset(x: origin.x, y: origin.y)
            .frame(width: viewport.size.width, height: viewport.size.height, alignment: .topLeading)
    }
    private func raster(_ diff: CrystalOrbDifference, mask: Bool, viewport: CrystalOrbViewport) -> some View {
        CrystalOrbRasterView(image: mask ? diff.nativeRedMask : diff.nativeDifference, preview: mask ? diff.redMask : diff.difference,
            identity: diff.id.uuidString + (mask ? ":mask" : ":diff"), canvas: canvas, scale: scale,
            pan: CGSize(width: (canvas.width / 2 - center.x) * scale, height: (center.y - canvas.height / 2) * scale))
            .frame(width: viewport.size.width, height: viewport.size.height)
    }
    private func annotation(_ viewport: CrystalOrbViewport) -> some View {
        Canvas { context, size in
            if model.showPixelGrid, scale >= 12 {
                let visible = viewport.visible.intersection(pair.alignment.extent)
                if !visible.isNull {
                    var path = Path()
                    for x in Int(floor(visible.minX))...Int(ceil(visible.maxX)) {
                        let a = viewport.screen(CGPoint(x: CGFloat(x), y: visible.minY)), b = viewport.screen(CGPoint(x: CGFloat(x), y: visible.maxY))
                        path.move(to: a); path.addLine(to: b)
                    }
                    for y in Int(floor(visible.minY))...Int(ceil(visible.maxY)) {
                        path.move(to: viewport.screen(CGPoint(x: visible.minX, y: CGFloat(y)))); path.addLine(to: viewport.screen(CGPoint(x: visible.maxX, y: CGFloat(y))))
                    }
                    context.stroke(path, with: .color(.white.opacity(0.25)), lineWidth: 0.5)
                }
            }
            if let roi = model.roi {
                let a = viewport.screen(CGPoint(x: roi.minX, y: roi.maxY))
                context.stroke(Path(CGRect(origin: a, size: CGSize(width: roi.width * scale, height: roi.height * scale))), with: .color(.cyan), style: StrokeStyle(lineWidth: 1, dash: [4,3]))
            }
            if model.showCrosshair, let point = model.cursor {
                let p = viewport.screen(CGPoint(x: floor(point.x)+0.5, y: floor(point.y)+0.5))
                var path = Path(); path.move(to: CGPoint(x: p.x, y: 0)); path.addLine(to: CGPoint(x: p.x, y: size.height))
                path.move(to: CGPoint(x: 0, y: p.y)); path.addLine(to: CGPoint(x: size.width, y: p.y))
                context.stroke(path, with: .color(.white.opacity(0.8)), lineWidth: 0.5)
            }
        }
    }
    private func drag(_ start: CGPoint, _ current: CGPoint, _ ended: Bool, viewport: CrystalOrbViewport) {
        if dragCenter == nil { dragCenter = center; dragRegistration = model.registration; dragStart = viewport.comparison(start) }
        let dx = (current.x - start.x) / scale, dy = -(current.y - start.y) / scale
        if model.selectingRegion, let anchor = dragStart {
            let rect = CGRect(x: min(anchor.x,anchor.x+dx), y: min(anchor.y,anchor.y+dy), width: abs(dx), height: abs(dy)).intersection(pair.alignment.extent)
            model.roi = rect.isNull || rect.isEmpty ? nil : rect
        } else if model.alignmentEditing, !model.alignmentLocked, let initial = dragRegistration {
            var r = initial; r.x += dx; r.y += dy; model.registration = r
        } else if let initial = dragCenter { model.center(on: CGPoint(x: initial.x-dx, y: initial.y-dy)) }
        if ended { dragCenter = nil; dragRegistration = nil; dragStart = nil }
    }
    private func divider(_ size: CGSize) -> some View {
        Rectangle().fill(.white).frame(width: 2, height: size.height)
            .overlay { Image(systemName: "arrow.left.and.right").padding(6).background(.regularMaterial, in: Circle()) }
            .frame(width: 32).contentShape(Rectangle()).offset(x: size.width * model.swipe - 16)
            .gesture(DragGesture(minimumDistance: 0, coordinateSpace: .named("orb-pane-" + kind)).onChanged { value in
                model.swipe = min(1, max(0, value.location.x / max(1,size.width)))
            }).accessibilityLabel("Swipe divider")
            .accessibilityAdjustableAction { model.swipe = min(1, max(0, model.swipe + ($0 == .increment ? 0.05 : -0.05))) }
    }
    private func overview(_ viewport: CrystalOrbViewport) -> some View {
        GeometryReader { geometry in
            let factor = min(geometry.size.width / canvas.width, geometry.size.height / canvas.height)
            let offset = CGPoint(x: (geometry.size.width-canvas.width*factor)/2, y: (geometry.size.height-canvas.height*factor)/2)
            ZStack(alignment: .topLeading) {
                source(candidate: false, viewport: CrystalOrbViewport(size: geometry.size, center: CGPoint(x: canvas.width/2, y: canvas.height/2), scale: factor))
                let visible = viewport.visible.intersection(pair.alignment.extent)
                if !visible.isNull {
                    Rectangle().stroke(.yellow, lineWidth: 1).frame(width: visible.width*factor, height: visible.height*factor)
                        .offset(x: offset.x+visible.minX*factor, y: offset.y+(canvas.height-visible.maxY)*factor)
                }
            }.background(.black.opacity(0.7)).contentShape(Rectangle())
                .gesture(DragGesture(minimumDistance: 0).onChanged { value in
                    model.center(on: CGPoint(x: (value.location.x-offset.x)/factor, y: canvas.height-(value.location.y-offset.y)/factor))
                })
        }
    }
}

/// Local NSView owns pointer events only inside a pane, without global monitors.
struct CrystalOrbPointerSurface: NSViewRepresentable {
    var onMove: (CGPoint) -> Void
    var onExit: () -> Void = {}
    var onDrag: (CGPoint, CGPoint, Bool) -> Void
    var onPan: (CGSize) -> Void
    var onZoom: (CGFloat, CGPoint) -> Void
    func makeNSView(context: Context) -> Surface { Surface() }
    func updateNSView(_ view: Surface, context: Context) { view.actions = self }
    final class Surface: NSView {
        var actions: CrystalOrbPointerSurface?
        private var start: CGPoint?
        override var isFlipped: Bool { true }
        override var acceptsFirstResponder: Bool { false }
        override func updateTrackingAreas() {
            super.updateTrackingAreas(); trackingAreas.forEach(removeTrackingArea)
            addTrackingArea(NSTrackingArea(rect: bounds, options: [.mouseMoved, .mouseEnteredAndExited, .activeInKeyWindow, .inVisibleRect], owner: self))
        }
        private func point(_ event: NSEvent) -> CGPoint { convert(event.locationInWindow, from: nil) }
        override func mouseMoved(with event: NSEvent) { actions?.onMove(point(event)) }
        override func mouseEntered(with event: NSEvent) { actions?.onMove(point(event)) }
        override func mouseExited(with event: NSEvent) { actions?.onExit() }
        override func mouseDown(with event: NSEvent) { start = point(event); actions?.onMove(point(event)) }
        override func mouseDragged(with event: NSEvent) { if let start { actions?.onDrag(start, point(event), false) } }
        override func mouseUp(with event: NSEvent) { if let start { actions?.onDrag(start, point(event), true) }; start = nil }
        override func magnify(with event: NSEvent) { actions?.onZoom(max(0.1, 1+event.magnification), point(event)) }
        override func scrollWheel(with event: NSEvent) {
            if event.hasPreciseScrollingDeltas && !event.modifierFlags.contains(.command) {
                actions?.onPan(CGSize(width: event.scrollingDeltaX, height: event.scrollingDeltaY))
            } else { actions?.onZoom(exp(event.scrollingDeltaY * 0.025), point(event)) }
        }
    }
}
