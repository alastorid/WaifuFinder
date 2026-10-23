import SwiftUI

struct CrystalOrbComparisonView: View {
    @ObservedObject var model: WaifuCrystalOrbModel
    @Environment(\.displayScale) private var displayScale
    @GestureState private var drag = CGSize.zero
    @GestureState private var magnification: CGFloat = 1

    var body: some View {
        GeometryReader { geometry in
            if let pair = model.pair {
                let alignment = pair.difference?.alignment ?? CrystalOrbAlignment(
                    target: CGSize(width: pair.target.width, height: pair.target.height),
                    candidate: CGSize(width: pair.candidate.width, height: pair.candidate.height))
                let size = CGSize(width: model.mode == .sideBySide ? (geometry.size.width - 1) / 2 : geometry.size.width,
                                  height: geometry.size.height)
                let fit = min(size.width / alignment.size.width, size.height / alignment.size.height)
                let scale = (model.actualPixels ? 1 / max(1, displayScale) : fit) * model.zoom * magnification
                HStack(spacing: 1) {
                    viewport(pair: pair, alignment: alignment, size: size, scale: scale, target: true)
                    if model.mode == .sideBySide {
                        viewport(pair: pair, alignment: alignment, size: size, scale: scale, target: false)
                    }
                }
                .overlay(alignment: .top) {
                    if !model.pairIsCurrent {
                        Text(model.loading ? "Previous comparison · Loading selected candidate…" : "Previous comparison · Selected original unavailable")
                            .padding(8).background(.regularMaterial, in: Capsule()).padding(8)
                    }
                }
            } else {
                ContentUnavailableView("Image comparison", systemImage: "photo.on.rectangle",
                    description: Text(model.loading ? "Loading original…" : "Choose a query image and a candidate to compare."))
            }
        }
        .background(Color.black.opacity(0.92))
    }

    private func viewport(pair: WaifuCrystalOrbModel.Pair, alignment: CrystalOrbAlignment,
                          size: CGSize, scale: CGFloat, target: Bool) -> some View {
        ZStack(alignment: .topLeading) {
            Group {
                if model.mode == .difference {
                    if let difference = pair.difference {
                        differenceLayer(difference, mask: false, size: size, scale: scale)
                    } else { Text(model.comparisonError ?? "Computing pixel difference…").foregroundStyle(.white) }
                } else if model.mode == .swipe {
                    layer(pair.candidate, rect: alignment.candidate, canvas: alignment.size, viewport: size, scale: scale)
                    layer(pair.target, rect: alignment.target, canvas: alignment.size, viewport: size, scale: scale)
                        .mask(alignment: .leading) { Rectangle().frame(width: size.width * model.swipe) }
                } else {
                    let showingTarget = model.mode == .sideBySide ? target : !(model.showCandidate != model.holdingToggle)
                    layer(showingTarget ? pair.target : pair.candidate,
                          rect: showingTarget ? alignment.target : alignment.candidate, canvas: alignment.size, viewport: size, scale: scale)
                }
            }.allowsHitTesting(false)
            if model.redOverlay, model.mode != .difference, let diff = pair.difference {
                differenceLayer(diff, mask: true, size: size, scale: scale)
                    .opacity(model.opacity).allowsHitTesting(false)
            }

            Text(label(target: target)).font(.caption.weight(.medium)).padding(7)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 5)).padding(8)
                .allowsHitTesting(false)

            if model.mode == .swipe {
                Rectangle().fill(.white).frame(width: 2).frame(maxHeight: .infinity)
                    .overlay(alignment: .center) { Image(systemName: "arrow.left.and.right").padding(7).background(.regularMaterial, in: Circle()) }
                    .frame(width: 28).contentShape(Rectangle())
                    .offset(x: size.width * model.swipe - 14)
                    .gesture(DragGesture(minimumDistance: 0, coordinateSpace: .named("orb.viewport")).onChanged { value in
                        model.swipe = min(1, max(0, value.location.x / max(1, size.width)))
                    })
                    .accessibilityLabel("Comparison divider")
                    .accessibilityAdjustableAction { direction in model.swipe = min(1, max(0, model.swipe + (direction == .increment ? 0.05 : -0.05))) }
            }
        }
        .frame(width: size.width, height: size.height).clipped().coordinateSpace(name: "orb.viewport")
        .contentShape(Rectangle())
        .gesture(DragGesture().updating($drag) { value, state, _ in state = value.translation }
            .onEnded { value in model.pan.width += value.translation.width; model.pan.height += value.translation.height })
        .simultaneousGesture(MagnificationGesture().updating($magnification) { value, state, _ in state = value }
            .onEnded { value in model.zoom = min(32, max(0.1, model.zoom * value)) })
    }

    private func label(target: Bool) -> String {
        switch model.mode {
        case .sideBySide: return target ? "TARGET" : "CANDIDATE"
        case .swipe: return "TARGET ← Swipe → CANDIDATE"
        case .toggle: return model.showCandidate != model.holdingToggle ? "CANDIDATE" : "TARGET"
        case .difference: return "ΔRGB · Black = equal · Magenta = unmatched"
        }
    }

    private func differenceLayer(_ difference: CrystalOrbDifference, mask: Bool, size: CGSize, scale: CGFloat) -> some View {
        CrystalOrbRasterView(image: mask ? difference.nativeRedMask : difference.nativeDifference,
            preview: mask ? difference.redMask : difference.difference,
            identity: difference.id.uuidString + (mask ? ":red" : ":difference"),
            canvas: difference.alignment.size, scale: scale,
            pan: CGSize(width: model.pan.width + drag.width, height: model.pan.height + drag.height))
            .frame(width: size.width, height: size.height)
    }

    private func layer(_ image: CGImage, rect: CGRect, canvas: CGSize, viewport: CGSize, scale: CGFloat) -> some View {
        Image(decorative: image, scale: 1).resizable().interpolation(.high)
            .frame(width: rect.width, height: rect.height)
            .position(x: rect.midX, y: canvas.height - rect.midY)
            .frame(width: canvas.width, height: canvas.height)
            .scaleEffect(scale, anchor: .topLeading)
            .offset(x: (viewport.width - canvas.width * scale) / 2 + model.pan.width + drag.width,
                    y: (viewport.height - canvas.height * scale) / 2 + model.pan.height + drag.height)
            .frame(width: viewport.width, height: viewport.height, alignment: .topLeading)
    }
}
