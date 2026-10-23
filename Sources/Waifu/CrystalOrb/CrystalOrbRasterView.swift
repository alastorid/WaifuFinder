import AppKit
import SwiftUI
import CoreImage

/// Only the on-screen region is rasterized, at the window's backing resolution.
/// A transformed preview remains visible while the native-detail tile is prepared.
struct CrystalOrbRasterView: NSViewRepresentable {
    let image: CIImage
    let preview: CGImage
    let identity: String
    let canvas: CGSize
    let scale: CGFloat
    let pan: CGSize

    func makeNSView(context: Context) -> Surface { Surface() }
    func updateNSView(_ view: Surface, context: Context) {
        view.configure(image: image, preview: preview, identity: identity, canvas: canvas, scale: scale, pan: pan)
    }
    static func dismantleNSView(_ view: Surface, coordinator: ()) { view.stop() }

    @MainActor final class Surface: NSView {
        private let fallback = CALayer()
        private let detail = CALayer()
        private var task: Task<Void, Never>?
        private var ticket = UUID()
        private var key: Key?
        private var input: (CIImage, CGImage, String, CGSize, CGFloat, CGSize)?
        private struct Key: Equatable {
            let identity: String
            let size: CGSize
            let canvas: CGSize
            let scale: CGFloat
            let pan: CGSize
            let backing: CGFloat
        }
        override init(frame: NSRect) {
            super.init(frame: frame)
            wantsLayer = true
            layer?.masksToBounds = true
            layer?.addSublayer(fallback)
            layer?.addSublayer(detail)
        }
        required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }
        override func hitTest(_ point: NSPoint) -> NSView? { nil }
        func stop() { task?.cancel(); ticket = UUID(); key = nil; input = nil }
        override func layout() {
            super.layout()
            if let input { configure(image: input.0, preview: input.1, identity: input.2, canvas: input.3, scale: input.4, pan: input.5) }
        }

        func configure(image: CIImage, preview: CGImage, identity: String, canvas: CGSize, scale: CGFloat, pan: CGSize) {
            input = (image, preview, identity, canvas, scale, pan)
            let backing = window?.backingScaleFactor ?? 2
            let next = Key(identity: identity, size: bounds.size, canvas: canvas, scale: scale, pan: pan, backing: backing)
            guard key != next, bounds.width > 0, bounds.height > 0 else { return }
            task?.cancel()
            let request = UUID(); ticket = request
            key = next
            CATransaction.begin(); CATransaction.setDisableActions(true)
            fallback.isHidden = false
            fallback.contents = preview
            fallback.frame = CGRect(x: (bounds.width - canvas.width * scale) / 2 + pan.width,
                                    y: (bounds.height - canvas.height * scale) / 2 - pan.height,
                                    width: canvas.width * scale, height: canvas.height * scale)
            detail.contents = nil
            detail.frame = bounds
            detail.contentsScale = backing
            CATransaction.commit()
            task = Task { [weak self] in
                do {
                    try await Task.sleep(for: .milliseconds(20))
                    let pixels = try await CrystalOrbViewportRenderer.shared.render(image: image, canvas: canvas,
                        viewport: next.size, scale: scale, pan: pan, backing: backing, identity: identity)
                    guard !Task.isCancelled, let self, ticket == request else { return }
                    CATransaction.begin(); CATransaction.setDisableActions(true)
                    detail.contents = pixels
                    fallback.isHidden = true
                    CATransaction.commit()
                } catch { /* The bounded preview remains available. */ }
            }
        }
    }
}

actor CrystalOrbViewportRenderer {
    static let shared = CrystalOrbViewportRenderer()
    private let context = CIContext(options: [
        .workingColorSpace: CGColorSpace(name: CGColorSpace.sRGB)!,
        .outputColorSpace: CGColorSpace(name: CGColorSpace.sRGB)!, .cacheIntermediates: false])

    private var tiles: [String: CGImage] = [:]
    private var order: [String] = []
    private var cost = 0
    private var material: String?
    func render(image: CIImage, canvas: CGSize, viewport: CGSize, scale: CGFloat, pan: CGSize, backing: CGFloat, identity: String? = nil) throws -> CGImage {
        try Task.checkCancellation()
        let key = identity.map { "\($0):\(canvas):\(viewport):\(scale):\(pan):\(backing)" }
        if let key, let cached = tiles[key] { return cached }
        let nextMaterial = identity?.split(separator: ":").first.map(String.init)
        if nextMaterial != material { context.clearCaches(); material = nextMaterial }
        let offset = CGPoint(x: (viewport.width - canvas.width * scale) / 2 + pan.width,
                             y: (viewport.height - canvas.height * scale) / 2 - pan.height)
        let sampled = scale * backing >= 8 ? image.samplingNearest() : image
        let transformed = sampled.transformed(by: CGAffineTransform(scaleX: scale * backing, y: scale * backing))
            .transformed(by: CGAffineTransform(translationX: offset.x * backing, y: offset.y * backing))
        let extent = CGRect(x: 0, y: 0, width: ceil(viewport.width * backing), height: ceil(viewport.height * backing))
        guard let result = context.createCGImage(transformed, from: extent, format: .RGBA8,
                                                 colorSpace: CGColorSpace(name: CGColorSpace.sRGB)) else { throw CrystalOrbImageSource.failure }
        try Task.checkCancellation()
        let bytes = result.bytesPerRow * result.height
        if let key, bytes <= 64 * 1024 * 1024 {
            while cost + bytes > 64 * 1024 * 1024 || order.count >= 8 {
                guard !order.isEmpty else { break }
                let oldest = order.removeFirst()
                if let removed = tiles.removeValue(forKey: oldest) { cost -= removed.bytesPerRow * removed.height }
            }
            tiles[key] = result; order.append(key); cost += bytes
        }
        return result
    }
}
