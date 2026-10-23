import CoreGraphics
import Foundation

/// Images have already had EXIF orientation applied by ImageIO/PhotoKit.
/// Source and comparison coordinates are pixel-edge coordinates, origin bottom-left.
/// Viewport coordinates are points, origin top-left. Pixel centers are n + 0.5.
struct CrystalOrbRegistration: Sendable, Equatable {
    var x: CGFloat = 0
    var y: CGFloat = 0
    var scale: CGFloat = 1
    var degrees: CGFloat = 0
}

struct CrystalOrbViewport: Sendable {
    let size: CGSize
    let center: CGPoint
    let scale: CGFloat // points per comparison pixel; 1/backing is native 1:1
    func screen(_ point: CGPoint) -> CGPoint {
        CGPoint(x: size.width / 2 + (point.x - center.x) * scale,
                y: size.height / 2 - (point.y - center.y) * scale)
    }
    func comparison(_ point: CGPoint) -> CGPoint {
        CGPoint(x: center.x + (point.x - size.width / 2) / scale,
                y: center.y - (point.y - size.height / 2) / scale)
    }
    var visible: CGRect {
        CGRect(x: center.x - size.width / scale / 2, y: center.y - size.height / scale / 2,
               width: size.width / scale, height: size.height / scale)
    }

}

struct CrystalOrbAlignment: Sendable, Equatable {
    let size: CGSize
    let target: CGRect
    let candidate: CGRect
    let targetSize: CGSize
    let candidateSize: CGSize
    let registration: CrystalOrbRegistration
    var extent: CGRect { CGRect(origin: .zero, size: size) }
    init(target: CGSize, candidate: CGSize, registration: CrystalOrbRegistration = .init()) {
        targetSize = target; candidateSize = candidate; self.registration = registration
        let canvas = CGSize(width: max(target.width, candidate.width), height: max(target.height, candidate.height))
        func fit(_ image: CGSize) -> CGRect {
            let scale = min(canvas.width / max(1, image.width), canvas.height / max(1, image.height))
            let width = image.width * scale, height = image.height * scale
            return CGRect(x: (canvas.width - width) / 2, y: (canvas.height - height) / 2, width: width, height: height)
        }
        size = canvas; self.target = fit(target); self.candidate = fit(candidate)
    }
    func transform(candidate isCandidate: Bool, decodedSize: CGSize? = nil) -> CGAffineTransform {
        let rect = isCandidate ? candidate : target
        let source = decodedSize ?? (isCandidate ? candidateSize : targetSize)
        let base = CGAffineTransform(scaleX: rect.width / max(1, source.width), y: rect.height / max(1, source.height))
            .concatenating(CGAffineTransform(translationX: rect.minX, y: rect.minY))
        guard isCandidate else { return base }
        let r = registration
        return base.concatenating(CGAffineTransform(translationX: -rect.midX, y: -rect.midY))
            .concatenating(CGAffineTransform(scaleX: r.scale, y: r.scale))
            .concatenating(CGAffineTransform(rotationAngle: r.degrees * .pi / 180))
            .concatenating(CGAffineTransform(translationX: rect.midX + r.x, y: rect.midY + r.y))
    }
    func source(_ point: CGPoint, candidate: Bool) -> CGPoint? {
        let p = point.applying(transform(candidate: candidate).inverted())
        let s = candidate ? candidateSize : targetSize
        return p.x >= 0 && p.y >= 0 && p.x < s.width && p.y < s.height ? p : nil
    }
    func comparison(_ point: CGPoint, candidate: Bool) -> CGPoint { point.applying(transform(candidate: candidate)) }
    func contains(_ point: CGPoint, candidate: Bool) -> Bool { source(point, candidate: candidate) != nil }
}
