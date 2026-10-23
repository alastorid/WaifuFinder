import CoreImage
import CoreGraphics
import Foundation

struct CrystalOrbAlignment: Sendable, Equatable {
    let size: CGSize
    let target: CGRect
    let candidate: CGRect

    init(target: CGSize, candidate: CGSize) {
        let canvas = CGSize(width: max(target.width, candidate.width), height: max(target.height, candidate.height))
        func fit(_ image: CGSize) -> CGRect {
            let scale = min(canvas.width / max(1, image.width), canvas.height / max(1, image.height))
            let width = image.width * scale, height = image.height * scale
            return CGRect(x: (canvas.width - width) / 2, y: (canvas.height - height) / 2, width: width, height: height)
        }
        size = canvas
        self.target = fit(target)
        self.candidate = fit(candidate)
    }
}

struct CrystalOrbDifference: Sendable {
    let id = UUID()
    let alignment: CrystalOrbAlignment
    let difference: CGImage
    let redMask: CGImage
    let nativeDifference: CIImage
    let nativeRedMask: CIImage
}

enum CrystalOrbDifferenceRenderer {
    /// GPU-backed sRGB absolute difference, retained as a lazy native graph.
    /// Small previews paint immediately; viewport rendering preserves native detail.
    /// Magenta denotes pixels without a correspondence.
    static func render(target: CGImage, candidate: CGImage) throws -> CrystalOrbDifference {
        let alignment = CrystalOrbAlignment(target: CGSize(width: target.width, height: target.height),
                                             candidate: CGSize(width: candidate.width, height: candidate.height))
        let extent = CGRect(origin: .zero, size: alignment.size)
        let color = CGColorSpace(name: CGColorSpace.sRGB)!
        func aligned(_ image: CGImage, rect: CGRect) -> CIImage {
            // Clamp resampling at source edges, and compare transparency over
            // the same opaque black canvas used by the inspection view. Equal
            // transparent pixels are valid matches, not unmatched regions.
            let scaled = CIImage(cgImage: image).clampedToExtent()
                .applyingFilter("CILanczosScaleTransform", parameters: [
                    kCIInputScaleKey: rect.width / CGFloat(image.width), kCIInputAspectRatioKey: 1])
                .transformed(by: CGAffineTransform(translationX: rect.minX, y: rect.minY)).cropped(to: rect)
            return scaled.composited(over: CIImage(color: .black).cropped(to: rect))
        }
        let a = aligned(target, rect: alignment.target)
        let b = aligned(candidate, rect: alignment.candidate)
        let overlap = alignment.target.intersection(alignment.candidate).integral.intersection(extent)
        let delta = a.applyingFilter("CIDifferenceBlendMode", parameters: [kCIInputBackgroundImageKey: b]).cropped(to: overlap)
        let difference = delta.composited(over: CIImage(color: CIColor(red: 1, green: 0, blue: 1)).cropped(to: extent))
        let outside = CIImage(color: CIColor(red: 1, green: 0, blue: 0)).cropped(to: extent)
            .applyingFilter("CISourceOutCompositing", parameters: [kCIInputBackgroundImageKey:
                CIImage(color: .white).cropped(to: overlap)])
        // Use the strongest changed channel with a square-root display curve.
        // Small, real pixel changes remain visible; an exact match stays clear.
        let intensity = delta.applyingFilter("CIMaximumComponent")
            .applyingFilter("CIGammaAdjust", parameters: ["inputPower": 0.5])
        let red = intensity.applyingFilter("CIColorMatrix", parameters: [
            "inputRVector": CIVector(x: 0, y: 0, z: 0, w: 0),
            "inputGVector": CIVector(x: 0, y: 0, z: 0, w: 0),
            "inputBVector": CIVector(x: 0, y: 0, z: 0, w: 0),
            "inputAVector": CIVector(x: 1, y: 0, z: 0, w: 0),
            "inputBiasVector": CIVector(x: 1, y: 0, z: 0, w: 0)]).composited(over: outside)
        // Fit previews have a bounded cost. Native CI graphs stay lazy and are
        // evaluated only for the visible viewport, including at actual-pixel zoom.
        let previewScale = min(1, 1600 / max(extent.width, extent.height))
        let previewExtent = CGRect(x: 0, y: 0, width: ceil(extent.width * previewScale), height: ceil(extent.height * previewScale))
        let transform = CGAffineTransform(scaleX: previewScale, y: previewScale)
        guard let diff = context.createCGImage(difference.transformed(by: transform), from: previewExtent, format: .RGBA8, colorSpace: color),
              let redMask = context.createCGImage(red.transformed(by: transform), from: previewExtent, format: .RGBA8, colorSpace: color) else { throw CrystalOrbImageSource.failure }
        return CrystalOrbDifference(alignment: alignment, difference: diff, redMask: redMask,
                                    nativeDifference: difference, nativeRedMask: red)
    }

    private static let context = CIContext(options: [
        .workingColorSpace: CGColorSpace(name: CGColorSpace.sRGB)!,
        .outputColorSpace: CGColorSpace(name: CGColorSpace.sRGB)!, .cacheIntermediates: false])
}
