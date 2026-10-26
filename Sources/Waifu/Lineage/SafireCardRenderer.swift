import AppKit
import CoreImage

struct SafireCardMaterial: Sendable {
    let highlighted: CGImage
    let mask: CGImage
}

/// Bounded-size previews. Threshold bytes are prepared off the main actor;
/// scaling, mask compositing and the bright boundary run through Core Image.
actor SafireCardRenderer {
    static let shared = SafireCardRenderer()
    private let context = CIContext(options: [.cacheIntermediates: false])

    func make(image: CGImage, result: SafireSpatialResult, maximumSide: Int = 560) throws -> SafireCardMaterial {
        try Task.checkCancellation()
        let binary = result.probabilities.map { $0 >= 0.5 ? UInt8(255) : UInt8(0) }
        guard let provider = CGDataProvider(data: Data(binary) as CFData),
              let gray = CGImage(width: result.width, height: result.height, bitsPerComponent: 8,
                  bitsPerPixel: 8, bytesPerRow: result.width, space: CGColorSpaceCreateDeviceGray(),
                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue), provider: provider,
                  decode: nil, shouldInterpolate: false, intent: .defaultIntent) else {
            throw PhotosMedia.failure("Unable to render SAFIRE mask")
        }
        let factor = min(1, CGFloat(maximumSide) / CGFloat(max(image.width, image.height)))
        let width = max(1, Int(CGFloat(image.width) * factor))
        let height = max(1, Int(CGFloat(image.height) * factor))
        let rect = CGRect(x: 0, y: 0, width: width, height: height)
        let photo = CIImage(cgImage: image).transformed(by: CGAffineTransform(
            scaleX: CGFloat(width) / CGFloat(image.width), y: CGFloat(height) / CGFloat(image.height))).cropped(to: rect)
        let mask = CIImage(cgImage: gray).transformed(by: CGAffineTransform(
            scaleX: CGFloat(width) / CGFloat(result.width), y: CGFloat(height) / CGFloat(result.height))).cropped(to: rect)
        let red = CIImage(color: CIColor(red: 1, green: 0.08, blue: 0, alpha: 0.68)).cropped(to: rect).composited(over: photo)
        let highlighted = red.applyingFilter("CIBlendWithMask", parameters: [
            kCIInputBackgroundImageKey: photo, kCIInputMaskImageKey: mask
        ])
        let boundary = mask.applyingFilter("CIMorphologyGradient", parameters: [kCIInputRadiusKey: 1.5])
        let outlined = CIImage(color: CIColor(red: 1, green: 0.95, blue: 0.1)).cropped(to: rect)
            .applyingFilter("CIBlendWithMask", parameters: [kCIInputBackgroundImageKey: highlighted, kCIInputMaskImageKey: boundary])
        guard let overlay = context.createCGImage(outlined, from: rect, format: .RGBA8, colorSpace: CrystalOrbDifferenceRenderer.color),
              let raster = context.createCGImage(mask, from: rect, format: .RGBA8, colorSpace: CrystalOrbDifferenceRenderer.color) else {
            throw PhotosMedia.failure("Unable to display SAFIRE evidence")
        }
        return .init(highlighted: overlay, mask: raster)
    }
}
