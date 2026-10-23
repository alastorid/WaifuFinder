import CoreImage
import CoreGraphics
import Foundation

struct CrystalOrbDifferenceSettings: Sendable, Equatable {
    enum Channel: String, CaseIterable, Sendable { case rgb = "RGB", luminance = "Luminance", red = "Red", green = "Green", blue = "Blue" }
    enum Display: String, CaseIterable, Sendable { case color = "Color difference", heatmap = "Heatmap", luminance = "Luminance difference", red = "Red overlay" }
    var tolerance: Double = 0 // gamma-encoded sRGB channel units, 0...255
    var gain: Double = 1
    var channel: Channel = .rgb
    var display: Display = .color
}

struct CrystalOrbDifference: Sendable {
    let id = UUID()
    let measurementID: UUID
    let alignment: CrystalOrbAlignment
    let difference: CGImage
    let redMask: CGImage
    let nativeDifference: CIImage
    let nativeRedMask: CIImage
    let target: CIImage
    let candidate: CIImage
    let targetMask: CIImage
    let candidateMask: CIImage
    let raw: CIImage
    let luminance: CIImage
    let overlap: CIImage
    let union: CIImage
    let settings: CrystalOrbDifferenceSettings
}

enum CrystalOrbDifferenceRenderer {
    // Explicit gamma-encoded sRGB, flattened over black. Raw measurements are
    // absolute channel differences in [0,1], quantized to 8-bit for statistics.
    // Luminance here is sRGB luma Y' (Rec.709 weights), not linear-light luminance.
    static let color = CGColorSpace(name: CGColorSpace.sRGB)!
    static let context = CIContext(options: [.workingColorSpace: color, .outputColorSpace: color, .cacheIntermediates: false])

    static func source(_ image: CGImage, alignment: CrystalOrbAlignment, candidate: Bool) -> (CIImage, CIImage) {
        let native = CGRect(x: 0, y: 0, width: image.width, height: image.height)
        let transform = alignment.transform(candidate: candidate, decodedSize: native.size)
        let pixels = CIImage(cgImage: image).composited(over: CIImage(color: .black).cropped(to: native))
            .clampedToExtent().transformed(by: transform)
        let mask = CIImage(color: .white).cropped(to: native).transformed(by: transform)
            .composited(over: CIImage(color: .black).cropped(to: alignment.extent)).cropped(to: alignment.extent)
        let clipped = pixels.cropped(to: native.applying(transform).intersection(alignment.extent))
            .applyingFilter("CIBlendWithMask", parameters: [kCIInputBackgroundImageKey: CIImage(color: .black).cropped(to: alignment.extent), kCIInputMaskImageKey: mask])
        return (clipped.cropped(to: alignment.extent), mask)
    }
    static func render(target: CGImage, candidate: CGImage, alignment declared: CrystalOrbAlignment? = nil,
                       settings: CrystalOrbDifferenceSettings = .init()) throws -> CrystalOrbDifference {
        // A new material must not inherit same-sized zero tiles cached by a
        // previous comparison. This only drops CI intermediates, never sources.
        context.clearCaches()
        let alignment = declared ?? CrystalOrbAlignment(target: CGSize(width: target.width, height: target.height), candidate: CGSize(width: candidate.width, height: candidate.height))
        let a = source(target, alignment: alignment, candidate: false)
        let b = source(candidate, alignment: alignment, candidate: true)
        let raw = a.0.applyingFilter("CIDifferenceBlendMode", parameters: [kCIInputBackgroundImageKey: b.0])
        func luma(_ image: CIImage) -> CIImage { monochrome(image, vector: CIVector(x: 0.2126, y: 0.7152, z: 0.0722, w: 0)) }
        let luminance = luma(a.0).applyingFilter("CIDifferenceBlendMode", parameters: [kCIInputBackgroundImageKey: luma(b.0)])
        let overlap = a.1.applyingFilter("CIMinimumCompositing", parameters: [kCIInputBackgroundImageKey: b.1])
        let union = a.1.applyingFilter("CIMaximumCompositing", parameters: [kCIInputBackgroundImageKey: b.1])
        return try visualize(alignment: alignment, a: a.0, b: b.0, am: a.1, bm: b.1, raw: raw, luminance: luminance, overlap: overlap, union: union, settings: settings)
    }
    static func remap(_ input: CrystalOrbDifference, settings: CrystalOrbDifferenceSettings) throws -> CrystalOrbDifference {
        context.clearCaches()
        return try visualize(alignment: input.alignment, a: input.target, b: input.candidate, am: input.targetMask, bm: input.candidateMask,
                      raw: input.raw, luminance: input.luminance, overlap: input.overlap, union: input.union, settings: settings, measurementID: input.measurementID)
    }
    private static func monochrome(_ image: CIImage, vector: CIVector) -> CIImage {
        image.applyingFilter("CIColorMatrix", parameters: ["inputRVector": vector, "inputGVector": vector, "inputBVector": vector])
    }
    static func magnitude(_ raw: CIImage, luminance: CIImage, channel: CrystalOrbDifferenceSettings.Channel) -> CIImage {
        switch channel {
        case .rgb: raw.applyingFilter("CIMaximumComponent")
        case .luminance: luminance
        case .red: monochrome(raw, vector: CIVector(x: 1, y: 0, z: 0, w: 0))
        case .green: monochrome(raw, vector: CIVector(x: 0, y: 1, z: 0, w: 0))
        case .blue: monochrome(raw, vector: CIVector(x: 0, y: 0, z: 1, w: 0))
        }
    }
    private static func visualize(alignment: CrystalOrbAlignment, a: CIImage, b: CIImage, am: CIImage, bm: CIImage,
                                  raw: CIImage, luminance: CIImage, overlap: CIImage, union: CIImage,
                                  settings: CrystalOrbDifferenceSettings, measurementID: UUID = UUID()) throws -> CrystalOrbDifference {
        let extent = alignment.extent
        let black = CIImage(color: .black).cropped(to: extent)
        let metric = magnitude(raw, luminance: luminance, channel: settings.channel)
        // Threshold only the presentation; preserve raw deltas for samples/statistics.
        let threshold = metric.applyingFilter("CIColorThreshold", parameters: ["inputThreshold": (settings.tolerance + 0.01) / 255])
        let significant = threshold.applyingFilter("CIMultiplyCompositing", parameters: [kCIInputBackgroundImageKey: overlap])
        let gain = Float(settings.gain)
        func amplify(_ image: CIImage) -> CIImage {
            image.applyingFilter("CIColorMatrix", parameters: ["inputRVector": CIVector(x: CGFloat(gain), y: 0, z: 0, w: 0),
                "inputGVector": CIVector(x: 0, y: CGFloat(gain), z: 0, w: 0), "inputBVector": CIVector(x: 0, y: 0, z: CGFloat(gain), w: 0)])
                .applyingFilter("CIColorClamp")
        }
        let mapped: CIImage
        switch settings.display {
        case .color: mapped = amplify(settings.channel == .rgb ? raw : metric)
        case .luminance: mapped = amplify(luminance)
        case .heatmap:
            mapped = amplify(metric).applyingFilter("CIFalseColor", parameters: ["inputColor0": CIColor.black, "inputColor1": CIColor(red: 1, green: 0.78, blue: 0.12)])
        case .red: mapped = CIImage(color: CIColor(red: 1, green: 0, blue: 0)).cropped(to: extent)
        }
        let visible = mapped.applyingFilter("CIBlendWithMask", parameters: [kCIInputBackgroundImageKey: black, kCIInputMaskImageKey: significant])
        let unmatched = union.applyingFilter("CIDifferenceBlendMode", parameters: [kCIInputBackgroundImageKey: overlap])
        let mappedDifference = CIImage(color: CIColor(red: 1, green: 0, blue: 1)).cropped(to: extent)
            .applyingFilter("CIBlendWithMask", parameters: [kCIInputBackgroundImageKey: visible, kCIInputMaskImageKey: unmatched])
        let intensity = amplify(metric).applyingFilter("CIGammaAdjust", parameters: ["inputPower": 0.5])
            .applyingFilter("CIMultiplyCompositing", parameters: [kCIInputBackgroundImageKey: significant])
            .applyingFilter("CIMaximumCompositing", parameters: [kCIInputBackgroundImageKey: unmatched])
        let red = intensity.applyingFilter("CIColorMatrix", parameters: ["inputRVector": CIVector(x: 0, y: 0, z: 0, w: 0),
            "inputGVector": CIVector(x: 0, y: 0, z: 0, w: 0), "inputBVector": CIVector(x: 0, y: 0, z: 0, w: 0),
            "inputAVector": CIVector(x: 1, y: 0, z: 0, w: 0), "inputBiasVector": CIVector(x: 1, y: 0, z: 0, w: 0)])
        let redPresentation = CIImage(color: CIColor(red: 1, green: 0, blue: 1)).cropped(to: extent)
            .applyingFilter("CIBlendWithMask", parameters: [kCIInputBackgroundImageKey: red.composited(over: a), kCIInputMaskImageKey: unmatched])
        let difference = settings.display == .red ? redPresentation : mappedDifference
        return CrystalOrbDifference(measurementID: measurementID, alignment: alignment, difference: try preview(difference, extent: extent), redMask: try preview(red, extent: extent),
            nativeDifference: difference, nativeRedMask: red, target: a, candidate: b, targetMask: am, candidateMask: bm,
            raw: raw, luminance: luminance, overlap: overlap, union: union, settings: settings)
    }
    static func preview(_ image: CIImage, extent: CGRect) throws -> CGImage {
        let scale = min(1, 1600 / max(extent.width, extent.height))
        let rect = CGRect(x: 0, y: 0, width: ceil(extent.width * scale), height: ceil(extent.height * scale))
        guard let result = context.createCGImage(image.transformed(by: CGAffineTransform(scaleX: scale, y: scale)), from: rect, format: .RGBA8, colorSpace: color) else { throw CrystalOrbImageSource.failure }
        return result
    }
}
