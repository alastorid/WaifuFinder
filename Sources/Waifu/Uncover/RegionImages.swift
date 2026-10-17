import CoreGraphics
import CoreImage
import Foundation
import ImageIO
import UniformTypeIdentifiers

enum RegionImages {
    static func resize(_ image: CGImage, to size: CGSize, context: CIContext) throws -> CGImage {
        let input = CIImage(cgImage: image)
        let sx = size.width / Double(image.width)
        let sy = size.height / Double(image.height)
        let resized = input.applyingFilter("CILanczosScaleTransform", parameters: [
            kCIInputScaleKey: sy, kCIInputAspectRatioKey: sx / sy,
        ])
        guard let result = context.createCGImage(resized, from: CGRect(origin: .zero, size: size)) else {
            throw UncoverError.invalid("Could not resize the region.")
        }
        return result
    }

    static func write(_ image: CGImage, to url: URL) throws {
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw UncoverError.invalid("Could not write a prepared frame.")
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw UncoverError.invalid("Could not finish a prepared frame.") }
    }

    static func read(_ url: URL) throws -> CGImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw UncoverError.backend("Missing or corrupt frame: \(url.lastPathComponent)")
        }
        return image
    }

    static func frameURL(_ directory: URL, _ name: String, _ index: Int) -> URL {
        directory.appendingPathComponent(name).appendingPathComponent(String(format: "%06d.png", index))
    }

    /// An inward feather: the mask is strictly zero outside the selection.
    static func maskWeight(x: Double, y: Double, width: Int, height: Int,
                           feather: Double, strength: Double) -> Double {
        guard x >= 0, y >= 0, x < Double(width), y < Double(height) else { return 0 }
        let edge = min(x + 0.5, y + 0.5, Double(width) - x - 0.5, Double(height) - y - 0.5)
        let radius = min(max(0, feather), Double(min(width, height)) / 2)
        let t = radius == 0 ? 1 : min(1, max(0, edge / radius))
        return min(1, max(0, strength)) * t * t * (3 - 2 * t)
    }

    static func composite(original: CGImage, region: CGImage, rect: CGRect,
                          feather: Double, strength: Double, context: CIContext,
                          outputScale: Int = 1) throws -> CGImage {
        guard outputScale > 0,
              UncoverOutputLimits.accepts(width: original.width, height: original.height, scale: outputScale),
              region.width == Int(rect.width) * outputScale,
              region.height == Int(rect.height) * outputScale else {
            throw UncoverError.invalid("Reconstruction dimensions do not match the selected region.")
        }
        let width = region.width, height = region.height
        let outputWidth = original.width * outputScale, outputHeight = original.height * outputScale
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let value = UInt8((255 * maskWeight(x: Double(x), y: Double(y), width: width,
                                                   height: height, feather: feather * Double(outputScale), strength: strength)).rounded())
                let offset = (y * width + x) * 4
                bytes[offset] = value; bytes[offset + 1] = value; bytes[offset + 2] = value; bytes[offset + 3] = 255
            }
        }
        let mask = CIImage(bitmapData: Data(bytes), bytesPerRow: width * 4,
                           size: CGSize(width: width, height: height), format: .RGBA8,
                           colorSpace: nil)
        let scaledRect = CGRect(x: rect.minX * CGFloat(outputScale), y: rect.minY * CGFloat(outputScale),
                                width: rect.width * CGFloat(outputScale), height: rect.height * CGFloat(outputScale))
        let transform = CGAffineTransform(translationX: scaledRect.minX,
                                          y: Double(outputHeight) - scaledRect.maxY)
        let background = CIImage(cgImage: original).transformed(
            by: CGAffineTransform(scaleX: CGFloat(outputScale), y: CGFloat(outputScale)))
        let result = CIImage(cgImage: region).transformed(by: transform)
            .applyingFilter("CIBlendWithMask", parameters: [
                kCIInputBackgroundImageKey: background,
                kCIInputMaskImageKey: mask.transformed(by: transform),
            ])
        guard let output = context.createCGImage(result, from: CGRect(x: 0, y: 0,
            width: outputWidth, height: outputHeight)) else {
            throw UncoverError.invalid("Could not composite the region.")
        }
        return output
    }
}
