// Apple-native generation for the native-ROI visibility hypothesis; no UI is opened.
import AppKit
import CoreGraphics
import CoreText
import Foundation
import ImageIO
import UniformTypeIdentifiers

let input = URL(fileURLWithPath: CommandLine.arguments[1])
let output = URL(fileURLWithPath: CommandLine.arguments[2])
try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
let source = CGImageSourceCreateWithURL(input as CFURL, nil)!
let image = CGImageSourceCreateImageAtIndex(source, 0, nil)!
let rect = CGRect(x: (image.width - 448) / 2, y: (image.height - 448) / 2, width: 448, height: 448)
// CGImage crop coordinates are top-left pixel edges; record the equivalent bottom-left rect below.
let clean = image.cropping(to: rect)!
func write(_ image: CGImage, _ name: String) {
    let target = CGImageDestinationCreateWithURL(output.appendingPathComponent(name) as CFURL, UTType.png.identifier as CFString, 1, nil)!
    CGImageDestinationAddImage(target, image, nil)
    precondition(CGImageDestinationFinalize(target))
}
write(clean, "clean.png")
let context = CGContext(data: nil, width: 448, height: 448, bitsPerComponent: 8, bytesPerRow: 448 * 4,
    space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
context.draw(clean, in: CGRect(x: 0, y: 0, width: 448, height: 448))
context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 0.78))
context.fill(CGRect(x: 0, y: 172, width: 448, height: 104))
let text = NSAttributedString(string: "WATERMARK", attributes: [.font: NSFont.boldSystemFont(ofSize: 44), .foregroundColor: NSColor.red])
let line = CTLineCreateWithAttributedString(text)
let width = CTLineGetTypographicBounds(line, nil, nil, nil)
context.textPosition = CGPoint(x: (448-width)/2, y: 207)
CTLineDraw(line, context)
write(context.makeImage()!, "watermarked.png")
let metadata: [String: Any] = ["input": input.lastPathComponent, "originalWidth": image.width, "originalHeight": image.height,
    "referenceRectBottomLeft": ["x": Int(rect.minX), "y": image.height - Int(rect.maxY), "width": 448, "height": 448],
    "cropWidth": 448, "cropHeight": 448, "watermarkBandBottomLeft": ["x": 0, "y": 172, "width": 448, "height": 104],
    "knownCleanSource": "clean.png", "knownContaminatedSource": "watermarked.png", "generation": "Apple CoreGraphics/CoreText; no resizing or generated detail"]
try JSONSerialization.data(withJSONObject: metadata, options: [.prettyPrinted, .sortedKeys]).write(to: output.appendingPathComponent("fixture.json"))
