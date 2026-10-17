// Apple frameworks only. Render at 2x; TIFF point size keeps Finder at 800 × 600.
// Only the color fills 800 × 600; all content fits the 640 × 420 default window.
// Usage: swift tools/make-dmg-background.swift output.tiff
import AppKit
import Foundation

guard CommandLine.arguments.count == 2 else {
    fputs("Usage: make-dmg-background.swift output.tiff\n", stderr)
    exit(2)
}
let width = 800, height = 600
let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: width * 2,
    pixelsHigh: height * 2, bitsPerSample: 8, samplesPerPixel: 4,
    hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
    bytesPerRow: 0, bitsPerPixel: 0)!
rep.size = NSSize(width: width, height: height)
NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
let context = NSGraphicsContext.current!.cgContext
// Top-down coordinates match Finder icon positions in configure-dmg-layout.py.
context.translateBy(x: 0, y: CGFloat(height))
context.scaleBy(x: 1, y: -1)
let canvas = NSRect(x: 0, y: 0, width: width, height: height)
NSGradient(starting: NSColor(srgbRed: 0.97, green: 0.95, blue: 0.99, alpha: 1),
           ending: NSColor(srgbRed: 0.88, green: 0.94, blue: 0.93, alpha: 1))!.draw(in: canvas, angle: 25)
func text(_ value: String, x: CGFloat, y: CGFloat, width: CGFloat,
          size: CGFloat, weight: NSFont.Weight = .regular, color: NSColor) {
    let paragraph = NSMutableParagraphStyle()
    paragraph.alignment = .center
    // NSString uses AppKit's flipped flag for upright text in this context.
    NSGraphicsContext.current!.saveGraphicsState()
    let transform = NSAffineTransform()
    transform.translateX(by: 0, yBy: y * 2 + size * 1.5)
    transform.scaleX(by: 1, yBy: -1)
    transform.concat()
    (value as NSString).draw(in: NSRect(x: x, y: y, width: width, height: size * 1.6),
        withAttributes: [.font: NSFont.systemFont(ofSize: size, weight: weight),
                         .foregroundColor: color, .paragraphStyle: paragraph])
    NSGraphicsContext.current!.restoreGraphicsState()
}
let ink = NSColor(srgbRed: 0.21, green: 0.17, blue: 0.29, alpha: 1)
let muted = NSColor(srgbRed: 0.43, green: 0.38, blue: 0.50, alpha: 1)
let accent = NSColor(srgbRed: 0.47, green: 0.34, blue: 0.62, alpha: 1)
text("W A I F U F I N D E R", x: 20, y: 20, width: 600, size: 12, weight: .bold, color: accent)
text("Your next favorite frame awaits.", x: 20, y: 50, width: 600, size: 26, weight: .semibold, color: ink)
text("1  Drag to Applications", x: 20, y: 108, width: 380, size: 16, weight: .medium, color: muted)
text("2  Open Start Here", x: 440, y: 108, width: 160, size: 16, weight: .medium, color: muted)
for x: CGFloat in [60, 260, 460] {
    NSColor(white: 1, alpha: 0.55).setFill()
    NSBezierPath(roundedRect: NSRect(x: x, y: 143, width: 120, height: 124), xRadius: 25, yRadius: 25).fill()
}
let arrow = NSBezierPath()
arrow.move(to: NSPoint(x: 205, y: 205)); arrow.line(to: NSPoint(x: 235, y: 205))
arrow.move(to: NSPoint(x: 221, y: 191)); arrow.line(to: NSPoint(x: 235, y: 205))
arrow.line(to: NSPoint(x: 221, y: 219))
arrow.lineWidth = 3; arrow.lineCapStyle = .round; arrow.lineJoinStyle = .round
accent.withAlphaComponent(0.6).setStroke(); arrow.stroke()
text("Apple silicon  ·  macOS 15+", x: 20, y: 308, width: 600, size: 12, color: muted)
NSGraphicsContext.restoreGraphicsState()
guard let data = rep.representation(using: .tiff, properties: [.compressionFactor: 1]) else {
    fputs("Could not render DMG background\n", stderr); exit(1)
}
try data.write(to: URL(fileURLWithPath: CommandLine.arguments[1]))
