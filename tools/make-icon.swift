// Draws WaifuFinder's application icon from code and writes it as an `.icns`.
//
// The source paths are the icon master; docs/assets holds only a generated preview.
// Every standard macOS rendition is generated together. `build.sh` caches the
// result until this source or the toolchain changes, then copies it into Resources.
// A rendering failure is reported and not fatal, because an app without an icon still runs.
//
// An abstract lens, viewfinder corners and sparkle suggest finding the best frame.
// Geometry is scaled for every standard macOS icon rendition.
//
// Usage: swift tools/make-icon.swift <output.icns>

import AppKit
import Foundation

// MARK: - Drawing

/// Renders the icon at `pixels` square and returns PNG data.
func renderIcon(pixels: Int) -> Data? {
    guard let rep = NSBitmapImageRep(
        bitmapDataPlanes: nil,
        pixelsWide: pixels, pixelsHigh: pixels,
        bitsPerSample: 8, samplesPerPixel: 4,
        hasAlpha: true, isPlanar: false,
        colorSpaceName: .deviceRGB,
        bytesPerRow: 0, bitsPerPixel: 0
    ) else { return nil }

    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
    drawIcon(size: CGFloat(pixels))
    NSGraphicsContext.restoreGraphicsState()

    return rep.representation(using: .png, properties: [:])
}

/// Abstract viewfinder corners, a lens and a best-frame sparkle.
func drawIcon(size: CGFloat) {
    func point(_ x: CGFloat, _ y: CGFloat) -> NSPoint { NSPoint(x: x * size, y: y * size) }
    func rect(_ x: CGFloat, _ y: CGFloat, _ w: CGFloat, _ h: CGFloat) -> NSRect {
        NSRect(x: x * size, y: y * size, width: w * size, height: h * size)
    }
    func color(_ r: CGFloat, _ g: CGFloat, _ b: CGFloat) -> NSColor {
        NSColor(srgbRed: r, green: g, blue: b, alpha: 1)
    }
    let cream = color(1, 0.93, 0.87)
    let mint = color(0.63, 0.91, 0.85)
    let body = NSBezierPath(roundedRect: rect(0.055, 0.055, 0.89, 0.89),
                            xRadius: size * 0.20, yRadius: size * 0.20)
    NSGradient(starting: color(0.21, 0.17, 0.37), ending: color(0.48, 0.34, 0.61))?
        .draw(in: body, angle: 65)

    let handle = NSBezierPath()
    handle.move(to: point(0.61, 0.39)); handle.line(to: point(0.79, 0.21))
    handle.lineWidth = size * 0.09; handle.lineCapStyle = .round
    mint.setStroke(); handle.stroke()

    // A quiet lens halo separates the mark from the tile.
    let lens = NSBezierPath(ovalIn: rect(0.235, 0.345, 0.43, 0.43))
    NSColor(white: 1, alpha: 0.06).setFill(); lens.fill()
    lens.lineWidth = max(1, size * 0.035)
    cream.setStroke(); lens.stroke()

    // Cropped corners are the frame finder, rather than a literal character.
    let corners: [[NSPoint]] = [
        [point(0.17, 0.67), point(0.17, 0.83), point(0.33, 0.83)],
        [point(0.57, 0.83), point(0.73, 0.83), point(0.73, 0.67)],
        [point(0.17, 0.45), point(0.17, 0.29), point(0.33, 0.29)]
    ]
    for points in corners {
        let path = NSBezierPath()
        path.move(to: points[0]); path.line(to: points[1]); path.line(to: points[2])
        path.lineWidth = max(1, size * 0.03); path.lineCapStyle = .round; path.lineJoinStyle = .round
        mint.setStroke(); path.stroke()
    }
    let sparkle = NSBezierPath()
    for vertex in 0..<8 {
        let radius = size * (vertex.isMultiple(of: 2) ? 0.125 : 0.032)
        let angle = CGFloat.pi / 2 + CGFloat(vertex) * CGFloat.pi / 4
        let p = NSPoint(x: size * 0.45 + cos(angle) * radius,
                        y: size * 0.56 + sin(angle) * radius)
        if vertex == 0 { sparkle.move(to: p) } else { sparkle.line(to: p) }
    }
    sparkle.close(); cream.setFill(); sparkle.fill()
}

// MARK: - Iconset assembly

/// The (name, pixels) pairs `iconutil` expects in an `.iconset` directory.
let iconsetEntries: [(name: String, pixels: Int)] = [
    ("icon_16x16.png", 16),
    ("icon_16x16@2x.png", 32),
    ("icon_32x32.png", 32),
    ("icon_32x32@2x.png", 64),
    ("icon_128x128.png", 128),
    ("icon_128x128@2x.png", 256),
    ("icon_256x256.png", 256),
    ("icon_256x256@2x.png", 512),
    ("icon_512x512.png", 512),
    ("icon_512x512@2x.png", 1024),
]

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("make-icon: \(message)\n".utf8))
    exit(1)
}

let arguments = CommandLine.arguments
guard arguments.count >= 2 else { fail("usage: make-icon.swift <output.icns>") }
let outputURL = URL(fileURLWithPath: arguments[1])

let fileManager = FileManager.default
let iconsetURL = fileManager.temporaryDirectory
    .appendingPathComponent("WaifuFinderIcon-\(UUID().uuidString).iconset", isDirectory: true)
try? fileManager.removeItem(at: iconsetURL)
do {
    try fileManager.createDirectory(at: iconsetURL, withIntermediateDirectories: true)
} catch {
    fail("could not create the iconset directory: \(error.localizedDescription)")
}

for entry in iconsetEntries {
    guard let data = renderIcon(pixels: entry.pixels) else {
        fail("could not render \(entry.name)")
    }
    do {
        try data.write(to: iconsetURL.appendingPathComponent(entry.name))
    } catch {
        fail("could not write \(entry.name): \(error.localizedDescription)")
    }
}

// `iconutil` is the only supported way to build an `.icns`, and it is part of
// macOS. Running it here keeps the tool's output a real icon file rather than a
// PNG some other step has to remember to wrap.
let iconutil = Process()
iconutil.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
iconutil.arguments = ["--convert", "icns", "--output", outputURL.path, iconsetURL.path]
let pipe = Pipe()
iconutil.standardOutput = pipe
iconutil.standardError = pipe
do {
    try iconutil.run()
} catch {
    fail("could not run iconutil: \(error.localizedDescription)")
}
iconutil.waitUntilExit()
try? fileManager.removeItem(at: iconsetURL)

guard iconutil.terminationStatus == 0 else {
    let output = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
    fail("iconutil failed: \(output)")
}

print("make-icon: wrote \(outputURL.path)")
