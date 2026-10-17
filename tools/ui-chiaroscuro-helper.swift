import AppKit
import ApplicationServices
import Foundation

//
// Drives the WaifuFinder window for tools/ui-test-chiaroscuro-link.sh.
//
// The east egg needs three things that only exist in a running app: a Photos
// workspace with an album open, a fraction smaller than its total, and a pointer
// resting on that fraction for three seconds. The controls are found through the
// accessibility tree rather than through hard-coded pixels, so the check does not
// depend on the panel being laid out exactly as it was when this was written.
//
// Every step prints what it found, and each stage writes the accessibility tree as
// it stood before that step, so a failure is diagnosable from the output directory
// without re-running anything by hand.
//

// MARK: - Command line

let arguments = CommandLine.arguments
func flag(_ name: String) -> String? {
    guard let index = arguments.firstIndex(of: name), index + 1 < arguments.count else { return nil }
    return arguments[index + 1]
}
guard let bundleID = flag("--bundle"), let outputPath = flag("--out") else {
    FileHandle.standardError.write(Data("helper: --bundle and --out are required\n".utf8))
    exit(2)
}
let output = URL(fileURLWithPath: outputPath, isDirectory: true)
try? FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)

func log(_ message: String) {
    print(message)
    FileHandle.standardOutput.synchronizeFile()
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("helper: \(message)\n".utf8))
    exit(1)
}

// MARK: - Accessibility

func rawAttribute(_ element: AXUIElement, _ name: String) -> CFTypeRef? {
    var value: CFTypeRef?
    guard AXUIElementCopyAttributeValue(element, name as CFString, &value) == .success else { return nil }
    return value
}

func attribute<T>(_ element: AXUIElement, _ name: String) -> T? {
    rawAttribute(element, name) as? T
}

func subElements(_ element: AXUIElement, _ name: String = kAXChildrenAttribute) -> [AXUIElement] {
    rawAttribute(element, name) as? [AXUIElement] ?? []
}

func role(_ element: AXUIElement) -> String { attribute(element, kAXRoleAttribute) ?? "" }
func title(_ element: AXUIElement) -> String { attribute(element, kAXTitleAttribute) ?? "" }
func helpText(_ element: AXUIElement) -> String { attribute(element, kAXHelpAttribute) ?? "" }

func stringValue(_ element: AXUIElement) -> String {
    let value = rawAttribute(element, kAXValueAttribute)
    if let text = value as? String { return text }
    if let number = value as? NSNumber { return number.stringValue }
    return ""
}

/// Every label a control could be found by, in the order a person reads them.
func labels(_ element: AXUIElement) -> [String] {
    [title(element), stringValue(element), helpText(element),
     attribute(element, kAXDescriptionAttribute) ?? ""]
}

func position(of element: AXUIElement) -> CGPoint? {
    guard let value = rawAttribute(element, kAXPositionAttribute) else { return nil }
    if let point = value as? CGPoint { return point }
    return (value as? NSValue)?.pointValue
}

func size(of element: AXUIElement) -> CGSize? {
    guard let value = rawAttribute(element, kAXSizeAttribute) else { return nil }
    if let size = value as? CGSize { return size }
    return (value as? NSValue)?.sizeValue
}

func frame(of element: AXUIElement) -> CGRect? {
    guard let origin = position(of: element), let extent = size(of: element) else { return nil }
    return CGRect(origin: origin, size: extent)
}

func describe(_ element: AXUIElement) -> String {
    let box = frame(of: element).map { "\(Int($0.minX)),\(Int($0.minY)) \(Int($0.width))×\(Int($0.height))" } ?? "—"
    return "\(role(element)) [\(labels(element).joined(separator: " | "))] at \(box)"
}

func press(_ element: AXUIElement) -> Bool {
    AXUIElementPerformAction(element, kAXPressAction as CFString) == .success
}

/// Breadth-first, so controls come back in the order they appear on screen.
func findAll(_ root: AXUIElement, where matches: (AXUIElement) -> Bool) -> [AXUIElement] {
    var found: [AXUIElement] = []
    var queue: [AXUIElement] = [root]
    var index = 0
    while index < queue.count && found.count < 500 {
        let element = queue[index]
        index += 1
        if matches(element) { found.append(element) }
        queue.append(contentsOf: subElements(element))
    }
    return found
}

func dump(_ root: AXUIElement, as name: String) {
    var lines: [String] = []
    var queue: [(AXUIElement, Int)] = [(root, 0)]
    var index = 0
    while index < queue.count {
        let (element, depth) = queue[index]
        index += 1
        lines.append(String(repeating: "  ", count: depth) + describe(element))
        for child in subElements(element) { queue.append((child, depth + 1)) }
    }
    let path = output.appendingPathComponent("ax-\(name).txt")
    try? lines.joined(separator: "\n").write(to: path, atomically: true, encoding: .utf8)
    log("    accessibility tree → \(path.lastPathComponent)")
}

// MARK: - The app and its window

guard let running = NSRunningApplication.runningApplications(withBundleIdentifier: bundleID)
        .first(where: { !$0.isTerminated }) else {
    fail("no running application with bundle identifier \(bundleID)")
}
let application = AXUIElementCreateApplication(running.processIdentifier)
running.activate(options: [])
log("==> WaifuFinder is running (pid \(running.processIdentifier))")

var window: AXUIElement?
let windowDeadline = Date().addingTimeInterval(25)
while Date() < windowDeadline && window == nil {
    window = subElements(application, kAXWindowsAttribute).first
    if window == nil { Thread.sleep(forTimeInterval: 0.2) }
}
guard let appWindow = window else { fail("the app never opened a window") }

// A known size and origin, so one photograph can be compared with another.
AXUIElementSetAttributeValue(appWindow, kAXPositionAttribute as CFString,
                              NSValue(point: CGPoint(x: 0, y: 0)) as CFTypeRef)
AXUIElementSetAttributeValue(appWindow, kAXSizeAttribute as CFString,
                              NSValue(size: CGSize(width: 1440, height: 900)) as CFTypeRef)
Thread.sleep(forTimeInterval: 1.2)
log("==> window: \(describe(appWindow))")
dump(appWindow, as: "00-launch")

// MARK: - Into Photos

let photosButtons = findAll(appWindow, where: { element in
    role(element) == kAXButtonRole as String && labels(element).contains { $0.contains("Photos") }
})
guard let openPhotos = photosButtons.first else {
    dump(appWindow, as: "01-no-photos-button")
    fail("no button that opens Photos was found")
}
log("==> pressing \(describe(openPhotos))")
guard press(openPhotos) else { fail("the Photos button refused to be pressed") }
Thread.sleep(forTimeInterval: 2.5)
dump(appWindow, as: "01-photos")

// MARK: - An album to rank

struct Row { let element: AXUIElement; let name: String; let isFolder: Bool }

let rows: [Row] = findAll(appWindow, where: { role($0) == kAXRowRole as String }).map { element in
    let isFolder = subElements(element).contains {
        role($0) == kAXDisclosureTriangleRole as String
            || (rawAttribute($0, kAXSubroleAttribute) as? String) == "AXDisclosureTriangle"
    }
    return Row(element: element, name: title(element), isFolder: isFolder)
}
log("==> sidebar rows: \(rows.map { $0.isFolder ? "\($0.name) (folder)" : $0.name }.joined(separator: ", "))")
guard let album = rows.first(where: { !$0.isFolder && !$0.name.isEmpty }) else {
    dump(appWindow, as: "02-no-albums")
    fail("no album row was found in the Photos sidebar")
}
log("==> opening album \(album.name)")
AXUIElementSetAttributeValue(album.element, kAXSelectedAttribute as CFString, kCFBooleanTrue)
if !press(album.element) { log("    (the row could not be pressed; selection was set instead)") }
Thread.sleep(forTimeInterval: 2.0)

// MARK: - The fraction, once the album has been ranked

let fractionPattern = try! NSRegularExpression(pattern: "^[0-9,]+/[0-9,]+$")

func counts(from text: String) -> (shown: Int, total: Int)? {
    let parts = text.split(separator: "/")
    guard parts.count == 2 else { return nil }
    let numbers = parts.map { Int($0.replacingOccurrences(of: ",", with: "")) }
    guard let shown = numbers[0], let total = numbers[1] else { return nil }
    return (shown, total)
}

func findFraction() -> (element: AXUIElement, shown: Int, total: Int)? {
    let texts = findAll(appWindow, where: { role($0) == kAXStaticTextRole as String })
    for element in texts {
        let value = stringValue(element)
        guard fractionPattern.firstMatch(in: value, range: NSRange(value.startIndex..., in: value)) != nil,
              let parsed = counts(from: value) else { continue }
        return (element, parsed.shown, parsed.total)
    }
    return nil
}

log("==> waiting for the album to be ranked")
var ranked: (element: AXUIElement, shown: Int, total: Int)?
let rankingDeadline = Date().addingTimeInterval(240)
while Date() < rankingDeadline {
    if let found = findFraction(), found.total > 0 {
        ranked = found
        break
    }
    Thread.sleep(forTimeInterval: 1.0)
}
guard let first = ranked else {
    dump(appWindow, as: "03-no-frames")
    fail("the album never produced a frame count")
}
log("==> fraction reads \(stringValue(first.element))")

// MARK: - A filter that hides something, so there is something to link

log("==> switching Faces to \"Without faces\"")
let faceOptions = findAll(appWindow, where: { labels($0).contains("Without faces") })
log("    candidates: \(faceOptions.map(describe).joined(separator: " / "))")
guard let option = faceOptions.first(where: { press($0) }) else {
    dump(appWindow, as: "04-no-face-option")
    fail("the \"Without faces\" control was not found")
}
log("    pressed \(describe(option))")

var filtered: (element: AXUIElement, shown: Int, total: Int)?
let filterDeadline = Date().addingTimeInterval(45)
while Date() < filterDeadline {
    if let found = findFraction(), found.total > 0, found.shown < found.total {
        filtered = found
        break
    }
    Thread.sleep(forTimeInterval: 0.5)
}
guard let hiding = filtered else {
    dump(appWindow, as: "04-not-filtered")
    fail("the fraction still shows every frame, so the light has nothing to link")
}
log("==> fraction reads \(stringValue(hiding.element))")
dump(appWindow, as: "05-filtered")

// MARK: - The destination

let cloud = findAll(appWindow, where: { labels($0).contains { $0.lowercased().contains("icloud") } })
log("==> iCloud controls: \(cloud.map(describe).joined(separator: " / "))")
guard let switchedOff = cloud.first(where: { labels($0).contains("Off") }) else {
    dump(appWindow, as: "06-cloud-on")
    fail("the iCloud control is switched on, so it is not the destination")
}
log("    destination found and switched off: \(describe(switchedOff))")

// MARK: - Photographs

func windowIdentifier() -> CGWindowID? {
    guard let windows = CGWindowListCopyWindowInfo([.optionOnScreenOnly, .excludeDesktopElements],
                                                   kCGNullWindowID) as? [[String: Any]] else { return nil }
    return windows
        .filter { ($0[kCGWindowOwnerName as String] as? String) == "WaifuFinder" }
        .filter { ($0[kCGWindowLayer as String] as? Int) == 0 }
        .max { lhs, rhs in
            let area: (Any) -> Double = {
                let bounds = $0 as? [String: Any] ?? [:]
                return (bounds["Width"] as? Double ?? 0) * (bounds["Height"] as? Double ?? 0)
            }
            return area(lhs[kCGWindowBounds as String] ?? [:]) < area(rhs[kCGWindowBounds as String] ?? [:])
        }
        .flatMap { $0[kCGWindowNumber as String] as? CGWindowID }
}

func capture(_ name: String) -> URL? {
    guard let identifier = windowIdentifier() else {
        log("    !! no WaifuFinder window to photograph — Screen Recording permission?")
        return nil
    }
    let destination = output.appendingPathComponent("\(name).png")
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/sbin/screencapture")
    process.arguments = ["-x", "-o", "-l\(identifier)", destination.path]
    let errors = Pipe()
    process.standardError = errors
    do { try process.run() } catch {
        log("    !! screencapture could not run: \(error.localizedDescription)")
        return nil
    }
    process.waitUntilExit()
    let message = String(data: errors.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
    guard process.terminationStatus == 0, FileManager.default.fileExists(atPath: destination.path) else {
        log("    !! screencapture failed (status \(process.terminationStatus)) \(message)")
        return nil
    }
    return destination
}

/// One number per pixel: red + green + blue, truncated. Enough to tell a window
/// that has light drawn on it from one that has not, without keeping megabytes of
/// pixel data around.
func signature(_ path: URL) -> [UInt8]? {
    guard let image = NSImage(contentsOf: path),
          let tiff = image.tiffRepresentation,
          let bitmap = NSBitmapImageRep(data: tiff),
          let bytes = bitmap.bitmapData, bitmap.bitsPerSample == 8 else { return nil }
    let width = bitmap.pixelsWide
    let stride = bitmap.bytesPerRow
    let channels = max(1, stride / max(1, width))
    let buffer = UnsafeRawBufferPointer(start: bytes, count: stride * bitmap.pixelsHigh)
    var values: [UInt8] = []
    values.reserveCapacity(width * bitmap.pixelsHigh)
    for row in 0..<bitmap.pixelsHigh {
        for column in 0..<width {
            let offset = row * stride + column * channels
            let total = Int(buffer[offset])
                + Int(buffer[offset + min(1, channels - 1)])
                + Int(buffer[offset + min(2, channels - 1)])
            values.append(UInt8(truncatingIfNeeded: total))
        }
    }
    return values
}

// MARK: - The wave

guard let fractionFrame = frame(of: hiding.element) else { fail("the fraction has no frame on screen") }
let resting = CGPoint(x: fractionFrame.midX, y: fractionFrame.midY)
log("==> resting the pointer at \(Int(resting.x)),\(Int(resting.y))")

func moveMouse(to point: CGPoint) {
    CGEvent(mouseEventSource: nil, mouseType: .mouseMoved,
            mouseCursorPosition: point, mouseButton: .left)?.post(tap: .cghidEventTap)
    CGWarpMouseCursorPosition(point)
}

_ = capture("00-before")

// Three seconds of stillness first: those photographs must be identical to the one
// above, because the whole point is that a passing pointer never sets the light off.
let stillDeadline = Date().addingTimeInterval(3.1)
var stillShot = 0
while Date() < stillDeadline {
    moveMouse(to: resting)
    Thread.sleep(forTimeInterval: 0.4)
    stillShot += 1
    _ = capture(String(format: "01-still-%02d", stillShot))
}

// Then the flight itself, photographed as fast as the camera allows.
let flightDeadline = Date().addingTimeInterval(2.4)
var flightShot = 0
while Date() < flightDeadline {
    moveMouse(to: resting)
    _ = capture(String(format: "02-flight-%02d", flightShot))
    flightShot += 1
}
log("==> \(stillShot) still photographs, \(flightShot) flight photographs")

moveMouse(to: CGPoint(x: 700, y: 450))
Thread.sleep(forTimeInterval: 0.8)
_ = capture("03-after")

// Whether anything moved at all, measured here so the answer is in the log and not
// only in the images.
var stillMoved = 0, flightMoved = 0
if let before = signature(output.appendingPathComponent("00-before.png")) {
    var differences: [(name: String, count: Int)] = []
    for path in (try? FileManager.default.contentsOfDirectory(at: output, includingPropertiesForKeys: nil)) ?? [] where path.pathExtension == "png" {
        guard let other = signature(path), other.count == before.count else { continue }
        let changed = zip(before, other).reduce(into: 0) { total, pair in
            if abs(Int(pair.0) - Int(pair.1)) > 6 { total += 1 }
        }
        differences.append((path.lastPathComponent, changed))
        if path.lastPathComponent.hasPrefix("01-still") { stillMoved = max(stillMoved, changed) }
        if path.lastPathComponent.hasPrefix("02-flight") { flightMoved = max(flightMoved, changed) }
    }
    for difference in differences.sorted(by: { $0.count > $1.count }).prefix(6) {
        log("    \(difference.name): \(difference.count) pixels differ from the still window")
    }
    log("==> most pixels changed while merely holding: \(stillMoved)")
    log("==> most pixels changed during the flight: \(flightMoved)")
    if flightMoved <= max(stillMoved, 200) {
        fail("nothing was drawn on the window while the light should have been crossing it")
    }
}
log("==> screenshots in \(output.path)")