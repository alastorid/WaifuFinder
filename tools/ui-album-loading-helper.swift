import AppKit
import ApplicationServices
import Foundation

// Named-album UI verification. Invoked only by ui-test-album-loading.sh
// through run-ui-tests.sh. Reads accessibility and captures the app window;
// clicks only the Photos entry and each explicitly supplied album name.

setvbuf(stdout, nil, _IOLBF, 0)

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
    while index < queue.count && index < 20000 {
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


// Only run through tools/ui-test-album-loading.sh and run-ui-tests.sh.
let albumNames = arguments.enumerated().compactMap { index, value in
    value == "--album" && index + 1 < arguments.count ? arguments[index + 1] : nil
}
guard !albumNames.isEmpty else { fail("at least one --album is required") }
guard AXIsProcessTrusted() else { fail("Accessibility permission is required for this UI check") }
guard let running = NSRunningApplication.runningApplications(withBundleIdentifier: bundleID)
    .first(where: { !$0.isTerminated }) else { fail("app is not running") }
try? String(running.processIdentifier).write(to: output.appendingPathComponent("app-pid.txt"), atomically: true, encoding: .utf8)
let application = AXUIElementCreateApplication(running.processIdentifier)
running.activate(options: [])
var window: AXUIElement?
let launchDeadline = Date().addingTimeInterval(20)
while Date() < launchDeadline && window == nil {
    window = subElements(application, kAXWindowsAttribute).first
    if window == nil { Thread.sleep(forTimeInterval: 0.2) }
}
guard let appWindow = window else { fail("no app window") }
Thread.sleep(forTimeInterval: 1)
func exactLabel(_ element: AXUIElement, _ text: String) -> Bool { labels(element).contains(text) }
if let photos = findAll(appWindow, where: {
    role($0) == kAXButtonRole as String && (exactLabel($0, "Or open Photos") || exactLabel($0, "Photos"))
}).first {
    guard press(photos) else { fail("could not open Photos workspace") }
    Thread.sleep(forTimeInterval: 2)
}
func capture(_ name: String) {
    guard let entries = CGWindowListCopyWindowInfo([.optionOnScreenOnly, .excludeDesktopElements], kCGNullWindowID) as? [[String: Any]],
          let entry = entries.first(where: { ($0[kCGWindowOwnerPID as String] as? pid_t) == running.processIdentifier && ($0[kCGWindowLayer as String] as? Int) == 0 }),
          let id = entry[kCGWindowNumber as String] as? Int else { log("no window available for screenshot"); return }
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/sbin/screencapture")
    process.arguments = ["-x", "-o", "-l\(id)", output.appendingPathComponent(name + ".png").path]
    do { try process.run(); process.waitUntilExit() } catch { log("screenshot unavailable: \(error)") }
}
func clickRow(_ row: AXUIElement) -> Bool {
    _ = AXUIElementSetAttributeValue(row, kAXSelectedAttribute as CFString, kCFBooleanTrue)
    if press(row) { return true }
    guard let originValue = rawAttribute(row, kAXPositionAttribute),
          let sizeValue = rawAttribute(row, kAXSizeAttribute),
          CFGetTypeID(originValue) == AXValueGetTypeID(), CFGetTypeID(sizeValue) == AXValueGetTypeID() else { return false }
    var origin = CGPoint.zero, extent = CGSize.zero
    guard AXValueGetValue(unsafeDowncast(originValue, to: AXValue.self), .cgPoint, &origin),
          AXValueGetValue(unsafeDowncast(sizeValue, to: AXValue.self), .cgSize, &extent), extent.width > 0 else { return false }
    let point = CGPoint(x: origin.x + extent.width * 0.5, y: origin.y + extent.height * 0.5)
    CGEvent(mouseEventSource: nil, mouseType: .leftMouseDown, mouseCursorPosition: point, mouseButton: .left)?.post(tap: .cghidEventTap)
    CGEvent(mouseEventSource: nil, mouseType: .leftMouseUp, mouseCursorPosition: point, mouseButton: .left)?.post(tap: .cghidEventTap)
    return true
}
func isInsideRow(_ element: AXUIElement) -> Bool {
    var current = element
    for _ in 0..<12 {
        guard let raw = rawAttribute(current, kAXParentAttribute), CFGetTypeID(raw) == AXUIElementGetTypeID() else { break }
        current = unsafeDowncast(raw, to: AXUIElement.self)
        if role(current) == kAXRowRole as String { return true }
    }
    return false
}
var failures = 0
for (index, name) in albumNames.enumerated() {
    log("Opening named album: \(name)")
    let sidebarDeadline = Date().addingTimeInterval(20)
    var matchingRow: AXUIElement?
    while Date() < sidebarDeadline && matchingRow == nil {
        matchingRow = findAll(appWindow, where: { role($0) == kAXRowRole as String }).first { row in
            labels(row).contains { $0 == name || $0.hasPrefix(name + " ·") }
                || !findAll(row, where: { exactLabel($0, name) }).isEmpty
        }
        if matchingRow == nil { Thread.sleep(forTimeInterval: 0.5) }
    }
    guard let row = matchingRow, clickRow(row) else {
        dump(appWindow, as: "\(index)-album-not-found")
        log("FAIL \(name): album row not found or could not be selected")
        failures += 1
        continue
    }
    Thread.sleep(forTimeInterval: 1)
    capture("\(index)-initial")
    let deadline = Date().addingTimeInterval(120)
    var ready = 0
    var lastLog = Date.distantPast
    while Date() < deadline {
        let thumbnails = findAll(appWindow, where: { element in
            let identifier: String = attribute(element, kAXIdentifierAttribute) ?? ""
            return identifier.hasPrefix("frame-thumbnail-")
        })
        let selectedTitle = !findAll(appWindow, where: {
            role($0) == kAXStaticTextRole as String && exactLabel($0, name) && !isInsideRow($0)
        }).isEmpty
        ready = selectedTitle ? thumbnails.count : 0
        let texts = findAll(appWindow, where: { role($0) == kAXStaticTextRole as String }).map(stringValue)
        let fraction = texts.first { $0.range(of: "^[0-9,]+/[0-9,]+$", options: .regularExpression) != nil } ?? "unavailable"
        let status = texts.filter { $0.contains("Loading") || $0.contains("Preparing") || $0.contains("match") || $0.contains("ranked") }
        if Date().timeIntervalSince(lastLog) >= 5 {
            log("\(name): ready thumbnail cells=\(ready), shown fraction=\(fraction), status=\(status)")
            lastLog = Date()
        }
        // Wait briefly after the first visible pixels to catch a disappearing batch.
        if ready > 0 {
            Thread.sleep(forTimeInterval: 3)
            let stillReady = findAll(appWindow, where: { element in
                let identifier: String = attribute(element, kAXIdentifierAttribute) ?? ""
                return identifier.hasPrefix("frame-thumbnail-")
            }).count
            if stillReady > 0 { ready = stillReady; break }
        }
        Thread.sleep(forTimeInterval: 1)
    }
    dump(appWindow, as: "\(index)-final")
    capture("\(index)-final")
    if ready > 0 { log("PASS \(name): \(ready) ready thumbnail cells") }
    else { log("FAIL \(name): no ready thumbnail cells after 120 seconds"); failures += 1 }
}
log("Album checks complete; failures=\(failures)")
exit(failures == 0 ? 0 : 1)
