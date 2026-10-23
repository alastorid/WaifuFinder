import AppKit
import ApplicationServices
import AVFoundation
import ImageIO

@main
struct OrbUICheck {
    @MainActor static func main() async {
        do { try await run() }
        catch {
            FileHandle.standardError.write(Data((error.localizedDescription + "\n").utf8))
            exit(1)
        }
    }

    @MainActor static func run() async throws {
        guard AXIsProcessTrusted() else { throw failure("Accessibility permission is required") }
        let output = URL(fileURLWithPath: CommandLine.arguments[1])
        let video = try VideoFixture.makeMP4(in: output, durationSeconds: 3, width: 640, height: 360, frameRate: 12)
        let generator = AVAssetImageGenerator(asset: AVURLAsset(url: video))
        generator.appliesPreferredTrackTransform = true
        let (image, _) = try await generator.image(at: .zero)
        let query = output.appendingPathComponent("query.png")
        let destination = CGImageDestinationCreateWithURL(query as CFURL, "public.png" as CFString, 1, nil)!
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw failure("Could not encode query fixture") }
        let pasteboard = NSPasteboard.general
        let saved: [[NSPasteboard.PasteboardType: Data]] = (pasteboard.pasteboardItems ?? []).map { item in
            Dictionary(uniqueKeysWithValues: item.types.compactMap { type in item.data(forType: type).map { (type, $0) } })
        }
        defer {
            pasteboard.clearContents()
            pasteboard.writeObjects(saved.map { values in
                let item = NSPasteboardItem(); for (type, data) in values { item.setData(data, forType: type) }; return item
            })
        }
        let running = try await wait("running app") {
            NSRunningApplication.runningApplications(withBundleIdentifier: "com.alastorid.waifufinder").first { !$0.isTerminated }
        }
        running.activate(options: [])
        let app = AXUIElementCreateApplication(running.processIdentifier)
        let window = try await wait("app window") { children(app, kAXWindowsAttribute).first }
        try await press(window, "Browse files…")
        _ = try await wait("file picker") { all(window).first { attribute($0, kAXRoleAttribute) as? String == kAXSheetRole } }
        key(5, flags: [.maskCommand, .maskShift]) // Go to Folder
        try await Task.sleep(for: .milliseconds(400))
        pasteboard.clearContents(); pasteboard.setString(video.path, forType: .string)
        key(9, flags: .maskCommand); key(36)
        try await Task.sleep(for: .milliseconds(400)); key(36)
        _ = try await wait("video toolbar", seconds: 90) { find(window, "WaifuCrystalOrb — Find Visually Similar Images") }
        dump(window, to: output.appendingPathComponent("video-toolbar.txt"))
        pasteboard.clearContents(); pasteboard.setData(try Data(contentsOf: query), forType: .png)
        key(9, flags: .maskCommand)
        _ = try await wait("automatic paste activation") { find(window, "WaifuCrystalOrb") }
        _ = try await wait("ranked search result", seconds: 90) {
            all(window).first { labels($0).contains { $0.hasPrefix("Candidate 1,") } }
        }
        for (code, name) in [(1, "side-by-side"), (13, "swipe"), (17, "toggle"), (2, "difference")] {
            key(CGKeyCode(code))
            try await Task.sleep(for: .milliseconds(500))
            dump(window, to: output.appendingPathComponent(name + ".txt"))
            screenshot(pid: running.processIdentifier, to: output.appendingPathComponent(name + ".png"))
        }
        key(53)
        _ = try await wait("restored grid") {
            all(window).first { labels($0).contains { $0.contains("frame-thumbnail-") } }
        }
        dump(window, to: output.appendingPathComponent("restored-grid.txt"))
        try await press(window, "Back to welcome screen")
        try await press(window, "Or open Photos")
        _ = try await wait("Photos toolbar") { find(window, "WaifuCrystalOrb — Find Visually Similar Images") }
        dump(window, to: output.appendingPathComponent("photos-toolbar.txt"))
        key(9, flags: .maskCommand)
        _ = try await wait("Photos paste activation") { find(window, "WaifuCrystalOrb") }
        dump(window, to: output.appendingPathComponent("photos-orb.txt"))
        key(53)
        print("PASS: video import, paste activation, ranked candidates, four modes, grid restoration, Photos toolbar and paste activation")
    }

    static func failure(_ message: String) -> NSError { NSError(domain: "OrbUICheck", code: 1, userInfo: [NSLocalizedDescriptionKey: message]) }
    @MainActor static func wait<T>(_ name: String, seconds: Double = 20, value: () -> T?) async throws -> T {
        let deadline = Date().addingTimeInterval(seconds)
        while Date() < deadline {
            if let result = value() { return result }
            try await Task.sleep(for: .milliseconds(200))
        }
        throw failure("Timed out waiting for \(name)")
    }
    static func attribute(_ element: AXUIElement, _ name: String) -> CFTypeRef? {
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, name as CFString, &value) == .success else { return nil }
        return value
    }
    static func children(_ element: AXUIElement, _ name: String = kAXChildrenAttribute) -> [AXUIElement] { attribute(element, name) as? [AXUIElement] ?? [] }
    static func labels(_ element: AXUIElement) -> [String] {
        [kAXTitleAttribute, kAXValueAttribute, kAXDescriptionAttribute, kAXHelpAttribute, kAXIdentifierAttribute].compactMap { attribute(element, $0) as? String }
    }
    static func all(_ root: AXUIElement) -> [AXUIElement] {
        var nodes = [root], index = 0
        while index < nodes.count && index < 15000 { nodes.append(contentsOf: children(nodes[index])); index += 1 }
        return nodes
    }
    static func find(_ root: AXUIElement, _ label: String) -> AXUIElement? { all(root).first { labels($0).contains(label) } }
    @MainActor static func press(_ root: AXUIElement, _ label: String) async throws {
        let button = try await wait(label) { find(root, label) }
        guard AXUIElementPerformAction(button, kAXPressAction as CFString) == .success else { throw failure("Could not press \(label)") }
    }
    static func key(_ code: CGKeyCode, flags: CGEventFlags = []) {
        for down in [true, false] {
            let event = CGEvent(keyboardEventSource: nil, virtualKey: code, keyDown: down)
            event?.flags = flags; event?.post(tap: .cghidEventTap)
        }
    }
    static func dump(_ root: AXUIElement, to url: URL) {
        try? all(root).map { labels($0).joined(separator: " | ") }.joined(separator: "\n").write(to: url, atomically: true, encoding: .utf8)
    }
    static func screenshot(pid: pid_t, to url: URL) {
        guard let windows = CGWindowListCopyWindowInfo([.optionOnScreenOnly], kCGNullWindowID) as? [[String: Any]],
              let window = windows.first(where: { ($0[kCGWindowOwnerPID as String] as? pid_t) == pid && ($0[kCGWindowLayer as String] as? Int) == 0 }),
              let number = window[kCGWindowNumber as String] as? Int else { return }
        let process = Process(); process.executableURL = URL(fileURLWithPath: "/usr/sbin/screencapture")
        process.arguments = ["-x", "-o", "-l\(number)", url.path]
        try? process.run(); process.waitUntilExit()
    }
}
