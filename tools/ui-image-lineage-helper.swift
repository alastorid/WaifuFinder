import AppKit
import ApplicationServices

@main struct LineageUICheck {
    @MainActor static func main() async {
        do { try await run() } catch {
            FileHandle.standardError.write(Data((error.localizedDescription+"\n").utf8));exit(1)
        }
    }
    static func failure(_ text:String)->NSError { NSError(domain:"LineageUICheck",code:1,userInfo:[NSLocalizedDescriptionKey:text]) }
    static func attr(_ element:AXUIElement,_ name:String)->CFTypeRef? {
        var value:CFTypeRef?
        guard AXUIElementCopyAttributeValue(element,name as CFString,&value) == .success else { return nil };return value
    }
    static func labels(_ element:AXUIElement)->[String] {
        [kAXTitleAttribute,kAXValueAttribute,kAXDescriptionAttribute,kAXHelpAttribute].compactMap { attr(element,$0) as? String }
    }
    static func children(_ element:AXUIElement)->[AXUIElement] { attr(element,kAXChildrenAttribute) as? [AXUIElement] ?? [] }
    static func all(_ element:AXUIElement)->[AXUIElement] {
        var nodes=[element],index=0
        while index<nodes.count && index<20000 { nodes.append(contentsOf:children(nodes[index]));index+=1 };return nodes
    }
    static func find(_ root:AXUIElement,_ text:String)->AXUIElement? { all(root).first { labels($0).contains(text) } }
    @MainActor static func wait<T>(_ text:String,seconds:Double=40,_ closure:()->T?) async throws -> T {
        let deadline=Date().addingTimeInterval(seconds)
        while Date()<deadline {
            if let value=closure() { return value };try await Task.sleep(for:.milliseconds(250))
        };throw failure("Timed out: "+text)
    }
    @MainActor static func press(_ root:AXUIElement,_ text:String) async throws {
        let element=try await wait(text) { find(root,text) }
        guard AXUIElementPerformAction(element,kAXPressAction as CFString) == .success else { throw failure("Cannot press "+text) }
    }
    static func key(_ code:CGKeyCode,flags:CGEventFlags=[]) {
        for down in [true,false] { let event=CGEvent(keyboardEventSource:nil,virtualKey:code,keyDown:down);event?.flags=flags;event?.post(tap:.cghidEventTap) }
    }
    static func dump(_ window:AXUIElement,_ url:URL) throws {
        try all(window).map { labels($0).joined(separator:" | ") }.joined(separator:"\n").write(to:url,atomically:true,encoding:.utf8)
    }
    @MainActor static func run() async throws {
        guard AXIsProcessTrusted() else { throw failure("Accessibility permission is required") }
        let appURL=URL(fileURLWithPath:CommandLine.arguments[1]),out=URL(fileURLWithPath:CommandLine.arguments[2])
        let previous=NSWorkspace.shared.frontmostApplication
        let configuration=NSWorkspace.OpenConfiguration();configuration.createsNewApplicationInstance=true
        let running=try await NSWorkspace.shared.openApplication(at:appURL,configuration:configuration)
        defer { running.terminate();previous?.activate() }
        running.activate()
        let app=AXUIElementCreateApplication(running.processIdentifier)
        let window=try await wait("new test window") { (attr(app,kAXWindowsAttribute) as? [AXUIElement])?.first }
        try await press(window,"Or open Photos")
        let row=try await wait("small nonempty album") {
            all(window).filter { attr($0,kAXRoleAttribute) as? String == kAXRowRole }.compactMap { row -> (AXUIElement,Int)? in
                let values=all(row).flatMap(labels)
                guard let count=values.compactMap(Int.init).filter({$0>0 && $0<=50}).min() else { return nil }
                return (row,count)
            }.min { $0.1<$1.1 }?.0
        }
        if AXUIElementSetAttributeValue(row,kAXSelectedAttribute as CFString,kCFBooleanTrue) != .success {
            guard AXUIElementPerformAction(row,kAXPressAction as CFString) == .success else { throw failure("Cannot select the small album") }
        }
        try await press(window,"Image Lineage")
        _=try await wait("Forensic Story") { find(window,"Forensic Story") }
        try dump(window,out.appendingPathComponent("lineage.txt"))
        if let windows=CGWindowListCopyWindowInfo([.optionOnScreenOnly],kCGNullWindowID) as? [[String:Any]],
           let info=windows.first(where:{($0[kCGWindowOwnerPID as String] as? pid_t)==running.processIdentifier && ($0[kCGWindowLayer as String] as? Int)==0}),
           let number=info[kCGWindowNumber as String] as? Int {
            let capture=Process();capture.executableURL=URL(fileURLWithPath:"/usr/sbin/screencapture")
            capture.arguments=["-x","-l",String(number),out.appendingPathComponent("lineage.png").path]
            try capture.run();capture.waitUntilExit()
        }
        // Keyboard routing must address the selected lineage photo, never a hidden
        // ranking selection. Cancel the exact confirmation; no PhotoKit mutation.
        _=try await wait("selected photo") { all(window).first { labels($0).contains(where:{$0.contains("Delete selected photo from Photos library")}) } }
        key(51,flags:.maskCommand)
        _=try await wait("library deletion confirmation") { find(window,"Delete from your Photos library?") }
        try dump(window,out.appendingPathComponent("deletion-confirmation.txt"))
        try await press(window,"Cancel")
        try await press(window,"Back")
        _=try await wait("ranking restored") { find(window,"Forensic Story")==nil ? true : nil }
        guard find(window,"Image Lineage") != nil else { throw failure("Photos toolbar did not survive mode exit") }
        try dump(window,out.appendingPathComponent("ranking-restored.txt"))
        print("PASS: Photos-only entry, album selection, graph/story, keyboard deletion confirmation cancelled, ranking restored; no Photos mutations")
    }
}
