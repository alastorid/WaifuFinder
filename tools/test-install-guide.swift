// Offscreen WebKit layout checks. No visible windows, browser automation, or app launch.
// Usage: xcrun swift tools/test-install-guide.swift docs/DMG-Read-Me.html
import AppKit
import WebKit

@MainActor
final class GuideTest: NSObject, WKNavigationDelegate {
    let web = WKWebView(frame: NSRect(x: 0, y: 0, width: 760, height: 800))
    let cases: [(CGFloat, CGFloat)] = [(320, 1), (480, 1), (760, 1), (1280, 1), (760, 2)]
    var index = 0
    func start(_ path: String) {
        web.navigationDelegate = self
        let url = URL(fileURLWithPath: path)
        web.loadFileURL(url, allowingReadAccessTo: url.deletingLastPathComponent())
    }
    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) { checkNext() }
    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        fputs("Guide failed to load: \(error)\n", stderr); exit(1)
    }
    func checkNext() {
        guard index < cases.count else { print("PASS: command block fits narrow/wide viewports and 200% zoom."); exit(0) }
        let (width, zoom) = cases[index]
        web.frame.size = NSSize(width: width, height: 800)
        web.pageZoom = zoom
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { self.check(width, zoom) }
    }
    func check(_ width: CGFloat, _ zoom: CGFloat) {
        web.evaluateJavaScript("""
          (() => {
            const command = document.getElementById('command');
            return { viewport: innerWidth, page: document.documentElement.scrollWidth,
              commandWidth: command.clientWidth, commandContent: command.scrollWidth,
              steps: document.querySelectorAll('ol > li').length,
              block: command.closest('pre').getBoundingClientRect().toJSON() };
          })()
        """) { value, error in
            guard error == nil, let result = value as? [String: Any],
                  let viewport = result["viewport"] as? Double,
                  let page = result["page"] as? Double,
                  let commandWidth = result["commandWidth"] as? Double,
                  let content = result["commandContent"] as? Double,
                  let block = result["block"] as? [String: Double],
                  page <= viewport + 1, content <= commandWidth + 1,
                  block["left", default: -1] >= 0,
                  block["right", default: .infinity] <= viewport + 1,
                  result["steps"] as? Int == 3 else {
                fputs("Guide overflow at \(width) px / \(zoom)x: \(String(describing: value)); \(String(describing: error))\n", stderr)
                exit(1)
            }
            print("Verified guide at \(Int(width)) px, \(Int(zoom * 100))% zoom.")
            self.index += 1
            self.checkNext()
        }
    }
}
NSApplication.shared.setActivationPolicy(.prohibited)
guard CommandLine.arguments.count == 2 else { exit(2) }
let test = MainActor.assumeIsolated {
    let test = GuideTest()
    test.start(CommandLine.arguments[1])
    return test
}
DispatchQueue.main.asyncAfter(deadline: .now() + 30) { fputs("Guide test timed out\n", stderr); exit(1) }
NSApplication.shared.run()
