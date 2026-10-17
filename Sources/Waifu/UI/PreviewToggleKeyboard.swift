import AppKit
import SwiftUI

/// Lives with the window, so preview teardown and focus cannot interrupt a press.
struct PreviewToggleKeyboard: NSViewRepresentable {
    let session: WaifuSession

    func makeNSView(context: Context) -> KeyboardView {
        let view = KeyboardView()
        view.session = session
        return view
    }

    func updateNSView(_ view: KeyboardView, context: Context) {
        view.session = session
    }

    static func dismantleNSView(_ view: KeyboardView, coordinator: ()) {
        view.removeMonitor()
    }

    final class KeyboardView: NSView {
        var session: WaifuSession?
        private var monitor: Any?
        private var resignObserver: NSObjectProtocol?
        private var router = PreviewToggleKeyRouter()

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            removeMonitor()
            Diag.trace(.focus, .keys, message: "preview.toggle attachment window=\(window?.windowNumber.description ?? "nil") size=\(frame.size)")
            guard window != nil else { return }
            resignObserver = NotificationCenter.default.addObserver(
                forName: NSWindow.didResignKeyNotification, object: window, queue: .main
            ) { [weak self] _ in
                MainActor.assumeIsolated { self?.router = PreviewToggleKeyRouter() }
            }
            monitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown, .keyUp],
                handler: Diag.tracedKeyHandler("preview.toggle") { [weak self] event in
                    MainActor.assumeIsolated {
                        if event.keyCode == 49 || event.keyCode == 53 {
                            Diag.trace(.keys, .preview, message: "preview.toggle context ownerWindow=\(self?.window?.windowNumber.description ?? "nil") eventWindow=\(event.window?.windowNumber.description ?? "nil") keyWindow=\(self?.window?.isKeyWindow ?? false) previewUp=\(self?.session?.previewFrame != nil)")
                        }
                        guard let self, let window = self.window,
                              event.window === window, window.isKeyWindow,
                              event.keyCode == 49 || event.keyCode == 53,
                              let session = self.session else { return false }
                        let outcome = self.router.route(
                            code: event.keyCode, isUp: event.type == .keyUp,
                            isRepeat: event.isARepeat,
                            previewIsUp: session.previewFrame != nil,
                            canOpen: session.focusedFrameID.flatMap { session.frame(id: $0) } != nil,
                            blocked: session.uncover != nil || session.progress.isRunning || session.taggingProgress.isRunning ||
                                session.showExportSheet || session.errorMessage != nil || window.attachedSheet != nil,
                            hasTextFocus: window.firstResponder is NSTextView,
                            modifiers: event.modifierFlags)
                        switch outcome {
                        case .passThrough: return false
                        case .consume: return true
                        case .toggle:
                            session.togglePreview()
                            return true
                        case .close:
                            session.previewFrame = nil
                            return true
                        }
                    }
                })
        }

        func removeMonitor() {
            if let monitor { NSEvent.removeMonitor(monitor) }
            if let resignObserver { NotificationCenter.default.removeObserver(resignObserver) }
            monitor = nil
            resignObserver = nil
            router = PreviewToggleKeyRouter()
        }
    }
}
