import AppKit
import SwiftUI

/// Lives with the window, so preview teardown and focus cannot interrupt a press.
struct PreviewToggleKeyboard: NSViewRepresentable {
    let session: WaifuSession
    let isPhotosMode: Bool
    var orb: WaifuCrystalOrbModel? = nil

    func makeNSView(context: Context) -> KeyboardView {
        let view = KeyboardView()
        view.orb = orb
        view.session = session
        view.isPhotosMode = isPhotosMode
        return view
    }

    func updateNSView(_ view: KeyboardView, context: Context) {
        view.orb = orb
        view.session = session
        view.isPhotosMode = isPhotosMode
    }

    static func dismantleNSView(_ view: KeyboardView, coordinator: ()) {
        view.removeMonitor()
    }

    final class KeyboardView: NSView {
        var session: WaifuSession?
        var isPhotosMode = false
        var orb: WaifuCrystalOrbModel?
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
                MainActor.assumeIsolated { self?.router = PreviewToggleKeyRouter(); self?.orb?.holdingToggle = false }
            }
            monitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown, .keyUp],
                handler: Diag.tracedKeyHandler("preview.toggle") { [weak self] event in
                    MainActor.assumeIsolated {
                        guard let self, let window = self.window,
                              event.window === window, window.isKeyWindow,
                              let session = self.session else { return false }

                        let hasTextFocus = window.firstResponder is NSTextView
                            || (window.firstResponder as? NSTextField)?.isEditable == true
                        let modifiers = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
                        if !hasTextFocus, window.attachedSheet == nil, !session.showExportSheet,
                           session.errorMessage == nil {
                            if event.type == .keyDown, modifiers == .command,
                               event.charactersIgnoringModifiers?.lowercased() == "v",
                               let orb = self.orb, session.uncover == nil, session.previewFrame == nil {
                                return orb.paste(session: session)
                            }
                            if let orb = self.orb, orb.isOpen { return orb.handleKey(event) }
                        }
                        if self.orb?.isOpen == true { return false }
                        if event.type == .keyDown, event.keyCode == 51, modifiers == .command,
                           self.isPhotosMode, !hasTextFocus,
                           session.uncover == nil, !session.progress.isRunning,
                           !session.showExportSheet, session.errorMessage == nil,
                           window.attachedSheet == nil {
                            if !event.isARepeat { session.deleteSelectedPhotos() }
                            return true
                        }

                        if event.keyCode == 49 || event.keyCode == 53 {
                            Diag.trace(.keys, .preview, message: "preview.toggle context ownerWindow=\(window.windowNumber) eventWindow=\(event.window?.windowNumber.description ?? "nil") keyWindow=\(window.isKeyWindow) previewUp=\(session.previewFrame != nil)")
                        }
                        let outcome = self.router.route(
                            code: event.keyCode, isUp: event.type == .keyUp,
                            isRepeat: event.isARepeat,
                            previewIsUp: session.previewFrame != nil,
                            canOpen: session.focusedFrameID.flatMap { session.frame(id: $0) } != nil,
                            blocked: session.uncover != nil || session.progress.isRunning || session.taggingProgress.isRunning ||
                                session.showExportSheet || session.errorMessage != nil || window.attachedSheet != nil,
                            hasTextFocus: hasTextFocus,
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
