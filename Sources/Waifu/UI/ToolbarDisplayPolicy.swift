import AppKit
import SwiftUI

/// Keep the window toolbar fixed to icons, including when SwiftUI rebuilds it.
struct ToolbarDisplayPolicy: NSViewRepresentable {
    func makeNSView(context: Context) -> PolicyView { PolicyView() }

    func updateNSView(_ view: PolicyView, context: Context) {
        view.applyPolicy()
        // Toolbar reconciliation can finish after the content update.
        DispatchQueue.main.async { [weak view] in view?.applyPolicy() }
    }

    final class PolicyView: NSView {
        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            applyPolicy()
            DispatchQueue.main.async { [weak self] in self?.applyPolicy() }
        }

        override func layout() {
            super.layout()
            applyPolicy()
        }

        func applyPolicy() {
            guard let toolbar = window?.toolbar else { return }
            toolbar.allowsDisplayModeCustomization = false
            toolbar.allowsUserCustomization = false
            if toolbar.displayMode != .iconOnly { toolbar.displayMode = .iconOnly }
        }
    }
}
