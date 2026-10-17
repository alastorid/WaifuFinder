import AppKit
import SwiftUI

extension Diag {
    /// A traced `NSEvent` local-monitor handler.
    ///
    /// Every key monitor in the app installs one of these rather than a raw closure,
    /// so that *every* monitor reports the same things: the press it saw, what it
    /// claimed, and what it let through. That uniformity is what makes a trace
    /// readable — a press that vanishes is visible as a line that stops.
    ///
    /// The type matches what `NSEvent.addLocalMonitorForEvents` wants back:
    /// `nil` to consume the event, the event to let it through.
    typealias KeyHandler = @MainActor (NSEvent) -> NSEvent?

    /// Wraps a key handler so tracing is a property of installing it rather than
    /// something each call site remembers to add.
    ///
    /// The returned closure is what goes to `NSEvent.addLocalMonitorForEvents`; it
    /// converts the handler's "consumed" answer into the `NSEvent?` that API wants
    /// (`nil` consumes, non-`nil` passes through) and logs both halves.
    ///
    /// - Parameters:
    ///   - name: identifies the monitor in the trace, so two monitors bound to the
    ///     same key can be told apart.
    ///   - categories: defaults to `keys`; pass `focus` too where liveness matters.
    ///   - handler: the real logic, returning whether it consumed the event.
    static func tracedKeyHandler(_ name: String,
                                 categories: Set<Category> = [.keys],
                                 _ handler: @escaping @MainActor (NSEvent) -> Bool) -> KeyHandler {
        { event in
            // `addLocalMonitorForEvents` delivers on the main thread, and the closure
            // is `@MainActor`, so the non-`Sendable` `NSEvent` stays on one actor
            // instead of being carried across — the same reasoning the app's own
            // monitors follow when they take `keyCode` out before dispatching.
            let code = event.keyCode
            let type = event.type == .keyUp ? "up" : (event.isARepeat ? "repeat" : "down")
            let mods = modifiers(event.modifierFlags)
            let tag = "\(name) \(keyName(code)) \(type) mods=\(mods)"

            let consumed = handler(event)
            trace(categories, "\(tag) -> \(consumed ? "consumed" : "passed through") | \(responderDescription())")
            return consumed ? nil : event
        }
    }

    /// Logs a press arriving at an `onKeyPress` handler and the result it returned.
    ///
    /// `onKeyPress` has no interception point of its own, so the *call site* has to
    /// say what it decided. Passing the result back through here keeps "saw it" and
    /// "claimed it" on one line, which is the pairing that matters when hunting a
    /// swallowed press.
    @discardableResult
    @MainActor
    static func keyResult(_ name: String,
                          _ key: KeyPress,
                          categories: Set<Category> = [.keys],
                          _ result: KeyPress.Result) -> KeyPress.Result {
        let phase = key.phase == .up ? "up" : (key.phase == .repeat ? "repeat" : "down")
        let verdict = result == .handled ? "handled" : "ignored"
        let name = keyName(UInt16(key.characters.unicodeScalars.first?.value ?? 0))
        trace(categories, "\(name) \(phase) -> \(verdict) | \(responderDescription())")
        return result
    }

    /// A `Bool` version of `keyResult` for handlers that report through a closure.
    @MainActor
    static func logClaim(_ name: String, _ key: String, consumed: Bool) {
        trace([.keys], "\(name) -> \(consumed ? "consumed" : "passed through") | \(responderDescription())")
    }
}