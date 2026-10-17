import AppKit
import Foundation

/// Who owns the keyboard right now, and whether a given key press is the overlay's
/// to take.
///
/// This exists as a pure value rather than as logic inside `FramePreviewView` for one
/// reason: the rule that decides whether a press is claimed was previously spread
/// across a view's `onKeyPress` handlers and its `NSEvent` monitor, and the bug it
/// caused could not be tested at all — reproducing it needs a window, a focus engine
/// and real key delivery. As a plain function the whole decision is one line per case
/// and `Tests/` can pin it.
///
/// ## The rule
///
/// Ownership is derived from *state*, never from view lifetime. Closing a preview
/// clears `previewFrame` first and SwiftUI tears the overlay down afterwards, so for
/// a moment the overlay is still mounted — still in the responder chain, still
/// running its handlers — while the preview it belongs to no longer exists. A handler
/// that claims a press during that moment swallows the Space that reopens a preview,
/// which is why leaving took two presses.
///
/// So: when there is no preview up, this overlay owns nothing and must let every key
/// through.
enum PreviewKeyOwnership: Equatable, Sendable {
    /// The frame preview overlay is up and owns the navigation keys.
    case overlay
    /// No preview is up. The overlay — mounted or not — owns nothing.
    case none

    /// Whether `previewFrame` being non-nil means the overlay is the rightful owner.
    init(previewFrameIsUp: Bool) {
        self = previewFrameIsUp ? .overlay : .none
    }

    /// Whether a key-down on `key` should be consumed by the preview.
    ///
    /// Arrows and Space/Escape are the overlay's while it is up. Anything else — and
    /// everything at all once it is down — passes through to the grid.
    func claims(_ key: PreviewOwnedKey) -> Bool {
        self == .overlay
    }
}

/// The keys the preview overlay is responsible for while it is up.
enum PreviewOwnedKey: String, CaseIterable, Sendable {
    case left, right, up, down, space, escape

    /// Maps a hardware key code to the key the preview owns, or `nil` if it owns none.
    ///
    /// Returning `nil` for everything else is what lets the monitor decline a key it has
    /// no business touching — a ⌘-chord, a function key, anything typed into a field.
    init?(hardwareKeyCode code: UInt16) {
        switch code {
        case 123: self = .left
        case 124: self = .right
        case 126: self = .up
        case 125: self = .down
        case 49: self = .space
        case 53: self = .escape
        default: return nil
        }
    }
}

/// The whole decision, as one testable expression.
///
/// This is what the preview's event monitor calls, so the tested behaviour and the
/// shipped behaviour cannot drift apart.
enum PreviewKeyRouter {
    /// What the monitor should do with a key-down.
    enum Outcome: Equatable, Sendable {
        /// Act on it, and consume it so nothing downstream also acts.
        case consume(PreviewOwnedKey)
        /// Leave it alone and let it continue to the responder chain.
        case passThrough
    }

    /// Decides whether a key-down is the preview's to take.
    ///
    /// - Parameters:
    ///   - code: the hardware key code.
    ///   - previewFrameIsUp: whether a preview is currently open. This is the only
    ///     input that decides ownership, which is the entire point.
    ///   - hasTextFocus: whether the window's first responder is a text view. A space
    ///     typed into a field belongs to the field.
    ///   - modifiers: the event's modifier flags.
    static func outcome(forCode code: UInt16,
                        previewFrameIsUp: Bool,
                        hasTextFocus: Bool = false,
                        modifiers: NSEvent.ModifierFlags = []) -> Outcome {
        // Ownership first: with no preview up the overlay owns nothing, whatever the key
        // was. This is the line that stops a lingering overlay eating the press that
        // reopens a preview.
        guard PreviewKeyOwnership(previewFrameIsUp: previewFrameIsUp) == .overlay else {
            return .passThrough
        }
        guard let key = PreviewOwnedKey(hardwareKeyCode: code) else { return .passThrough }
        // A space typed into a text field belongs to the field, and ⌘Space/⌃Space/⌥Space
        // belong to the system — an input source switch, say — not to this overlay.
        if hasTextFocus { return .passThrough }
        guard modifiers.intersection([.command, .control, .option]).isEmpty else { return .passThrough }
        return .consume(key)
    }
}

/// Tracks the whole Space/Escape gesture across preview insertion and removal.
/// A claimed down owns its release even when that down has just closed the preview.
struct PreviewToggleKeyRouter {
    enum Outcome: Equatable {
        case passThrough, consume, toggle, close
    }

    private var pressed: Set<UInt16> = []

    mutating func route(code: UInt16, isUp: Bool = false, isRepeat: Bool = false,
                        previewIsUp: Bool, canOpen: Bool = true, blocked: Bool = false,
                        hasTextFocus: Bool = false,
                        modifiers: NSEvent.ModifierFlags = []) -> Outcome {
        guard code == 49 || code == 53 else { return .passThrough }
        if isUp {
            return pressed.remove(code) != nil ? .consume : .passThrough
        }
        // Never toggle again while a claimed key is held, even if focus/state changes.
        if isRepeat { return pressed.contains(code) ? .consume : .passThrough }
        // A fresh down also recovers from a release lost when the window deactivated.
        pressed.remove(code)
        guard !blocked, !hasTextFocus,
              modifiers.intersection([.command, .control, .option]).isEmpty else { return .passThrough }
        guard previewIsUp || (code == 49 && canOpen) else { return .passThrough }
        pressed.insert(code)
        return code == 49 ? .toggle : .close
    }
}
