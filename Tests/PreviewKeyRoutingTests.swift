import AppKit
import Combine
import Foundation

/// The key-routing rule that the "Space needed two presses to leave the preview" bug
/// lived in.
///
/// These are the tests that could not be written before the decision was lifted out of
/// the view: reproducing the bug needed a window, a focus engine and real key
/// delivery, none of which the headless harness has.
func registerPreviewKeyRoutingTests() {
    Registry.shared.add(suite: "Preview keys", TestCase(name: "background analysis leaves preview gestures available and keeps running", knownBug: nil) {
        let directory = try Temporary.directory("preview-during-loading")
        let url = try VideoFixture.makeMP4(in: directory, durationSeconds: 2, width: 160, height: 120, frameRate: 6)
        await Task { @MainActor in
            let session = WaifuSession(settingsURL: directory.appendingPathComponent("settings.json"))
            session.loadVideo(at: url)
            defer { session.cancel() }
            check(session.progress.isRunning, "background import has started")
            let frame = FrameFixture.make(id: "f30")
            session.updateAnalyzedFrames([frame], isProgressive: true)
            session.focus(on: frame)
            var router = PreviewToggleKeyRouter()
            checkEqual(router.route(code: 49, previewIsUp: false,
                                    blocked: session.isPreviewInteractionBlocked), .toggle,
                       "Space reaches preview while background work runs")
            session.togglePreview()
            checkEqual(session.previewFrame?.id, frame.id, "available frame opens")
            check(session.progress.isRunning, "preview does not cancel import")
            session.togglePreview()
            checkNil(session.previewFrame, "preview closes while import continues")
            session.showExportSheet = true
            check(session.isPreviewInteractionBlocked, "export still owns keyboard input")
            session.showExportSheet = false
            session.errorMessage = "Test error"
            check(session.isPreviewInteractionBlocked, "error dialog still owns keyboard input")
        }.value
    })
    Registry.shared.add(suite: "Preview keys", TestCase(name: "preview UI effects run after the stored state changes", knownBug: nil) {
        let directory = try Temporary.directory("preview-publication")
        try await Task { @MainActor in
            let session = WaifuSession(settingsURL: directory.appendingPathComponent("settings.json"))
            let frame = FrameFixture.make(id: "f30")
            var received: [FrameID?] = []
            var changing = false
            let subscription = session.previewFrameUpdates.sink { value in
                MainActor.assumeIsolated {
                    check(!changing, "scroll/focus effects cannot run inside the preview setter")
                    checkEqual(session.previewFrame?.id, value?.id, "render effects see committed preview state")
                    received.append(value?.id)
                }
            }
            try await Task.sleep(for: .milliseconds(20))
            received.removeAll()
            changing = true
            session.previewFrame = frame
            changing = false
            try await Task.sleep(for: .milliseconds(20))
            checkEqual(received, [frame.id], "opening publishes once after committing")
            received.removeAll()
            changing = true
            session.previewFrame = nil
            changing = false
            try await Task.sleep(for: .milliseconds(20))
            checkEqual(received, [nil], "closing publishes once after committing")
            withExtendedLifetime(subscription) {}
        }.value
    })
    Registry.shared.add(suite: "Preview keys", TestCase(name: "Space gestures toggle once and own their release across overlay teardown", knownBug: nil) {
        var router = PreviewToggleKeyRouter()
        var open = false
        for _ in 0..<10 {
            checkEqual(router.route(code: 49, previewIsUp: open), .toggle, "every fresh Space toggles")
            open.toggle()
            checkEqual(router.route(code: 49, isRepeat: true, previewIsUp: open), .consume,
                       "holding Space cannot undo the toggle")
            checkEqual(router.route(code: 49, isUp: true, previewIsUp: open), .consume,
                       "release is consumed even after closing and unmounting")
            checkEqual(router.route(code: 49, isUp: true, previewIsUp: open), .passThrough,
                       "an unrelated release is not claimed")
        }
        checkEqual(router.route(code: 53, previewIsUp: true), .close, "Escape closes on the first down")
        checkEqual(router.route(code: 53, isRepeat: true, previewIsUp: false), .consume, "Escape repeat remains owned")
        checkEqual(router.route(code: 53, isUp: true, previewIsUp: false), .consume, "Escape release remains owned")
        checkEqual(router.route(code: 53, previewIsUp: false), .passThrough, "Escape without a preview belongs elsewhere")
        checkEqual(router.route(code: 49, previewIsUp: false), .toggle, "Space reopens immediately after Escape")
    })

    Registry.shared.add(suite: "Preview keys", TestCase(name: "window toggle leaves typing, shortcuts and modal interaction alone", knownBug: nil) {
        var router = PreviewToggleKeyRouter()
        for open in [false, true] {
            checkEqual(router.route(code: 49, previewIsUp: open, blocked: true), .passThrough, "modal UI owns Space")
            checkEqual(router.route(code: 49, previewIsUp: open, hasTextFocus: true), .passThrough, "text owns Space")
            for modifier in [NSEvent.ModifierFlags.command, .control, .option] {
                checkEqual(router.route(code: 49, previewIsUp: open, modifiers: modifier), .passThrough, "shortcuts pass through")
            }
            checkEqual(router.route(code: 49, isUp: true, previewIsUp: open), .passThrough, "unclaimed down leaves release alone")
            checkEqual(router.route(code: 49, isRepeat: true, previewIsUp: open), .passThrough, "unclaimed repeats pass through")
            checkEqual(router.route(code: 123, previewIsUp: open), .passThrough, "navigation stays with its handler")
        }
        checkEqual(router.route(code: 49, previewIsUp: false, canOpen: false), .passThrough, "no frame means no preview gesture")
        checkEqual(router.route(code: 49, previewIsUp: true), .toggle, "closing claims the gesture")
        checkEqual(router.route(code: 49, isUp: true, previewIsUp: false, blocked: true, hasTextFocus: true), .consume,
                   "a claimed release cannot activate another control when focus changes")
        checkEqual(router.route(code: 49, previewIsUp: false), .toggle, "a new physical press can always reopen")
        checkEqual(router.route(code: 49, previewIsUp: true), .toggle, "a missed release cannot strand a fresh down")
    })

    Registry.shared.add(suite: "Preview keys", TestCase(name: "a lingering overlay claims nothing once the preview has closed", knownBug: nil) {
        // The bug, stated as a rule. Closing clears `previewFrame` and SwiftUI removes
        // the overlay afterwards, so for a moment the overlay's monitor is still
        // installed while the preview is gone. If it still claims Space there, the
        // press that reopens a preview is swallowed and the user presses twice.
        for key in PreviewOwnedKey.allCases {
            let code = key.hardwareKeyCode
            checkEqual(PreviewKeyRouter.outcome(forCode: code, previewFrameIsUp: false), .passThrough,
                       "\(key.rawValue) is not claimed once the preview has closed")
        }

        // ...and the mirror image: while it *is* up, the overlay does own them.
        for key in PreviewOwnedKey.allCases {
            checkEqual(PreviewKeyRouter.outcome(forCode: key.hardwareKeyCode, previewFrameIsUp: true),
                       .consume(key), "\(key.rawValue) is the overlay's while it is up")
        }

        // Opening a preview and closing it again must leave the grid able to take the
        // very next Space, which is the whole complaint.
        checkEqual(PreviewKeyRouter.outcome(forCode: PreviewOwnedKey.space.hardwareKeyCode, previewFrameIsUp: true),
                   .consume(.space), "space closes an open preview")
        checkEqual(PreviewKeyRouter.outcome(forCode: PreviewOwnedKey.space.hardwareKeyCode, previewFrameIsUp: false),
                   .passThrough, "the next space reaches the grid instead of being eaten")
    })

    Registry.shared.add(suite: "Preview keys", TestCase(name: "the overlay leaves keys it does not own alone", knownBug: nil) {
        // Unowned keys must pass through even while the preview is up, so the monitor
        // cannot swallow typing, function keys or a ⌘-chord.
        for code: UInt16 in [24, 48, 114, 116, 121, 8, 127, 53 &+ 100] {
            checkEqual(PreviewKeyRouter.outcome(forCode: code, previewFrameIsUp: true), .passThrough,
                       "key code \(code) is not the overlay's to take")
        }

        // ⌘Space and friends belong to the system — an input source switch, say.
        // Unmodified they close the preview; modified they do not.
        checkEqual(PreviewKeyRouter.outcome(forCode: 49, previewFrameIsUp: true), .consume(.space),
                   "a bare space closes the preview")
        for modifier in [NSEvent.ModifierFlags.command, .control, .option] {
            checkEqual(PreviewKeyRouter.outcome(forCode: 49, previewFrameIsUp: true, modifiers: modifier),
                       .passThrough, "a modified space is left to the system")
        }
        // Shift is not one of them: Shift-Space is still just a space.
        checkEqual(PreviewKeyRouter.outcome(forCode: 49, previewFrameIsUp: true, modifiers: .shift),
                   .consume(.space), "shift does not disqualify space")

        // A space typed into a text field belongs to the field.
        checkEqual(PreviewKeyRouter.outcome(forCode: 49, previewFrameIsUp: true, hasTextFocus: true),
                   .passThrough, "typing a space into a field is left alone")
    })

    Registry.shared.add(suite: "Preview keys", TestCase(name: "the closing key-down is not mistaken for the key-up that follows it", knownBug: nil) {
        // Regression, from a trace rather than from reasoning. AppKit activates a focused
        // control on Space *release*, so the monitor has to swallow the key-up to stop a
        // header button firing as the preview closes. Doing that without checking the
        // event's phase also swallowed the key-**down** — the press that does the closing
        // — and returned before the router ran. Space then did nothing at all while the
        // preview was open: the trace was an unbroken run of `space down -> consumed`
        // with no `previewFrame` ever clearing.
        //
        // The router only ever sees key-downs, so it must stay the single decision point;
        // the key-up swallow has to live strictly outside it.
        checkEqual(PreviewKeyRouter.outcome(forCode: 49, previewFrameIsUp: true), .consume(.space),
                   "the closing press reaches the router while the preview is up")
        checkEqual(PreviewKeyRouter.outcome(forCode: 53, previewFrameIsUp: true), .consume(.escape),
                   "escape closing an open preview also reaches the router")

        // Once closed, both must fall through or the next Space is swallowed again.
        checkEqual(PreviewKeyRouter.outcome(forCode: 49, previewFrameIsUp: false), .passThrough,
                   "after closing, space is not claimed by the stale overlay")
        checkEqual(PreviewKeyRouter.outcome(forCode: 53, previewFrameIsUp: false), .passThrough,
                   "after closing, escape is not claimed by the stale overlay either")
    })

    Registry.shared.add(suite: "Preview keys", TestCase(name: "an orphaned overlay stays transparent to keys it no longer owns", knownBug: nil) {
        // Second regression, also from a trace. `.focusable()` on the overlay made it
        // take keyboard focus on appear and *keep* it: SwiftUI unmounts the overlay a
        // beat after `previewFrame` goes nil, and a trace caught presses being passed
        // through the monitor only to die on that still-mounted, still-focused view —
        // which owned no handlers. Space and Escape were both dead for the whole linger.
        //
        // The invariant is that key routing never depends on the overlay being present or
        // focused. That is what `PreviewKeyOwnership` encodes: ownership is a pure
        // function of `previewFrame`, so it cannot differ between "overlay mounted" and
        // "overlay unmounted" — there is no state in which the overlay is mounted,
        // focused and still the right owner.
        for isUp in [true, false] {
            for key in PreviewOwnedKey.allCases {
                // Same input, same answer, regardless of what the view hierarchy is doing.
                checkEqual(PreviewKeyRouter.outcome(forCode: key.hardwareKeyCode, previewFrameIsUp: isUp),
                           isUp ? .consume(key) : .passThrough,
                           "\(key.rawValue) routing depends only on preview state, not on the overlay (preview up: \(isUp))")
            }
        }
        // And the specific sequence that broke: close, then press again while the overlay
        // is still tearing down.
        checkEqual(PreviewKeyRouter.outcome(forCode: 49, previewFrameIsUp: true), .consume(.space),
                   "the closing press is the overlay's")
        checkEqual(PreviewKeyRouter.outcome(forCode: 49, previewFrameIsUp: false), .passThrough,
                   "the next press belongs to the grid, even mid-teardown")
    })

    Registry.shared.add(suite: "Preview keys", TestCase(name: "ownership comes from state, never from the view existing", knownBug: nil) {
        // The invariant in one line: ownership tracks `previewFrame` and nothing else.
        for isUp in [true, false] {
            checkEqual(PreviewKeyOwnership(previewFrameIsUp: isUp),
                       isUp ? .overlay : .none,
                       "ownership mirrors the preview state exactly")
        }
        check(PreviewKeyOwnership(previewFrameIsUp: false).claims(.space) == false,
              "a closed preview claims no key at all")
        check(PreviewKeyOwnership(previewFrameIsUp: true).claims(.space),
              "an open preview claims its navigation keys")

        // Every key the app binds must be nameable, and nothing else may claim to be.
        for key in PreviewOwnedKey.allCases {
            check(PreviewOwnedKey(hardwareKeyCode: key.hardwareKeyCode) == key,
                  "\(key.rawValue) round-trips through its hardware code")
        }
        check(PreviewOwnedKey(hardwareKeyCode: 9999) == nil,
              "an unknown key code is not claimed")
    })
}

private extension PreviewOwnedKey {
    /// The hardware key code this key is delivered as.
    var hardwareKeyCode: UInt16 {
        switch self {
        case .left: return 123
        case .right: return 124
        case .up: return 126
        case .down: return 125
        case .space: return 49
        case .escape: return 53
        }
    }
}
