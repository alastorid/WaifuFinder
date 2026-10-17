import AppKit
import Foundation

/// Runtime tracing for UI and keyboard work.
///
/// **Off unless asked for.** Every call is a no-op returning `false`/empty unless
/// tracing is switched on, so the shipping app pays one branch per call site and
/// writes nothing. This exists because keyboard bugs in a SwiftUI app are routinely
/// invisible from the source: a press is swallowed by whichever handler happens to be
/// live, and the only reliable way to find out is a timestamped record of what saw
/// the press and what the responder chain thought at that moment.
///
/// ## Turning it on
///
/// Three ways, checked in this order — useful in different situations:
///
///   * `WAIFU_TRACE=1` in the environment — for `open -a WaifuFinder --args`-style
///     runs and for scripts that drive the app.
///   * `--waifu-trace` in `CommandLine.arguments` — for a double-clicked app, since
///     the environment of a Finder launch is awkward to set.
///   * `Diag.enable()` called from code, or the **Trace** item in the app's Debug
///     menu — for attaching mid-session to a bug that only sometimes reproduces.
///
/// With no argument, or `WAIFU_TRACE=0`, tracing stays off. A bare `--waifu-trace`
/// traces everything; `--waifu-trace keys,focus` narrows it to those categories, and
/// an unmatched category name is reported at startup rather than silently ignored,
/// because a typo that disables logging is worse than no logging.
///
/// ## Where output goes
///
/// A timestamped file in the system temporary directory, named per process so two
/// runs cannot interleave, plus stderr. The file path is printed to stderr on
/// enablement, so it never has to be guessed:
///
///     WAIFU_TRACE=1 open -a WaifuFinder
///     # → /var/folders/…/waifu-trace-12345.log
///
/// Rotation is by size, not time: a trace left on during a long session would
/// otherwise grow without bound. `Diag.mark("section")` writes a visible separator
/// so a capture can be split into phases after the fact.
///
/// ## Categories
///
/// Filtering by category keeps a focused trace readable — a keyboard investigation
/// usually wants `keys,focus,preview` and nothing else. See `Category`.
enum Diag {
    /// The areas of the app that can be traced independently.
    ///
    /// Raw values so a command-line filter stays readable and stable across renames.
    enum Category: String, CaseIterable, Sendable {
        /// Key presses: what was pressed, who saw it, and who claimed it.
        case keys
        /// Focus moves: `onAppear`/`onDisappear`, `@FocusState` changes, first responder.
        case focus
        /// The frame preview overlay and its lifetime.
        case preview
        /// Uncover mode.
        case uncover
        /// The frame grid.
        case grid
        /// Analysis and grouping progress.
        case analysis
        /// Loading and decoding, including thumbnails and previews.
        case loading
        /// Export.
        case export
    }

    // MARK: - Configuration

    private static let lock = NSLock()
    nonisolated(unsafe) private static var enabled = false
    nonisolated(unsafe) private static var categories: Set<Category> = Set(Category.allCases)
    nonisolated(unsafe) private static var logURL: URL?
    nonisolated(unsafe) private static var pid = 0
    nonisolated(unsafe) private static var bytesWritten = 0
    /// Rotating at a few MB keeps a long session bounded while leaving far more room
    /// than any single investigation needs.
    private static let rotationLimit = 4 * 1024 * 1024

    /// Whether tracing is on. Call sites can test this to skip expensive argument
    /// construction — `Diag.trace(.keys) { "frame \(expensiveThing())" }` only pays
    /// for `expensiveThing()` when the category is live.
    static var isEnabled: Bool {
        lock.lock(); defer { lock.unlock() }
        return enabled
    }

    /// The categories currently being traced, for reporting in the UI.
    static var activeCategories: Set<Category> {
        lock.lock(); defer { lock.unlock() }
        return categories
    }

    /// The file this process is tracing to, or `nil` when tracing is off.
    static var currentLogURL: URL? {
        lock.lock(); defer { lock.unlock() }
        return logURL
    }

    /// Whether lines are echoed to stderr as well as the file.
    ///
    /// Set only by the environment/argument bootstrap. That path means someone is
    /// watching a terminal and wants the trace live; the Debug menu and the tests
    /// write the file alone, so a trace never lands in unrelated tools' stderr.
    nonisolated(unsafe) private static var mirrorToStandardError = false

    // MARK: - Enabling

    /// Reads the environment and arguments. Called once at launch.
    ///
    /// Safe to call more than once: it only ever *adds* tracing, and an explicit
    /// `--waifu-trace` re-reads its filter.
    static func bootstrapFromEnvironment() {
        var requested: String?
        if let env = ProcessInfo.processInfo.environment["WAIFU_TRACE"], env != "0" {
            requested = env
        }
        for argument in CommandLine.arguments where argument.hasPrefix("--waifu-trace") {
            // Both `--waifu-trace` and `--waifu-trace=keys,focus` are accepted.
            requested = argument.contains("=")
                ? String(argument.drop(while: { $0 != "=" }).dropFirst())
                : ""
        }
        guard let requested else { return }
        enable(selecting: requested, mirrorToStandardError: true)
    }

    /// Switches tracing on.
    ///
    /// - Parameters:
    ///   - filter: empty for every category, otherwise a comma-separated list of
    ///     `Category` raw values. An unrecognised name is reported and ignored rather
    ///     than silently disabling everything, so a typo is visible immediately.
    ///   - mirrorToStandardError: also echo to stderr. Only the launch-flag path sets
    ///     it; see `mirrorToStandardError`.
    static func enable(selecting filter: String = "", mirrorToStandardError mirror: Bool = false) {
        lock.lock()
        mirrorToStandardError = mirror
        lock.unlock()
        let trimmed = filter.trimmingCharacters(in: .whitespaces)
        let parsed: Set<Category>
        if trimmed.isEmpty {
            parsed = Set(Category.allCases)
        } else {
            var chosen: Set<Category> = []
            var unknown: [String] = []
            for name in trimmed.split(separator: ",") {
                let key = name.trimmingCharacters(in: .whitespaces)
                if let category = Category(rawValue: key) { chosen.insert(category) }
                else { unknown.append(key) }
            }
            if !unknown.isEmpty {
                let known = Category.allCases.map(\.rawValue).joined(separator: ",")
                let message = "waifu-trace: unknown category(s) \(unknown.joined(separator: ", ")); valid categories are \(known)"
                FileHandle.standardError.write(Data((message + "\n").utf8))
            }
            parsed = chosen
        }

        lock.lock()
        categories = parsed
        if logURL == nil {
            pid = Int(ProcessInfo.processInfo.processIdentifier)
            let url = URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent("waifu-trace-\(pid).log")
            logURL = url
            // Start each run with a clean file; a stale one from a crashed run would
            // otherwise be appended to and misread as this session's output.
            FileManager.default.createFile(atPath: url.path, contents: nil)
            bytesWritten = 0
        }
        enabled = true
        let url = logURL
        lock.unlock()

        guard let url else { return }
        let banner = "=== waifu-trace pid \(pid) categories: \(parsed.map(\.rawValue).sorted().joined(separator: ",")) ==="
        if mirror {
            FileHandle.standardError.write(Data((banner + "\n  -> " + url.path + "\n").utf8))
        }
        write(banner)
    }

    /// Switches tracing off. The file is left in place for reading.
    static func disable() {
        lock.lock()
        enabled = false
        lock.unlock()
    }

    // MARK: - Logging

    /// Records a line under one or more categories.
    ///
    /// Returns `false` when the category is not being traced, so a call site can skip
    /// building a message it would otherwise throw away.
    @discardableResult
    static func trace(_ categories: Category..., message: @autoclosure () -> String) -> Bool {
        log(categories, message())
    }

    /// The `Set` form, for a category computed at runtime.
    @discardableResult
    static func trace(_ categories: Set<Category>, _ message: @autoclosure () -> String) -> Bool {
        lock.lock()
        let live = enabled && !categories.isDisjoint(with: self.categories)
        lock.unlock()
        guard live else { return false }
        let label = categories.map(\.rawValue).sorted().joined(separator: "+")
        write("[\(label)] \(message())")
        return true
    }

    @discardableResult
    private static func log(_ wanted: [Category], _ message: @autoclosure () -> String) -> Bool {
        lock.lock()
        let live = enabled && !wanted.filter { categories.contains($0) }.isEmpty
        lock.unlock()
        guard live else { return false }
        let label = wanted.filter { categories.contains($0) }.map(\.rawValue).sorted().joined(separator: "+")
        write("[\(label)] \(message())")
        return true
    }

    /// Writes a separator so a long capture can be read in phases.
    static func mark(_ title: String) {
        guard isEnabled else { return }
        write("=== \(title) ===")
    }

    private static func write(_ line: String) {
        let stamp = DiagClock.now
        let text = "\(stamp) \(line)\n"

        lock.lock()
        let url = logURL
        if bytesWritten > rotationLimit, let url {
            // Rotate rather than truncate: the earlier half of a long trace is often
            // exactly the part being compared against.
            let previous = url.appendingPathExtension("1")
            try? FileManager.default.removeItem(at: previous)
            try? FileManager.default.moveItem(at: url, to: previous)
            FileManager.default.createFile(atPath: url.path, contents: nil)
            bytesWritten = 0
        }
        bytesWritten += text.utf8.count
        lock.unlock()

        if let url, let handle = try? FileHandle(forWritingTo: url) {
            defer { try? handle.close() }
            handle.seekToEndOfFile()
            handle.write(Data(text.utf8))
        }
        lock.lock()
        let mirror = mirrorToStandardError
        lock.unlock()
        if mirror { FileHandle.standardError.write(Data(text.utf8)) }
    }
}

/// A fixed-width, monotonic-enough timestamp.
///
/// `Date` alone is unreadable in a diff (`1791881723.66`) and its fractional part
/// jitters between calls. This prints `HH:MM:SS.mmm` plus milliseconds since launch,
/// which is what you actually want when comparing two events a few lines apart.
enum DiagClock {
    private static let start = Date()
    private static let formatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss.SSS"
        return formatter
    }()

    static var now: String {
        let elapsed = Int(Date().timeIntervalSince(start) * 1000)
        return "\(formatter.string(from: Date())) +\(String(format: "%6d", elapsed))ms"
    }
}

// MARK: - Keyboard and focus vocabulary

extension Diag {
    /// A readable name for a hardware key code, for trace lines.
    ///
    /// Unrecognised codes print as `key(0xNN)` rather than being omitted: an unknown
    /// code in a keyboard trace is information, not noise.
    static func keyName(_ code: UInt16) -> String {
        switch Int(code) {
        case 36: return "return"
        case 76, 96: return "enter"
        case 53: return "escape"
        case 49: return "space"
        case 51: return "delete"
        case 48: return "tab"
        case 50: return "`"
        case 12: return "q"
        case 14: return "w"
        case 15: return "e"
        case 1: return "s"
        case 2: return "d"
        case 3: return "f"
        case 4: return "h"
        case 5: return "g"
        case 6: return "z"
        case 7: return "x"
        case 8: return "c"
        case 9: return "v"
        case 11: return "b"
        case 13: return "r"
        case 16: return "y"
        case 17: return "t"
        case 18: return "1"
        case 19: return "2"
        case 20: return "3"
        case 21: return "4"
        case 23: return "6"
        case 25: return "8"
        case 26: return "9"
        case 27: return "0"
        case 28: return "]"
        case 30: return "o"
        case 31: return "u"
        case 32: return "["
        case 33: return "i"
        case 34: return "p"
        case 35: return "l"
        case 37: return "-"
        case 38: return "j"
        case 39: return "'"
        case 40: return "k"
        case 41: return ";"
        case 43: return ","
        case 45: return "/"
        case 44: return "="
        case 46: return "m"
        case 47: return "."
        case 24: return "="
        case 123: return "left"
        case 124: return "right"
        case 125: return "down"
        case 126: return "up"
        case 116: return "pageUp"
        case 121: return "pageDown"
        case 114: return "help"
        default: return "key(0x\(String(code, radix: 16)))"
        }
    }

    /// Modifier flags in a form that fits in a trace line.
    static func modifiers(_ flags: NSEvent.ModifierFlags) -> String {
        var parts: [String] = []
        if flags.contains(.command) { parts.append("cmd") }
        if flags.contains(.control) { parts.append("ctrl") }
        if flags.contains(.option) { parts.append("opt") }
        if flags.contains(.shift) { parts.append("shift") }
        if flags.contains(.function) { parts.append("fn") }
        return parts.isEmpty ? "none" : parts.joined(separator: "+")
    }

    /// What the window currently thinks its first responder is.
    ///
    /// The single most useful line when keys go nowhere: a trace that shows a press
    /// reaching a handler, with a first responder that explains why the handler was
    /// skipped, is usually the whole diagnosis.
    ///
    /// Main-actor isolated because `NSApp` is, and every caller is already on it
    /// (view bodies, key handlers); the isolation is stated rather than hopped so the
    /// compiler catches a future background caller instead of it silently reading nil.
    @MainActor
    static func responderDescription() -> String {
        guard let window = NSApp.keyWindow ?? NSApp.mainWindow ?? NSApp.windows.first else {
            return "no window"
        }
        guard let responder = window.firstResponder else {
            return "firstResponder=nil"
        }
        var parts = ["firstResponder=\(type(of: responder))"]
        if let button = responder as? NSButton {
            parts.append("title=\(button.title)")
            parts.append("key=\(window.initialFirstResponder === responder)")
        }
        if let view = responder as? NSView {
            parts.append("in=\(type(of: view.superview ?? view))")
        }
        return parts.joined(separator: " ")
    }

    /// A one-line summary of every window's responder, for when focus has escaped
    /// somewhere unexpected and the key window is not the interesting one.
    @MainActor
    static func windowSummary() -> String {
        NSApp.windows.map { window in
            "\(window.title.isEmpty ? "untitled" : window.title): \(window.isKeyWindow ? "key" : "not-key")/\(window.firstResponder.map { String(describing: type(of: $0)) } ?? "nil")"
        }.joined(separator: " | ")
    }
}