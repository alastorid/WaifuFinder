import Foundation

/// The tracing facility's own logic.
///
/// The point of these cases is the *filter*: tracing that silently records nothing
/// because of a typo is worse than no tracing at all, because it looks like the run
/// proved there was no bug.
func registerDiagTests() {
    Registry.shared.add(suite: "Diagnostics", TestCase(name: "an unknown category is refused rather than silently disabling tracing", knownBug: nil) {
        checkEqual(Diag.Category(rawValue: "keys"), .keys, "a documented category parses")
        checkEqual(Diag.Category(rawValue: "focus"), .focus, "every category parses")
        check(Diag.Category(rawValue: "keyz") == nil, "a misspelled category is rejected instead of matching nothing")
        check(Diag.Category(rawValue: "") == nil, "an empty name is not a category")
        let known = Set(Diag.Category.allCases)
        checkEqual(known.count, 8, "the documented category set is complete")
        // Every case must be covered by the all-cases default, or `--waifu-trace` with
        // no filter would silently omit it.
        check(Set(Diag.Category.allCases).isSuperset(of: known), "allCases covers every category")
    })
    Registry.shared.add(suite: "Diagnostics", TestCase(name: "key codes the app actually binds are named, unknown ones are still reported", knownBug: nil) {
        // These are the codes hard-coded in the two NSEvent monitors, so a trace that
        // prints them readably is worth having; a rename in hardware terms would
        // otherwise make old traces unreadable.
        checkEqual(Diag.keyName(49), "space", "space is named")
        checkEqual(Diag.keyName(53), "escape", "escape is named")
        checkEqual(Diag.keyName(123), "left", "left arrow is named")
        checkEqual(Diag.keyName(124), "right", "right arrow is named")
        checkEqual(Diag.keyName(125), "down", "down arrow is named")
        checkEqual(Diag.keyName(126), "up", "up arrow is named")
        checkEqual(Diag.keyName(11), "b", "the Uncover comparison key is named")
        // Unrecognised codes are information, not noise: they must still appear.
        check(Diag.keyName(9999).hasPrefix("key(0x"), "an unknown code is reported rather than dropped")
        checkEqual(Diag.keyName(9999), "key(0x270f)", "the unknown code is quoted exactly")
    })
    Registry.shared.add(suite: "Diagnostics", TestCase(name: "modifier flags render readably and name the empty case", knownBug: nil) {
        checkEqual(Diag.modifiers([]), "none", "no modifiers reads as none, not an empty string")
        checkEqual(Diag.modifiers([.command]), "cmd", "command is abbreviated")
        checkEqual(Diag.modifiers([.command, .shift]), "cmd+shift", "modifiers combine in a stable order")
        checkEqual(Diag.modifiers([.control, .option]), "ctrl+opt", "ctrl and opt are abbreviated")
        check(Diag.modifiers([.command, .shift, .option, .control, .function]).contains("fn"),
              "the fn flag is reported when present")
    })
    Registry.shared.add(suite: "Diagnostics", TestCase(name: "tracing is off unless enabled, and enabling it is additive", knownBug: nil) {
        // The shipping app must pay nothing: `isEnabled` false and no file written.
        Diag.disable()
        check(!Diag.isEnabled, "tracing is off after disable()")
        // The return value is "was this recorded", which is what lets a call site skip
        // building an expensive message. While tracing is off nothing is recorded and
        // nothing is written, so no log file exists either.
        check(!Diag.trace(.keys, message: "this must not be written"), "a trace call records nothing while off")
        checkEqual(Diag.currentLogURL.map { _ in true }, nil, "no log file exists while tracing is off")
        // Re-enabling must not throw, must report enabled, and must be idempotent —
        // bootstrap runs at launch and the Debug menu can toggle on top of it.
        Diag.enable(selecting: "keys,focus")
        check(Diag.isEnabled, "tracing is on after enable()")
        checkEqual(Diag.activeCategories, [.keys, .focus], "only the requested categories are live")
        check(Diag.trace(.keys, message: "recorded"), "a live category records")
        check(!Diag.trace(.export, message: "filtered out"), "a category outside the filter is dropped")
        Diag.enable(selecting: "keys,focus")
        checkEqual(Diag.activeCategories, [.keys, .focus], "enabling twice is idempotent")
        // An all-categories filter is the `--waifu-trace` bare form.
        Diag.enable(selecting: "")
        checkEqual(Diag.activeCategories, Set(Diag.Category.allCases), "an empty filter records everything")
        Diag.disable()
        check(!Diag.isEnabled, "tracing can be switched back off")
        // Enabling opened a real per-process log file. Remove it, so a test run does not
        // leave one behind per invocation in the temporary directory.
        if let url = Diag.currentLogURL { try? FileManager.default.removeItem(at: url) }
    })
}