import Foundation

/// A timestamp the way the app shows one: `1:02:03` past an hour, `2:03` below it.
///
/// One function rather than the four private copies this replaced (the grid, the
/// preview, the old sidebar each had their own, and they disagreed about whether a
/// zero-padded hour was shown). Anything a user can compare side by side — a grid
/// cell's timecode, the preview's header, a hidden group's label — has to be
/// formatted the same way or the comparison is a reading error waiting to happen.
func timecode(_ seconds: Double) -> String {
    guard seconds.isFinite, seconds >= 0 else { return "0:00" }
    let total = Int(seconds.rounded())
    let hours = total / 3600
    let minutes = (total % 3600) / 60
    let secs = total % 60
    return hours > 0
        ? String(format: "%d:%02d:%02d", hours, minutes, secs)
        : String(format: "%d:%02d", minutes, secs)
}
