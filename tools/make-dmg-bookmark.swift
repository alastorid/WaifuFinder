// Native, headless bookmark creation for modern Finder background references.
// Usage: xcrun swift tools/make-dmg-bookmark.swift BACKGROUND OUTPUT
import Foundation

guard CommandLine.arguments.count == 3 else { exit(2) }
let target = URL(fileURLWithPath: CommandLine.arguments[1])
let bookmark = try target.bookmarkData(options: .minimalBookmark,
    includingResourceValuesForKeys: nil, relativeTo: nil)
try bookmark.write(to: URL(fileURLWithPath: CommandLine.arguments[2]))
