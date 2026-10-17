// Read-only native resolution test. Neither opens Finder nor mounts volumes.
import Foundation
import CoreFoundation

guard CommandLine.arguments.count == 3 else { exit(2) }
let data = try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]))
let bookmark: CFData
if data.starts(with: Array("book".utf8)) {
    bookmark = data as CFData
} else {
    guard let converted = CFURLCreateBookmarkDataFromAliasRecord(nil, data as CFData)?.takeRetainedValue() else {
        fputs("Could not convert DMG background alias\n", stderr)
        exit(1)
    }
    bookmark = converted
}
var error: Unmanaged<CFError>?
// Without UI (0x100) and without mounting (0x200).
guard let resolved = CFURLCreateByResolvingBookmarkData(nil, bookmark,
    CFURLBookmarkResolutionOptions(rawValue: 0x300), nil, nil, nil, &error)?.takeRetainedValue() else {
    fputs("Could not resolve DMG background alias\n", stderr)
    exit(1)
}
let actual = (resolved as URL).standardizedFileURL.resolvingSymlinksInPath()
let expected = URL(fileURLWithPath: CommandLine.arguments[2]).standardizedFileURL.resolvingSymlinksInPath()
guard actual == expected else {
    fputs("Background alias resolved to \(actual.path), expected \(expected.path)\n", stderr)
    exit(1)
}
print("Native background alias resolves correctly after remount.")
