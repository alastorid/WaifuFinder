import Combine
import Foundation

/// A shared destination for frame, preview and clip exports, independent of a loaded session.
@MainActor
final class DefaultExportFolder: ObservableObject {
    static let shared = DefaultExportFolder()
    private static let bookmarkKey = "defaultExportFolderBookmark"
    private static let pathKey = "defaultExportFolderPath"
    private let defaults: UserDefaults
    private let makeBookmark: (URL) throws -> Data
    @Published private(set) var url: URL?

    init(defaults: UserDefaults = .standard,
         makeBookmark: @escaping (URL) throws -> Data = {
             try $0.bookmarkData(options: [.minimalBookmark],
                                 includingResourceValuesForKeys: nil, relativeTo: nil)
         }) {
        self.defaults = defaults
        self.makeBookmark = makeBookmark
        if let data = defaults.data(forKey: Self.bookmarkKey) {
            var stale = false
            if let resolved = try? URL(resolvingBookmarkData: data, options: [.withoutUI],
                                      relativeTo: nil, bookmarkDataIsStale: &stale) {
                url = resolved
                if stale { try? set(resolved) }
                return
            }
        }
        if let path = defaults.string(forKey: Self.pathKey) {
            url = URL(fileURLWithPath: path, isDirectory: true)
        }
    }

    func set(_ directory: URL) throws {
        guard directory.isFileURL else { throw CocoaError(.fileReadUnsupportedScheme) }
        guard try directory.resourceValues(forKeys: [.isDirectoryKey]).isDirectory == true else {
            throw CocoaError(.fileReadUnknown)
        }
        // This app is not sandboxed: a security-scoped bookmark is unnecessary
        // and can fail for a folder selected in NSOpenPanel. An ordinary bookmark
        // tracks moves when available; its failure must not reject a valid folder.
        if let data = try? makeBookmark(directory) {
            defaults.set(data, forKey: Self.bookmarkKey)
        } else {
            defaults.removeObject(forKey: Self.bookmarkKey)
        }
        defaults.set(directory.path, forKey: Self.pathKey)
        url = directory
    }

    func clear() {
        defaults.removeObject(forKey: Self.bookmarkKey)
        defaults.removeObject(forKey: Self.pathKey)
        url = nil
    }
}
