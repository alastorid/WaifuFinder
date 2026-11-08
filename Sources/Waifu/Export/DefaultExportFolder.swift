import Combine
import Foundation

/// A shared destination for frame, preview and clip exports, independent of a loaded session.
@MainActor
final class DefaultExportFolder: ObservableObject {
    static let shared = DefaultExportFolder()
    private static let bookmarkKey = "defaultExportFolderBookmark"
    private static let pathKey = "defaultExportFolderPath"
    @Published private(set) var url: URL?

    private init() {
        if let data = UserDefaults.standard.data(forKey: Self.bookmarkKey) {
            var stale = false
            if let resolved = try? URL(resolvingBookmarkData: data, options: [.withSecurityScope],
                                      relativeTo: nil, bookmarkDataIsStale: &stale) {
                url = resolved
                if stale { try? set(resolved) }
                return
            }
        }
        if let path = UserDefaults.standard.string(forKey: Self.pathKey) {
            url = URL(fileURLWithPath: path, isDirectory: true)
        }
    }

    func set(_ directory: URL) throws {
        let data = try directory.bookmarkData(options: [.withSecurityScope],
                                            includingResourceValuesForKeys: nil, relativeTo: nil)
        UserDefaults.standard.set(data, forKey: Self.bookmarkKey)
        UserDefaults.standard.set(directory.path, forKey: Self.pathKey)
        url = directory
    }

    func clear() {
        UserDefaults.standard.removeObject(forKey: Self.bookmarkKey)
        UserDefaults.standard.removeObject(forKey: Self.pathKey)
        url = nil
    }
}
