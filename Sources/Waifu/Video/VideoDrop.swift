import Foundation
import UniformTypeIdentifiers

/// Which files a drop onto the window is allowed to hand to the session.
///
/// The decision lives here, as a pure function, rather than inside the drop
/// handler, because a handler that mis-classifies a file is indistinguishable from
/// one that never ran: both produce no analysis and no error. Pinning the rule by
/// value is what makes "the drop did nothing" a testable statement.
enum VideoDrop {
    /// Resolve every provider before starting one analysis. Sequential resolution
    /// preserves the drop order even when providers take different times to load.
    @MainActor
    static func urls(from providers: [NSItemProvider]) async -> [URL] {
        var urls: [URL] = []
        for provider in providers {
            let type = provider.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier)
                ? UTType.fileURL.identifier : UTType.url.identifier
            let url: URL? = await withCheckedContinuation { continuation in
                provider.loadItem(forTypeIdentifier: type, options: nil) { item, _ in
                    let url: URL?
                    if let value = item as? URL { url = value }
                    else if let value = item as? String { url = URL(string: value) }
                    else if let value = item as? Data { url = URL(dataRepresentation: value, relativeTo: nil) }
                    else { url = nil }
                    continuation.resume(returning: url)
                }
            }
            if let url, accepts(url) { urls.append(url) }
        }
        return urls
    }

    /// Accept candidates by location; the import pipeline validates their contents.
    static func accepts(_ url: URL) -> Bool {
        if MediaImport.webURL(url.absoluteString) != nil { return true }
        guard url.isFileURL else { return false }
        var isDirectory: ObjCBool = false
        return FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) && !isDirectory.boolValue
    }
}
