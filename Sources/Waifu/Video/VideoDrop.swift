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
            // A load that errors or yields nothing is often transient (see
            // DropRetry), so it is re-asked before the provider is given up on.
            let url: URL? = await DropRetry.value {
                await withCheckedContinuation { continuation in
                    provider.loadItem(forTypeIdentifier: type, options: nil) { item, _ in
                        let url: URL?
                        if let value = item as? URL { url = value }
                        else if let value = item as? String { url = URL(string: value) }
                        else if let value = item as? Data { url = URL(dataRepresentation: value, relativeTo: nil) }
                        else { url = nil }
                        continuation.resume(returning: url)
                    }
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

/// Re-asks a drag provider for its representation before the drop is declared
/// unreadable. Representation loads cross a process boundary: the source app
/// can be busy or a promised file not yet materialised, and asking again
/// moments later often succeeds where the first attempt errored, returned
/// nothing, or delivered bytes ImageIO rejects — roughly half of these drops
/// import fine when dragged again, so the import performs that second drag
/// itself instead of sending the error to the user. The schedule bounds the
/// added wait to under two seconds before a drop finally fails.
@MainActor
enum DropRetry {
    static let delays: [TimeInterval] = [0.2, 0.5, 1]

    /// Runs `load` until it returns non-nil or the attempts run out. Nil means
    /// every attempt failed, or the surrounding task was cancelled while
    /// waiting between attempts.
    static func value<T>(from load: @MainActor () async -> T?, delays: [TimeInterval] = delays) async -> T? {
        for attempt in 0...delays.count {
            if attempt > 0 {
                do { try await Task.sleep(for: .seconds(delays[attempt - 1])) }
                catch { return nil }
            }
            if Task.isCancelled { return nil }
            if let value = await load() { return value }
        }
        return nil
    }
}
