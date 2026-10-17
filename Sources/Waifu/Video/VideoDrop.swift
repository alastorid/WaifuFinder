import Foundation
import UniformTypeIdentifiers

/// Which files a drop onto the window is allowed to hand to the session.
///
/// The decision lives here, as a pure function, rather than inside the drop
/// handler, because a handler that mis-classifies a file is indistinguishable from
/// one that never ran: both produce no analysis and no error. Pinning the rule by
/// value is what makes "the drop did nothing" a testable statement.
enum VideoDrop {
    /// The first URL in `urls` that Waifu will analyze, or `nil` when none is a
    /// candidate.
    static func firstAcceptableVideo(in urls: [URL]) -> URL? {
        urls.first(where: accepts)
    }

    /// Accept candidates by location; the import pipeline validates their contents.
    static func accepts(_ url: URL) -> Bool {
        if MediaImport.webURL(url.absoluteString) != nil { return true }
        guard url.isFileURL else { return false }
        var isDirectory: ObjCBool = false
        return FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) && !isDirectory.boolValue
    }
}
