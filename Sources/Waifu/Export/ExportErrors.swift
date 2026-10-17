import Foundation

/// Errors that can occur during frame export.
///
/// Each case corresponds to a distinct failure mode that the UI can present
/// differently: a missing source file is a user error, a decode failure is a
/// codec limitation, and a write failure is a disk/permission problem.
enum ExportError: Error, Sendable, Equatable, CustomStringConvertible {
    /// The source video file could not be opened for reading.
    ///
    /// This covers both "file does not exist" and "file exists but AVFoundation
    /// cannot read it" (wrong codec, corrupted container, permissions).
    case sourceUnreadable(URL)

    /// The video track has no readable frames at the requested timestamp.
    ///
    /// This happens when the timestamp is past the track's duration, or when
    /// the track uses a codec that `AVAssetReader` cannot decode at native
    /// resolution (e.g. some ProRes variants on older macOS).
    case seekFailed(frameID: FrameID, timeSeconds: Double, underlying: String)

    /// The decoder produced no image for the requested frame.
    ///
    /// Distinct from `seekFailed` because the seek succeeded — the reader
    /// returned a sample buffer — but `CGImage` creation from that buffer
    /// returned `nil`. This can happen with damaged frames or pixel formats
    /// CoreGraphics does not support.
    case decodeFailed(frameID: FrameID, timeSeconds: Double)

    /// The destination file could not be written.
    ///
    /// Covers permission denied, disk full, invalid path, and any
    /// `CGImageDestination` failure (unsupported format, bad properties).
    case writeFailed(frameID: FrameID, url: URL, underlying: String)

    /// The destination URL exists and is not a directory.
    ///
    /// Export requires a directory because it writes multiple files with
    /// generated names. A file URL here is a programming error in the caller.
    case destinationNotADirectory(URL)

    /// A unique suffix could not be generated after 10,000 attempts.
    ///
    /// Practically impossible unless the directory contains millions of
    /// similarly-named files, but included for completeness.
    case suffixExhausted(baseURL: URL)

    public var description: String {
        switch self {
        case .sourceUnreadable(let url):
            return "Cannot read source video: \(url.path)"
        case .seekFailed(let frameID, let time, let underlying):
            return "Seek to \(time)s for frame \(frameID) failed: \(underlying)"
        case .decodeFailed(let frameID, let time):
            return "Decode failed for frame \(frameID) at \(time)s"
        case .writeFailed(let frameID, let url, let underlying):
            return "Write failed for frame \(frameID) to \(url.path): \(underlying)"
        case .destinationNotADirectory(let url):
            return "Destination is not a directory: \(url.path)"
        case .suffixExhausted(let baseURL):
            return "Could not generate unique filename for \(baseURL.lastPathComponent)"
        }
    }
}
