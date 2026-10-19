import AVFoundation
import CoreMedia
import Foundation

/// Reads video metadata without decoding any frames.
///
/// `probe(_:)` is the first thing the engine does with a dropped file. It must be
/// fast (milliseconds), must not decode, and must reject files that cannot be
/// analysed — a file with no video track, a zero/NaN duration, or an invalid URL.
/// The modern async `load(...)` APIs are used throughout; the deprecated KVO-based
/// `AVAsset` property accessors are never touched.
///
/// ## Why a separate probe step
///
/// The decode-and-score pass needs `VideoInfo` to size its progress bar
/// and its frame-count estimate. Doing that up front means the pass
/// never has to handle "I don't know how long this is" mid-stream. A
/// probe failure is also the right
/// place to surface "this is an audio file" or "this URL is a directory" before
/// the UI shows a progress bar.
///
/// ## Rejection criteria
///
/// - **Invalid URLs** — local URLs must identify an existing file; remote URLs
///   must use HTTP or HTTPS. Directories and unsupported schemes are rejected
///   before AVFoundation opens the resource.
/// - **No video track** — `loadTracks(withMediaType: .video)` returns empty.
///   Rejected with `.noVideoTrack`.
/// - **Zero or NaN duration** — `load(.duration)` yields `CMTime` with
///   `seconds.isNaN || seconds <= 0`. Rejected with `.zeroOrInvalidDuration`.
///   A zero-duration video would divide by zero in the frame-count estimate.
///
/// ## What is not rejected
///
/// - Variable frame rate — `nominalFrameRate` is a hint, not a promise.
///   It is used only to estimate the frame count that sizes the progress
///   bar; the actual timestamps come from the sequential reader pass.
/// - Missing `naturalSize` — if the track reports zero size we treat it as
///   "no video track" because a 0×0 track cannot be decoded.
/// - Codec AVFoundation cannot decode — that failure surfaces later, in the
///   reader, where it belongs. The probe only answers "is there a video track
///   with a plausible duration and dimensions?".
enum VideoProbeError: Error, LocalizedError, Equatable {
    case notAFile
    case noVideoTrack
    case zeroOrInvalidDuration
    case noNaturalSize
    case underlying(Error)

    var errorDescription: String? {
        switch self {
        case .notAFile:
            return "URL is not a media file or HTTP(S) resource"
        case .noVideoTrack:
            return "File contains no video track"
        case .zeroOrInvalidDuration:
            return "Video duration is zero or invalid"
        case .noNaturalSize:
            return "Video track has no natural size"
        case .underlying(let error):
            return "Failed to probe video: \(error.localizedDescription)"
        }
    }

    static func == (lhs: VideoProbeError, rhs: VideoProbeError) -> Bool {
        switch (lhs, rhs) {
        case (.notAFile, .notAFile),
             (.noVideoTrack, .noVideoTrack),
             (.zeroOrInvalidDuration, .zeroOrInvalidDuration),
             (.noNaturalSize, .noNaturalSize):
            return true
        case (.underlying(let l), .underlying(let r)):
            return l.localizedDescription == r.localizedDescription
        default:
            return false
        }
    }
}

/// Probes a video file for the metadata the sampling layer needs.
///
/// - Parameter url: A local media file or an HTTP(S) media resource.
/// - Returns: `VideoInfo` with duration, dimensions, frame rate, and estimated
///   frame count. `estimatedFrameCount` is `round(duration * frameRate)` — it is
///   an estimate because `nb_frames` is not reliably present in MP4 metadata, and
///   the estimate is only ever used to bound work, never to address a frame.
/// - Throws: `VideoProbeError` for the rejection cases above, or a wrapped
///   `AVError` for anything else.
struct VideoProbe {
    static func probe(_ url: URL) async throws -> VideoInfo {
        if let collection = VideoCollection.get(url) { return collection.info }
        if let album = PhotosMedia.shared.album(url) {
            return VideoInfo(url: url, durationSeconds: album.duration,
                width: album.items.map { $0.asset.pixelWidth }.max() ?? 1,
                height: album.items.map { $0.asset.pixelHeight }.max() ?? 1,
                frameRate: 30, estimatedFrameCount: album.items.reduce(0) { $0 + ($1.isStill ? 1 : Int($1.duration * 30)) })
        }
        if let item = PhotosMedia.shared.item(url) {
            if item.isStill {
                return VideoInfo(url: url, durationSeconds: 1, width: item.asset.pixelWidth,
                    height: item.asset.pixelHeight, frameRate: 1, estimatedFrameCount: 1)
            }
            let asset = try await PhotosMedia.shared.videoAsset(url).value
            guard let track = try await asset.loadTracks(withMediaType: .video).first else { throw VideoProbeError.noVideoTrack }
            let rate = Double(try await track.load(.nominalFrameRate))
            let size = try await track.load(.naturalSize)
            let seconds = try await asset.load(.duration).seconds
            return VideoInfo(url: url, durationSeconds: seconds, width: Int(size.width), height: Int(size.height),
                frameRate: rate, estimatedFrameCount: Int(seconds * (rate > 0 ? rate : 30)))
        }
        if url.isFileURL {
            guard FileManager.default.fileExists(atPath: url.path) else { throw VideoProbeError.notAFile }
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), !isDirectory.boolValue else {
                throw VideoProbeError.notAFile
            }
        } else if MediaImport.webURL(url.absoluteString) == nil {
            throw VideoProbeError.notAFile
        }

        let asset = VideoAssetFactory.asset(for: url)

        // Load video tracks using the modern async API.
        let videoTracks = try await asset.loadTracks(withMediaType: .video)
        guard let videoTrack = videoTracks.first else { throw VideoProbeError.noVideoTrack }

        // Load the properties sequentially to avoid Swift 6 data race warnings
        // with non-Sendable AVAsset/AVAssetTrack. The loads are fast enough
        // that parallel execution doesn't meaningfully improve probe latency.
        let loadedDuration = try await asset.load(.duration)
        let loadedSize = try await videoTrack.load(.naturalSize)
        let loadedFrameRate = try await videoTrack.load(.nominalFrameRate)

        let durationSeconds = loadedDuration.seconds
        guard durationSeconds > 0, !durationSeconds.isNaN else { throw VideoProbeError.zeroOrInvalidDuration }

        let width = Int(loadedSize.width)
        let height = Int(loadedSize.height)
        guard width > 0, height > 0 else { throw VideoProbeError.noNaturalSize }

        let frameRate = Double(loadedFrameRate)
        // nominalFrameRate can be 0 for variable-frame-rate content. We still accept
        // the file — the sampling pass reads sequentially and uses actual timestamps.
        // But we must not divide by zero when estimating frame count.
        let effectiveFrameRate = frameRate > 0 ? frameRate : 30.0
        let estimatedFrameCount = Int((durationSeconds * effectiveFrameRate).rounded())

        return VideoInfo(
            url: url,
            durationSeconds: durationSeconds,
            width: width,
            height: height,
            frameRate: frameRate,
            estimatedFrameCount: estimatedFrameCount
        )
    }
}
