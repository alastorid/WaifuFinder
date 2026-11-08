import AVFoundation
import Foundation

struct VideoClipRange: Sendable, Equatable {
    let sourceURL: URL
    let seconds: ClosedRange<Double>
}

enum VideoClipExporter {
    /// Merge touching ±3-second windows within each original video. Collection
    /// boundaries never pull unrelated frames from the next or previous source.
    static func ranges(video: VideoInfo, frames: [AnalyzedFrame]) -> [VideoClipRange] {
        if let collection = VideoCollection.get(video.url) {
            return collection.members.enumerated().flatMap { index, member in
                let times = frames.filter { VideoCollection.sourceIndex($0.id.sampleIndex) == index }
                    .map { $0.timeSeconds - member.offset }
                return FocusedAnalysisRanges.make(times: times, duration: member.video.durationSeconds)
                    .filter { $0.upperBound > $0.lowerBound }
                    .map { VideoClipRange(sourceURL: member.video.url, seconds: $0) }
            }
        }
        return FocusedAnalysisRanges.make(times: frames.map(\.timeSeconds), duration: video.durationSeconds)
            .filter { $0.upperBound > $0.lowerBound }
            .map { VideoClipRange(sourceURL: video.url, seconds: $0) }
    }

    /// AVFoundation re-encodes the requested interval, preserving its audio and
    /// avoiding keyframe-aligned passthrough cuts that can include extra footage.
    static func export(_ clip: VideoClipRange, to directory: URL) async throws -> URL {
        try Task.checkCancellation()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let asset = VideoAssetFactory.asset(for: clip.sourceURL)
        guard let exporter = AVAssetExportSession(asset: asset, presetName: AVAssetExportPresetHighestQuality) else {
            throw ClipExportError.unavailable
        }
        let start = CMTime(seconds: clip.seconds.lowerBound, preferredTimescale: 60_000)
        let end = CMTime(seconds: clip.seconds.upperBound, preferredTimescale: 60_000)
        exporter.timeRange = CMTimeRange(start: start, end: end)
        let stem = clip.sourceURL.deletingPathExtension().lastPathComponent
        let interval = String(format: "%.3f-%.3f", locale: Locale(identifier: "en_US_POSIX"),
                              clip.seconds.lowerBound, clip.seconds.upperBound)
        let base = directory.appendingPathComponent("\(stem)-clip-\(interval).mov")
        let temporary = directory.appendingPathComponent(".waifu-clip-\(UUID().uuidString).mov")
        defer { try? FileManager.default.removeItem(at: temporary) }
        let cancellation = ClipExportCancellation(exporter)
        try await withTaskCancellationHandler {
            try await exporter.export(to: temporary, as: .mov)
        } onCancel: { cancellation.cancel() }
        try Task.checkCancellation()
        // Publish only a completed movie. Moving fails if another export has
        // claimed this name in the meantime, so retry without overwriting it.
        for suffix in 0...10_000 {
            let destination = suffix == 0 ? base : directory.appendingPathComponent("\(stem)-clip-\(interval)-\(suffix).mov")
            if FileManager.default.fileExists(atPath: destination.path) { continue }
            do {
                try FileManager.default.moveItem(at: temporary, to: destination)
                return destination
            } catch {
                if FileManager.default.fileExists(atPath: destination.path) { continue }
                throw error
            }
        }
        throw ClipExportError.namesExhausted
    }
}

private enum ClipExportError: LocalizedError {
    case unavailable, namesExhausted
    var errorDescription: String? {
        switch self {
        case .unavailable: "AVFoundation cannot export this video as a clip."
        case .namesExhausted: "No unused clip file name is available in the export folder."
        }
    }
}

private final class ClipExportCancellation: @unchecked Sendable {
    private let exporter: AVAssetExportSession
    init(_ exporter: AVAssetExportSession) { self.exporter = exporter }
    func cancel() { exporter.cancelExport() }
}
