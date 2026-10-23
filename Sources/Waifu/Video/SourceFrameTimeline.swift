import AVFoundation
import Foundation

/// Actual presentation timestamps, independent of nominal FPS and analysis mode.
/// Sample cursors read timing metadata without decoding or retaining image pixels.
struct SourceFrameTimeline: Sendable {
    let timestamps: [Double]

    init(timestamps: [Double]) {
        var previous: Double?
        self.timestamps = timestamps.filter { $0.isFinite && $0 >= 0 }.sorted().filter { time in
            defer { previous = time }
            return previous != time
        }
    }

    func index(at seconds: Double) -> Int {
        guard !timestamps.isEmpty else { return 0 }
        let low = lowerBound(seconds)
        if low == 0 { return 0 }
        if low == timestamps.count { return low - 1 }
        return seconds - timestamps[low - 1] <= timestamps[low] - seconds ? low - 1 : low
    }

    func frameCount(from start: Double, until end: Double) -> Int {
        guard start.isFinite, end.isFinite, end > start else { return 0 }
        return max(0, lowerBound(end) - lowerBound(start))
    }

    private func lowerBound(_ seconds: Double) -> Int {
        var low = 0, high = timestamps.count
        while low < high {
            let middle = (low + high) / 2
            if timestamps[middle] < seconds { low = middle + 1 } else { high = middle }
        }
        return low
    }

    static func load(source: URL) async throws -> SourceFrameTimeline {
        if PhotosMedia.shared.isStill(source) { return SourceFrameTimeline(timestamps: [0]) }
        let asset = try await PhotosMedia.asset(source).value
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw FrameSamplingError.noVideoTrack
        }
        var times: [Double] = []
        if let cursor = track.makeSampleCursor(presentationTimeStamp: .zero) {
            let segments = try await track.load(.segments)
            repeat {
                try Task.checkCancellation()
                // Cursor time is in the media's timeline. Container edit lists
                // (including ordinary B-frame lead-in) map it into the asset timeline.
                let mediaTime = cursor.presentationTimeStamp
                for segment in segments where !segment.isEmpty {
                    let mapping = segment.timeMapping
                    if CMTimeRangeContainsTime(mapping.source, time: mediaTime) {
                        times.append(CMTimeMapTimeFromRangeToRange(mediaTime, fromRange: mapping.source,
                                                                toRange: mapping.target).seconds)
                        break
                    }
                }
            } while cursor.stepInPresentationOrder(byCount: 1) == 1
        } else {
            // Some containers lack sample cursors. Read compressed packet timing;
            // nil output settings avoid image decompression and pixel allocation.
            let reader = try AVAssetReader(asset: asset)
            let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
            output.alwaysCopiesSampleData = false
            guard reader.canAdd(output) else { throw FrameSamplingError.noVideoTrack }
            reader.add(output)
            guard reader.startReading() else {
                throw FrameSamplingError.assetReaderFailed(reader.error?.localizedDescription ?? "Cannot read timing")
            }
            defer { reader.cancelReading() }
            while let sample = output.copyNextSampleBuffer() {
                try Task.checkCancellation()
                times.append(CMSampleBufferGetPresentationTimeStamp(sample).seconds)
            }
            if reader.status == .failed {
                throw FrameSamplingError.assetReaderFailed(reader.error?.localizedDescription ?? "Cannot read timing")
            }
        }
        try Task.checkCancellation()
        let timeline = SourceFrameTimeline(timestamps: times)
        guard !timeline.timestamps.isEmpty else { throw FrameSamplingError.decodingFailed("No frame timestamps") }
        return timeline
    }

    /// Reads only the sample timestamps around a requested interval. Sample
    /// cursors seek by presentation time, so this work stays local to the range
    /// instead of walking a long video's entire sample table.
    static func load(source: URL, around range: ClosedRange<Double>) async throws -> SourceFrameTimeline {
        let asset = try await PhotosMedia.asset(source).value
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw FrameSamplingError.noVideoTrack
        }
        let segments = try await track.load(.segments)
        var times: [Double] = []
        let targetStart = CMTime(seconds: max(0, range.lowerBound), preferredTimescale: 600)
        let targetEnd = CMTime(seconds: max(range.lowerBound, range.upperBound), preferredTimescale: 600)

        if let segment = segment(containingOrBefore: targetStart, in: segments) {
            let mediaStart = CMTimeMapTimeFromRangeToRange(targetStart,
                fromRange: segment.timeMapping.target, toRange: segment.timeMapping.source)
            if let cursor = track.makeSampleCursor(presentationTimeStamp: mediaStart) {
                func mappedTime() -> Double? {
                    let mediaTime = cursor.presentationTimeStamp
                    for segment in segments where !segment.isEmpty {
                        let mapping = segment.timeMapping
                        if CMTimeRangeContainsTime(mapping.source, time: mediaTime) {
                            return CMTimeMapTimeFromRangeToRange(mediaTime, fromRange: mapping.source,
                                                                toRange: mapping.target).seconds
                        }
                    }
                    return nil
                }

                // Include one preceding sample when the cursor lands just after
                // the requested start. It gives timestamp-to-ordinal mapping a
                // stable local origin for frames at the range boundary.
                while let seconds = mappedTime(), seconds > range.lowerBound,
                      cursor.stepInPresentationOrder(byCount: -1) == -1 {}

                while true {
                    try Task.checkCancellation()
                    if let seconds = mappedTime() {
                        if seconds > targetEnd.seconds { break }
                        if seconds.isFinite, seconds >= 0 { times.append(seconds) }
                    }
                    guard cursor.stepInPresentationOrder(byCount: 1) == 1 else { break }
                }
            }
        }

        // Some formats do not provide sample cursors. AVAssetReader's compressed
        // passthrough still avoids image decoding and honors the requested range.
        if times.isEmpty {
            let reader = try AVAssetReader(asset: asset)
            let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
            output.alwaysCopiesSampleData = false
            guard reader.canAdd(output) else { throw FrameSamplingError.noVideoTrack }
            reader.add(output)
            let duration = CMTimeSubtract(targetEnd, targetStart)
            reader.timeRange = CMTimeRange(start: targetStart, duration: duration)
            guard reader.startReading() else {
                throw FrameSamplingError.assetReaderFailed(reader.error?.localizedDescription ?? "Cannot read timing")
            }
            defer { reader.cancelReading() }
            while let sample = output.copyNextSampleBuffer() {
                try Task.checkCancellation()
                let seconds = CMSampleBufferGetPresentationTimeStamp(sample).seconds
                if seconds.isFinite, seconds >= 0 { times.append(seconds) }
            }
            if reader.status == .failed {
                throw FrameSamplingError.assetReaderFailed(reader.error?.localizedDescription ?? "Cannot read timing")
            }
        }

        try Task.checkCancellation()
        let timeline = SourceFrameTimeline(timestamps: times)
        guard !timeline.timestamps.isEmpty else { throw FrameSamplingError.decodingFailed("No frame timestamps in the requested range") }
        return timeline
    }

    private static func segment(containingOrBefore time: CMTime,
                                in segments: [AVAssetTrackSegment]) -> AVAssetTrackSegment? {
        let usable = segments.filter { !$0.isEmpty }
        if let containing = usable.first(where: { CMTimeRangeContainsTime($0.timeMapping.target, time: time) }) {
            return containing
        }
        return usable.filter { CMTimeCompare($0.timeMapping.target.start, time) <= 0 }
            .max { CMTimeCompare($0.timeMapping.target.start, $1.timeMapping.target.start) < 0 }
            ?? usable.first
    }
}
