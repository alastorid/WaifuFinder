import AVFoundation
import CoreImage
import CoreMedia
import Foundation

// Errors that can occur during frame sampling.
//
// Each case corresponds to a distinct failure mode that the caller may want to
// handle differently — e.g., `.unreadableFile` means the container is corrupt
// or the codec is unsupported, while `.noVideoTrack` means the file is valid
// but contains no video (which `probe` should have caught, but the reader can
// still fail if the track disappears between probe and sample).
enum FrameSamplingError: Error, LocalizedError, Equatable {
    case unreadableFile(String)
    case noVideoTrack
    case assetReaderFailed(String)
    case decodingFailed(String)
    case cancelled

    var errorDescription: String? {
        switch self {
        case .unreadableFile(let detail):
            return "Cannot read video file: \(detail)"
        case .noVideoTrack:
            return "Video track unavailable during sampling"
        case .assetReaderFailed(let detail):
            return "Asset reader failed: \(detail)"
        case .decodingFailed(let detail):
            return "Frame decoding failed: \(detail)"
        case .cancelled:
            return "Sampling cancelled"
        }
    }

    static func == (lhs: FrameSamplingError, rhs: FrameSamplingError) -> Bool {
        switch (lhs, rhs) {
        case (.unreadableFile(let l), .unreadableFile(let r)),
             (.assetReaderFailed(let l), .assetReaderFailed(let r)),
             (.decodingFailed(let l), .decodingFailed(let r)):
            return l == r
        case (.noVideoTrack, .noVideoTrack), (.cancelled, .cancelled):
            return true
        default:
            return false
        }
    }
}

// A frame sampler that performs **one sequential `AVAssetReader` pass** over
// the file, decoding native pixels once and scaling the cropped picture, yielding
// **every frame the decoder produces**. The pass is frame by frame: no frame
// is skipped, dropped, or sampled away.
//
// ## Bounded parallel preparation
//
// The source reader advances sequentially. Each selected pixel buffer is then
// cropped and resized on a background task, with a memory- and core-bounded
// preparation window. Tasks can complete in either order; `next()` returns them
// in source order. This overlaps raster work with decoding and scoring without
// buffering the video. FrameProcessingConcurrency reduces concurrency for large
// sources.
// Each task drains its temporary image objects before returning its two images.
//
// Native picture pixels supply both face scores; a smaller raster supplies
// aesthetics, FeaturePrint and thumbnails. Its longest edge is analysisWidth.
// A caller that stops pulling triggers no more decoding; already queued work is
// limited to the bounded preparation window.
//
// ## Frame ownership
//
// The mode decides which decoded frames become `SampledFrame`s. The
// full pass takes every one; the quick pass takes one frame every
// `frameStep` decoded frames, always including the first; the
// keyframe pass takes the sync samples the reader flags, which the
// encoder places at its own cadence rather than on any grid. A kept
// frame carries its *decoded* ordinal as its `sampleIndex`, so
// `sampleIndex` is the frame's position in the file — the same
// frame has the same identifier in every mode. Its actual decoded
// timestamp is retained separately, including variable-rate timing. The stream's length is the
// file's *actual* kept-frame count: a variable-frame-rate file that
// yields more frames than `duration × frameRate` still has all of
// them considered, because the pass ends when the reader is
// exhausted, not when a planned count is reached.
final class AVFoundationFrameSampler: FrameSampling {
    func probe(_ url: URL) async throws -> VideoInfo {
        try await VideoProbe.probe(url)
    }

    func sample(_ url: URL,
                analysisWidth: Int,
                mode: AnalysisMode,
                onProgress: @escaping @Sendable (Int) -> Void) -> AsyncThrowingStream<SampledFrame, Error> {
        sample(url, analysisWidth: analysisWidth, mode: mode, timeRange: nil,
               excludingSampleIndices: [], onProgress: onProgress)
    }

    func sample(
        _ url: URL,
        analysisWidth: Int,
        mode: AnalysisMode,
        timeRange: ClosedRange<Double>?,
        excludingSampleIndices: Set<Int>,
        sampleIndexMaps: [Int: FrameSampleIndexMap] = [:],
        onProgress: @escaping @Sendable (Int) -> Void
    ) -> AsyncThrowingStream<SampledFrame, Error> {
        if let collection = VideoCollection.get(url) {
            let producer = VideoCollectionSampler(collection: collection, width: analysisWidth, mode: mode,
                                                  timeRange: timeRange,
                                                  excludingSampleIndices: excludingSampleIndices,
                                                  sampleIndexMaps: sampleIndexMaps,
                                                  onProgress: onProgress)
            return AsyncThrowingStream(unfolding: { try await producer.next() })
        }
        if let album = PhotosMedia.shared.album(url) {
            let producer = PhotosAlbumSampler(album: album, width: analysisWidth, mode: mode)
            return AsyncThrowingStream(unfolding: { try await producer.next() })
        }
        let producer = FrameSamplingProducer(
            url: url,
            analysisWidth: analysisWidth,
            mode: mode,
            timeRange: timeRange,
            excludingSampleIndices: excludingSampleIndices,
            sampleIndexMap: sampleIndexMaps[0],
            onProgress: onProgress
        )
        // `AsyncThrowingStream` has no `unfolding:onCancel:` (only the non-throwing
        // `AsyncStream` does), so cancellation is observed where it matters: the
        // producer checks for it before each frame and throws, which ends the
        // stream, releases it, and with it the reader.
        return AsyncThrowingStream(unfolding: { try await producer.next() })
    }
}

private struct FrameTrackContext: @unchecked Sendable {
    let asset: AVAsset
    let track: AVAssetTrack
    let transform: CGAffineTransform
}

/// The pixel buffer is retained until raster preparation finishes and is only
/// read by that task. The serial reader cannot reuse a retained buffer.
private struct DecodedFrame: @unchecked Sendable {
    let sampleIndex: Int
    let timeSeconds: Double
    let pixels: CVPixelBuffer
}

/// CIContext supports concurrent rendering. Reuse one context across the small
/// preparation window without caching intermediates from unrelated video frames.
private final class FrameRasterizer: @unchecked Sendable {
    private let context = CIContext(options: [.cacheIntermediates: false])

    func render(_ frame: DecodedFrame, transform: CGAffineTransform,
                analysisWidth: Int) throws -> SampledFrame {
        try autoreleasepool {
            try Task.checkCancellation()
            let native = CIImage(cvPixelBuffer: frame.pixels).transformed(by: transform)
            guard let decodedImage = context.createCGImage(native, from: native.extent) else {
                throw FrameSamplingError.decodingFailed("could not rasterise native pixels at \(frame.timeSeconds)s")
            }
            let picture = FramePictureArea.bounds(decodedImage)
            let nativeImage = decodedImage.cropping(to: picture) ?? decodedImage
            let input = CIImage(cgImage: nativeImage)
            let scale = CGFloat(analysisWidth) / max(input.extent.width, input.extent.height)
            let scaled = input.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
            guard let image = context.createCGImage(scaled, from: scaled.extent) else {
                throw FrameSamplingError.decodingFailed("could not rasterise the frame at \(frame.timeSeconds)s")
            }
            try Task.checkCancellation()
            return SampledFrame(sampleIndex: frame.sampleIndex, timeSeconds: frame.timeSeconds,
                                image: image, nativeImage: nativeImage)
        }
    }
}

/// The presentation times of a video's sync samples — the frames the
/// encoder marked decodable on their own.
///
/// ## Why this is its own pass
///
/// The sync flag is not available where it would be convenient. On a
/// **decoded** `AVAssetReaderTrackOutput` every buffer reports *no*
/// sample attachments at all, so reading `kCMSampleAttachmentKey_NotSync`
/// there yields nothing and every frame looks like a keyframe — a pass
/// that silently did the opposite of what it was asked. The flag only
/// survives on the **compressed** passthrough (`outputSettings: nil`),
/// which is what `AVAssetReader` returns when asked not to decode.
///
/// Learning where the keyframes are therefore costs one extra read of
/// the sample table. That read decodes nothing and allocates no pixels
/// — the same kind of walk `SourceFrameTimeline` makes for timing — and
/// its result is a list of timestamps, kilobytes rather than gigabytes,
/// so the bounded-memory guarantee of the sampling pass is untouched.
///
/// ## Matching by timestamp, not by ordinal
///
/// The compressed pass emits a leading sample the decode pass does not
/// show, so a keyframe's position among compressed samples is offset
/// from its position among decoded frames. Presentation times carry no
/// such offset, which is why the decoded pass matches on `CMTime`.
///
/// ## Matching is a cursor, not a search
///
/// Both sequences ascend, so the index carries its own forward cursor
/// instead of being searched per frame. A linear scan would be
/// O(decoded frames × keyframes): on the 47-minute reference that is
/// ~84,000 × ~1,400, or 118 million comparisons — a cost of the same
/// order as the Vision work this mode exists to save, spent on
/// bookkeeping. The cursor makes the whole pass O(frames).
struct SyncSampleIndex: Sendable {
    /// Ascending presentation times. Empty means the scan found no
    /// usable sync flags, which callers read as "cannot filter".
    private let times: [CMTime]
    /// Half a frame at the source rate: the slack allowed when matching
    /// a decoded timestamp against a recorded one. Both come from the
    /// same asset and normally compare exactly, but a timescale
    /// difference between the two readers must not cost a frame.
    private let toleranceSeconds: Double
    /// The one sync sample the producer has not yet passed. Advances
    /// only forward, so the scan across a pass is linear overall.
    private var cursor = 0

    init(times: [CMTime], toleranceSeconds: Double) {
        self.times = times
        self.toleranceSeconds = toleranceSeconds
    }

    /// Whether the file exposed any sync flags to filter on.
    var isEmpty: Bool { times.isEmpty }

    /// Whether `pts` sits on one of the indexed sync samples.
    ///
    /// Nearest-within-tolerance rather than exact equality, so a frame
    /// survives a reader rounding its timestamps differently from the
    /// scan. Skips past any sync sample that ended up behind `pts` —
    /// which happens when the compressed pass reported a sample the
    /// decode pass never shows — rather than stalling on it.
    ///
    /// `false` for an empty index, which the producer reads as "this
    /// file does not expose sync flags".
    mutating func contains(_ pts: CMTime) -> Bool {
        guard !isEmpty, pts.isNumeric else { return false }
        let seconds = pts.seconds
        guard seconds.isFinite else { return false }
        while cursor < times.count, times[cursor].seconds < seconds - toleranceSeconds { cursor += 1 }
        guard cursor < times.count else { return false }
        return abs(times[cursor].seconds - seconds) <= toleranceSeconds
    }
}

/// Maps a short, locally scanned presentation timeline back to the decoded
/// ordinals already used by an analysis. Calibrating at an existing frame also
/// accounts for any leading compressed samples the decoder does not emit.
struct FrameSampleIndexMap: Sendable {
    let timeline: SourceFrameTimeline
    let ordinalOffset: Int

    func index(at seconds: Double) -> Int {
        max(0, timeline.index(at: seconds) + ordinalOffset)
    }
}

// Holds the reader and does the work of walking the file, one frame per `next()`.
//
// An actor because the pipeline may pull from more than one task, and the reader
// is a single mutable resource. `next()` returning `nil` ends the stream.
private actor FrameSamplingProducer {
    private let url: URL
    private let analysisWidth: Int
    private let mode: AnalysisMode
    private let timeRange: ClosedRange<Double>?
    private let excludingSampleIndices: Set<Int>
    private let sampleIndexMap: FrameSampleIndexMap?
    private let onProgress: @Sendable (Int) -> Void

    private let rasterizer = FrameRasterizer()
    private var preparationWorkers = 1
    private var prepared: [Task<SampledFrame, Error>] = []

    private var reader: AVAssetReader?
    private var output: AVAssetReaderTrackOutput?
    private var frameStep = 1
    private var decodedCount = 0
    private var framesEmitted = 0
    private var isPrepared = false
    private var isFinished = false
    private var nativeTransform = CGAffineTransform.identity
    /// Keyframe mode only, and empty for the two modes that never look
    /// at it. See `SyncSampleIndex`.
    private var syncSamples = SyncSampleIndex(times: [], toleranceSeconds: 0)

    init(url: URL, analysisWidth: Int, mode: AnalysisMode, timeRange: ClosedRange<Double>?,
         excludingSampleIndices: Set<Int>,
         sampleIndexMap: FrameSampleIndexMap?,
         onProgress: @escaping @Sendable (Int) -> Void) {
        self.url = url
        self.analysisWidth = analysisWidth
        self.mode = mode
        self.timeRange = timeRange
        self.excludingSampleIndices = excludingSampleIndices
        self.sampleIndexMap = sampleIndexMap
        self.onProgress = onProgress
    }

    /// Load AVFoundation's non-Sendable track references outside actor isolation;
    /// the resulting reader and track remain owned by this producer actor.
    nonisolated private static func loadTrack(url: URL) async throws -> FrameTrackContext {
        let asset = try await PhotosMedia.asset(url).value
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw FrameSamplingError.noVideoTrack
        }
        let transform = try await track.load(.preferredTransform)
        return FrameTrackContext(asset: asset, track: track, transform: transform)
    }

    /// Whether the reader marked `buffer` a sync sample: a frame the
    /// encoder can decode without reference to any earlier one.
    ///
    /// Attachments are read with `createIfNecessary: false`. Asking
    /// CoreMedia to *build* that table for every buffer would allocate
    /// across tens of thousands of samples, which is the cost this
    /// filter exists to avoid.
    ///
    /// CoreMedia documents that **absence** of `NotSync` means the
    /// sample *is* sync, and that is read literally here. A buffer
    /// carrying no attachments at all — the compressed pass opens with
    /// one — is treated as sync for the same reason, so a container
    /// that omits the table entirely reads as all-sync rather than as
    /// none.
    nonisolated private static func isSyncSample(_ buffer: CMSampleBuffer) -> Bool {
        guard let attachments = CMSampleBufferGetSampleAttachmentsArray(buffer, createIfNecessary: false) as? [[String: Any]],
              let first = attachments.first else { return true }
        return (first[kCMSampleAttachmentKey_NotSync as String] as? Bool) != true
    }

    /// Walks the compressed sample table and records every sync sample's
    /// presentation time. Decodes nothing and allocates no pixels — see
    /// `SyncSampleIndex` for why this has to be a separate read.
    ///
    /// A failure here is not a failure of the pass. The index is simply
    /// left empty, which the producer reads as "this file does not
    /// expose sync flags" and answers by analysing every frame.
    nonisolated private static func readSyncSamples(track: AVAssetTrack,
                                                    asset: AVAsset,
                                                    frameRate: Double) -> SyncSampleIndex {
        let tolerance = max(1.0 / max(frameRate, 1) / 2, 0.0005)
        let empty = SyncSampleIndex(times: [], toleranceSeconds: tolerance)
        guard let reader = try? AVAssetReader(asset: asset) else { return empty }
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { return empty }
        reader.add(output)
        guard reader.startReading() else { return empty }
        defer { reader.cancelReading() }
        var times: [CMTime] = []
        while let buffer = output.copyNextSampleBuffer() {
            guard isSyncSample(buffer) else { continue }
            let pts = CMSampleBufferGetPresentationTimeStamp(buffer)
            guard pts.isNumeric else { continue }
            times.append(pts)
        }
        return SyncSampleIndex(times: times, toleranceSeconds: tolerance)
    }

    /// The next frame of the file, or `nil` at its end.
    func next() async throws -> SampledFrame? {
        do {
            try Task.checkCancellation()
            if !isPrepared { try await prepare() }
            // Read serially, then prepare a bounded number of native frames in parallel.
            // A slow consumer never causes additional decoding or queued work.
            while !isFinished, prepared.count < preparationWorkers {
                guard let decoded = try nextDecodedFrame() else { break }
                let rasterizer = self.rasterizer
                let transform = nativeTransform
                let width = analysisWidth
                prepared.append(Task.detached(priority: .userInitiated) {
                    try rasterizer.render(decoded, transform: transform, analysisWidth: width)
                })
            }
            guard !prepared.isEmpty else { return nil }
            // Tasks can finish in either order. Await submission order so source
            // ordinals and timestamps stay deterministic in every analysis mode.
            let task = prepared.removeFirst()
            let frame = try await withTaskCancellationHandler {
                try await task.value
            } onCancel: {
                task.cancel()
            }
            try Task.checkCancellation()
            framesEmitted += 1
            onProgress(framesEmitted)
            return frame
        } catch {
            for task in prepared { task.cancel() }
            prepared.removeAll(keepingCapacity: false)
            reader?.cancelReading()
            isFinished = true
            throw error
        }
    }

    /// AVAssetReader remains owned by this actor; expensive raster work does not.
    private func nextDecodedFrame() throws -> DecodedFrame? {
        try Task.checkCancellation()
        if isFinished { return nil }
        guard let output else { isFinished = true; return nil }

        while true {
            try Task.checkCancellation()
            // The reader is exhausted: the pass is over. Every frame
            // the mode selects has been emitted.
            guard let buffer = output.copyNextSampleBuffer() else {
                isFinished = true
                if let reader, reader.status == .failed {
                    throw FrameSamplingError.assetReaderFailed(reader.error?.localizedDescription ?? "unknown error")
                }
                return nil
            }

            let decodedIndex = decodedCount
            decodedCount += 1

            let presentation = CMSampleBufferGetPresentationTimeStamp(buffer)
            guard presentation.isValid, presentation != .invalid else { continue }
            let seconds = presentation.seconds
            guard seconds >= 0, seconds.isFinite else { continue }
            if let timeRange, !timeRange.contains(seconds) { continue }
            let sampleIndex = sampleIndexMap?.index(at: seconds) ?? decodedIndex
            // Apply the mode's grid to source ordinals. A focused pass can seek
            // into the file, so its local reader count is not the source ordinal.
            guard sampleIndex % frameStep == 0 else { continue }
            if excludingSampleIndices.contains(sampleIndex) { continue }

            // Keyframe mode keeps only the frames the encoder made
            // self-sufficient, matched by timestamp against the index
            // the compressed pass built. An empty index means the file
            // did not expose sync flags; falling through keeps the
            // pass *complete* rather than silently empty, at full
            // mode's cost. That is the safe direction to fail.
            if mode == .keyframe, !syncSamples.isEmpty, !syncSamples.contains(presentation) {
                continue
            }

            guard let imageBuffer = CMSampleBufferGetImageBuffer(buffer) else {
                throw FrameSamplingError.decodingFailed("sample buffer has no image buffer at \(seconds)s")
            }
            return DecodedFrame(sampleIndex: sampleIndex, timeSeconds: seconds, pixels: imageBuffer)
        }
    }

    // MARK: - Setup and teardown

    private func prepare() async throws {
        isPrepared = true

        let videoInfo = try await VideoProbe.probe(url)
        guard videoInfo.hasVideo else { throw FrameSamplingError.noVideoTrack }
        frameStep = mode.frameStep(frameRate: videoInfo.frameRate)
        preparationWorkers = FrameProcessingConcurrency(
            width: videoInfo.width, height: videoInfo.height,
            analysisWidth: analysisWidth).preparationWorkers

        let trackContext = try await Self.loadTrack(url: url)
        let asset = trackContext.asset
        let track = trackContext.track
        nativeTransform = trackContext.transform

        // Only the keyframe pass needs to know where the keyframes are,
        // and that costs an extra read of the sample table. Full and
        // quick select by ordinal and must not pay for a pass they
        // never read.
        if mode == .keyframe {
            syncSamples = Self.readSyncSamples(track: track, asset: asset, frameRate: videoInfo.frameRate)
        }

        let reader: AVAssetReader
        do {
            reader = try AVAssetReader(asset: asset)
        } catch {
            throw FrameSamplingError.unreadableFile(error.localizedDescription)
        }

        // Decode once at native resolution. The retained native frame supplies
        // both face scores; the smaller image supplies aesthetics and grouping.
        let outputSettings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]

        let readerOutput = AVAssetReaderTrackOutput(track: track, outputSettings: outputSettings)
        readerOutput.alwaysCopiesSampleData = false // zero-copy where possible

        guard reader.canAdd(readerOutput) else {
            throw FrameSamplingError.assetReaderFailed("cannot add the video output")
        }
        reader.add(readerOutput)

        // Filtering timestamps after reading still makes AVFoundation decode
        // the entire file. Constrain the reader itself so it can seek to the
        // nearest decodable point before the requested interval.
        if let timeRange {
            let start = CMTime(seconds: max(0, timeRange.lowerBound), preferredTimescale: 600)
            let requestedEnd = CMTime(seconds: max(timeRange.lowerBound, timeRange.upperBound),
                                      preferredTimescale: 600)
            let frameDuration = CMTime(seconds: 1 / max(videoInfo.frameRate, 1), preferredTimescale: 600)
            let end = CMTimeAdd(requestedEnd, frameDuration)
            reader.timeRange = CMTimeRange(start: start, duration: CMTimeSubtract(end, start))
        }

        guard reader.startReading() else {
            throw FrameSamplingError.assetReaderFailed(reader.error?.localizedDescription ?? "unknown error")
        }

        self.reader = reader
        self.output = readerOutput
    }

}
