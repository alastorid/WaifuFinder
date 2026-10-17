import AVFoundation
import CoreGraphics
import CoreImage
import CoreMedia
import Foundation
import UniformTypeIdentifiers

/// Protocol for reading analysis-resolution thumbnails from the in-memory cache.
///
/// The export layer does not own the thumbnail store — it is provided by the
/// orchestration layer (or a test double). The store is keyed by the
/// `thumbnailKey` from `AnalyzedFrame`, which is of the form
/// `"thumb-fXXXXXX"` where `XXXXXX` is the zero-padded sample index.
protocol ThumbnailStore: Sendable {
    /// Returns cached JPEG data or regenerates an evicted frame from its source.
    func thumbnailData(forKey key: String) async -> Data?
}

/// Exports frames from a video at native resolution.
///
/// For each frame in the request, this class:
/// 1. Opens the source video with `AVAssetReader`
/// 2. Creates a video track output that decompresses to a pixel buffer at source resolution
/// 3. Seeks to the frame's `timeSeconds` using `reader.timeRange`
/// 4. Reads one sample buffer and converts it to a `CGImage`
/// 5. Writes the image via `CGImageDestination` in the requested format
///
/// **Seek strategy**: since there are only a handful of frames to export (user
/// selection, typically <50), we seek to each timestamp individually rather than
/// a sequential pass. `AVAssetReader.timeRange` starts at the timestamp; we read
/// one frame and close the reader.
///
/// **Thumbnails mode**: if `request.useThumbnails` is true, reads the cached
/// analysis-resolution JPEG from `ThumbnailStore` instead of re-extracting. This
/// is far cheaper and always available, but loses detail (analysis resolution is
/// typically 512 px wide vs. native 1920+ px). The trade-off is explicit: the UI
/// presents this as "Export thumbnails" vs. "Export full resolution".
///
/// **Overwrite**: if `request.overwrite` is false and a file exists, appends
/// `-1`, `-2`, etc. before the extension.
///
/// Returns an `AsyncThrowingStream<ExportResult, Error>` — one result per frame
/// in request order. A failed frame yields an error in the stream (not a silent
/// skip), so the UI can report which selections failed.
final class FrameExporter: FrameExporting {
    private let thumbnailStore: ThumbnailStore?

    /// Creates an exporter with an optional thumbnail store.
    ///
    /// - Parameter thumbnailStore: If provided, `request.useThumbnails = true`
    ///   will read from this store instead of re-extracting frames.
    init(thumbnailStore: ThumbnailStore? = nil) {
        self.thumbnailStore = thumbnailStore
    }

    /// Exports frames according to the request.
    ///
    /// - Parameters:
    ///   - request: The export specification (frames, destination, format, options).
    ///   - sourceURL: The source video file URL.
    /// - Returns: A stream of `ExportResult`, one per frame in request order.
    ///   Errors for individual frames are emitted in the stream so the caller
    ///   can present per-frame failures.
    func export(_ request: ExportRequest, sourceURL: URL) -> AsyncThrowingStream<ExportResult, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    // Validate destination
                    var isDirectory: ObjCBool = false
                    let exists = FileManager.default.fileExists(atPath: request.destinationDirectory.path, isDirectory: &isDirectory)
                    if exists && !isDirectory.boolValue {
                        throw ExportError.destinationNotADirectory(request.destinationDirectory)
                    }
                    if !exists {
                        try FileManager.default.createDirectory(at: request.destinationDirectory, withIntermediateDirectories: true)
                    }

                    // Get source stem for naming
                    let sourceStem = sourceURL.deletingPathExtension().lastPathComponent

                    // If using thumbnails, we don't need to open the video
                    if request.useThumbnails {
                        guard let store = thumbnailStore else {
                            // No store configured — emit errors for all frames
                            for frameID in request.frameIDs {
                                _ = "thumb-\(frameID.raw)"
                                continuation.yield(with: .failure(ExportError.writeFailed(
                                    frameID: frameID,
                                    url: request.destinationDirectory.appendingPathComponent(request.fileName(for: frameID, sourceStem: sourceStem)),
                                    underlying: "Thumbnails requested but no thumbnail store configured"
                                )))
                            }
                            continuation.finish()
                            return
                        }

                        // Export each frame from thumbnail store
                        for frameID in request.frameIDs {
                            let key = "thumb-\(frameID.raw)"
                            let fileName = request.fileName(for: frameID, sourceStem: sourceStem)
                            let destURL = request.destinationDirectory.appendingPathComponent(fileName)

                            let finalURL = request.overwrite ? destURL : uniqueDestinationURL(base: destURL)

                            if let data = await store.thumbnailData(forKey: key) {
                                do {
                                    try data.write(to: finalURL, options: .atomic)
                                    let byteCount = data.count
                                    continuation.yield(ExportResult(frameID: frameID, url: finalURL, byteCount: byteCount))
                                } catch {
                                    continuation.yield(with: .failure(ExportError.writeFailed(
                                        frameID: frameID,
                                        url: finalURL,
                                        underlying: error.localizedDescription
                                    )))
                                }
                            } else {
                                continuation.yield(with: .failure(ExportError.writeFailed(
                                    frameID: frameID,
                                    url: finalURL,
                                    underlying: "Thumbnail not found in store for key: \(key)"
                                )))
                            }
                        }
                        continuation.finish()
                        return
                    }

                    if PhotosMedia.isPhotos(sourceURL) {
                        for frameID in request.frameIDs {
                            try Task.checkCancellation()
                            let image = try await PhotosMedia.shared.image(sourceURL, seconds: request.frameTimes[frameID] ?? Double(frameID.sampleIndex))
                            let destination = request.destinationDirectory.appendingPathComponent(request.fileName(for: frameID, sourceStem: sourceStem))
                            let final = request.overwrite ? destination : uniqueDestinationURL(base: destination)
                            let count = try writeImage(image, to: final, format: request.format, quality: request.jpegQuality)
                            continuation.yield(ExportResult(frameID: frameID, url: final, byteCount: count))
                        }
                        continuation.finish()
                        return
                    }

                    // Full-resolution extraction path
                    let asset = VideoAssetFactory.asset(for: sourceURL)
                    let tracks = try await asset.loadTracks(withMediaType: .video)
                    guard let videoTrack = tracks.first else {
                        throw ExportError.sourceUnreadable(sourceURL)
                    }

                    // Export each frame by seeking to its timestamp
                    for frameID in request.frameIDs {
                        let frameTimeSeconds = if let time = request.frameTimes[frameID] { time } else { await frameTimeForFrameID(frameID, in: asset) }
                        let fileName = request.fileName(for: frameID, sourceStem: sourceStem)
                        let destURL = request.destinationDirectory.appendingPathComponent(fileName)

                        let finalURL: URL
                        if request.overwrite {
                            finalURL = destURL
                        } else {
                            finalURL = uniqueDestinationURL(base: destURL)
                        }

                        do {
                            let cgImage = try await extractFrameAtTime(
                                asset: asset,
                                videoTrack: videoTrack,
                                timeSeconds: frameTimeSeconds
                            )

                            let byteCount = try writeImage(cgImage, to: finalURL, format: request.format, quality: request.jpegQuality)
                            continuation.yield(ExportResult(frameID: frameID, url: finalURL, byteCount: byteCount))
                        } catch let error as ExportError {
                            continuation.yield(with: .failure(error))
                        } catch {
                            continuation.yield(with: .failure(ExportError.writeFailed(
                                frameID: frameID,
                                url: finalURL,
                                underlying: error.localizedDescription
                            )))
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    /// Returns the presentation time in seconds for a given FrameID.
    ///
    /// Since FrameID encodes only the sampleIndex (ordinal in the decode
    /// pass), we need to recover the original timestamp. The pass is
    /// frame by frame, so the ordinal maps back through the frame rate:
    /// `time = ordinal / frameRate`. That is what is reconstructed here,
    /// from the asset's own track.
    ///
    /// Legacy callers without `ExportRequest.frameTimes` use nominal FPS. The
    /// browser supplies actual decoded timestamps and never uses this VFR estimate.
    private func frameTimeForFrameID(_ frameID: FrameID, in asset: AVAsset) async -> Double {
        let duration = try? await asset.load(.duration)
        let durationSeconds = duration?.seconds ?? 0
        guard durationSeconds > 0 else { return 0 }

        // The same fallback the probe uses: a track that cannot report
        // its rate is treated as 30 fps, which is what the frame count
        // estimate the frame was numbered under was built on.
        var frameRate = 30.0
        if let track = (try? await asset.loadTracks(withMediaType: .video))?.first,
           let nominal = try? await track.load(.nominalFrameRate), nominal > 0 {
            frameRate = Double(nominal)
        }

        return min(Double(frameID.sampleIndex) / frameRate, durationSeconds)
    }

    /// One frame at native resolution, for the zoomable preview.
    func nativeImage(at timeSeconds: Double, in asset: AVURLAsset) async throws -> CGImage {
        let tracks = try await asset.loadTracks(withMediaType: .video)
        guard let videoTrack = tracks.first else { throw ExportError.sourceUnreadable(asset.url) }
        return try await extractFrameAtTime(asset: asset, videoTrack: videoTrack, timeSeconds: timeSeconds)
    }

    /// Extracts a single frame at the given time using AVAssetReader.
    ///
    /// Creates a new reader for each seek because `AVAssetReader` is not
    /// designed for repeated seeking — its `timeRange` is set once at start.
    /// With <50 frames, the overhead is negligible.
    private func extractFrameAtTime(asset: AVAsset, videoTrack: AVAssetTrack, timeSeconds: Double) async throws -> CGImage {
        let reader = try AVAssetReader(asset: asset)
        // A pixel format must be named. With `nil` output settings the reader vends
        // the track's *stored* format — for H.264/HEVC that is a compressed sample
        // buffer with no image buffer at all, so `CMSampleBufferGetImageBuffer`
        // returns nil and every export fails with "decode failed". Specifying only
        // the pixel format (not dimensions) makes the reader decompress and leaves
        // the frame at the source's native resolution, which is the whole point of
        // the full-resolution path.
        let outputSettings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        let output = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: outputSettings)
        output.alwaysCopiesSampleData = false

        guard reader.canAdd(output) else {
            throw ExportError.decodeFailed(frameID: FrameID(sampleIndex: 0), timeSeconds: timeSeconds)
        }
        reader.add(output)

        // Seek to the exact timestamp
        let startTime = CMTime(seconds: timeSeconds, preferredTimescale: 600)
        reader.timeRange = CMTimeRange(start: startTime, duration: .indefinite)

        guard reader.startReading() else {
            let error = reader.error?.localizedDescription ?? "unknown"
            throw ExportError.seekFailed(frameID: FrameID(sampleIndex: 0), timeSeconds: timeSeconds, underlying: error)
        }

        // Read exactly one frame
        guard let sampleBuffer = output.copyNextSampleBuffer() else {
            throw ExportError.decodeFailed(frameID: FrameID(sampleIndex: 0), timeSeconds: timeSeconds)
        }

        // Convert CMSampleBuffer to CGImage
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            throw ExportError.decodeFailed(frameID: FrameID(sampleIndex: 0), timeSeconds: timeSeconds)
        }

        let ciImage = CIImage(cvPixelBuffer: imageBuffer)
        let context = CIContext()
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
            throw ExportError.decodeFailed(frameID: FrameID(sampleIndex: 0), timeSeconds: timeSeconds)
        }

        reader.cancelReading()
        return cgImage
    }

    /// Writes a CGImage to the given URL in the specified format.
    ///
    /// - Returns: The number of bytes written.
    private func writeImage(_ image: CGImage, to url: URL, format: ExportFormat, quality: Float) throws -> Int {
        let dest = CGImageDestinationCreateWithURL(url as CFURL, format.contentType as CFString, 1, nil)
        guard let destination = dest else {
            throw ExportError.writeFailed(frameID: FrameID(sampleIndex: 0), url: url, underlying: "Failed to create image destination")
        }

        var properties: [CFString: Any] = [:]
        if format == .jpeg {
            properties[kCGImageDestinationLossyCompressionQuality] = quality
        }

        CGImageDestinationAddImage(destination, image, properties as CFDictionary)
        guard CGImageDestinationFinalize(destination) else {
            throw ExportError.writeFailed(frameID: FrameID(sampleIndex: 0), url: url, underlying: "CGImageDestinationFinalize failed")
        }

        // Get file size
        let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
        return attrs[.size] as? Int ?? 0
    }

    /// Generates a unique destination URL by appending `-1`, `-2`, etc.
    ///
    /// The suffix is inserted before the file extension. Stops after 10,000
    /// attempts (practically unreachable).
    ///
    /// Internal rather than private so the naming decision can be pinned by value
    /// without writing 10,001 files into a directory to reach the cap: the only
    /// other way to observe it is through a full export, which needs a real video
    /// and can only ever show the first suffix.
    func uniqueDestinationURL(base: URL) -> URL {
        let directory = base.deletingLastPathComponent()
        let stem = base.deletingPathExtension().lastPathComponent
        let ext = base.pathExtension

        var candidate = base
        var suffix = 1
        while FileManager.default.fileExists(atPath: candidate.path) && suffix <= 10000 {
            // Only join the extension when there is one. Interpolating an empty
            // extension would leave a trailing dot (`"README-1."`), which is not
            // the name any caller asked for and not a name a Finder listing shows
            // the same way as `"README-1"`.
            let newName = ext.isEmpty ? "\(stem)-\(suffix)" : "\(stem)-\(suffix).\(ext)"
            candidate = directory.appendingPathComponent(newName)
            suffix += 1
        }
        return candidate
    }
}
