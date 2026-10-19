import AVFoundation
import Foundation

extension MediaImport {
    /// Materializes one seekable timeline so all existing readers, previews and
    /// exporters use the same timestamps. Inputs stay alive until export finishes.
    static func prepare(_ sources: [URL]) async throws -> PreparedMedia {
        guard !sources.isEmpty else { throw failure("No videos were supplied.") }
        if sources.count == 1 { return try await prepare(sources[0]) }

        var inputs: [PreparedMedia] = []
        for source in sources {
            try Task.checkCancellation()
            do { inputs.append(try await prepare(source)) }
            catch {
                if Task.isCancelled { throw CancellationError() }
                throw failure("Could not prepare \(source.lastPathComponent): \(error.localizedDescription)")
            }
        }
        return try await concatenate(inputs)
    }

    private static func concatenate(_ inputs: [PreparedMedia]) async throws -> PreparedMedia {
        let composition = AVMutableComposition()
        guard let video = composition.addMutableTrack(withMediaType: .video,
                preferredTrackID: kCMPersistentTrackID_Invalid) else {
            throw failure("Could not create the joined video.")
        }
        let audio = composition.addMutableTrack(withMediaType: .audio,
                                                preferredTrackID: kCMPersistentTrackID_Invalid)
        var cursor = CMTime.zero
        var instructions: [AVMutableVideoCompositionInstruction] = []
        var geometry: [(CGSize, CGAffineTransform)] = []
        var canvas = CGSize.zero
        var frameRate: Double = 1
        for input in inputs {
            try Task.checkCancellation()
            let asset = VideoAssetFactory.asset(for: input.url)
            guard let track = try await asset.loadTracks(withMediaType: .video).first else {
                throw VideoProbeError.noVideoTrack
            }
            let duration = try await asset.load(.duration)
            let range = CMTimeRange(start: .zero, duration: duration)
            let size = try await track.load(.naturalSize)
            let transform = try await track.load(.preferredTransform)
            let bounds = CGRect(origin: .zero, size: size).applying(transform)
            canvas.width = max(canvas.width, bounds.width)
            canvas.height = max(canvas.height, bounds.height)
            let rate = Double(try await track.load(.nominalFrameRate))
            frameRate = max(frameRate, rate > 0 ? rate : 30)
            try video.insertTimeRange(range, of: track, at: cursor)
            if let sourceAudio = try await asset.loadTracks(withMediaType: .audio).first, let audio {
                let audioRange = CMTimeRangeGetIntersection(range, otherRange: try await sourceAudio.load(.timeRange))
                if audioRange.duration > .zero {
                    try audio.insertTimeRange(audioRange, of: sourceAudio,
                                              at: cursor + audioRange.start)
                }
            }
            let instruction = AVMutableVideoCompositionInstruction()
            instruction.timeRange = CMTimeRange(start: cursor, duration: duration)
            instructions.append(instruction)
            geometry.append((size, transform))
            cursor = cursor + duration
        }
        // Use an even canvas for the native encoder, preserving each clip's
        // orientation and fitting differing dimensions without stretching.
        canvas = CGSize(width: ceil(canvas.width / 2) * 2, height: ceil(canvas.height / 2) * 2)
        for (index, instruction) in instructions.enumerated() {
            let (size, transform) = geometry[index]
            let bounds = CGRect(origin: .zero, size: size).applying(transform)
            let scale = min(canvas.width / bounds.width, canvas.height / bounds.height)
            let fitted = transform
                .concatenating(CGAffineTransform(translationX: -bounds.minX, y: -bounds.minY))
                .concatenating(CGAffineTransform(scaleX: scale, y: scale))
                .concatenating(CGAffineTransform(translationX: (canvas.width - bounds.width * scale) / 2,
                                                 y: (canvas.height - bounds.height * scale) / 2))
            let layer = AVMutableVideoCompositionLayerInstruction(assetTrack: video)
            layer.setTransform(fitted, at: instruction.timeRange.start)
            instruction.layerInstructions = [layer]
        }
        if let audio, audio.segments.isEmpty { composition.removeTrack(audio) }
        let rendering = AVMutableVideoComposition()
        rendering.renderSize = canvas
        rendering.frameDuration = CMTime(seconds: 1 / frameRate, preferredTimescale: 60_000)
        rendering.instructions = instructions
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("waifu-joined-\(UUID())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let joined = PreparedMedia(url: directory.appendingPathComponent("Joined videos.mov"), directory: directory)
        guard let exporter = AVAssetExportSession(asset: composition, presetName: AVAssetExportPresetHighestQuality) else {
            throw failure("Could not create the joined video exporter.")
        }
        exporter.videoComposition = rendering
        try Task.checkCancellation()
        try await exporter.export(to: joined.url, as: .mov)
        try Task.checkCancellation()
        // Explicit lifetime: remote inputs own temporary files used by export.
        withExtendedLifetime(inputs) {}
        return joined
    }
}
