import Foundation
import CoreGraphics

/// The three requests that actually run a model.
///
/// Each one reads what the client already has, runs the same production code the
/// local UI runs, and writes the result back in the same form — so a remote
/// result is not a second-class version of a local one.
extension RemoteServerSession {
    static func describe(_ error: Error) -> RemoteError {
        if let remote = error as? RemoteError { return remote }
        if let tagger = error as? TaggerError { return .modelUnavailable(tagger.errorDescription ?? "Tagging failed.") }
        if let uncover = error as? UncoverError { return .invalidRequest(uncover.errorDescription ?? "Uncover failed.") }
        return .internalFailure(error.localizedDescription)
    }

    // MARK: Tagging

    /// `tagBatch` → `tagResult`.
    ///
    /// The response carries UInt8/LZ4 score rows, including all four rating probabilities.
    /// The client applies its own display threshold over the retained scores.
    func handleTagBatch(_ envelope: RemoteEnvelope) async {
        await resources.touch()
        do {
            try Task.checkCancellation()
            var reader = WireReader(envelope.payload)
            let batchCount = Int(try reader.u8())
            guard let codec = RemoteTagImageCodec(rawValue: try reader.u8()), try reader.u16() == 0 else {
                throw RemoteError.invalidRequest("Unknown image codec or reserved flags.")
            }
            guard (1...WDTaggerBackend.batchSize).contains(batchCount) else { throw TaggerError.invalidBatch }
            var images: [CGImage] = []
            images.reserveCapacity(batchCount)
            for _ in 0..<batchCount { images.append(try RemoteTagImageDecoder.decode(codec: codec, bytes: try reader.raw())) }
            guard reader.isAtEnd else { throw RemoteError.framing("trailing bytes after a tagging batch") }

            guard await ModelLibrary.shared.active == nil else { throw RemoteError.busy }
            let backend = try await catalog.tagger()
            let results = try await backend.tag(batch: images)
            try Task.checkCancellation()
            guard results.count == batchCount else { throw TaggerError.invalidProbabilities }
            let payload = try RemoteTagScorePayload.encode(results, fingerprint: await catalog.vocabularyFingerprint())
            try await connection.send(.tagResult, payload: payload, streamID: envelope.streamID)
        } catch {
            guard !Task.isCancelled else { return }
            try? await connection.sendError(envelope, Self.describe(error))
        }
    }

    // MARK: Vocabulary

    /// The labels CSV, so a client with no local tagger can still filter, search
    /// and render tags. Sent once per fingerprint and cached on the client.
    func handleVocabularyRequest(_ envelope: RemoteEnvelope) async {
        await resources.touch()
        do {
            let csv = try await catalog.vocabularyCSV()
            var payload = WireWriter(capacity: csv.count + 12)
            payload.u64(try await catalog.vocabularyFingerprint())
            payload.u32(UInt32(csv.count))
            payload.append(RemoteCompressedPayload.encode([UInt8](csv)))
            try await connection.send(.vocabularyResponse, payload: payload.bytes, streamID: envelope.streamID)
        } catch {
            try? await connection.sendError(envelope, Self.describe(error))
        }
    }

    // MARK: Uncover

    /// A whole reconstruction, start to finish.
    ///
    /// The client ships the manifest and the `degraded/` frames; the server runs
    /// the same backend the local UI would and ships back `reconstructed/`. The
    /// `original/` frames stay on the client, which is where compositing happens.
    ///
    /// This handler does the upload collection itself rather than returning, so
    /// the frames keep arriving on the same stream while the model runs. That is
    /// why the work is registered as a cancellable job rather than awaited
    /// inline: a `cancelJob` must be able to reach it.
    func handleUncover(_ envelope: RemoteEnvelope) async {
        do {
            guard let control = try? envelope.control(),
                  case .uncoverBegin(let manifest, let name, let allowUnsafeMemory) = control else {
                throw RemoteError.invalidRequest("A reconstruction must begin with a manifest.")
            }
            guard let algorithm = UncoverAlgorithm(rawValue: name) else {
                throw RemoteError.invalidRequest("Unknown algorithm \(name).")
            }
            guard ManagedModel.offered.contains(where: { $0.algorithm == algorithm }) else {
                throw RemoteError.modelUnavailable("This build does not offer \(algorithm.label).")
            }
            guard !manifest.frames.isEmpty, manifest.frames.count <= 300,
                  manifest.frames.enumerated().allSatisfy({ $0.offset == $0.element.index }),
                  manifest.width > 0, manifest.height > 0,
                  manifest.regionWidth > 0, manifest.regionHeight > 0 else {
                throw RemoteError.invalidRequest("Invalid reconstruction manifest.")
            }
            let backend = catalog.backend(algorithm)
            let capabilities = await backend.capabilities()
            guard capabilities.available else {
                throw RemoteError.modelUnavailable(capabilities.message.isEmpty
                    ? "\(algorithm.label) is not installed on this server." : capabilities.message)
            }
            let streamID = envelope.streamID
            try await resources.reserve(streamID: streamID)

            // Collect the upload before loading anything, so a truncated transfer
            // fails immediately instead of after a 225 MB model load.
            var collected: [Data] = []
            collected.reserveCapacity(manifest.frames.count)
            var complete = false
            var byteCount = 0
            while !complete {
                guard let frame = await connection.next(streamID: streamID) else {
                    throw RemoteError.invalidRequest("The reconstruction upload ended early.")
                }
                switch frame.opcode {
                case .uncoverInput:
                    byteCount += frame.payload.count
                    guard collected.count < manifest.frames.count,
                          byteCount <= RemoteLimits.maximumMessagePayload else { throw RemoteError.tooLarge }
                    collected.append(frame.data)
                case .uncoverInputEnd: complete = true
                default: throw RemoteError.invalidRequest("Unexpected \(frame.opcode.rawValue) during upload.")
                }
            }
            guard collected.count == manifest.frames.count else {
                throw RemoteError.invalidRequest("The upload sent \(collected.count) of \(manifest.frames.count) frames.")
            }
            // Immutable before the job closure captures it: a `var` would be a
            // concurrent capture, and a `let` array of `Data` is a value.
            let frames = collected

            await resources.run(streamID: streamID) { [weak self] in
                guard let self else { return }
                do {
                    try await self.reconstruct(streamID: streamID, manifest: manifest, algorithm: algorithm,
                                              allowUnsafeMemory: allowUnsafeMemory, frames: frames)
                } catch {
                    try? await self.connection.sendError(
                        RemoteEnvelope(opcode: .uncoverResultBegin, streamID: streamID, payload: []),
                        Self.describe(error))
                }
                await self.resources.finish(streamID: streamID)
                await self.noteJob(nil)
            }
        } catch {
            try? await connection.sendError(envelope, Self.describe(error))
        }
    }

    /// Runs the backend over a temporary job directory and streams the frames
    /// back one at a time, so a client can start compositing before the last
    /// frame has been reconstructed.
    private func reconstruct(streamID: UInt32, manifest: UncoverManifest, algorithm: UncoverAlgorithm,
                             allowUnsafeMemory: Bool, frames: [Data]) async throws {
        try Task.checkCancellation()
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("waifu-remote-uncover-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        for name in ["original", "degraded", "baseline", "reconstructed"] {
            try FileManager.default.createDirectory(at: directory.appendingPathComponent(name),
                                                    withIntermediateDirectories: true)
        }
        // Written as received: the same PNG bytes the client produced, so the
        // backend reads precisely what a local run would have read.
        for (index, data) in frames.enumerated() {
            try data.write(to: RegionImages.frameURL(directory, "degraded", index))
        }
        try JSONEncoder().encode(manifest).write(to: directory.appendingPathComponent("manifest.json"),
                                                 options: .atomic)

        await self.noteJob("\(algorithm.label) reconstructing")
        guard await ModelLibrary.shared.active == nil else { throw RemoteError.busy }
        await ModelLibrary.shared.acquire(algorithm)
        let progressRelay = RemoteProgressRelay(connection: connection, streamID: streamID)
        let result: UncoverResult
        do {
            result = try await catalog.backend(algorithm).reconstruct(
                UncoverPrepared(directory: directory, manifest: manifest),
                allowUnsafeMemory: allowUnsafeMemory,
                progress: { message in progressRelay.send(message) })
            await progressRelay.flush()
        } catch {
            await ModelLibrary.shared.release(algorithm)
            throw error
        }
        await ModelLibrary.shared.release(algorithm)

        try await connection.send(.uncoverResultBegin, control: .uncoverResultBegin(
            model: result.model, seconds: result.seconds, outputScale: result.outputScale), streamID: streamID)
        for frame in manifest.frames {
            let url = RegionImages.frameURL(directory, "reconstructed", frame.index)
            try await connection.send(.uncoverResult, payload: [UInt8](try Data(contentsOf: url)), streamID: streamID)
        }
        try await connection.send(.uncoverResultEnd, payload: [], streamID: streamID)
    }
}
/// Chains progress sends so the terminal result cannot overtake a callback.
private final class RemoteProgressRelay: @unchecked Sendable {
    private let lock = NSLock()
    private var tail: Task<Void, Never>?
    private let connection: RemoteConnection
    private let streamID: UInt32

    init(connection: RemoteConnection, streamID: UInt32) {
        self.connection = connection
        self.streamID = streamID
    }

    func send(_ message: String) {
        lock.withLock {
            let previous = tail
            tail = Task { [connection, streamID] in
                await previous?.value
                try? await connection.send(.uncoverProgress, control: .progress(message: message), streamID: streamID)
            }
        }
    }

    func flush() async {
        let task = lock.withLock { tail }
        await task?.value
    }
}
