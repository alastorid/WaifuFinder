import Foundation
import ImageIO

/// A `TemporalVideoUpscaler` that reconstructs on the server.
///
/// The contract is written in terms of a *prepared job directory*, and this
/// backend is what makes that contract survive the trip: the client owns its
/// prepared directory and holds the `original/` frames that compositing needs, so
/// the only thing that leaves the machine is `manifest.json` and the
/// `degraded/` frames the model actually reads. The returned frames are written
/// into the client's own `reconstructed/` directory, and every downstream stage
/// — `UncoverRenderer`, `UncoverVideoExporter`, the comparison modes — then works
/// on local files exactly as it does after a local run.
struct RemoteUncoverBackend: TemporalVideoUpscaler {
    private let client: RemoteClient
    private let algorithm: UncoverAlgorithm

    init(algorithm: UncoverAlgorithm, client: RemoteClient) {
        self.algorithm = algorithm
        self.client = client
    }

    /// Names the server and the algorithm, so two caches cannot collide and a
    /// result directory says where it came from.
    var cacheIdentity: String {
        "remote:\(algorithm.rawValue):\(client.remoteIdentity)"
    }

    /// The server's own limits, translated into the local type.
    ///
    /// Reporting the server's `scales` and `minimumFrames` rather than the local
    /// backend's is what makes `UncoverSession.canReconstruct` correct on the
    /// client: a region the server cannot reconstruct is refused here, before a
    /// job is ever opened.
    func capabilities() async -> UpscalerCapabilities {
        await client.capabilities?.localCapabilities(algorithm)
            ?? UpscalerCapabilities(available: false, message: "Not connected to an inference server.",
                                    scales: UncoverScale.allCases, minimumFrames: 5,
                                    requiresUnsafeMemoryOptIn: false)
    }

    /// Sends the job and receives the reconstructed frames.
    func reconstruct(_ prepared: UncoverPrepared, allowUnsafeMemory: Bool,
                     progress: @escaping @Sendable (String) -> Void) async throws -> UncoverResult {
        let manifest = prepared.manifest
        guard manifest.frames.count <= 300 else {
            throw UncoverError.invalid("The interval exceeds the 300-frame processing limit.")
        }
        progress("Sending \(manifest.frames.count) frames to the inference server")

        let outcome = try await client.runUncoverJob(
            manifest: manifest, algorithm: algorithm, allowUnsafeMemory: allowUnsafeMemory,
            prepared: prepared, progress: progress)

        // Written into the client's own prepared directory under
        // `reconstructed/`, which is exactly where a local run would have left
        // them — so the renderer and the exporter need no remote branch at all.
        try Task.checkCancellation()
        return UncoverResult(prepared: prepared, model: outcome.model,
                             seconds: outcome.serverSeconds, outputScale: outcome.outputScale)
    }
}

extension RemoteClient {
    /// The multi-message exchange a reconstruction is: a manifest and N input
    /// frames go up, a result header and N reconstructed frames come back, all on
    /// one stream so they cannot be interleaved with another request's.
    func runUncoverJob(manifest: UncoverManifest, algorithm: UncoverAlgorithm, allowUnsafeMemory: Bool,
                       prepared: UncoverPrepared,
                       progress: @escaping @Sendable (String) -> Void) async throws
        -> (model: String, serverSeconds: Double, outputScale: Int) {
        guard let connection else { throw RemoteError.framing("Not connected to an inference server.") }
        if let request = manifest.realESRGANImage {
            guard algorithm == request.variant.algorithm, capabilities?.offersRealESRGAN(request.variant) == true else {
                throw RemoteError.modelUnavailable("This server does not advertise the matching RealESRGAN photo model and runtime.")
            }
            try request.validate(request.variant)
        }
        let streamID = try await connection.beginRequest()
        let timeout: Task<Void,Never>? = manifest.realESRGANImage == nil ? nil : Task {
            do { try await Task.sleep(for:.seconds(120)) } catch { return }
            await connection.cancelRequest(streamID)
        }
        defer { timeout?.cancel() }

        do {
            let result = try await withTaskCancellationHandler {
                try await exchangeUncover(connection: connection, streamID: streamID, manifest: manifest,
                                          algorithm: algorithm, allowUnsafeMemory: allowUnsafeMemory,
                                          prepared: prepared, progress: progress)
            } onCancel: { Task {
                if manifest.realESRGANImage != nil { await connection.cancelRequest(streamID) }
                else { await connection.stop() }
            } }
            await connection.endRequest()
            return result
        } catch {
            if manifest.realESRGANImage != nil { await connection.cancelRequest(streamID) }
            await connection.endRequest()
            throw error
        }
    }

    private func exchangeUncover(connection: RemoteConnection, streamID: UInt32,
                                manifest: UncoverManifest, algorithm: UncoverAlgorithm,
                                allowUnsafeMemory: Bool, prepared: UncoverPrepared,
                                progress: @escaping @Sendable (String) -> Void) async throws
        -> (model: String, serverSeconds: Double, outputScale: Int) {
        guard (1...300).contains(manifest.frames.count) else { throw RemoteError.tooLarge }
        var uploaded = 0

        // Every input frame is read from the prepared directory and written to
        // the socket as the same PNG bytes that produced it. The `original/` and
        // `baseline/` directories are deliberately left behind: compositing and
        // export need them, and the model does not.
        try await connection.send(.uncoverBegin, control: .uncoverBegin(
            manifest: manifest, algorithm: algorithm.rawValue, allowUnsafeMemory: allowUnsafeMemory),
            streamID: streamID)

        for frame in manifest.frames {
            try Task.checkCancellation()
            let url = RegionImages.frameURL(prepared.directory, "degraded", frame.index)
            let data = try Data(contentsOf: url)
            uploaded += data.count
            guard uploaded <= RemoteLimits.maximumMessagePayload else { throw RemoteError.tooLarge }
            try await connection.send(.uncoverInput, payload: [UInt8](data), streamID: streamID)
        }
        try await connection.send(.uncoverInputEnd, payload: [], streamID: streamID)

        let directory = prepared.directory.appendingPathComponent("reconstructed", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        var model = algorithm.label
        var serverSeconds: Double = 0
        var outputScale = 1
        var received = 0
        var receivedHeader = false
        let expected = manifest.frames.count

        while received < expected {
            guard let envelope = await connection.next(streamID: streamID) else {
                throw RemoteError.framing("The server ended the reconstruction early.")
            }
            if envelope.isError { throw RemoteConnection.remoteError(from: envelope) }
            switch envelope.opcode {
            case .uncoverProgress:
                if case .progress(let message) = try envelope.control() { progress(message) }
            case .uncoverResultBegin:
                guard let control = try? envelope.control(),
                      case .uncoverResultBegin(let name, let seconds, let scale) = control else {
                    throw RemoteError.framing("The server sent a malformed result header.")
                }
                guard !receivedHeader, scale > 0, scale <= 64, seconds.isFinite, seconds >= 0 else {
                    throw RemoteError.framing("Invalid result metadata")
                }
                if let request = manifest.realESRGANImage {
                    guard name == request.variant.modelName, scale == request.variant.scale else {
                        throw RemoteError.framing("RealESRGAN response model identity or scale differs.")
                    }
                }
                receivedHeader = true
                model = name
                serverSeconds = seconds
                outputScale = scale
            case .uncoverResult:
                guard receivedHeader, received < expected else {
                    throw RemoteError.framing("The server sent more frames than the manifest declares.")
                }
                if let request = manifest.realESRGANImage {
                    guard let source = CGImageSourceCreateWithData(envelope.data as CFData,nil),
                          let properties = CGImageSourceCopyPropertiesAtIndex(source,0,nil) as? [CFString:Any],
                          properties[kCGImagePropertyPixelWidth] as? Int == manifest.regionWidth * request.variant.scale,
                          properties[kCGImagePropertyPixelHeight] as? Int == manifest.regionHeight * request.variant.scale else {
                        throw RemoteError.framing("RealESRGAN response has incorrect image dimensions.")
                    }
                }
                try envelope.data.write(to: RegionImages.frameURL(prepared.directory, "reconstructed", received))
                received += 1
                progress("Received \(received) of \(expected) reconstructed frames")
            default:
                throw RemoteError.framing("Unexpected \(envelope.opcode.rawValue) during reconstruction.")
            }
        }
        // The end marker must arrive; without it a truncated stream that happened
        // to have the right frame count would look complete.
        guard let end = await connection.next(streamID: streamID), end.opcode == .uncoverResultEnd else {
            throw RemoteError.framing("The server did not finish the reconstruction.")
        }
        try Task.checkCancellation()
        return (model, serverSeconds, outputScale)
    }
}
