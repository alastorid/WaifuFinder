import Foundation
import MLX
import MLXNN

/// Resident mode avoids reloading weights. Streamed mode retains only the active
/// VAE stage or transformer root plus one block, with real temporal context intact.
public final class SeedVR2VideoUpscaler {
    private let checkpoint: SeedVR2VideoCheckpoint
    private let vae: SeedVR2VAE?
    private let transformer: SeedVR2Transformer?
    private let text: MLXArray
    public let streamed: Bool
    public init(directory: URL, embedding: URL, streamWeights: Bool? = nil) throws {
        let checkpoint = try SeedVR2VideoCheckpoint(directory: directory)
        self.checkpoint = checkpoint
        streamed = streamWeights ?? (checkpoint.variant == "seedvr2-7b")
        if streamed { vae = nil; transformer = nil }
        else {
            let transformer = SeedVR2Transformer(checkpoint.config), vae = SeedVR2VAE(fusedAttention: checkpoint.fusedVAE)
            try checkpoint.apply(checkpoint.weights("transformer"), to: transformer)
            try checkpoint.apply(checkpoint.weights("vae"), to: vae)
            self.transformer = transformer; self.vae = vae
        }
        let embeddings = try loadArrays(url: embedding)
        guard let value = embeddings["embedding"] else { throw SeedVR2VideoCheckpoint.CheckpointError.invalid("Missing embedding") }
        text = value.ndim == 2 ? value.expandedDimensions(axis: 0) : value
        SeedVR2Runtime.materialize(text)
    }
    private func loadVAE(encoding: Bool) throws -> SeedVR2VAE {
        let model = SeedVR2VAE(fusedAttention: checkpoint.fusedVAE)
        // Encoding never uses decoder weights, and decoding never uses encoder
        // weights. Avoid reading/allocating the inactive half in streamed mode.
        let prefix = encoding ? "encoder." : "decoder."
        try checkpoint.apply(checkpoint.weights("vae") { $0.hasPrefix(prefix) }, to: model, verifyAll: false)
        return model
    }
    private func encode(_ frames: MLXArray) throws -> MLXArray {
        let model = try vae ?? loadVAE(encoding: true)
        let condition = SeedVR2LatentCreator.condition(model.encode(frames))
        SeedVR2Runtime.materialize(condition)
        return condition
    }
    private func predict(_ input: MLXArray) throws -> MLXArray {
        if let transformer { return transformer(input, text, timestep: MLXArray(Float(1000))) }
        let model = SeedVR2Transformer(checkpoint.config)
        try checkpoint.apply(checkpoint.weights("transformer") { !$0.hasPrefix("blocks.") }, to: model, verifyAll: false)
        let predicted = try model.streamed(input, text, timestep: MLXArray(Float(1000))) { index, block in
            try self.checkpoint.loadBlock(index, into: block)
        }
        SeedVR2Runtime.materialize(predicted)
        return predicted
    }
    private func decode(_ latent: MLXArray) throws -> MLXArray {
        let model = try vae ?? loadVAE(encoding: false)
        let result = model.decode(latent)
        SeedVR2Runtime.materialize(result)
        return result
    }
    public func restore(_ frames: MLXArray, noise: MLXArray) throws -> MLXArray {
        SeedVR2Runtime.progress?("Encoding temporal context")
        let condition = try SeedVR2Runtime.measure("encode") { try encode(frames) }
        Memory.clearCache()
        SeedVR2Runtime.progress?("Reconstructing temporal detail\(streamed ? " · streamed weights" : "")")
        let predicted = try SeedVR2Runtime.measure("transformer") {
            let value = try predict(concatenated([noise, condition], axis: 1))
            SeedVR2Runtime.materialize(value)
            return value
        }
        let latent = noise - predicted
        SeedVR2Runtime.materialize(latent)
        Memory.clearCache()
        SeedVR2Runtime.progress?("Decoding temporal result")
        return try SeedVR2Runtime.measure("decode") { try decode(latent) }
    }
}
