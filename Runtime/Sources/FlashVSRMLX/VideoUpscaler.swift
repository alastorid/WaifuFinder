import Foundation
import MLX

public final class FlashVSRVideoUpscaler {
    private let projection: FlashProjection, transformer: FlashTransformer, decoder: FlashDecoder
    public var progress: (String) -> Void = { print($0) }
    public var observer: ((String, MLXArray) -> Void)? {
        didSet {
            transformer.observer = observer
            projection.observer = observer
        }
    }
    public init(directory: URL) throws {
        let manifest =
            try JSONSerialization.jsonObject(
                with: Data(contentsOf: directory.appendingPathComponent("manifest.json"))) as? [String: Any]
        guard let hashes = manifest?["fileSHA256"] as? [String: String],
            let expected = hashes["prompt.safetensors"],
            try FlashWeights.digest(directory.appendingPathComponent("prompt.safetensors")) == expected
        else { throw FlashError.invalid("Prompt checksum mismatch") }
        projection = FlashProjection(try FlashWeights(directory: directory, component: "projection"))
        decoder = FlashDecoder(try FlashWeights(directory: directory, component: "decoder"))
        let context = try loadArrays(url: directory.appendingPathComponent("prompt.safetensors"))
        guard Set(context.keys) == ["context"], let prompt = context["context"],
            prompt.shape == [1, 512, 4096], prompt.dtype == .bfloat16
        else { throw FlashError.invalid("Invalid FlashVSR prompt") }
        transformer = FlashTransformer(
            weights: try FlashWeights(directory: directory, component: "dit"), context: prompt)
    }
    /// First chunk observes 25 conditioning frames and emits 21; subsequent chunks
    /// observe eight and emit eight. Four-frame lookahead and causal memories persist.
    /// `frame` supplies bicubic conditioning at the requested output geometry in THWC.
    public func restore(
        frameCount: Int, width: Int, height: Int, noise: MLXArray,
        frame: (Int) throws -> MLXArray, emit: (Int, MLXArray) throws -> Void
    ) throws {
        guard frameCount >= 5, width % 128 == 0, height % 128 == 0 else {
            throw FlashError.invalid("FlashVSR requires five frames and 128-aligned geometry")
        }
        projection.reset()
        decoder.reset()
        transformer.resetTemporal()
        let following = max(0, (frameCount - 21 + 7) / 8)
        let chunks = 1 + following
        precondition(noise.shape == [1, 6 + following * 2, height / 8, width / 8, 16])
        var visible = 0
        func read(_ start: Int, _ count: Int) throws -> MLXArray {
            let frames = try (start..<(start + count)).map { try frame(min($0, frameCount - 1)) }
            return stacked(frames, axis: 0).asType(.bfloat16)
        }
        for chunk in 0..<chunks {
            var parts: [MLXArray] = []
            if chunk == 0 {
                for i in 0..<7 {
                    let input = try read(max(0, i * 4 - 3), i == 0 ? 1 : 4)
                    if let value = projection(input) { parts.append(value) }
                }
            } else {
                for i in 0..<2 {
                    if let value = projection(try read(chunk * 8 + 17 + i * 4, 4)) { parts.append(value) }
                }
            }
            let projected = concatenated(parts, axis: 1)
            observer?("projection_\(chunk)", projected)
            let start = chunk == 0 ? 0 : 4 + chunk * 2
            let count = chunk == 0 ? 6 : 2
            let input = noise[0..., start..<(start + count)]
            let predicted = transformer.predict(
                noise: input, projection: projected, index: chunk, progress: progress)
            let latent = (input - predicted)[0]
            flashEval(latent)
            observer?("latent_\(chunk)", latent)
            let offset = chunk == 0 ? 0 : 21 + (chunk - 1) * 8
            let outputCount = chunk == 0 ? 21 : 8
            let condition = try read(offset, outputCount)
            // Per-frame ADAIN is the official Tiny/Long default, despite its wavelet name.
            var local = 0
            var decodeError: Error?
            decoder.decode(latents: latent, conditioning: condition) { output in
                defer { local += 1 }
                guard decodeError == nil else { return }
                let index = offset + local
                let style = condition[local]
                let content = output.asType(.float32)
                let reference = style.asType(.float32)
                let cm = mean(content, axes: [0, 1], keepDims: true)
                let sm = mean(reference, axes: [0, 1], keepDims: true)
                let cs = sqrt(
                    mean((content - cm) * (content - cm), axes: [0, 1], keepDims: true) + Float(1e-5))
                let ss = sqrt(
                    mean((reference - sm) * (reference - sm), axes: [0, 1], keepDims: true) + Float(1e-5))
                let corrected = clip((content - cm) / cs * ss + sm, min: -1, max: 1)
                flashEval(corrected)
                if index < frameCount {
                    do {
                        try emit(index, corrected)
                        visible += 1
                    } catch { decodeError = error }
                }
            }
            if let decodeError { throw decodeError }
            guard local == outputCount else {
                throw FlashError.invalid("Tiny decoder output count mismatch: \(local)")
            }
            Memory.clearCache()
        }
        guard visible == frameCount else { throw FlashError.invalid("FlashVSR dropped a visible frame") }
    }
}
