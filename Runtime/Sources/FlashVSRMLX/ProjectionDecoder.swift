import Foundation
import MLX
import MLXNN

/// Layout is [batch,time,height,width,channels]. Separate caches belong to this clip.
final class FlashProjection {
    let weights: FlashWeights
    var cache1: MLXArray?, cache2: MLXArray?
    var initialized = false
    var iteration = 0
    var observer: ((String, MLXArray) -> Void)?
    init(_ weights: FlashWeights) { self.weights = weights }
    func reset() {
        cache1 = nil
        cache2 = nil
        initialized = false
        iteration = 0
    }
    func callAsFunction(_ input: MLXArray) -> MLXArray? {
        defer { iteration += 1 }
        var video = input
        if !initialized { video = concatenated([repeated(input[0..<1], count: 3, axis: 0), input], axis: 0) }
        let t = video.shape[0]
        let h = video.shape[1]
        let w = video.shape[2]
        let shuffled = video.reshaped(t, h / 16, 16, w / 16, 16, 3).transposed(0, 1, 3, 5, 2, 4).reshaped(
            1, t, h / 16, w / 16, 768)
        let tail1 = shuffled[0..., (t - 2)...] + MLXArray(0).asType(shuffled.dtype)
        var x = convolve(shuffled, cache: cache1, name: "conv1")
        cache1 = tail1
        if iteration < 2 { observer?("projection_conv1_\(iteration)", x) }
        x = normalize(x, "norm1")
        x = flashSiLU(x)
        if iteration < 2 { observer?("projection_norm1_\(iteration)", x) }
        let tail2 = x[0..., (x.shape[1] - 2)...] + MLXArray(0).asType(x.dtype)
        if !initialized {
            cache2 = tail2
            initialized = true
            flashEval(cache1!, cache2!)
            return nil
        }
        x = convolve(x, cache: cache2, name: "conv2")
        cache2 = tail2
        if iteration == 1 { observer?("projection_conv2", x) }
        x = flashSiLU(normalize(x, "norm2"))
        if iteration == 1 { observer?("projection_norm2", x) }
        let projected = weights.linear(x.reshaped(1, -1, 3072), "linear_layers.0")
        flashEval(projected, cache1!, cache2!)
        return projected
    }
    private func convolve(_ input: MLXArray, cache: MLXArray?, name: String) -> MLXArray {
        let x: MLXArray
        if let cache {
            x = concatenated([cache, input], axis: 1)
        } else {
            x = padded(input, widths: [0, [2, 0], 0, 0, 0], mode: .edge)
        }
        let spatial = padded(x, widths: [0, 0, 1, 1, 0], mode: .edge)
        return convGeneral(spatial, weights[name + ".weight"], strides: [2, 1, 1], padding: [0, 0, 0])
            + weights[name + ".bias"]
    }
    private func normalize(_ x: MLXArray, _ name: String) -> MLXArray {
        let f = x.asType(.float32)
        let denominator = maximum(sqrt(sum(f * f, axis: -1, keepDims: true)), MLXArray(Float(1e-12)))
        return (f / denominator).asType(x.dtype) * Float(sqrt(Double(x.shape[4])))
            * weights[name + ".gamma"].reshaped(x.shape[4])
    }
}

/// Tiny Conditional Decoder's recurrent MemBlocks and both temporal TGrow stages.
/// Processes each latent/frame depth-first, retaining only each layer's preceding input.
final class FlashDecoder {
    let weights: FlashWeights
    var memories: [Int: MLXArray] = [:]
    var initialized = false
    init(_ weights: FlashWeights) { self.weights = weights }
    func reset() {
        memories.removeAll()
        initialized = false
    }
    func decode(latents: MLXArray, conditioning: MLXArray, emit: (MLXArray) -> Void) {
        let t = conditioning.shape[0]
        let h = conditioning.shape[1]
        let w = conditioning.shape[2]
        var cond = conditioning
        if t % 4 != 0 {
            cond = concatenated([repeated(cond[0..<1], count: 4 - t % 4, axis: 0), cond], axis: 0)
        }
        // PyTorch order: channel, temporal subpixel, vertical subpixel, horizontal subpixel.
        let packed = cond.reshaped(cond.shape[0] / 4, 4, h / 8, 8, w / 8, 8, 3).transposed(
            0, 2, 4, 6, 1, 3, 5
        ).reshaped(-1, h / 8, w / 8, 768)
        precondition(packed.shape[0] == latents.shape[0])
        let x = concatenated([packed, latents], axis: -1)
        var trim = initialized ? 0 : 3
        func execute(_ input: MLXArray, _ index: Int) {
            if index == 27 {
                if trim > 0 { trim -= 1 } else { emit(input[0] * 2 - 1) }
                return
            }
            var y: MLXArray
            switch index {
            case 0: y = tanh(input / 3) * 3
            case 2, 4, 23, 25: y = relu(input)
            case 5, 6, 7, 11, 12, 13, 17, 18, 19:
                let past = memories[index] ?? zeros(like: input)
                let p = "decoder.\(index).conv."
                y = weights.conv2(
                    relu(
                        weights.conv2(
                            relu(weights.conv2(concatenated([input, past], axis: -1), p + "0")), p + "2")),
                    p + "4")
                y = relu(y + input)
                memories[index] = input + MLXArray(0).asType(input.dtype)
                flashEval(y, memories[index]!)
            case 8, 14, 20: y = repeated(repeated(input, count: 2, axis: 1), count: 2, axis: 2)
            case 9, 15, 21:
                let stride = index == 9 ? 1 : 2
                let grown = weights.conv2(input, "decoder.\(index).conv", padding: 0)
                    .reshaped(1, input.shape[1], input.shape[2], stride, input.shape[3]).transposed(
                        0, 3, 1, 2, 4)
                flashEval(grown)
                for i in 0..<stride { execute(grown[0, i].expandedDimensions(axis: 0), index + 1) }
                return
            default: y = weights.conv2(input, "decoder.\(index)")
            }
            flashEval(y)
            execute(y, index + 1)
        }
        for i in 0..<x.shape[0] { execute(x[i].expandedDimensions(axis: 0), 0) }
        initialized = true
    }
}
