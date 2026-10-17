import Foundation
import MLX
import MLXNN

final class FlashTransformer {
    let weights: FlashWeights
    let attention = FlashSparseAttention()
    var crossK: [MLXArray] = [], crossV: [MLXArray] = []
    var selfK: [MLXArray?] = Array(repeating: nil, count: 30),
        selfV: [MLXArray?] = Array(repeating: nil, count: 30)
    let time: MLXArray, modulation: MLXArray
    var observer: ((String, MLXArray) -> Void)?

    init(weights: FlashWeights, context: MLXArray) {
        self.weights = weights
        // Reference uses double precision for the sinusoid before casting to BF16.
        let frequencies = (0..<128).map { 1000 * pow(10000, -Double($0) / 128) }
        let embedding = MLXArray(
            frequencies.map { Float(cos($0)) } + frequencies.map { Float(sin($0)) }, [1, 256]
        ).asType(.bfloat16)
        time = weights.linear(flashSiLU(weights.linear(embedding, "time_embedding.0")), "time_embedding.2")
        modulation = weights.linear(flashSiLU(time), "time_projection.1").reshaped(1, 6, 1536)
        let text = weights.linear(flashGELU(weights.linear(context, "text_embedding.0")), "text_embedding.2")
        flashEval(time, modulation, text)
        for i in 0..<30 {
            let p = "blocks.\(i).cross_attn."
            let k = rms(weights.linear(text, p + "k"), weights[p + "norm_k.weight"])
            let v = weights.linear(text, p + "v")
            flashEval(k, v)
            crossK.append(k)
            crossV.append(v)
        }
    }
    static func partition(_ x: MLXArray, f: Int, h: Int, w: Int) -> MLXArray {
        x.reshaped(1, f / 2, 2, h / 8, 8, w / 8, 8, 1536).transposed(0, 1, 3, 5, 2, 4, 6, 7).reshaped(
            -1, 128, 1536)
    }
    func resetTemporal() {
        selfK = Array(repeating: nil, count: 30)
        selfV = Array(repeating: nil, count: 30)
        Memory.clearCache()
    }
    static func reverse(_ x: MLXArray, f: Int, h: Int, w: Int) -> MLXArray {
        x.reshaped(1, f / 2, h / 8, w / 8, 2, 8, 8, 1536).transposed(0, 1, 4, 2, 5, 3, 6, 7).reshaped(
            1, f * h * w, 1536)
    }
    private func rotary(f: Int, h: Int, w: Int, start: Int) -> (MLXArray, MLXArray) {
        var cosine: [Float] = []
        var sine: [Float] = []
        cosine.reserveCapacity(f * h * w * 64)
        sine.reserveCapacity(f * h * w * 64)
        for t in start..<(start + f) {
            for y in 0..<h {
                for x in 0..<w {
                    for (position, dim) in [(t, 44), (y, 42), (x, 42)] {
                        for i in 0..<(dim / 2) {
                            let angle = Double(position) * pow(10000, -Double(i * 2) / Double(dim))
                            cosine.append(Float(cos(angle)))
                            sine.append(Float(sin(angle)))
                        }
                    }
                }
            }
        }
        return (MLXArray(cosine, [1, f * h * w, 1, 64]), MLXArray(sine, [1, f * h * w, 1, 64]))
    }
    private func rotate(_ x: MLXArray, _ c: MLXArray, _ s: MLXArray) -> MLXArray {
        let paired = x.asType(.float32).reshaped(1, x.shape[1], 12, 64, 2)
        let real = paired[.ellipsis, 0]
        let imag = paired[.ellipsis, 1]
        return stacked([real * c - imag * s, real * s + imag * c], axis: -1).reshaped(x.shape).asType(x.dtype)
    }
    func predict(noise: MLXArray, projection: MLXArray, index: Int, progress: (String) -> Void) -> MLXArray {
        let f = noise.shape[1]
        let h = noise.shape[2] / 2
        let w = noise.shape[3] / 2
        let spatial = h * w / 64
        precondition((index == 0 && f == 6) || (index > 0 && f == 2))
        var x =
            convGeneral(noise, weights["patch_embedding.weight"], strides: [1, 2, 2], padding: [0, 0, 0])
            + weights["patch_embedding.bias"]
        x = x.reshaped(1, f * h * w, 1536) + projection
        let (cosine, sine) = rotary(f: f, h: h, w: w, start: index == 0 ? 0 : 4 + index * 2)
        flashEval(x, cosine, sine)
        observer?("patch", x)
        observer?("time", time)
        observer?("time_modulation", modulation)
        let ratio = 2.0 * 768 * 1280 / Double(h * 16 * w * 16)
        let topK = Int(Double(spatial * spatial) * ratio) - 1
        for i in 0..<30 {
            progress("FlashVSR streaming chunk \(index+1) · transformer \(i+1)/30")
            let p = "blocks.\(i)."
            let m = weights[p + "modulation"] + modulation
            let qin = norm(x) * (1 + m[0..., 1]) + m[0..., 0]
            let a = p + "self_attn."
            let q = Self.partition(
                rotate(rms(weights.linear(qin, a + "q"), weights[a + "norm_q.weight"]), cosine, sine), f: f,
                h: h, w: w)
            var k = Self.partition(
                rotate(rms(weights.linear(qin, a + "k"), weights[a + "norm_k.weight"]), cosine, sine), f: f,
                h: h, w: w)
            var v = Self.partition(weights.linear(qin, a + "v"), f: f, h: h, w: w)
            if let previous = selfK[i], let pv = selfV[i] {
                k = concatenated([previous, k], axis: 0)
                v = concatenated([pv, v], axis: 0)
            }
            flashEval(q, k, v)
            let mask = attention.blockMask(
                q: q, k: k, height: h / 8, width: w / 8, temporalWindows: f / 2, topK: topK)
            if i == 0 {
                observer?("q", q)
                observer?("k", k)
                observer?("v", v)
                observer?("mask", mask)
            }
            let attended = attention.apply(q: q, k: k, v: v, mask: mask)
            if k.shape[0] / spatial > 3 {
                k = k[spatial...] + MLXArray(0).asType(k.dtype)
                v = v[spatial...] + MLXArray(0).asType(v.dtype)
            }
            selfK[i] = k
            selfV[i] = v
            x = x + m[0..., 2] * weights.linear(Self.reverse(attended, f: f, h: h, w: w), a + "o")
            let ca = p + "cross_attn."
            let query = rms(
                weights.linear(
                    norm(x, weight: weights[p + "norm3.weight"], bias: weights[p + "norm3.bias"]), ca + "q"),
                weights[ca + "norm_q.weight"]
            )
            .reshaped(1, -1, 12, 128).transposed(0, 2, 1, 3)
            let key = crossK[i].reshaped(1, -1, 12, 128).transposed(0, 2, 1, 3)
            let value = crossV[i].reshaped(1, -1, 12, 128).transposed(0, 2, 1, 3)
            let crossed = MLXFast.scaledDotProductAttention(
                queries: query, keys: key, values: value, scale: Float(1 / sqrt(128.0)), mask: nil
            )
            .transposed(0, 2, 1, 3).reshaped(1, -1, 1536)
            x = x + weights.linear(crossed, ca + "o")
            let input = norm(x) * (1 + m[0..., 4]) + m[0..., 3]
            x = x + m[0..., 5] * weights.linear(flashGELU(weights.linear(input, p + "ffn.0")), p + "ffn.2")
            flashEval(x, k, v)
            if i == 0 || i == 29 { observer?("block_\(i)", x) }
        }
        let hm = weights["head.modulation"] + time.expandedDimensions(axis: 1)
        x = weights.linear(norm(x) * (1 + hm[0..., 1]) + hm[0..., 0], "head.head")
        let predicted = x.reshaped(1, f, h, w, 1, 2, 2, 16).transposed(0, 1, 4, 2, 5, 3, 6, 7).reshaped(
            noise.shape)
        flashEval(predicted)
        observer?("predicted", predicted)
        return predicted
    }
}
