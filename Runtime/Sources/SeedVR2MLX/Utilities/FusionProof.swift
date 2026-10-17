import Foundation
import MLX
import MLXNN

public enum SeedVR2FusionProof {
    public static func run() {
        // 7B pixel positions must be normalized per window. A three-position
        // window's centre is zero; language positions would rotate it by pi.
        let pixel = SeedVR2RoPE(dim: 64, pixel: true)
        try! pixel.update(parameters: ModuleParameters.unflattened([
            "freqs": MLXArray(Array(repeating: Float.pi, count: 10))]), verify: [.all])
        let input = MLXArray.ones([3, 1, 128])
        let (rotated, _) = pixel.applyVid(input, input, windowShapes: [[1, 1, 3]])
        eval(rotated)
        precondition(abs(rotated[1, 0, 40].item(Float.self) - 1) < 1e-5)
        precondition(abs(rotated[0, 0, 40].item(Float.self) + 1) < 1e-5)
        print("7B normalized pixel-position regression passed")
        MLXRandom.seed(42)
        for bits in [4, 8] {
            for dim in [2560] {
                let hidden = ((2 * dim * 4 / 3 + 255) / 256) * 256
                let gate = QuantizedLinear(weight: MLXRandom.normal([hidden, dim]) * 0.01, bias: nil, groupSize: 64, bits: bits)
                let value = QuantizedLinear(weight: MLXRandom.normal([hidden, dim]) * 0.01, bias: nil, groupSize: 64, bits: bits)
                let fused = SeedVR2ProjectionFusion.join([gate, value])!
                eval(gate, value, fused)
                for tokens in [64, 512] {
                    let x = MLXRandom.normal([1, tokens, dim]).asType(.bfloat16)
                    eval(x)
                    func reference() -> MLXArray { silu(gate(x)) * value(x) }
                    func optimized() -> MLXArray {
                        let both = fused(x)
                        return SeedVR2FusedOps.swiGLU(both[.ellipsis, 0..<hidden], both[.ellipsis, hidden..<(2 * hidden)])
                    }
                    let a = reference(), b = optimized(); eval(a, b)
                    let error = abs(a - b).max().item(Float.self)
                    precondition(error <= 0.016, "Packed projection fusion mismatch: \(error)")
                    func seconds(_ body: () -> MLXArray) -> Double {
                        for _ in 0..<2 { eval(body()) }
                        let start = Date()
                        for _ in 0..<8 { eval(body()) }
                        return Date().timeIntervalSince(start) / 8
                    }
                    let original = seconds(reference), joined = seconds(optimized)
                    print("Fusion q\(bits), dim \(dim), tokens \(tokens): max error \(error), separate \(original)s, joined \(joined)s, speedup \(original / joined)")
                }
                Memory.clearCache()
            }
        }
    }
}
