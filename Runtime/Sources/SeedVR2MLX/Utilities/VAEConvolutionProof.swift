import Foundation
import MLX
import MLXNN
import MLXRandom

public enum SeedVR2VAEConvolutionProof {
    public static func run() {
        Memory.cacheLimit = 64 * 1024 * 1024
        MLXRandom.seed(91)
        defer {
            SeedVR2Runtime.causalState = .disabled
            SeedVR2Runtime.disableVAECausalFusion = false
            SeedVR2Runtime.disableVAEPointwiseConv = false
            SeedVR2Runtime.disableVAEResidualFusion = false
        }
        for shape in [[2, 1, 7, 9, 32], [1, 4, 16, 16, 128], [1, 5, 256, 256, 128]] {
            let x = MLXRandom.normal(shape).asType(.float16).transposed(0, 1, 3, 2, 4)
            for active in [false, true] {
                let history = active ? MLXRandom.normal([shape[0], 2, shape[3], shape[2], shape[4]]).asType(.float16) : nil
                let prefix = history ?? repeated(x[0..., 0..<1], count: 2, axis: 1)
                let reference = concatenated([prefix, x], axis: 1)
                for tailCount in [0, 1, 2] {
                    let (prepared, tail) = SeedVR2VAECausalPreparation.apply(x, history: history, padding: 2, tailCount: tailCount)
                    eval(reference, prepared)
                    precondition(max(abs(reference - prepared)).item(Float.self) == 0, "Causal input must agree exactly")
                    if let tail {
                        let expected = reference[0..., (reference.shape[1] - tailCount)...]
                        precondition(max(abs(expected - tail)).item(Float.self) == 0, "Compact causal history must agree exactly")
                    }
                }
            }
            Memory.clearCache()
        }
        print("Causal padding/history preparation agrees exactly for all shapes and states")

        for (shape, outputs) in [([2, 3, 7, 9, 128], 256), ([1, 2, 32, 32, 512], 4096), ([1, 5, 64, 64, 128], 512)] {
            let channels = shape.last!
            let x = MLXRandom.normal(shape).asType(.float16)
            let weight = (MLXRandom.normal([outputs, 1, 1, 1, channels]) * 0.02).asType(.float16)
            eval(x, weight)
            func reference() -> MLXArray { convGeneral(x, weight, strides: [1, 1, 1], padding: [0, 0, 0]) }
            func candidate() -> MLXArray {
                matmul(x.reshaped([-1, channels]), weight.reshaped([outputs, channels]).T)
                    .reshaped(Array(shape[..<4]) + [outputs])
            }
            let a = reference(), b = candidate(); eval(a, b)
            let error = max(abs(a.asType(.float32) - b.asType(.float32))).item(Float.self)
            let mae = mean(abs(a.asType(.float32) - b.asType(.float32))).item(Float.self)
            precondition(error.isFinite && error <= 0.01 && mae <= 0.001, "Pointwise convolution parity failure")
            func time(_ op: () -> MLXArray) -> Double {
                let start = ProcessInfo.processInfo.systemUptime
                for _ in 0..<3 { eval(op()) }
                return (ProcessInfo.processInfo.systemUptime - start) / 3
            }
            print("Pointwise \(shape) → \(outputs): max \(error), mean \(mae), convolution \(time(reference)), matmul \(time(candidate))")
            Memory.clearCache()
        }

        // Exercise the real module across initialization and two history-consuming slices.
        for (stride, padding, paddingCausal) in [(1, 1, false), (2, 1, false), (1, 0, true)] {
            let conv = CausalConv3d(32, 32, stride: (stride, 1, 1),
                padding: (padding, 1, 1), usePaddingCausal: paddingCausal)
            try! conv.update(parameters: ModuleParameters.unflattened([
                "weight": (MLXRandom.normal([32, 3, 3, 3, 32]) * 0.01).asType(.float16),
                "bias": (MLXRandom.normal([32]) * 0.01).asType(.float16)]), verify: [.all])
            let inputs = [padding == 0 ? 3 : 1, 4, 4].map { MLXRandom.normal([2, 32, $0, 9, 7]).asType(.float16) }
            func sequence(disabled: Bool) -> [MLXArray] {
                conv.memory = nil
                SeedVR2Runtime.disableVAECausalFusion = disabled
                return inputs.enumerated().map { index, x in
                    SeedVR2Runtime.causalState = index == 0 ? .initializing : .active
                    return conv(x)
                }
            }
            let a = sequence(disabled: true), b = sequence(disabled: false)
            for (left, right) in zip(a, b) {
                precondition(max(abs(left - right)).item(Float.self) == 0, "Causal module sequence must agree exactly")
            }
        }
        for shape in [[2, 3, 7, 9, 128], [1, 4, 256, 256, 128]] {
            let x = (MLXRandom.normal(shape) * 10).asType(.float16)
            let bias = (MLXRandom.normal([shape.last!]) * 0.2).asType(.float16)
            let residual = MLXRandom.normal(shape).asType(.float16)
            let a = x + bias
            eval(a) // Preserve the original F16 intermediate rounding boundary.
            let reference = a + residual, fused = SeedVR2FusedOps.vaeBiasResidual(x, bias, residual)
            eval(reference, fused)
            precondition(max(abs(reference - fused)).item(Float.self) == 0, "Bias/residual fusion must retain F16 rounding")
            Memory.clearCache()
        }
        print("Fused VAE bias/residual preserves both F16 additions exactly")
        print("Native VAE convolution/preparation proof passed")
    }
}
