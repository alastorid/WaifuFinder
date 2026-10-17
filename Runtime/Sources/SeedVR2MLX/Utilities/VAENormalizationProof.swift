import Foundation
import MLX
import MLXNN
import MLXRandom

public enum SeedVR2VAENormalizationProof {
    public static func run() {
        Memory.cacheLimit = 64 * 1024 * 1024
        MLXRandom.seed(71)
        for shape in [[2, 3, 7, 64], [1, 5, 17, 19, 128], [1, 1, 1, 512], [2, 4097, 32], [1, 5, 256, 256, 128]] {
            let channels = shape.last!
            let norm = groupNorm32(channels)
            try! norm.update(parameters: ModuleParameters.unflattened([
                "weight": (MLXRandom.normal([channels]) * 0.1 + 1).asType(.float16),
                "bias": (MLXRandom.normal([channels]) * 0.1).asType(.float16)]), verify: [.all])
            for offset: Float in [0, 100] {
                let x = (MLXRandom.normal(shape) + offset).asType(.float16)
                eval(x)
                for activate in [false, true] {
                    func reference() -> MLXArray {
                        let h = norm(x.asType(.float32)).asType(.float16)
                        return activate ? silu(h) : h
                    }
                    func fused() -> MLXArray { SeedVR2VAENormalization.apply(x, norm: norm, activate: activate) }
                    let a = reference(), b = fused()
                    eval(a, b)
                    let error = max(abs(a.asType(.float32) - b.asType(.float32))).item(Float.self)
                    let mae = mean(abs(a.asType(.float32) - b.asType(.float32))).item(Float.self)
                    precondition(error.isFinite && error <= 0.02 && mae <= 0.002, "Normalization parity failure")
                    func time(_ operation: () -> MLXArray) -> Double {
                        let start = ProcessInfo.processInfo.systemUptime
                        for _ in 0..<3 { eval(operation()) }
                        return (ProcessInfo.processInfo.systemUptime - start) / 3
                    }
                    print("VAE norm \(shape), offset \(offset), activation \(activate): max \(error), mean \(mae), reference \(time(reference)), fused \(time(fused))")
                }
                Memory.clearCache()
            }
            // Constant activations exercise zero variance without cancellation.
            let flat = MLXArray.full(shape, values: MLXArray(Float(100)), dtype: .float16)
            let a = norm(flat.asType(.float32)).asType(.float16)
            let b = SeedVR2VAENormalization.apply(flat, norm: norm, activate: false)
            precondition(max(abs(a - b)).item(Float.self) == 0, "Flat normalization must agree exactly")
            Memory.clearCache()
        }
        print("Native VAE normalization/activation proof passed")
    }
}
