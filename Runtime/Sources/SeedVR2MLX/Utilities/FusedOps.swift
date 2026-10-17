import MLX
import MLXNN

/// Compile small elementwise chains, rather than retaining an entire video graph.
enum SeedVR2FusedOps {
    static let swiGLU = compile { (gate: MLXArray, value: MLXArray) in silu(gate) * value }
    static let vaeBiasResidual = compile { (x: MLXArray, bias: MLXArray, residual: MLXArray) in
        (x + bias) + residual
    }
}
