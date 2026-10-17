import MLX
import MLXNN

/// Inference-only shape placeholders. Checkpoint loading replaces every parameter;
/// constructing random multi-billion-parameter training weights serves no purpose.
func seedVR2Linear(_ input: Int, _ output: Int, bias: Bool = true) -> Linear {
    Linear(weight: MLXArray.zeros([output, input], dtype: .bfloat16),
           bias: bias ? MLXArray.zeros([output], dtype: .bfloat16) : nil)
}
