// Attention and AdaIN adapted from belisoful/InferKit, MIT.
// Copyright (c) 2026 Belisoful. See THIRD_PARTY_LICENSES.md.
// This is a PGTFormer component, not a complete restoration backend.
import Foundation
import MLX
import MLXNN

public enum FaceReferenceError: Error { case incompatibleWeights(String), incompatibleInput }

/// Original PGTFormer TransformerSALayer: pre-norm, positional Q/K only,
/// packed QKV projection, exact GELU, 512 dimensions and eight heads.
/// Operates on [tokens, dimensions] for one temporal window (batch one).
public struct PGTCodeFormerLayer {
    private let weights: [String: MLXArray]
    public init(weights: [String: MLXArray]) throws {
        let shapes = ["self_attn.in_proj_weight": [1536,512], "self_attn.in_proj_bias": [1536],
            "self_attn.out_proj.weight": [512,512], "self_attn.out_proj.bias": [512],
            "norm1.weight": [512], "norm1.bias": [512], "norm2.weight": [512], "norm2.bias": [512],
            "linear1.weight": [1024,512], "linear1.bias": [1024],
            "linear2.weight": [512,1024], "linear2.bias": [512]]
        guard Set(weights.keys) == Set(shapes.keys) else {
            throw FaceReferenceError.incompatibleWeights("Missing or unexpected TransformerSALayer tensors")
        }
        for (key, shape) in shapes {
            guard weights[key]?.shape == shape, weights[key]?.dtype == .float32 else {
                throw FaceReferenceError.incompatibleWeights(key)
            }
        }
        self.weights = weights
    }
    private func norm(_ x: MLXArray, _ name: String) -> MLXArray {
        let center = x - mean(x, axis: -1, keepDims: true)
        return center * rsqrt(mean(center * center, axis: -1, keepDims: true) + 1e-5)
            * weights[name + ".weight"]! + weights[name + ".bias"]!
    }
    private func linear(_ x: MLXArray, _ name: String) -> MLXArray {
        matmul(x, weights[name + ".weight"]!.transposed()) + weights[name + ".bias"]!
    }
    public func callAsFunction(_ x: MLXArray, positions: MLXArray) throws -> MLXArray {
        guard x.ndim == 2, x.shape[1] == 512, x.shape == positions.shape else {
            throw FaceReferenceError.incompatibleInput
        }
        let tokens = x.shape[0], normalized = norm(x, "norm1")
        func project(_ input: MLXArray, _ slot: Int) -> MLXArray {
            let range = slot * 512 ..< (slot + 1) * 512
            return (matmul(input, weights["self_attn.in_proj_weight"]![range].transposed())
                + weights["self_attn.in_proj_bias"]![range])
                .reshaped([tokens,8,64]).transposed(1,0,2)
        }
        let q = project(normalized + positions, 0), k = project(normalized + positions, 1)
        let v = project(normalized, 2)
        let probabilities = softmax(matmul(q, k.transposed(0,2,1)) * Float(0.125), axis: -1)
        let attended = matmul(probabilities, v).transposed(1,0,2).reshaped([tokens,512])
        let residual = x + linear(attended, "self_attn.out_proj")
        return residual + linear(gelu(linear(norm(residual, "norm2"), "linear1")), "linear2")
    }
}

/// NHWC; PyTorch's unbiased variance, epsilon inside the square root.
public func pgtAdaptiveInstanceNormalize(_ content: MLXArray, style: MLXArray) throws -> MLXArray {
    guard content.ndim == 4, content.shape == style.shape, content.shape[1] * content.shape[2] > 1 else {
        throw FaceReferenceError.incompatibleInput
    }
    func statistics(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let average = mean(x, axes: [1,2], keepDims: true)
        let centered = x - average
        let variance = sum(centered * centered, axes: [1,2], keepDims: true)
            / Float(x.shape[1] * x.shape[2] - 1)
        return (average, sqrt(variance + 1e-5))
    }
    let (cm, cs) = statistics(content), (sm, ss) = statistics(style)
    return (content - cm) / cs * ss + sm
}
