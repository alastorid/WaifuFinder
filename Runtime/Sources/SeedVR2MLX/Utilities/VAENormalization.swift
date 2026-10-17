import MLX
import MLXFast
import MLXNN

/// Hierarchical F32 Welford statistics over native channel-last F16 activations.
/// Avoids full-size F32 casts and grouped layout copies. Only tiny partials live
/// beside the input/output, and the epilogue preserves the F16 activation boundary.
enum SeedVR2VAENormalization {
    private static let reduce = """
        uint lane = tid % 32;
        for (uint offset = 16; offset > 0; offset >>= 1) {
            float rn = simd_shuffle_down(n, offset);
            float rm = simd_shuffle_down(mu, offset);
            float rv = simd_shuffle_down(m2, offset);
            if (lane + offset < 32 && rn > 0) {
                float total = n + rn, delta = rm - mu;
                m2 += rv + delta * delta * (n * rn / total);
                mu += delta * (rn / total); n = total;
            }
        }
        threadgroup float means[8], variances[8], counts[8];
        if (lane == 0) { means[tid / 32] = mu; variances[tid / 32] = m2; counts[tid / 32] = n; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 32) {
            mu = lane < 8 ? means[lane] : 0;
            m2 = lane < 8 ? variances[lane] : 0;
            n = lane < 8 ? counts[lane] : 0;
            for (uint offset = 16; offset > 0; offset >>= 1) {
                float rn = simd_shuffle_down(n, offset);
                float rm = simd_shuffle_down(mu, offset);
                float rv = simd_shuffle_down(m2, offset);
                if (lane + offset < 32 && rn > 0) {
                    float total = n + rn, delta = rm - mu;
                    m2 += rv + delta * delta * (n * rn / total);
                    mu += delta * (rn / total); n = total;
                }
            }
        }
        """
    private static let partial = MLXFast.metalKernel(name: "seedvr2_gn_welford_partial",
        inputNames: ["x"], outputNames: ["partial"], source: """
        uint tid = thread_position_in_threadgroup.x;
        uint part = threadgroup_position_in_grid.x;
        uint bg = thread_position_in_grid.y;
        uint batch = bg / 32, group = bg % 32;
        float n = 0, mu = 0, m2 = 0;
        for (uint j = part * 4096 + tid; j < min((part + 1) * 4096, uint(N)); j += 256) {
            uint index = batch * S * C + (j / (C / 32)) * C + group * (C / 32) + j % (C / 32);
            float value = float(x[index]), delta = value - mu;
            n += 1; mu += delta / n; m2 += delta * (value - mu);
        }
        """ + reduce + """
        if (tid == 0) { uint o = (bg * P + part) * 3; partial[o] = mu; partial[o+1] = m2; partial[o+2] = n; }
        """)
    private static let finish = MLXFast.metalKernel(name: "seedvr2_gn_welford_finish",
        inputNames: ["partial", "epsilon"], outputNames: ["stats"], source: """
        uint tid = thread_position_in_threadgroup.x, bg = thread_position_in_grid.y;
        float n = 0, mu = 0, m2 = 0;
        for (uint j = tid; j < P; j += 256) {
            uint o = (bg * P + j) * 3;
            float rn = partial[o+2], rm = partial[o], rv = partial[o+1];
            if (rn > 0) {
                float total = n + rn, delta = rm - mu;
                m2 += rv + delta * delta * (n * rn / total);
                mu += delta * (rn / total); n = total;
            }
        }
        """ + reduce + """
        if (tid == 0) { stats[bg*2] = mu; stats[bg*2+1] = metal::precise::rsqrt(m2 / n + epsilon); }
        """)
    private static let epilogue = MLXFast.metalKernel(name: "seedvr2_gn_affine_silu",
        inputNames: ["x", "stats", "weight", "bias"], outputNames: ["out"], source: """
        uint i = thread_position_in_grid.x;
        if (i >= B * S * C) return;
        uint c = i % C, bg = (i / (S * C)) * 32 + c / (C / 32);
        float h = (float(x[i]) - stats[bg*2]) * stats[bg*2+1];
        half z = half(h * float(weight[c]) + float(bias[c]));
        if (ACTIVATE) {
            auto q = 1 / (1 + metal::exp(metal::abs(z)));
            half sigmoid = z < 0 ? q : 1 - q;
            out[i] = z * sigmoid;
        } else { out[i] = z; }
        """)

    static func apply(_ x: MLXArray, norm: GroupNorm, activate: Bool) -> MLXArray {
        precondition(x.dtype == .float16 && norm.groupCount == 32 && norm.pytorchCompatible)
        let batch = x.shape[0], channels = x.shape.last!, spatial = x.size / (batch * channels)
        let count = spatial * (channels / 32), parts = (count + 4095) / 4096
        let templates: [(String, any KernelTemplateArg)] = [("C", channels), ("S", spatial), ("N", count), ("P", parts), ("B", batch)]
        let partials = partial([x], template: templates, grid: (parts * 256, batch * 32, 1),
            threadGroup: (256, 1, 1), outputShapes: [[batch, 32, parts, 3]], outputDTypes: [.float32])[0]
        let stats = finish([partials, MLXArray(norm.eps)], template: templates,
            grid: (256, batch * 32, 1), threadGroup: (256, 1, 1), outputShapes: [[batch, 32, 2]], outputDTypes: [.float32])[0]
        return epilogue([x, stats, norm.weight!, norm.bias!], template: templates + [("ACTIVATE", activate)],
            grid: (x.size, 1, 1), threadGroup: (256, 1, 1), outputShapes: [x.shape], outputDTypes: [.float16])[0]
    }
}
