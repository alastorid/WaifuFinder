import MLX
import MLXFast

/// Produce the causal convolution input and compact next-slice history together.
/// Inputs and history stay channel-last, matching the convolution's native layout.
enum SeedVR2VAECausalPreparation {
    private static let kernel = MLXFast.metalKernel(name: "seedvr2_vae_causal_input_tail",
        inputNames: ["x", "history"], outputNames: ["out", "tail"], source: """
        uint i = thread_position_in_grid.x;
        if (i < B * (T + PAD) * SPAN) {
            uint b = i / ((T + PAD) * SPAN), t = (i / SPAN) % (T + PAD), p = i % SPAN;
            if (t < PAD) {
                out[i] = HISTORY ? history[(b * PAD + t) * SPAN + p] : x[b * T * SPAN + p];
            } else { out[i] = x[(b * T + t - PAD) * SPAN + p]; }
        }
        if (i < B * TAIL * SPAN) {
            uint b = i / (TAIL * SPAN), t = T + PAD - TAIL + (i / SPAN) % TAIL, p = i % SPAN;
            if (t < PAD) {
                tail[i] = HISTORY ? history[(b * PAD + t) * SPAN + p] : x[b * T * SPAN + p];
            } else { tail[i] = x[(b * T + t - PAD) * SPAN + p]; }
        }
        """)

    static func apply(_ x: MLXArray, history: MLXArray?, padding: Int, tailCount: Int) -> (MLXArray, MLXArray?) {
        precondition(x.ndim == 5 && padding > 0 && tailCount >= 0 && tailCount <= x.shape[1] + padding)
        let b = x.shape[0], t = x.shape[1], span = x.shape[2] * x.shape[3] * x.shape[4]
        if let history { precondition(history.shape == [b, padding] + Array(x.shape[2...]) && history.dtype == x.dtype) }
        var shape = x.shape; shape[1] += padding
        var tailShape = x.shape; tailShape[1] = tailCount
        let result = kernel([x, history ?? MLXArray.zeros([0], dtype: x.dtype)],
            template: [("B", b), ("T", t), ("SPAN", span), ("PAD", padding), ("TAIL", tailCount), ("HISTORY", history != nil)],
            grid: (b * (t + padding) * span, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [shape, tailShape], outputDTypes: [x.dtype, x.dtype])
        return (result[0], tailCount > 0 ? result[1] : nil)
    }
}
