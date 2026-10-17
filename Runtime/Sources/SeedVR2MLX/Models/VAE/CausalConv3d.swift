// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
// Port of mflux seedvr2_vae/common/conv3d.py.
import MLX
import MLXNN

/// VAE activation precision after group-norms (mflux `ModelConfig.precision`).
enum VAEPrecision { static var dtype: DType = .bfloat16 }

/// Causal (in time) 3D conv. Weight layout [O, kt, kh, kw, I] (MLX NDHWC) — load directly.
public final class CausalConv3d: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray
    var memory: MLXArray? // compact NDHWC history
    let k: (Int, Int, Int), s: (Int, Int, Int), p: (Int, Int, Int)
    let causalTemporal: Bool, usePaddingCausal: Bool

    public init(_ inCh: Int, _ outCh: Int, kernel: (Int, Int, Int) = (3, 3, 3),
                stride: (Int, Int, Int) = (1, 1, 1), padding: (Int, Int, Int) = (1, 1, 1),
                causalTemporal: Bool = true, usePaddingCausal: Bool = false) {
        self.k = kernel; self.s = stride; self.p = padding
        self.causalTemporal = causalTemporal; self.usePaddingCausal = usePaddingCausal
        self._weight.wrappedValue = MLXArray.zeros([outCh, kernel.0, kernel.1, kernel.2, inCh])
        self._bias.wrappedValue = MLXArray.zeros([outCh])
        super.init()
    }

    public func callAsFunction(_ xIn: MLXArray, residual: MLXArray? = nil) -> MLXArray {
        var x = xIn  // [B,C,T,H,W]
        var tPad = p.0
        let hasCausalPadding = !usePaddingCausal || p.0 > 0
            || (SeedVR2Runtime.causalState == .active && memory != nil)
        if causalTemporal && k.0 > 1 && hasCausalPadding && !SeedVR2Runtime.disableVAECausalFusion {
            let history = SeedVR2Runtime.causalState == .active ? memory : nil
            let padding = history?.shape[1] ?? (usePaddingCausal ? 2 * p.0 : k.0 - 1)
            let tailCount = SeedVR2Runtime.causalState != .disabled && s.0 != k.0 ? k.0 - s.0 : 0
            if padding > 0 {
                let prepared = SeedVR2VAECausalPreparation.apply(x.transposed(0, 2, 3, 4, 1),
                    history: history, padding: padding, tailCount: tailCount)
                x = prepared.0.transposed(0, 4, 1, 2, 3)
                if let tail = prepared.1 { memory = tail }
            }
            tPad = 0
        } else if causalTemporal && k.0 > 1 {
            if SeedVR2Runtime.causalState == .active, let memory {
                x = concatenated([memory.asType(x.dtype).transposed(0, 4, 1, 2, 3), x], axis: 2)
            } else {
                let causalPad = usePaddingCausal ? 2 * p.0 : k.0 - 1
                if causalPad > 0 {
                    let first = x[0..., 0..., 0 ..< 1]
                    let pad = repeated(first, count: causalPad, axis: 2)
                    x = concatenated([pad, x], axis: 2)
                }
            }
            if SeedVR2Runtime.causalState != .disabled && s.0 != k.0 {
                // Materialize a compact causal tail, rather than retaining a full activation.
                let count = k.0 - s.0
                memory = (x[0..., 0..., (x.shape[2] - count)...] + MLXArray(0).asType(x.dtype)).transposed(0, 2, 3, 4, 1)
                eval(memory!)
            }
            tPad = 0
        }
        x = x.transposed(0, 2, 3, 4, 1)        // NDHWC
        var out: MLXArray
        // A unit-stride, unpadded 1×1×1 convolution is exactly a channel projection.
        if !SeedVR2Runtime.disableVAEPointwiseConv && k == (1, 1, 1) && s == (1, 1, 1) && p == (0, 0, 0) {
            let channels = x.shape[4], outputs = weight.shape[0]
            out = matmul(x.reshaped([-1, channels]), weight.reshaped([outputs, channels]).T)
                .reshaped(Array(x.shape[..<4]) + [outputs])
        } else {
            out = convGeneral(x, weight, strides: [s.0, s.1, s.2], padding: [tPad, p.1, p.2])
        }
        if let residual {
            out = SeedVR2FusedOps.vaeBiasResidual(out, bias, residual.transposed(0, 2, 3, 4, 1))
        } else { out = out + bias }
        out = out.transposed(0, 4, 1, 2, 3)
        SeedVR2Runtime.materialize(out)
        return out                        // back to [B,O,T,H,W]
    }
}

/// GroupNorm over channels (input [B,C,T,H,W]) done in fp32, cast back to VAE precision.
func vaeGroupNorm(_ x: MLXArray, _ norm: GroupNorm, activate: Bool = false) -> MLXArray {
    var h = x.transposed(0, 2, 3, 4, 1)                              // NDHWC
    if !SeedVR2Runtime.disableVAENormalizationFusion && x.dtype == .float16 && VAEPrecision.dtype == .float16 {
        return SeedVR2VAENormalization.apply(h, norm: norm, activate: activate).transposed(0, 4, 1, 2, 3)
    }
    h = norm(h.asType(.float32)).asType(VAEPrecision.dtype)
    if activate { h = silu(h) }
    return h.transposed(0, 4, 1, 2, 3)
}
