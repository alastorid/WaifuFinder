// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
// Port of mflux seedvr2_transformer/swiglu_mlp.py + mm_swiglu.py.
import MLX
import MLXNN

/// 3B SwiGLU (bias-free), or 7B's ordinary biased GELU MLP.
public final class SwiGLUMLP: Module, UnaryLayer {
    @ModuleInfo(key: "proj_in") var projIn: Linear
    @ModuleInfo(key: "proj_in_gate") var projInGate: Linear?
    @ModuleInfo(key: "proj_out") var projOut: Linear
    private let usesGELU: Bool

    public init(dim: Int, expandRatio: Int = 4, multipleOf: Int = 256, fusedInputs: Bool = false, gelu: Bool = false) {
        self.usesGELU = gelu
        var hidden = gelu ? dim * expandRatio : (2 * dim * expandRatio) / 3
        hidden = multipleOf * ((hidden + multipleOf - 1) / multipleOf)
        self._projIn.wrappedValue = seedVR2Linear(dim, fusedInputs && !gelu ? 2 * hidden : hidden, bias: gelu)
        self._projInGate.wrappedValue = fusedInputs || gelu ? nil : seedVR2Linear(dim, hidden, bias: false)
        self._projOut.wrappedValue = seedVR2Linear(hidden, dim, bias: gelu)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        if usesGELU { return projOut(geluApproximate(projIn(x))) }
        if let gate = projInGate { return projOut(silu(gate(x)) * projIn(x)) }
        let both = projIn(x)
        let hidden = both.shape.last! / 2
        return projOut(SeedVR2FusedOps.swiGLU(both[.ellipsis, 0..<hidden], both[.ellipsis, hidden..<(2 * hidden)]))
    }

}

/// Dual-stream MLP wrapper (video + text branches), mflux MMSwiGLU.
/// Shared blocks (i >= mm_layers) use a single `all` MLP for both streams.
public final class MMSwiGLU: Module {
    @ModuleInfo(key: "all") var all: SwiGLUMLP?
    @ModuleInfo(key: "vid") var vid: SwiGLUMLP?
    @ModuleInfo(key: "txt") var txt: SwiGLUMLP?
    let isLastLayer: Bool
    let shared: Bool
    private let batchSharedInputs: Bool

    public init(vidDim: Int, txtDim: Int, expandRatio: Int = 4, shared: Bool = false, isLastLayer: Bool = false, fusedInputs: Bool = false, gelu: Bool = false, fuseSharedInputs: Bool = false) {
        self.isLastLayer = isLastLayer
        self.shared = shared
        self.batchSharedInputs = shared && !isLastLayer && fuseSharedInputs
        if shared {
            self._all.wrappedValue = SwiGLUMLP(dim: vidDim, expandRatio: expandRatio, fusedInputs: fusedInputs, gelu: gelu)
        } else {
            self._vid.wrappedValue = SwiGLUMLP(dim: vidDim, expandRatio: expandRatio, fusedInputs: fusedInputs, gelu: gelu)
            self._txt.wrappedValue = isLastLayer ? nil : SwiGLUMLP(dim: txtDim, expandRatio: expandRatio, fusedInputs: fusedInputs, gelu: gelu)
        }
        super.init()
    }

    public func callAsFunction(_ vidIn: MLXArray, _ txtIn: MLXArray) -> (MLXArray, MLXArray) {
        if batchSharedInputs {
            let length = vidIn.shape[1]
            let both = all!(concatenated([vidIn, txtIn], axis: 1))
            return (both[0..., 0..<length], both[0..., length...])
        }
        let mlpVid = shared ? all! : vid!
        let v = mlpVid(vidIn)
        if isLastLayer { return (v, txtIn) }
        let t = (shared ? all! : txt!)(txtIn)
        return (v, t)
    }

}
