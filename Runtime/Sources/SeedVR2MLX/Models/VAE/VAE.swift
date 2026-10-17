// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
// Port of mflux seedvr2_vae/{encoder,decoder,vae}.py.
import MLX
import MLXNN

public final class Encoder3D: Module {
    @ModuleInfo(key: "conv_in") var convIn: CausalConv3d
    @ModuleInfo(key: "down_blocks") var downBlocks: [DownBlock3D]
    @ModuleInfo(key: "mid_block") var midBlock: MidBlock3D
    @ModuleInfo(key: "conv_norm_out") var convNormOut: GroupNorm
    @ModuleInfo(key: "conv_out") var convOut: CausalConv3d

    public init(inChannels: Int = 3, outChannels: Int = 16,
                blockOut: [Int] = [128, 256, 512, 512], layersPerBlock: Int = 2, temporalDownBlocks: Int = 2, fusedAttention: Bool = false) {
        self._convIn.wrappedValue = CausalConv3d(inChannels, blockOut[0])
        var blocks: [DownBlock3D] = []
        var outCh = blockOut[0]
        let n = blockOut.count
        for (i, ch) in blockOut.enumerated() {
            let inCh = outCh; outCh = ch
            let isFinal = i == n - 1
            let temporalDown = (i >= n - temporalDownBlocks - 1) && !isFinal
            blocks.append(DownBlock3D(inCh, outCh, numLayers: layersPerBlock, addDownsample: !isFinal, temporalDown: temporalDown))
        }
        self._downBlocks.wrappedValue = blocks
        self._midBlock.wrappedValue = MidBlock3D(blockOut[n - 1], fusedAttention: fusedAttention)
        self._convNormOut.wrappedValue = groupNorm32(blockOut[n - 1])
        self._convOut.wrappedValue = CausalConv3d(blockOut[n - 1], 2 * outChannels)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = convIn(x)
        for b in downBlocks { h = b(h) }
        h = midBlock(h)
        h = vaeGroupNorm(h, convNormOut, activate: true)
        return convOut(h)
    }
}

public final class Decoder3D: Module {
    @ModuleInfo(key: "conv_in") var convIn: CausalConv3d
    @ModuleInfo(key: "mid_block") var midBlock: MidBlock3D
    @ModuleInfo(key: "up_blocks") var upBlocks: [UpBlock3D]
    @ModuleInfo(key: "conv_norm_out") var convNormOut: GroupNorm
    @ModuleInfo(key: "conv_out") var convOut: CausalConv3d

    public init(inChannels: Int = 16, outChannels: Int = 3,
                blockOut: [Int] = [128, 256, 512, 512], layersPerBlock: Int = 3, temporalUpBlocks: Int = 2, fusedAttention: Bool = false) {
        let rev = Array(blockOut.reversed())
        self._convIn.wrappedValue = CausalConv3d(inChannels, rev[0])
        self._midBlock.wrappedValue = MidBlock3D(rev[0], fusedAttention: fusedAttention)
        var blocks: [UpBlock3D] = []
        var outCh = rev[0]
        let n = rev.count
        for (i, ch) in rev.enumerated() {
            let inCh = outCh; outCh = ch
            let isFinal = i == n - 1
            let temporalUp = i < temporalUpBlocks
            blocks.append(UpBlock3D(inCh, outCh, numLayers: layersPerBlock, addUpsample: !isFinal, temporalUp: temporalUp))
        }
        self._upBlocks.wrappedValue = blocks
        self._convNormOut.wrappedValue = groupNorm32(rev[n - 1])
        self._convOut.wrappedValue = CausalConv3d(rev[n - 1], outChannels)
        super.init()
    }

    public func callAsFunction(_ z: MLXArray) -> MLXArray {
        var h = convIn(z)
        h = midBlock(h)
        for b in upBlocks { h = b(h) }
        h = vaeGroupNorm(h, convNormOut, activate: true)
        return convOut(h)
    }
}

public final class SeedVR2VAE: Module {
    @ModuleInfo(key: "encoder") var encoder: Encoder3D
    @ModuleInfo(key: "decoder") var decoder: Decoder3D
    let scalingFactor: Float = 0.9152
    let latentChannels = 16

    public override convenience init() { self.init(fusedAttention: false) }
    public init(fusedAttention: Bool) {
        // The published VAE stores F16 convolution weights. Matching activations
        // avoids BF16 + F16 promotion to F32 in every convolution.
        VAEPrecision.dtype = fusedAttention ? .float16 : .bfloat16
        self._encoder.wrappedValue = Encoder3D(fusedAttention: fusedAttention)
        self._decoder.wrappedValue = Decoder3D(fusedAttention: fusedAttention)
        super.init()
    }

    /// x [B,3,T,H,W] (or [B,3,H,W]) -> latent [B,16,T,H/8,W/8].
    public func encode(_ xIn: MLXArray) -> MLXArray {
        let expanded = xIn.ndim == 4 ? xIn.expandedDimensions(axis: 2) : xIn
        let x = VAEPrecision.dtype == .float16 ? expanded.asType(.float16) : expanded
        let h = causalSlices(x, isEncode: true)
        let mean = h[0..., 0 ..< latentChannels]
        return mean * scalingFactor
    }

    /// z [B,16,T,H,W] -> image [B,3,T,H*8,W*8].
    public func decode(_ zIn: MLXArray) -> MLXArray {
        var z = zIn.ndim == 4 ? zIn.expandedDimensions(axis: 2) : zIn
        z = z / scalingFactor
        if VAEPrecision.dtype == .float16 { z = z.asType(.float16) }
        return causalSlices(z, isEncode: false)
    }
}


extension SeedVR2VAE {
    private func resetCausalMemory() {
        for module in modules() { (module as? CausalConv3d)?.memory = nil }
    }
    private func causalSlices(_ input: MLXArray, isEncode: Bool) -> MLXArray {
        resetCausalMemory()
        defer { resetCausalMemory(); SeedVR2Runtime.causalState = .disabled }
        let stride = isEncode ? 4 : 1
        let firstCount = stride + 1
        if input.shape[2] <= firstCount {
            SeedVR2Runtime.causalState = .disabled
            let output = isEncode ? encoder(input) : decoder(input)
            SeedVR2Runtime.materialize(output)
            return output
        }
        var slices: [MLXArray] = []
        var start = 0
        while start < input.shape[2] {
            let end = min(input.shape[2], start + (start == 0 ? firstCount : stride))
            SeedVR2Runtime.causalState = start == 0 ? .initializing : .active
            let chunk = input[0..., 0..., start ..< end]
            let output = isEncode ? encoder(chunk) : decoder(chunk)
            SeedVR2Runtime.materialize(output)
            slices.append(output)
            start = end
        }
        let output = concatenated(slices, axis: 2)
        SeedVR2Runtime.materialize(output)
        return output
    }
}
