// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
// Port of mflux seedvr2_transformer/transformer.py (top-level assembly).
import MLX
import MLXNN

public final class SeedVR2Transformer: Module {
    @ModuleInfo(key: "vid_in") var vidIn: PatchIn
    @ModuleInfo(key: "txt_in") var txtIn: Linear
    @ModuleInfo(key: "emb_in") var embIn: TimeEmbedding
    @ModuleInfo(key: "blocks") var blocks: [TransformerBlock]
    @ModuleInfo(key: "vid_out_norm") var vidOutNorm: SeedVR2RMSNorm?
    @ParameterInfo(key: "out_shift") var outShift: MLXArray?
    @ParameterInfo(key: "out_scale") var outScale: MLXArray?
    @ModuleInfo(key: "vid_out") var vidOut: PatchOut

    let vidDim: Int
    private let config: SeedVR2Config

    public init(_ cfg: SeedVR2Config = .r3B) {
        self.vidDim = cfg.vidDim
        self.config = cfg
        let embDim = 6 * cfg.vidDim
        self._vidIn.wrappedValue = PatchIn(inChannels: cfg.vidInChannels, patchSize: cfg.patchSize, dim: cfg.vidDim)
        self._txtIn.wrappedValue = seedVR2Linear(cfg.txtInDim, cfg.vidDim)
        self._embIn.wrappedValue = TimeEmbedding(sinusoidalDim: 256, hiddenDim: cfg.vidDim, outputDim: embDim)

        var blk: [TransformerBlock] = []
        for i in 0 ..< cfg.numLayers {
            blk.append(Self.makeBlock(i, cfg))
        }
        self._blocks.wrappedValue = blk

        self._vidOutNorm.wrappedValue = cfg.useOutputAda ? SeedVR2RMSNorm(cfg.vidDim, eps: cfg.normEps) : nil
        self._outShift.wrappedValue = cfg.useOutputAda ? MLXArray.zeros([cfg.vidDim]) : nil
        self._outScale.wrappedValue = cfg.useOutputAda ? MLXArray.ones([cfg.vidDim]) : nil
        self._vidOut.wrappedValue = PatchOut(outChannels: cfg.vidOutChannels, patchSize: cfg.patchSize, dim: cfg.vidDim)
        super.init()
    }

    /// vid [B,Cin,T,H,W], txt [B,Lt,txtInDim], timestep scalar. Returns [B,Cout,T,H,W].
    public func callAsFunction(_ vidIn0: MLXArray, _ txt0: MLXArray, timestep: MLXArray) -> MLXArray {
        // Without a checkpoint callback, forward cannot throw.
        try! forward(vidIn0, txt0, timestep: timestep, loadBlock: nil)
    }

    public func streamed(_ video: MLXArray, _ text: MLXArray, timestep: MLXArray,
                         loadBlock: @escaping (Int, TransformerBlock) throws -> Void) throws -> MLXArray {
        try forward(video, text, timestep: timestep, loadBlock: loadBlock)
    }

    private static func makeBlock(_ i: Int, _ cfg: SeedVR2Config) -> TransformerBlock {
        TransformerBlock(vidDim: cfg.vidDim, txtDim: cfg.vidDim, heads: cfg.heads,
            headDim: cfg.headDim, expandRatio: cfg.expandRatio, normEps: cfg.normEps,
            ropeDim: cfg.ropeDim, ropeOnText: cfg.ropeOnText, shared: i >= cfg.mmLayers,
            isLastLayer: cfg.lastLayerVideoOnly && i == cfg.numLayers - 1, window: cfg.window, shift: i % 2 == 1,
            pixelRoPE: cfg.pixelRoPE,
            fuseSharedProjections: cfg.sharedAttentionFusion && !SeedVR2Runtime.disableSharedAttentionFusion,
            fusedInputs: cfg.fusedMLP, geluMLP: cfg.geluMLP)
    }

    private func forward(_ vidIn0: MLXArray, _ txt0: MLXArray, timestep: MLXArray,
                         loadBlock: ((Int, TransformerBlock) throws -> Void)?) throws -> MLXArray {
        let video = config.halfPrecision ? vidIn0.asType(.float16) : vidIn0
        let text = config.halfPrecision ? txt0.asType(.float16) : txt0
        var txt = txtIn(text)
        let txtLen = txt.shape[1]
        var (vid, vidShape) = vidIn(video)            // [B,L,dim], [[t,h,w]]
        var emb = embIn(timestep)
        emb = emb.reshaped([-1, vidDim, 2, 3])
        if config.halfPrecision { emb = emb.asType(.float16) }

        for index in blocks.indices {
            if let loadBlock {
                // Keep loaded blocks out of the model's registered children/cache.
                // This scope owns exactly one block and releases it after evaluation.
                let block = Self.makeBlock(index, config)
                try loadBlock(index, block)
                (vid, txt) = block(vid, txt, emb: emb, vidShape: [vidShape], txtLen: txtLen)
                SeedVR2Runtime.materialize(vid, txt)
            } else {
                (vid, txt) = blocks[index](vid, txt, emb: emb, vidShape: [vidShape], txtLen: txtLen)
                SeedVR2Runtime.materialize(vid, txt)
            }
            Memory.clearCache()
        }

        if let vidOutNorm, let outShift, let outScale {
            vid = vidOutNorm(vid)
            // Output Ada is present in 3B, absent from the 7B architecture.
            let mod = emb[0..., 0..., 0]
            let shiftA = mod[.ellipsis, 0].expandedDimensions(axis: 1)
            let scaleA = mod[.ellipsis, 1].expandedDimensions(axis: 1)
            vid = vid * (scaleA + outScale) + (shiftA + outShift)
        }

        return vidOut(vid, vidShape: vidShape)
    }
}
