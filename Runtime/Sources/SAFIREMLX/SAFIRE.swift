// Native adaptation of SAFIRE (Kwon et al.) and SAM (Meta Platforms).
// See Runtime/SAFIRE-NOTICE.md for provenance and Apache-2.0 notices.
import Foundation
import CryptoKit
import MLX
import Metal

public enum SAFIREError: Error { case invalid(String) }

public enum SAFIREPrecision: String, Codable {
    case fp32, fp16
}

/// Serial worker-owned native port. Window attention and chunked global
/// relative-position attention preserve the upstream computation. Relative
/// position tables, the 16×16 prompt tokens and the dense positional encoding
/// are gathered once at load; linear weights are pre-transposed; the automatic
/// 256-prompt path runs as fixed-shape compiled chunks whose image keys stay
/// unbatched in the first layer and whose mask post-processing is a fused
/// kernel. Production uses FP16 only inside the two-way decoder transformer;
/// the encoder and spatial upscaling tail stay FP32.
public final class SAFIREModel {
    public static let revision = "safire-native-v2"
    private var weights: [String:MLXArray]
    private var transposed: [String:MLXArray] = [:]
    private var relTables: [String:(MLXArray,MLXArray)] = [:]
    private var upscaleWeights: [String:MLXArray] = [:]
    private var upscaleBiases: [String:MLXArray] = [:]
    private let profile: SAFIREProfile
    private let precision: SAFIREPrecision
    private var encoderDType: DType {
        precision == .fp16 && ["all", "encoder"].contains(profile.fp16Scope) ? .float16 : .float32
    }
    private var promptDType: DType {
        precision == .fp16 && ["all", "decoder"].contains(profile.fp16Scope) ? .float16 : .float32
    }
    private var transformerDType: DType {
        precision == .fp16 && ["all", "decoder", "transformer"].contains(profile.fp16Scope) ? .float16 : .float32
    }
    private var decoderTailDType: DType {
        precision == .fp16 && ["all", "decoder"].contains(profile.fp16Scope) ? .float16 : .float32
    }
    private let highpass: MLXArray
    private let densePE: MLXArray
    private let gridTokens: MLXArray
    private var compiledDecoders: [Int:([MLXArray])->[MLXArray]] = [:]
    private var compiledGrid: [Int:([MLXArray])->[MLXArray]] = [:]
    private var compiledEncoder: ((MLXArray) -> MLXArray)?
    private var compiledBlocks: [Int:([MLXArray]) -> [MLXArray]] = [:]
    private var peFolded: [String:MLXArray] = [:]
    /// Optional per-stage wall-clock recorder (seconds, active MLX bytes) for
    /// the benchmark harness; nil in production inference.
    public var stageRecorder: ((String, Double, Int, Int) -> Void)?
    /// True while a compiled graph replays or traces: per-op recorders cannot
    /// run there, so the outer stage barriers carry the attribution.
    private var inCompiledGraph=false

    public init(directory: URL) throws {
        profile = try SAFIREProfile.current()
        let modelURL=directory.appendingPathComponent("model.safetensors")
        let manifest=try JSONSerialization.jsonObject(with:Data(contentsOf:directory.appendingPathComponent("manifest.json"))) as? [String:Any]
        guard let expected=(manifest?["fileSHA256"] as? [String:String])?["model.safetensors"],expected.count==64 else {
            throw SAFIREError.invalid("Missing native checkpoint fingerprint")
        }
        let handle=try FileHandle(forReadingFrom:modelURL)
        var hash=SHA256()
        while let bytes=try handle.read(upToCount:8*1024*1024),!bytes.isEmpty { hash.update(data:bytes) }
        try handle.close()
        let actual=hash.finalize().map { String(format:"%02x",$0) }.joined()
        guard actual==expected else { throw SAFIREError.invalid("Native SAFIRE checkpoint checksum mismatch") }
        let (loaded,metadata)=try loadArraysAndMetadata(url:modelURL)
        weights=loaded
        guard weights.count == 348, weights["image_encoder.pos_embed"]?.shape == [1,64,64,768],
              weights["prompt_encoder.pe_layer.positional_encoding_gaussian_matrix"]?.shape == [2,128] else {
            throw SAFIREError.invalid("Expected the official 348-tensor SAFIRE ViT-B adapter checkpoint.")
        }
        // A prepared FP16 checkpoint keeps the published tensor layouts; only
        // the element type differs. The original FP32 publication stays
        // lossless on disk and remains the higher-precision fallback.
        let prepared=metadata["waifu.experiment"] == "safire-native-fp16-v1"
        if let experiment=metadata["waifu.experiment"],!prepared,experiment != "safire-extract-f32-v1" {
            throw SAFIREError.invalid("Unknown SAFIRE checkpoint preparation: \(experiment)")
        }
        if prepared && !weights.values.allSatisfy({ $0.dtype == .float16 }) {
            throw SAFIREError.invalid("Prepared SAFIRE checkpoint must hold FP16 weights")
        }
        precision = profile.precision
        if prepared && precision == .fp16 && profile.fp16Scope != "all" {
            throw SAFIREError.invalid("Mixed-precision scopes require the original FP32 SAFIRE checkpoint")
        }
        for (name,value) in weights {
            let useHalf: Bool
            if precision != .fp16 {
                useHalf = false
            } else {
                switch profile.fp16Scope {
                case "all": useHalf = true
                case "encoder": useHalf = name.hasPrefix("image_encoder.")
                case "decoder": useHalf = name.hasPrefix("prompt_encoder.") || name.hasPrefix("mask_decoder.")
                case "transformer": useHalf = name.hasPrefix("mask_decoder.transformer.")
                default: useHalf = false
                }
            }
            let target: DType = useHalf ? .float16 : .float32
            if value.dtype != target { weights[name] = value.asType(target) }
        }
        // One-time kernel layout preparation; original safetensors stay lossless.
        for key in Array(weights.keys) where key.hasSuffix(".weight") && weights[key]!.ndim == 4 {
            let transpose = key.contains("output_upscaling") ? [1,2,3,0] : [0,2,3,1]
            weights[key] = weights[key]!.transposed(axes:transpose)
        }
        // Linear weights are read exclusively through their transposed
        // copies; the published originals stay on disk. Freeing them here
        // avoids a second ~385 MB resident copy. Embedding/token tables and
        // the positional gaussian are read directly and keep their originals.
        let directReads: Set<String> = ["prompt_encoder.no_mask_embed.weight",
            "prompt_encoder.pe_layer.positional_encoding_gaussian_matrix",
            "prompt_encoder.not_a_point_embed.weight",
            "prompt_encoder.point_embeddings.0.weight","prompt_encoder.point_embeddings.1.weight",
            "prompt_encoder.point_embeddings.2.weight","prompt_encoder.point_embeddings.3.weight",
            "mask_decoder.iou_token.weight","mask_decoder.mask_tokens.weight"]
        for key in Array(weights.keys) where key.hasSuffix(".weight") && weights[key]!.ndim == 2 {
            transposed[String(key.dropLast(7))]=weights[key]!.transposed()
            if !directReads.contains(key) { weights[key]=nil }
        }
        // 2×2 stride-2 transposed convolutions run as per-pixel GEMMs: the
        // runtime [out,kh,kw,in] weight becomes [in, kh*kw*out] and the output
        // blocks are pixel-shuffled back to the double-resolution grid — in the
        // fused FP32 Metal kernels the shuffle is folded into the GEMM epilogue
        // store address (see SAFIREUpscale); the (kh,kw,out) column order needs
        // no further permutation. Biases are pre-tiled 4× for the same kernels.
        for key in Array(weights.keys) where key.contains("output_upscaling") && key.hasSuffix(".weight") && weights[key]!.ndim == 4 {
            let weight=weights[key]!
            let name=String(key.dropLast(7))
            upscaleWeights[name]=weight.transposed(axes:[3,1,2,0]).reshaped(weight.shape[3],-1)
            if let bias=weights[name+".bias"] {
                upscaleBiases[name]=concatenated([bias,bias,bias,bias],axis:0)
            }
        }
        var mask = [Float](repeating:1,count:1024*1024)
        // torch fftshift's centered [256,768) rejection, expressed unshifted.
        for y in 0..<1024 where y < 256 || y >= 768 {
            for x in 0..<1024 where x < 256 || x >= 768 { mask[y*1024+x] = 0 }
        }
        // The FFT branch stays FP32 in every precision mode.
        highpass = MLXArray(mask,[1,1024,1024,1])
        let encoderType: DType = precision == .fp16 && ["all", "encoder"].contains(profile.fp16Scope) ? .float16 : .float32
        let promptType: DType = precision == .fp16 && ["all", "decoder"].contains(profile.fp16Scope) ? .float16 : .float32
        let transformerType: DType = precision == .fp16 && ["all", "decoder", "transformer"].contains(profile.fp16Scope) ? .float16 : .float32
        let edt = encoderType
        let ddt = promptType
        for i in 0..<12 {
            let prefix="image_encoder.blocks.\(i).attn"
            let side=[2,5,8,11].contains(i) ? 64 : 14
            let indices=MLXArray((0..<side).flatMap { y in (0..<side).map { Int32(y-$0+side-1) } },[side,side])
            relTables[prefix]=(take(weights[prefix+".rel_pos_h"]!,indices.reshaped(-1),axis:0).reshaped(side,side,64).asType(edt),
                               take(weights[prefix+".rel_pos_w"]!,indices.reshaped(-1),axis:0).reshaped(side,side,64).asType(edt))
        }
        var grid: [Float] = []
        for i in 0..<4096 { grid.append(Float(i%64)+0.5); grid.append(Float(i/64)+0.5) }
        let gaussian=weights["prompt_encoder.pe_layer.positional_encoding_gaussian_matrix"]!
        let coords = MLXArray(grid,[1,4096,2]).asType(ddt)/64
        let phase = matmul(coords*2-1,gaussian)*(2*Float.pi)
        densePE = concatenated([sin(phase),cos(phase)],axis:-1)
        // Fixed automatic-prompt tokens: the 16×16 grid points are constants,
        // so every prompt's [7,256] token row is immutable per checkpoint.
        var points: [Float] = []
        for i in 0..<256 { points.append(Float(i%16)*64+32); points.append(Float(i/16)*64+32) }
        let promptPhase=matmul((MLXArray(points,[256,2]).asType(ddt)+0.5)/1024*2-1,gaussian)*(2*Float.pi)
        let positive=concatenated([sin(promptPhase),cos(promptPhase)],axis:-1)+weights["prompt_encoder.point_embeddings.1.weight"]!
        let padding=broadcast(weights["prompt_encoder.not_a_point_embed.weight"]!,to:[256,1,256])
        let outputTokens=concatenated([weights["mask_decoder.iou_token.weight"]!,weights["mask_decoder.mask_tokens.weight"]!],axis:0)
        let tokens=concatenated([broadcast(outputTokens.expandedDimensions(axis:0),to:[256,5,256]),
                                 positive.expandedDimensions(axis:1),padding],axis:1)
        // The dense positional encoding enters every cross-attention q/k
        // projection additively: (keys+pe)@W = keys@W + pe@W, and pe@W is a
        // checkpoint constant. Folding removes six [batch,4096,256]
        // materializations per grid chunk.
        var folded:[String:MLXArray]=[:]
        let transformerPE=densePE.asType(transformerType)
        // Folded constants carry the projection bias as well, so the runtime
        // form keys@W^T + (pe@W^T + bias) equals the upstream (keys+pe)@W^T + bias.
        for layer in 0..<2 {
            let p="mask_decoder.transformer.layers.\(layer)"
            let kBias=weights[p+".cross_attn_token_to_image.k_proj.bias"]
            let kFolded=matmul(transformerPE,transposed[p+".cross_attn_token_to_image.k_proj"]!)
            folded[p+".cross_attn_token_to_image.k"]=kBias.map { kFolded+$0.asType(transformerType) } ?? kFolded
            let qBias=weights[p+".cross_attn_image_to_token.q_proj.bias"]
            let qFolded=matmul(transformerPE,transposed[p+".cross_attn_image_to_token.q_proj"]!)
            folded[p+".cross_attn_image_to_token.q"]=qBias.map { qFolded+$0.asType(transformerType) } ?? qFolded
        }
        let finalFolded=matmul(transformerPE,transposed["mask_decoder.transformer.final_attn_token_to_image.k_proj"]!)
        let finalBiasFold=weights["mask_decoder.transformer.final_attn_token_to_image.k_proj.bias"].map { finalFolded+$0.asType(transformerType) }
        folded["mask_decoder.transformer.final_attn_token_to_image.k"]=finalBiasFold ?? finalFolded
        peFolded=folded
        peFolded=folded
        eval(Array(weights.values));eval(Array(transposed.values));eval(Array(upscaleWeights.values));eval(Array(upscaleBiases.values))
        eval(highpass,densePE,tokens);eval(Array(peFolded.values))
        if ProcessInfo.processInfo.environment["WAIFU_SAFIRE_DEBUG"] != nil {
            FileHandle.standardError.write(Data(String(format:"MEM lib post-init active %.0f cache %.0f peak %.0f\n",Memory.activeMemory/1048576,Memory.cacheMemory/1048576,Memory.peakMemory/1048576).utf8))
        }
        eval(relTables.values.flatMap { [$0.0,$0.1] })
        gridTokens=tokens
    }
    public var diagnostics: SAFIREProfile { profile }

    /// One-time deterministic FP16 checkpoint preparation, mirroring the WD
    /// Tagger native conversion: the published FP32 tensors are cast on the
    /// CPU (keeping the GPU out of conversion), the source SHA-256 and
    /// conversion revision are recorded in the safetensors metadata, and the
    /// original package stays untouched as the higher-precision fallback.
    public static func prepare(source: URL, destination: URL) throws {
        let manifest=try JSONSerialization.jsonObject(with:Data(contentsOf:source.appendingPathComponent("manifest.json"))) as? [String:Any]
        guard let expected=(manifest?["fileSHA256"] as? [String:String])?["model.safetensors"],expected.count==64 else {
            throw SAFIREError.invalid("Source package has no native checkpoint fingerprint")
        }
        let handle=try FileHandle(forReadingFrom:source.appendingPathComponent("model.safetensors"))
        var hash=SHA256()
        while let bytes=try handle.read(upToCount:8*1024*1024),!bytes.isEmpty { hash.update(data:bytes) }
        try handle.close()
        let actual=hash.finalize().map { String(format:"%02x",$0) }.joined()
        guard actual==expected else { throw SAFIREError.invalid("Source checkpoint checksum mismatch") }
        guard !FileManager.default.fileExists(atPath:destination.appendingPathComponent("model.safetensors").path) else {
            throw SAFIREError.invalid("Destination exists; use a new versioned package directory")
        }
        let half=try Stream.withNewDefaultStream(device: .cpu) {
            try Self.prepareOnCPU(source:source)
        }
        try FileManager.default.createDirectory(at:destination,withIntermediateDirectories:true)
        try save(arrays:half,metadata:[
            "waifu.experiment":"safire-native-fp16-v1",
            "sourceSHA256":actual,
            "conversion":"deterministic FP32→FP16 cast of the published SAFIRE weights; layouts unchanged",
        ],url:destination.appendingPathComponent("model.safetensors"),stream:.cpu)
        var architecture:[String:Any]=[: ]
        if let original=try? JSONSerialization.jsonObject(with:Data(contentsOf:source.appendingPathComponent("architecture.json"))) as? [String:Any] {
            architecture=original
        }
        architecture["conversionVersion"]="safire-prepare-fp16-v1"
        architecture["weightLayout"]="pytorch-f16"
        architecture["preparedFromSHA256"]=actual
        architecture["runtimeRevision"]=SAFIREModel.revision
        architecture["validationStatus"]="pending"
        try JSONSerialization.data(withJSONObject:architecture,options:[.prettyPrinted,.sortedKeys])
            .write(to:destination.appendingPathComponent("architecture.json"))
        for extra in ["NOTICE","README.md","LICENSE","APACHE-LICENSE"] {
            let url=source.appendingPathComponent(extra)
            if FileManager.default.fileExists(atPath:url.path) {
                try FileManager.default.copyItem(at:url,to:destination.appendingPathComponent(extra))
            }
        }
        try JSONSerialization.data(withJSONObject:["fileSHA256":Self.fileHashes(directory:destination)],options:[.prettyPrinted,.sortedKeys])
            .write(to:destination.appendingPathComponent("manifest.json"))
    }
    private static func prepareOnCPU(source: URL) throws -> [String:MLXArray] {
        let loaded=try loadArrays(url:source.appendingPathComponent("model.safetensors"),stream:.cpu)
        guard loaded.count == 348 else { throw SAFIREError.invalid("Expected the official 348-tensor SAFIRE checkpoint") }
        var converted:[String:MLXArray]=[:]
        for (name,value) in loaded {
            let casted=value.asType(.float16)
            eval(casted)
            converted[name]=casted
        }
        return converted
    }
    private static func fileHashes(directory: URL) -> [String:String] {
        var hashes:[String:String]=[:]
        let enumerator=FileManager.default.enumerator(at:directory,includingPropertiesForKeys:nil)
        while let url=enumerator?.nextObject() as? URL, url.isFileURL {
            guard let handle=try? FileHandle(forReadingFrom:url) else { continue }
            var hash=SHA256()
            while let bytes=try? handle.read(upToCount:8*1024*1024),!bytes.isEmpty { hash.update(data:bytes) }
            try? handle.close()
            hashes[url.lastPathComponent]=hash.finalize().map { String(format:"%02x",$0) }.joined()
        }
        return hashes
    }
    private func w(_ name:String) -> MLXArray {
        guard let value = weights[name] else { preconditionFailure("Missing SAFIRE parameter: \(name)") }
        return value
    }
    private func wt(_ name:String) -> MLXArray {
        guard let value = transposed[name] else { preconditionFailure("Missing SAFIRE parameter: \(name)") }
        return value
    }
    private func linear(_ x:MLXArray,_ name:String) -> MLXArray {
        guard let bias=weights[name+".bias"] else { return matmul(x,wt(name)) }
        return addMM(bias,x,wt(name))
    }
    /// FP32 rows run through the fused one-pass Metal norm (optional residual
    /// and GELU epilogue); other precisions keep the MLX primitives.
    private func norm(_ x:MLXArray,_ name:String,eps:Float=1e-6,residual:MLXArray?=nil,geluAfter:Bool=false) -> MLXArray {
        if x.dtype == .float32, residual != nil || geluAfter {
            // The fused kernel indexes both rows directly; broadcast-add first
            // when the residual carries a larger batch (unbatched image keys).
            if let residual,residual.shape != x.shape {
                return SAFIRENorm.norm(x+residual,residual:nil,weight:w(name+".weight"),bias:w(name+".bias"),eps:eps,gelu:geluAfter)
            }
            return SAFIRENorm.norm(x,residual:residual,weight:w(name+".weight"),bias:w(name+".bias"),eps:eps,gelu:geluAfter)
        }
        var value=x
        if let residual { value=value+residual }
        if x.dtype == .float32, geluAfter { return gelu(MLXFast.layerNorm(value,weight:w(name+".weight"),bias:w(name+".bias"),eps:eps)) }
        return MLXFast.layerNorm(value,weight:w(name+".weight"),bias:w(name+".bias"),eps:eps)
    }
    private func conv(_ x:MLXArray,_ name:String,stride:Int=1,padding:Int=0,transpose:Bool=false) -> MLXArray {
        let y = transpose ? convTransposed2d(x,w(name+".weight"),stride:.init(stride),padding:.init(padding))
                          : conv2d(x,w(name+".weight"),stride:.init(stride),padding:.init(padding))
        return weights[name+".bias"].map { y + $0 } ?? y
    }
    private func gelu(_ x:MLXArray) -> MLXArray {
        if x.dtype == .float32 { return SAFIRENorm.gelu(x) }
        return x * 0.5 * (1+erf(x / Float(sqrt(2.0))))
    }

    /// Two-way attention over token/image streams. Head count stays 8; the
    /// per-head dimension and softmax scale follow the projected query width
    /// (256 for self-attention, 128 for cross-attention), as upstream. When
    /// query and key/value batches already match, SDPA runs as upstream.
    /// Shared singleton-batch image keys stay unexpanded: per-head 2-D matmuls
    /// avoid materializing per-prompt key copies.
    private func attention(q:MLXArray,k:MLXArray,v:MLXArray,prefix:String) -> MLXArray {
        attentionProjected(a:linear(q,prefix+".q_proj"),b:linear(k,prefix+".k_proj"),
                           c:linear(v,prefix+".v_proj"),prefix:prefix)
    }
    private func attentionProjected(a:MLXArray,b:MLXArray,c:MLXArray,prefix:String) -> MLXArray {
        let headDim=a.dim(2)/8, scale=1/sqrt(Float(headDim))
        func split(_ x:MLXArray) -> MLXArray { x.reshaped(x.dim(0),x.dim(1),8,headDim).transposed(axes:[0,2,1,3]) }
        let aH=split(a),bH=split(b),cH=split(c)
        if a.dim(0)==b.dim(0) && b.dim(0)==c.dim(0) {
            let out=MLXFast.scaledDotProductAttention(queries:aH,keys:bH,values:cH,scale:scale,mask:nil)
            return linear(out.transposed(axes:[0,2,1,3]).reshaped(a.dim(0),a.dim(1),a.dim(2)),prefix+".out_proj")
        }
        let qSingleton=a.dim(0)==1,kSingleton=b.dim(0)==1
        var heads:[MLXArray]=[]
        for h in 0..<8 {
            let ah=aH[0...,h,0...,0...],bh=bH[0...,h,0...,0...],ch=cH[0...,h,0...,0...]
            let scores: MLXArray
            switch (qSingleton,kSingleton) {
            case (true,true): scores=matmul(ah[0]*scale,bh[0].transposed())
            case (true,false): scores=matmul(ah[0]*scale,bh.transposed(axes:[0,2,1]))
            case (false,true): scores=matmul(ah*scale,bh[0].transposed())
            case (false,false): scores=matmul(ah*scale,bh.transposed(axes:[0,2,1]))
            }
            // SDPA semantics: softmax in FP32, output in the score dtype.
            let probs=softmax(scores.asType(.float32),axis:-1).asType(scores.dtype)
            let out: MLXArray
            switch (qSingleton,kSingleton) {
            case (true,true): out=matmul(probs,ch[0])
            case (true,false): out=matmul(probs,ch)
            case (false,true): out=matmul(probs,ch[0])
            case (false,false): out=matmul(probs,ch)
            }
            heads.append(out.expandedDimensions(axis:1))
        }
        let batch=heads[0].dim(0),sequence=heads[0].dim(2)
        let merged=concatenated(heads,axis:1).transposed(axes:[0,2,1,3]).reshaped(batch,sequence,a.dim(2))
        return linear(merged,prefix+".out_proj")
    }
    private func mlp(_ x:MLXArray,_ prefix:String) -> MLXArray {
        var value=x
        for i in 0..<3 { value=linear(value,prefix+".layers.\(i)"); if i<2 { value=maximum(value,0) } }
        return value
    }
    public func decode(embedding:MLXArray,points:MLXArray) throws -> (masks:MLXArray,quality:MLXArray) {
        guard embedding.shape == [1,64,64,256], points.ndim == 2, points.dim(1) == 2,
              (1...16).contains(points.dim(0)) else { throw SAFIREError.invalid("Invalid prompt batch") }
        let embedding=embedding.asType(promptDType)
        let batch=points.dim(0)
        let graph:([MLXArray])->[MLXArray]
        if let cached=compiledDecoders[batch] { graph=cached }
        else {
            graph=compile { [unowned self] (inputs:[MLXArray]) -> [MLXArray] in
                let result=self.decodeGraph(embedding:inputs[0],points:inputs[1])
                return [result.0,result.1]
            }
            compiledDecoders[batch]=graph
        }
        inCompiledGraph=true
        let result=graph([embedding,points])
        inCompiledGraph=false
        eval(result)
        return (result[0],result[1])
    }
    private func decodeGraph(embedding:MLXArray,points:MLXArray) -> (MLXArray,MLXArray) {
        let batch=points.dim(0)
        let phase=matmul((points.asType(promptDType)+0.5)/1024*2-1,w("prompt_encoder.pe_layer.positional_encoding_gaussian_matrix"))*(2*Float.pi)
        let positive=concatenated([sin(phase),cos(phase)],axis:-1)+w("prompt_encoder.point_embeddings.1.weight")
        let padding=broadcast(w("prompt_encoder.not_a_point_embed.weight"),to:[batch,1,256])
        let sparse=concatenated([positive.expandedDimensions(axis:1),padding],axis:1)
        let outputTokens=concatenated([w("mask_decoder.iou_token.weight"),w("mask_decoder.mask_tokens.weight")],axis:0)
        let tokens=concatenated([broadcast(outputTokens.expandedDimensions(axis:0),to:[batch,5,256]),sparse],axis:1)
        let (queries,keys)=twoWayTransformer(embedding:embedding,tokens:tokens,sharedKeys:nil)
        return decoderTail(queries:queries,keys:keys,batch:batch)
    }
    /// Upstream TwoWayTransformer. `sharedKeys` keeps the image key/value
    /// stream unbatched through the first layer, where it is still identical
    /// for every prompt of the automatic grid.
    private func twoWayTransformer(embedding:MLXArray,tokens:MLXArray,sharedKeys:MLXArray?) -> (MLXArray,MLXArray) {
        let embedding=embedding.asType(transformerDType)
        let tokens=tokens.asType(transformerDType)
        let batch=tokens.dim(0)
        var queries=tokens
        var keys=sharedKeys?.asType(transformerDType) ?? broadcast(embedding.reshaped(1,4096,256)+w("prompt_encoder.no_mask_embed.weight").asType(transformerDType),to:[batch,4096,256])
        // q/k projections consume keys+pe via the pre-folded constant
        // (peFolded); v projections take keys directly, as upstream.
        // `folded` is the precomputed pe@W for whichever side consumes
        // keys+pe: the key side for token→image, the query side for image→token.
        func attentionFolded(foldOnQ:Bool,x:MLXArray,keys:MLXArray,v:MLXArray,folded:MLXArray,prefix:String) -> MLXArray {
            let a: MLXArray, b: MLXArray
            if foldOnQ {
                a=matmul(keys,transposed[prefix+".q_proj"]!)+folded
                b=linear(x,prefix+".k_proj")
            } else {
                a=linear(x,prefix+".q_proj")
                b=matmul(keys,transposed[prefix+".k_proj"]!)+folded
            }
            let c=linear(v,prefix+".v_proj")
            return attentionProjected(a:a,b:b,c:c,prefix:prefix)
        }
        func attention(q:MLXArray,k:MLXArray,v:MLXArray,prefix:String) -> MLXArray {
            attentionProjected(a:linear(q,prefix+".q_proj"),b:linear(k,prefix+".k_proj"),
                               c:linear(v,prefix+".v_proj"),prefix:prefix)
        }
        for i in 0..<2 {
            let p="mask_decoder.transformer.layers.\(i)"
            if i == 0 { queries=attention(q:queries,k:queries,v:queries,prefix:p+".self_attn") }
            else { queries=norm(queries,p+".norm1",eps:1e-5,
                residual:attention(q:queries+tokens,k:queries+tokens,v:queries,prefix:p+".self_attn")) }
            if i == 0 { queries=norm(queries,p+".norm1",eps:1e-5) }
            queries=norm(queries,p+".norm2",eps:1e-5,
                residual:attentionFolded(foldOnQ:false,x:queries+tokens,keys:keys,v:keys,
                    folded:peFolded[p+".cross_attn_token_to_image.k"]!,prefix:p+".cross_attn_token_to_image"))
            queries=norm(queries,p+".norm3",eps:1e-5,residual:linear(maximum(linear(queries,p+".mlp.lin1"),0),p+".mlp.lin2"))
            keys=norm(keys,p+".norm4",eps:1e-5,
                residual:attentionFolded(foldOnQ:true,x:queries+tokens,keys:keys,v:queries,
                    folded:peFolded[p+".cross_attn_image_to_token.q"]!,prefix:p+".cross_attn_image_to_token"))
        }
        queries=norm(queries,"mask_decoder.transformer.norm_final_attn",eps:1e-5,
            residual:attentionFolded(foldOnQ:false,x:queries+tokens,keys:keys,v:keys,
                folded:peFolded["mask_decoder.transformer.final_attn_token_to_image.k"]!,
                prefix:"mask_decoder.transformer.final_attn_token_to_image"))
        return (queries,keys)
    }
    /// 2×2 stride-2 transposed convolution as GEMM + pixel shuffle; identical
    /// arithmetic to convTransposed2d with these fixed shapes.
    private func upscale(_ x:MLXArray,_ name:String) -> MLXArray {
        let side=x.dim(1), channels=x.dim(3)
        let weight=upscaleWeights[name]!
        var block=matmul(x.reshaped(-1,channels),weight)
        if let bias=weights[name+".bias"] { block=block+concatenated([bias,bias,bias,bias],axis:0) }
        let outputChannels=block.dim(1)/4
        return block.reshaped(x.dim(0),side,side,2,2,outputChannels)
            .transposed(axes:[0,1,3,2,4,5]).reshaped(x.dim(0),side*2,side*2,outputChannels)
    }
    /// Decoder-tail upsampling stages. FP32 rows take the fused Metal GEMM
    /// (bias + pixel shuffle + LayerNorm/GELU epilogue in one pass; measured
    /// 1.4×/2.6× the MLX pipeline per stage on M3); WAIFU_SAFIRE_UPSCALE=mlx,
    /// FP16 checkpoints and any shape the kernels do not cover fall back to
    /// the MLX pipeline above.
    private func upscaleStage(_ x:MLXArray,_ name:String,lnName:String?=nil,geluAfter:Bool=false) -> MLXArray {
        if x.dtype == .float32, ProcessInfo.processInfo.environment["WAIFU_SAFIRE_UPSCALE"] != "mlx" {
            if let lnName, x.dim(1) == 64, x.dim(3) == 256 {
                return SAFIREUpscale.upscale1(x,weight:upscaleWeights[name]!,
                    bias:upscaleBiases[name]!,lnWeight:w(lnName+".weight"),lnBias:w(lnName+".bias"),eps:1e-6)
            }
            if lnName == nil, geluAfter, x.dim(1) == 128, x.dim(3) == 64 {
                return SAFIREUpscale.upscale2(x,weight:upscaleWeights[name]!,bias:upscaleBiases[name]!)
            }
        }
        var value=upscale(x,name)
        if let lnName { value=norm(value,lnName,geluAfter:true) }
        else if geluAfter { value=gelu(value) }
        return value
    }
    private func decoderTail(queries:MLXArray,keys:MLXArray,batch:Int) -> (MLXArray,MLXArray) {
        let tailQueries=queries.asType(decoderTailDType)
        let tailKeys=keys.asType(decoderTailDType)
        var mark=Date()
        var up=upscaleStage(tailKeys.reshaped(batch,64,64,256),"mask_decoder.output_upscaling.0",
            lnName:"mask_decoder.output_upscaling.1")
        record("tail.upscale1",mark,up); mark=Date()
        up=upscaleStage(up,"mask_decoder.output_upscaling.3",geluAfter:true)
        record("tail.upscale2",mark,up); mark=Date()
        // multimask_output=False selects mask token zero. Other tokens still take
        // part in two-way attention, just as upstream; unused hypernets are skipped.
        let hyper=mlp(tailQueries[0...,1,0...],"mask_decoder.output_hypernetworks_mlps.0")
        record("tail.hyper",mark,hyper); mark=Date()
        // Mask logits are the threshold-bearing output: evaluate the final
        // projection in FP32 in every precision mode.
        // Mask logits are the threshold-bearing output: project in FP32 in
        // every precision mode via the fused per-prompt dot kernel.
        let masks=SAFIRENorm.maskProject(up.reshaped(batch,65536,32).asType(.float32),
            hyper:hyper.asType(.float32),batch:batch).reshaped(batch,256,256)
        record("tail.maskProject",mark,masks); mark=Date()
        let quality=mlp(tailQueries[0...,0,0...],"mask_decoder.iou_prediction_head")[0...,0].asType(.float32)
        record("tail.quality",mark,quality)
        return (masks,quality)
    }
    /// One compiled automatic-grid chunk: decode `chunk` prompts, then produce
    /// thresholded areas, stride-16 binary samples, pooled features and quality
    /// entirely on the GPU. Nothing larger than the masks is read back.
    private func gridChunkGraph(embedding:MLXArray,keys:MLXArray,tokens:MLXArray) -> [MLXArray] {
        let chunk=tokens.dim(0)
        var mark=Date()
        let (queries,keysOut)=twoWayTransformer(embedding:embedding,tokens:tokens,sharedKeys:keys)
        record("decodeChunk.transformer",mark,keysOut); mark=Date()
        let (masks,quality)=decoderTail(queries:queries,keys:keysOut,batch:chunk)
        record("decodeChunk.tail",mark,masks); mark=Date()
        let post=SAFIREMaskPost.apply(masks,batch:chunk)
        let counts=sum(post.samples,axes:[1,2])
        let pooledFlat=matmul(post.samples.reshaped(chunk,4096),embedding.reshaped(4096,256).asType(.float32))
        let pooled=pooledFlat/maximum(counts.expandedDimensions(axis:1),MLXArray(Float(1)))
        let areas=sum(post.partials,axes:[1,2])
        return [masks,areas,pooled,counts,quality]
    }
    private func gridGraph(_ chunk:Int) -> ([MLXArray])->[MLXArray] {
        if let cached=compiledGrid[chunk] { return cached }
        let graph: ([MLXArray])->[MLXArray]
        if profile.compileEncoder {
            graph=compile { [unowned self] (inputs:[MLXArray]) -> [MLXArray] in
                self.gridChunkGraph(embedding:inputs[0],keys:inputs[1],tokens:inputs[2])
            }
        } else {
            // Eager escape hatch for stage profiling.
            graph={ [unowned self] inputs in
                self.gridChunkGraph(embedding:inputs[0],keys:inputs[1],tokens:inputs[2])
            }
        }
        compiledGrid[chunk]=graph
        return graph
    }
    /// Automatic-grid decode of all 256 prompts with a single GPU→CPU
    /// synchronization; masks stay GPU-resident for the representatives.
    func gridPredictions(embedding:MLXArray) -> (masks:MLXArray,areas:[Int32],counts:[Float],features:[[Float]],qualities:[Float]) {
        let chunk=profile.promptBatch
        precondition(256%chunk==0,"Prompt chunk must divide the 16×16 grid")
        let embedding=embedding.asType(promptDType)
        let keys0=embedding.reshaped(1,4096,256)+w("prompt_encoder.no_mask_embed.weight")
        let graph=gridGraph(chunk)
        var chunkMasks:[MLXArray]=[],chunkAreas:[MLXArray]=[],chunkPooled:[MLXArray]=[],chunkCounts:[MLXArray]=[],chunkQuality:[MLXArray]=[]
        var mark=Date()
        if profile.compileEncoder { inCompiledGraph=true }
        for start in stride(from:0,to:256,by:chunk) {
            let result=graph([embedding,keys0,gridTokens[start..<start+chunk]])
            chunkMasks.append(result[0]);chunkAreas.append(result[1]);chunkPooled.append(result[2])
            chunkCounts.append(result[3]);chunkQuality.append(result[4])
            // Per-chunk completion fence: bounds the allocator's live set to one
            // chunk working set (a single deferred sweep peaked at ~2 GiB),
            // without any GPU→CPU readback.
            eval(result)
            if let recorder=stageRecorder {
                recorder("decode.chunk\(start/chunk)",Date().timeIntervalSince(mark),Memory.activeMemory,Memory.peakMemory)
            }
        }
        inCompiledGraph=false
        record("decode.grid",mark,concatenated(chunkMasks,axis:0)); mark=Date()
        let areas=concatenated(chunkAreas,axis:0),counts=concatenated(chunkCounts,axis:0)
        let pooled=concatenated(chunkPooled,axis:0),quality=concatenated(chunkQuality,axis:0)
        eval(areas,counts,pooled,quality)
        let areaValues=areas.asArray(Int32.self),countValues=counts.asArray(Float.self)
        let pooledValues=pooled.asArray(Float.self),qualityValues=quality.asArray(Float.self)
        record("decode.readback",mark,pooled)
        var features:[[Float]]=[]
        for j in 0..<256 { features.append(Array(pooledValues[(j*256)..<((j+1)*256)])) }
        return (concatenated(chunkMasks,axis:0),areaValues,countValues,features,qualityValues)
    }
    private func relAttention(_ x:MLXArray,_ prefix:String) -> MLXArray {
        let batch=x.dim(0), side=x.dim(1), n=side*side
        let qkv=linear(x,prefix+".qkv").reshaped(batch,n,3,12,64).transposed(axes:[2,0,3,1,4])
        let q=qkv[0], k=qkv[1], v=qkv[2]
        guard let (rh,rw)=relTables[prefix] else { preconditionFailure("Missing relative position tables: \(prefix)") }
        let rows=side==64 ? profile.queryRows : n
        var chunks: [MLXArray] = []
        for start in stride(from:0,to:n,by:rows) {
            let end=min(n,start+rows)
            let qc=q[0...,0...,start..<end,0...]
            // Query rows are (row,column) pairs; whole side-sized row bands
            // share one gathered height table, so the chunked matmuls run
            // against init-time tables with no runtime gather.
            let grouped=qc.reshaped(batch,12,(end-start)/side,side,64)
            let height: MLXArray, width: MLXArray
            if rows<n {
                // Chunked global path: per-query rel-position terms as batched
                // matmuls against the init-time gathered tables (the einsum
                // lowering measured slower for the same 25 MFLOP).
                let bands=(end-start)/side
                height=matmul(grouped.reshaped(12,bands,64,64),
                    rh[start/side..<end/side].transposed(axes:[0,2,1]).expandedDimensions(axis:0))
                    .reshaped(1,12,bands,64,64)
                width=matmul(grouped.transposed(axes:[3,1,2,0,4]).reshaped(side,12*bands,64),
                    rw.transposed(axes:[0,2,1]))
                    .reshaped(side,12,bands,64).transposed(axes:[1,2,0,3]).expandedDimensions(axis:0)
            } else {
                // Window blocks: single chunk inside their own compiled block
                // graph; the einsum forms trace into the compiled graph.
                height=einsum("bhgxc,gkc->bhgxk",grouped,rh)
                width=einsum("bhgxc,xkc->bhgxk",grouped,rw)
            }
            let bias: MLXArray
            if rows<n && !inCompiledGraph {
                // Chunked global attention: the fused outer-sum kernel writes
                // the 50 MB bias tile at streaming rate from the two small
                // per-query table rows (the MLX broadcast add runs ~26 GB/s).
                bias=SAFIREBias.outerSum(height:height,width:width,rows:end-start)
            } else {
                // Window blocks: single chunk inside their own compiled block
                // graph, where the MLX broadcast add is traced and fused.
                bias=(height.expandedDimensions(axis:-1)+width.expandedDimensions(axis:-2)).reshaped(batch,12,end-start,n)
            }
            let attended=MLXFast.scaledDotProductAttention(queries:qc,keys:k,values:v,scale:0.125,mask:bias)
            // Fence each relative-position tile so its 50 MB bias and scores
            // return to the allocator before the next tile.
            if !inCompiledGraph { eval(attended) }
            chunks.append(attended)
        }
        let joined=concatenated(chunks,axis:2).transposed(axes:[0,2,1,3]).reshaped(batch,side,side,768)
        return linear(joined,prefix+".proj")
    }
    private func encoderBlock(_ input:MLXArray,index:Int) -> MLXArray {
        let prefix="image_encoder.blocks.\(index)"
        var x=norm(input,prefix+".norm1")
        let global=[2,5,8,11].contains(index)
        if !global {
            x=padded(x,widths:[.init(0),.init((0,6)),.init((0,6)),.init(0)])
            x=x.reshaped(1,5,14,5,14,768).transposed(axes:[0,1,3,2,4,5]).reshaped(25,14,14,768)
        }
        x=relAttention(x,prefix+".attn")
        if !global {
            x=x.reshaped(1,5,5,14,14,768).transposed(axes:[0,1,3,2,4,5]).reshaped(1,70,70,768)
            x=x[0...,0..<64,0..<64,0...]
        }
        x=input+x
        return x+linear(gelu(linear(norm(x,prefix+".norm2"),prefix+".mlp.lin1")),prefix+".mlp.lin2")
    }
    /// Input is normalized FP32 NHWC, exactly [1,1024,1024,3]. The FFT
    /// high-pass branch stays FP32; later stages run in the checkpoint
    /// precision with FP32 softmax and FP32 mask-logit projection.
    public func encode(_ input:MLXArray,activation: ((String,MLXArray)throws->Void)? = nil) throws -> MLXArray {
        guard input.shape == [1,1024,1024,3], input.dtype == .float32 else { throw SAFIREError.invalid("Invalid encoder input") }
        if activation==nil,profile.compileEncoder {
            // Window blocks compile per block; the four global blocks keep
            // their eager chunk loop (compiled whole-block replays accumulate
            // every 50 MB bias/score tile and peaked at ~2 GiB) with the fused
            // bias outer-sum kernel feeding SDPA per chunk.
            let (patched,tuned,_)=encodeStem(input)
            var hidden=patched
            for i in 0..<12 {
                if [2,5,8,11].contains(i) {
                    hidden=adaptorBlock(hidden,tuned:tuned,index:i)
                    eval(hidden)
                } else {
                    let block=compiledBlocks[i] ?? compile { [unowned self] (inputs:[MLXArray]) -> [MLXArray] in
                        [self.adaptorBlock(inputs[0],tuned:inputs[1],index:i)]
                    }
                    if compiledBlocks[i]==nil { compiledBlocks[i]=block }
                    inCompiledGraph=true
                    hidden=block([hidden,tuned])[0]
                    inCompiledGraph=false
                    eval(hidden)
                }
            }
            return encodeNeck(hidden,tuned:tuned)
        }
        return try encodeCore(input,activation:activation)
    }
    /// Stage attribution barrier: with a recorder attached, the stage output
    /// is materialized so wall-clock times attribute to the producing stage.
    /// Production inference (no recorder) stays fully lazy.
    func record(_ name:String,_ start:Date,_ value:MLXArray? = nil) {
        guard let recorder=stageRecorder,!inCompiledGraph else { return }
        if let value { eval(value) }
        recorder(name,Date().timeIntervalSince(start),Memory.activeMemory,Memory.peakMemory)
    }
    private func encodeCore(_ input:MLXArray,activation: ((String,MLXArray)throws->Void)? = nil) throws -> MLXArray {
        var mark=Date()
        let (patched,tuned,high)=encodeStem(input)
        var x=patched
        record("encoder.patchEmbed",mark,tuned); mark=Date()
        try activation?("highpass",high); try activation?("patch_embedding",x); try activation?("adapter_input",tuned)
        for i in 0..<12 {
            x=adaptorBlock(x,tuned:tuned,index:i)
            // Completion fence per block: without it the single deferred sweep
            // denies the allocator cross-block reuse and peaks at ~2 GiB.
            // Compiled replays trace this body once; they must stay eval-free.
            if stageRecorder==nil && !inCompiledGraph { eval(x) }
            record("encoder.block_\(i)",mark,x); mark=Date()
            try activation?("block_\(i)",x)
        }
        x=encodeNeck(x,tuned:tuned)
        record("encoder.neck",mark,x)
        try activation?("embedding",x)
        return x
    }
    /// FFT high-pass branch, patch embeddings and the adapter injection input.
    private func encodeStem(_ input:MLXArray) -> (x:MLXArray,tuned:MLXArray,high:MLXArray) {
        let frequency=MLXFFT.fft2(input,axes:[1,2])*highpass
        let high=abs(MLXFFT.ifft2(frequency,axes:[1,2]).realPart()).asType(encoderDType)
        let x=conv(input.asType(encoderDType),"image_encoder.patch_embed.proj",stride:16)+w("image_encoder.pos_embed")
        let y=conv(high,"image_encoder.hfp_overlap_patch_embed.proj",stride:16,padding:8)
        let tuned=linear(x,"image_encoder.adaptor_patch_embed_tune")+linear(y,"image_encoder.hfp_patch_embed_tune")
        return (x,tuned,high)
    }
    private func adaptorBlock(_ x:MLXArray,tuned:MLXArray,index:Int) -> MLXArray {
        let injected=x+linear(gelu(linear(tuned,"image_encoder.adaptor_blocks.\(index).0")),"image_encoder.adaptor_shared_up_projection_layer")
        return encoderBlock(injected,index:index)
    }
    private func encodeNeck(_ x:MLXArray,tuned:MLXArray) -> MLXArray {
        var value=x+linear(gelu(linear(tuned,"image_encoder.adaptor_final_block.0")),"image_encoder.adaptor_shared_up_projection_layer")
        value=norm(conv(value,"image_encoder.neck.0"),"image_encoder.neck.1")
        value=norm(conv(value,"image_encoder.neck.2",padding:1),"image_encoder.neck.3")
        return value
    }
}

public struct SAFIREProfile: Codable {
    public let device: String
    public let provenance: String
    public let queryRows: Int
    /// Legacy name kept for the benchmark harness; the automatic 16×16 prompt
    /// grid is decoded in chunks of this many prompts.
    public let promptBatch: Int
    public var decodeChunk: Int { promptBatch }
    public let precision: SAFIREPrecision
    public let precisionOverridden: Bool
    public let fp16Scope: String
    public let compileEncoder: Bool
    public let runtimeRevision: String
    public let operations: [String:String]
    public let evidence: String
    public let maximumThreadgroupBytes: Int

    init(device:String,provenance:String,queryRows:Int,promptBatch:Int,precision:SAFIREPrecision,precisionOverridden:Bool,fp16Scope:String,
         compileEncoder:Bool,runtimeRevision:String,operations:[String:String],evidence:String,maximumThreadgroupBytes:Int) {
        self.device=device;self.provenance=provenance;self.queryRows=queryRows;self.promptBatch=promptBatch
        self.precision=precision;self.precisionOverridden=precisionOverridden;self.fp16Scope=fp16Scope
        self.compileEncoder=compileEncoder;self.runtimeRevision=runtimeRevision
        self.operations=operations;self.evidence=evidence;self.maximumThreadgroupBytes=maximumThreadgroupBytes
    }
    public init(from decoder:Decoder) throws {
        let container=try decoder.container(keyedBy:CodingKeys.self)
        device=try container.decode(String.self,forKey:.device)
        provenance=try container.decode(String.self,forKey:.provenance)
        queryRows=try container.decode(Int.self,forKey:.queryRows)
        promptBatch=try container.decode(Int.self,forKey:.promptBatch)
        precision=try container.decodeIfPresent(SAFIREPrecision.self,forKey:.precision) ?? .fp32
        precisionOverridden=try container.decodeIfPresent(Bool.self,forKey:.precisionOverridden) ?? false
        fp16Scope=try container.decodeIfPresent(String.self,forKey:.fp16Scope) ?? "all" 
        compileEncoder=try container.decodeIfPresent(Bool.self,forKey:.compileEncoder) ?? false
        runtimeRevision=try container.decode(String.self,forKey:.runtimeRevision)
        operations=try container.decode([String:String].self,forKey:.operations)
        evidence=try container.decode(String.self,forKey:.evidence)
        maximumThreadgroupBytes=try container.decode(Int.self,forKey:.maximumThreadgroupBytes)
    }
    public static func current() throws -> Self {
        guard let device=MTLCreateSystemDefaultDevice(), device.supportsFamily(.apple7),
              device.maxThreadsPerThreadgroup.width >= 256 else { throw SAFIREError.invalid("An Apple Silicon Metal GPU is required") }
        let environment=ProcessInfo.processInfo.environment
        let rows=Int(environment["WAIFU_SAFIRE_QUERY_ROWS"] ?? Self.defaultQueryRows(device:device.name)) ?? 0
        let batch=Int(environment["WAIFU_SAFIRE_PROMPT_BATCH"] ?? Self.defaultPromptBatch(device:device.name)) ?? 0
        let precisionRaw=environment["WAIFU_SAFIRE_PRECISION"]
        let precision=SAFIREPrecision(rawValue:precisionRaw ?? "fp16") ?? .fp32
        let fp16Scope=environment["WAIFU_SAFIRE_FP16_SCOPE"] ?? "transformer"
        guard ["all","encoder","decoder","transformer"].contains(fp16Scope) else {
            throw SAFIREError.invalid("Invalid FP16 scope")
        }
        let compile=Int(environment["WAIFU_SAFIRE_COMPILE"] ?? "1") != 0
        guard [64,128,256].contains(rows),[1,2,4,8,16,32,64].contains(batch),
              device.maxBufferLength>=128*1024*1024 else {
            throw SAFIREError.invalid("Invalid or unsupported SAFIRE workload profile")
        }
        let overridden=environment["WAIFU_SAFIRE_QUERY_ROWS"] != nil || environment["WAIFU_SAFIRE_PROMPT_BATCH"] != nil
            || environment["WAIFU_SAFIRE_PRECISION"] != nil || environment["WAIFU_SAFIRE_COMPILE"] != nil
            || environment["WAIFU_SAFIRE_FP16_SCOPE"] != nil
        let provenance: String
        if overridden { provenance="benchmark-override" }
        else if device.name == "Apple M4" { provenance="conservative-unmeasured-grid-inheritance" }
        else if device.name == "Apple M3" { provenance="measured-m3-rows256-chunk4" }
        else { provenance="conservative-unmeasured-fallback" }
        return .init(device:device.name,provenance:provenance,queryRows:rows,promptBatch:batch,precision:precision,
            precisionOverridden:precisionRaw != nil,fp16Scope:fp16Scope,
            compileEncoder:compile,runtimeRevision:SAFIREModel.revision,
            operations:["preprocessing":"fused fixed-point RGBA/normalize, 256 threads",
                "encoderAttention":"relative-position SDPA from init-time tables, \(rows) query rows, fused bias outer-sum kernel",
                "decoder":"compiled fixed-shape grid chunks of \(batch); unbatched first-layer keys; pe folded into q/k projections",
                "maskPost":"fused resize/threshold/sample/area Metal kernel",
                "normalization":"fused one-pass residual/LayerNorm/GELU Metal kernel (FP32 rows)",
                "upscaling":"fused FP32 upsample GEMM kernels (bias + pixel shuffle + LayerNorm/GELU epilogue, one pass; WAIFU_SAFIRE_UPSCALE=mlx reverts to MLX matmul + separate passes)",
                "maskProject":"fused per-prompt hypernet dot kernel",
                "GEMM":"MLX library device-aware tiles, pre-transposed weights",
                "precision":precision == .fp16 && fp16Scope == "transformer"
                    ? "FP16 two-way decoder transformer; FP32 encoder, prompt encoder, upscaling tail, FFT, softmax and mask logits"
                    : "\(precision.rawValue.uppercased()) precision scope: \(fp16Scope); FP32 FFT, softmax and mask logits"],
            evidence:"docs/ImageLineage-native-validation.json; docs/ImageLineage-native-benchmark.json",
            maximumThreadgroupBytes:device.maxThreadgroupMemoryLength)
    }
    /// M3 paired sweep (2026-10-24): 256-row global-attention chunks beat
    /// 128 rows 3/3 interleaved rounds (2.304–2.310s vs 2.331–2.341s) — half
    /// the SDPA dispatches per global block. M4 inherits conservatively.
    private static func defaultQueryRows(device:String) -> String {
        "256"
    }
    private static func defaultPromptBatch(device:String) -> String {
        // M3 measured chunk sweep (2026-10-24, warm worker, gradient fixture,
        // eager global blocks + compiled window blocks): chunk 4 runs 2.28s at
        // a 1110 MB MLX peak; larger chunks only grow the upscale working set.
        // M4 has not been measured on this graph revision and inherits the
        // same chunk, unmeasured.
        "4"
    }
}
