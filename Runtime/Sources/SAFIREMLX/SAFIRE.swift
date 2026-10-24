// Native adaptation of SAFIRE (Kwon et al.) and SAM (Meta Platforms).
// See Runtime/SAFIRE-NOTICE.md for provenance and Apache-2.0 notices.
import Foundation
import CryptoKit
import MLX
import Metal

public enum SAFIREError: Error { case invalid(String) }

/// Serial worker-owned, full precision reference port. Window attention and
/// chunked global relative-position attention preserve the upstream computation.
/// No Swin routing, cosine normalization, quantization or approximate GELU.
public final class SAFIREModel {
    public static let revision = "safire-native-v1"
    private var weights: [String:MLXArray]
    private let profile: SAFIREProfile
    private let highpass: MLXArray
    private let densePE: MLXArray
    private var compiledDecoders: [Int:([MLXArray])->[MLXArray]] = [:]

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
        weights = try loadArrays(url:modelURL)
        guard weights.count == 348, weights["image_encoder.pos_embed"]?.shape == [1,64,64,768],
              weights["prompt_encoder.pe_layer.positional_encoding_gaussian_matrix"]?.shape == [2,128] else {
            throw SAFIREError.invalid("Expected the official 348-tensor SAFIRE ViT-B adapter checkpoint.")
        }
        // One-time kernel layout preparation; original safetensors stay lossless.
        for key in Array(weights.keys) where key.hasSuffix(".weight") && weights[key]!.ndim == 4 {
            let transpose = key.contains("output_upscaling") ? [1,2,3,0] : [0,2,3,1]
            weights[key] = weights[key]!.transposed(axes:transpose)
        }
        var mask = [Float](repeating:1,count:1024*1024)
        // torch fftshift's centered [256,768) rejection, expressed unshifted.
        for y in 0..<1024 where y < 256 || y >= 768 {
            for x in 0..<1024 where x < 256 || x >= 768 { mask[y*1024+x] = 0 }
        }
        highpass = MLXArray(mask,[1,1024,1024,1])
        var grid: [Float] = []
        for i in 0..<4096 { grid.append(Float(i%64)+0.5); grid.append(Float(i/64)+0.5) }
        let coords = MLXArray(grid,[1,4096,2])/64
        let phase = matmul(coords*2-1,weights["prompt_encoder.pe_layer.positional_encoding_gaussian_matrix"]!)*(2*Float.pi)
        densePE = concatenated([sin(phase),cos(phase)],axis:-1)
        eval(Array(weights.values)); eval(highpass,densePE)
    }
    public var diagnostics: SAFIREProfile { profile }
    private func w(_ name:String) -> MLXArray {
        guard let value = weights[name] else { preconditionFailure("Missing SAFIRE parameter: \(name)") }
        return value
    }
    private func linear(_ x:MLXArray,_ name:String) -> MLXArray {
        let output = matmul(x,w(name+".weight").transposed())
        return weights[name+".bias"].map { output + $0 } ?? output
    }
    private func norm(_ x:MLXArray,_ name:String,eps:Float=1e-6) -> MLXArray {
        MLXFast.layerNorm(x,weight:w(name+".weight"),bias:w(name+".bias"),eps:eps)
    }
    private func conv(_ x:MLXArray,_ name:String,stride:Int=1,padding:Int=0,transpose:Bool=false) -> MLXArray {
        let y = transpose ? convTransposed2d(x,w(name+".weight"),stride:.init(stride),padding:.init(padding))
                          : conv2d(x,w(name+".weight"),stride:.init(stride),padding:.init(padding))
        return weights[name+".bias"].map { y + $0 } ?? y
    }
    private func gelu(_ x:MLXArray) -> MLXArray { x * 0.5 * (1+erf(x / Float(sqrt(2.0)))) }
    private func relAttention(_ x:MLXArray,_ prefix:String) -> MLXArray {
        let batch=x.dim(0), side=x.dim(1), n=side*side
        let qkv=linear(x,prefix+".qkv").reshaped(batch,n,3,12,64).transposed(axes:[2,0,3,1,4])
        let q=qkv[0], k=qkv[1], v=qkv[2]
        let indices = MLXArray((0..<side).flatMap { y in (0..<side).map { Int32(y-$0+side-1) } },[side,side])
        let rh=w(prefix+".rel_pos_h")[indices], rw=w(prefix+".rel_pos_w")[indices]
        var chunks: [MLXArray] = []
        for start in stride(from:0,to:n,by:profile.queryRows) {
            let end=min(n,start+profile.queryRows), count=end-start
            let row=MLXArray((start..<end).map { Int32($0/side) }), col=MLXArray((start..<end).map { Int32($0%side) })
            let qc=q[0...,0...,start..<end,0...]
            let height=einsum("bhqc,qkc->bhqk",qc,rh[row])
            let width=einsum("bhqc,qkc->bhqk",qc,rw[col])
            let bias=(height.expandedDimensions(axis:-1)+width.expandedDimensions(axis:-2)).reshaped(batch,12,count,n)
            let result=MLXFast.scaledDotProductAttention(queries:qc,keys:k,values:v,scale:0.125,mask:bias)
            // Release each relative-position score tile before the next tile.
            eval(result); chunks.append(result)
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
    /// Input is normalized FP32 NHWC, exactly [1,1024,1024,3].
    public func encode(_ input:MLXArray,activation: ((String,MLXArray)throws->Void)? = nil) throws -> MLXArray {
        guard input.shape == [1,1024,1024,3], input.dtype == .float32 else { throw SAFIREError.invalid("Invalid encoder input") }
        let frequency=MLXFFT.fft2(input,axes:[1,2])*highpass
        let high=abs(MLXFFT.ifft2(frequency,axes:[1,2]).realPart())
        var x=conv(input,"image_encoder.patch_embed.proj",stride:16)+w("image_encoder.pos_embed")
        let y=conv(high,"image_encoder.hfp_overlap_patch_embed.proj",stride:16,padding:8)
        let tuned=linear(x,"image_encoder.adaptor_patch_embed_tune")+linear(y,"image_encoder.hfp_patch_embed_tune")
        eval(x,tuned)
        try activation?("highpass",high); try activation?("patch_embedding",x); try activation?("adapter_input",tuned)
        for i in 0..<12 {
            x=x+linear(gelu(linear(tuned,"image_encoder.adaptor_blocks.\(i).0")),"image_encoder.adaptor_shared_up_projection_layer")
            x=encoderBlock(x,index:i); eval(x)
            try activation?("block_\(i)",x)
        }
        x=x+linear(gelu(linear(tuned,"image_encoder.adaptor_final_block.0")),"image_encoder.adaptor_shared_up_projection_layer")
        x=norm(conv(x,"image_encoder.neck.0"),"image_encoder.neck.1")
        x=norm(conv(x,"image_encoder.neck.2",padding:1),"image_encoder.neck.3")
        eval(x); try activation?("embedding",x)
        return x
    }
    private func decoderAttention(q:MLXArray,k:MLXArray,v:MLXArray,prefix:String) -> MLXArray {
        let a=linear(q,prefix+".q_proj"), b=linear(k,prefix+".k_proj"), c=linear(v,prefix+".v_proj")
        func heads(_ x:MLXArray) -> MLXArray { x.reshaped(x.dim(0),x.dim(1),8,x.dim(2)/8).transposed(axes:[0,2,1,3]) }
        let output=MLXFast.scaledDotProductAttention(queries:heads(a),keys:heads(b),values:heads(c),scale:1/sqrt(Float(a.dim(2)/8)),mask:nil)
        return linear(output.transposed(axes:[0,2,1,3]).reshaped(a.dim(0),a.dim(1),a.dim(2)),prefix+".out_proj")
    }
    private func mlp(_ x:MLXArray,_ prefix:String) -> MLXArray {
        var value=x
        for i in 0..<3 { value=linear(value,prefix+".layers.\(i)"); if i<2 { value=maximum(value,0) } }
        return value
    }
    public func decode(embedding:MLXArray,points:MLXArray) throws -> (masks:MLXArray,quality:MLXArray) {
        guard embedding.shape == [1,64,64,256], points.ndim == 2, points.dim(1) == 2,
              (1...16).contains(points.dim(0)) else { throw SAFIREError.invalid("Invalid prompt batch") }
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
        let result=graph([embedding,points]);eval(result)
        return (result[0],result[1])
    }
    private func decodeGraph(embedding:MLXArray,points:MLXArray) -> (MLXArray,MLXArray) {
        let batch=points.dim(0)
        let phase=matmul((points+0.5)/1024*2-1,w("prompt_encoder.pe_layer.positional_encoding_gaussian_matrix"))*(2*Float.pi)
        let positive=concatenated([sin(phase),cos(phase)],axis:-1)+w("prompt_encoder.point_embeddings.1.weight")
        let padding=broadcast(w("prompt_encoder.not_a_point_embed.weight"),to:[batch,1,256])
        let sparse=concatenated([positive.expandedDimensions(axis:1),padding],axis:1)
        let outputTokens=concatenated([w("mask_decoder.iou_token.weight"),w("mask_decoder.mask_tokens.weight")],axis:0)
        let tokens=concatenated([broadcast(outputTokens.expandedDimensions(axis:0),to:[batch,5,256]),sparse],axis:1)
        var queries=tokens
        var keys=broadcast(embedding.reshaped(1,4096,256)+w("prompt_encoder.no_mask_embed.weight"),to:[batch,4096,256])
        let pe=broadcast(densePE,to:[batch,4096,256])
        for i in 0..<2 {
            let p="mask_decoder.transformer.layers.\(i)"
            if i == 0 { queries=decoderAttention(q:queries,k:queries,v:queries,prefix:p+".self_attn") }
            else { queries=queries+decoderAttention(q:queries+tokens,k:queries+tokens,v:queries,prefix:p+".self_attn") }
            queries=norm(queries,p+".norm1",eps:1e-5)
            queries=norm(queries+decoderAttention(q:queries+tokens,k:keys+pe,v:keys,prefix:p+".cross_attn_token_to_image"),p+".norm2",eps:1e-5)
            queries=norm(queries+linear(maximum(linear(queries,p+".mlp.lin1"),0),p+".mlp.lin2"),p+".norm3",eps:1e-5)
            keys=norm(keys+decoderAttention(q:keys+pe,k:queries+tokens,v:queries,prefix:p+".cross_attn_image_to_token"),p+".norm4",eps:1e-5)
        }
        queries=norm(queries+decoderAttention(q:queries+tokens,k:keys+pe,v:keys,prefix:"mask_decoder.transformer.final_attn_token_to_image"),"mask_decoder.transformer.norm_final_attn",eps:1e-5)
        var up=conv(keys.reshaped(batch,64,64,256),"mask_decoder.output_upscaling.0",stride:2,transpose:true)
        up=gelu(norm(up,"mask_decoder.output_upscaling.1"))
        up=gelu(conv(up,"mask_decoder.output_upscaling.3",stride:2,transpose:true))
        // multimask_output=False selects mask token zero. Other tokens still take
        // part in two-way attention, just as upstream; unused hypernets are skipped.
        let hyper=mlp(queries[0...,1,0...],"mask_decoder.output_hypernetworks_mlps.0")
        let masks=matmul(up.reshaped(batch,65536,32),hyper.expandedDimensions(axis:-1)).reshaped(batch,256,256)
        let quality=mlp(queries[0...,0,0...],"mask_decoder.iou_prediction_head")[0...,0]
        return (masks,quality)
    }
}

public struct SAFIREProfile: Codable {
    public let device: String
    public let provenance: String
    public let queryRows: Int
    public let promptBatch: Int
    public let runtimeRevision: String
    public let operations: [String:String]
    public let evidence: String
    public let maximumThreadgroupBytes: Int
    public static func current() throws -> Self {
        guard let device=MTLCreateSystemDefaultDevice(), device.supportsFamily(.apple7),
              device.maxThreadsPerThreadgroup.width >= 256 else { throw SAFIREError.invalid("An Apple Silicon Metal GPU is required") }
        // Both M3 and M4 deliberately use an unmeasured conservative configuration
        // until independent full-model reports justify an exact-device entry.
        let environment=ProcessInfo.processInfo.environment
        let rows=Int(environment["WAIFU_SAFIRE_QUERY_ROWS"] ?? "128") ?? 0
        // M4's independently repeated complete-inference trials favored one
        // prompt per compiled decoder graph. M3 retains its conservative batch 4.
        let defaultBatch=device.name == "Apple M4" ? "1" : "4"
        let batch=Int(environment["WAIFU_SAFIRE_PROMPT_BATCH"] ?? defaultBatch) ?? 0
        guard [64,128,256].contains(rows),[1,2,4,8,16].contains(batch),device.maxBufferLength>=128*1024*1024 else {
            throw SAFIREError.invalid("Invalid or unsupported SAFIRE workload profile")
        }
        let overridden=environment["WAIFU_SAFIRE_QUERY_ROWS"] != nil || environment["WAIFU_SAFIRE_PROMPT_BATCH"] != nil
        let provenance: String
        if overridden { provenance="benchmark-override" }
        else if device.name == "Apple M4" { provenance="measured-m4-paired-full-inference" }
        else if device.name == "Apple M3" { provenance="measured-m3-conservative-baseline" }
        else { provenance="conservative-unmeasured-fallback" }
        return .init(device:device.name,provenance:provenance,queryRows:rows,promptBatch:batch,runtimeRevision:SAFIREModel.revision,
            operations:["preprocessing":"fused fixed-point RGBA/normalize, 256 threads", "encoderAttention":"relative-position SDPA, \(rows) query rows", "decoder":"compiled fixed-batch \(batch)", "GEMM":"MLX library device-aware tiles", "normalization":"MLX fused FP32 LayerNorm", "precision":"FP32"],
            evidence:"docs/ImageLineage-native-validation.json",maximumThreadgroupBytes:device.maxThreadgroupMemoryLength)
    }
}
