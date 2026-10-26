// Native adaptation of GRIP-UNINA TruFor (CVPR 2023), informational/nonprofit only.
// Copyright (c) 2023 GRIP-UNINA. CMX Copyright (c) 2022 Huayao Liu (MIT).
// Retain Runtime/TRUFOR-LICENSE.txt and TRUFOR-CMX-LICENSE.txt with adaptations.
import Foundation
import CryptoKit
import CoreML
import MLX
import Metal

public enum TruForError:Error { case invalid(String) }
public struct TruForOutput {
    public let score:MLXArray
    public let anomaly:MLXArray?
    public let reliability:MLXArray?
    public let noiseprint:MLXArray?
}
public final class TruForModel {
    public static let revision="trufor-native-v1"
    public static let preprocessing="trufor-native-rgb-div256-v1"
    public static let sourceSHA="ac1d90e329a72e0d66e8665e123a19e94bfae3209c3ef8a4f9ca3b91578c7844"
    public enum Precision:String { case fp32,fp16 }
    /// Production defaults to the FP32 path validated against the official
    /// detection score and spatial outputs. FP16 remains an explicit benchmark
    /// experiment: its Noiseprint residual and localization exceed strict gates.
    public let precision:Precision
    private var weights:[String:MLXArray]
    private var fuseParts:[String:[MLXArray]]=[:]
    private var coreMLNoiseprint:CoreMLNoiseprint?
    private var coreMLFailureReported=false
    public let device:String
    public let queryRows:Int
    public let noiseTile:Int
    public var parameterBytes:Int { weights.values.reduce(0){$0+$1.nbytes}+fuseParts.values.flatMap{$0}.reduce(0){$0+$1.nbytes} }
    private static let residentCopy=MLXFast.metalKernel(name:"trufor_resident_weight_copy",inputNames:["input","count"],outputNames:["output"],source:"""
        uint i=thread_position_in_grid.x;
        if(i<uint(count[0])) output[i]=input[i];
        """)
    public init(directory:URL) throws {
        guard let gpu=MTLCreateSystemDefaultDevice(),gpu.supportsFamily(.apple7),gpu.maxThreadsPerThreadgroup.width>=256 else { throw TruForError.invalid("Unsupported GPU") }
        device=gpu.name
        queryRows=Int(ProcessInfo.processInfo.environment["WAIFU_TRUFOR_QUERY_ROWS"] ?? "512") ?? 512
        noiseTile=Int(ProcessInfo.processInfo.environment["WAIFU_TRUFOR_NOISE_TILE"] ?? "512") ?? 512
        guard [128,256,512].contains(queryRows),[128,256,512].contains(noiseTile) else { throw TruForError.invalid("Unsupported profile override") }
        precision=Precision(rawValue:ProcessInfo.processInfo.environment["WAIFU_TRUFOR_PRECISION"] ?? "") ?? .fp32
        if let path=ProcessInfo.processInfo.environment["WAIFU_TRUFOR_COREML_FP16_MODEL"],
           FileManager.default.fileExists(atPath:path) {
            do { coreMLNoiseprint=try CoreMLNoiseprint(modelURL:URL(fileURLWithPath:path)) }
            catch { FileHandle.standardError.write(Data("TruFor Core ML FP16 noiseprint unavailable; using native MLX: \(error)\n".utf8)) }
        }
        let file=directory.appendingPathComponent("model.safetensors")
        let meta=try JSONSerialization.jsonObject(with:Data(contentsOf:directory.appendingPathComponent("architecture.json"))) as? [String:Any]
        guard meta?["sourceSHA256"] as? String == Self.sourceSHA,meta?["preprocessingVersion"] as? String == Self.preprocessing else { throw TruForError.invalid("Expected the verified official TruFor checkpoint") }
        let manifest=try JSONSerialization.jsonObject(with:Data(contentsOf:directory.appendingPathComponent("manifest.json"))) as? [String:Any]
        guard let expected=(manifest?["fileSHA256"] as? [String:String])?["model.safetensors"] else { throw TruForError.invalid("Missing model hash") }
        let handle=try FileHandle(forReadingFrom:file);var hash=SHA256()
        while let data=try handle.read(upToCount:8*1024*1024),!data.isEmpty { hash.update(data:data) };try handle.close()
        guard hash.finalize().map({String(format:"%02x",$0)}).joined()==expected else { throw TruForError.invalid("Damaged TruFor checkpoint") }
        weights=try loadArrays(url:file)
        guard weights.count==927,weights["detection.0.weight"]?.shape == [128,8],weights["backbone.patch_embed1.proj.weight"]?.shape == [64,3,7,7] else { throw TruForError.invalid("Unsupported TruFor architecture") }
        // Materialize parameters into Metal-owned buffers once. Keeping the
        // default CPU load arrays alongside their GPU views inflates footprint.
        for key in Array(weights.keys) {
            autoreleasepool {
                let value=weights[key]!
                let copied=Self.residentCopy([value,MLXArray([Int32(value.size)])],grid:(value.size,1,1),threadGroup:(256,1,1),outputShapes:[value.shape],outputDTypes:[value.dtype])[0]
                eval(copied);weights[key]=copied
            }
        }
        for key in Array(weights.keys) where key.hasSuffix(".weight") && weights[key]!.ndim==4 { weights[key]=weights[key]!.transposed(axes:[0,2,3,1]).contiguous() }
        if precision == .fp16 {
            for key in Array(weights.keys) where !Self.isFP32Island(key) { weights[key]=weights[key]!.asType(.float16,stream:.cpu) }
        }
        // The fused decoder kernel is sliced per source part on every call; slice
        // once at load so each part convolves with a contiguous weight.
        for head in ["decode_head","decode_head_conf"] {
            let kernel=weights[head+".linear_fuse.0.weight"]!
            guard kernel.dim(3)%4==0,kernel.dim(3)/4==kernel.dim(0) else { throw TruForError.invalid("Unsupported decoder kernel") }
            let partWidth=kernel.dim(3)/4
            fuseParts[head]=(0..<4).map{kernel[0...,0...,0...,$0*partWidth..<($0+1)*partWidth].contiguous()}
            weights.removeValue(forKey:head+".linear_fuse.0.weight")
        }
        eval(Array(weights.values));eval(fuseParts.values.flatMap{$0})
    }
    /// FP32 islands: LayerNorm affines (every `norm` call), BN statistics and
    /// affines, the DnCNN BN rows, and the trained detection head.
    private static func isFP32Island(_ key:String)->Bool {
        if key.contains(".norm") || key.contains(".running_") || key.hasPrefix("detection.") { return true }
        if key.hasPrefix("dncnn."),key.hasSuffix(".weight") || key.hasSuffix(".bias") {
            guard let index=Int(key.split(separator:".")[1]),index>=3,index<=45,index%3==0 else { return false }
            return true
        }
        if key.hasPrefix("decode_head.linear_fuse.1.") || key.hasPrefix("decode_head_conf.linear_fuse.1.") { return true }
        return false
    }
    private func w(_ key:String)->MLXArray { weights[key]! }
    private func linear(_ x:MLXArray,_ p:String)->MLXArray {
        let value=matmul(x,w(p+".weight").T)
        return weights[p+".bias"].map{value+$0} ?? value
    }
    private func conv(_ x:MLXArray,_ p:String,stride:Int=1,padding:Int=0,groups:Int=1)->MLXArray {
        let value=conv2d(x,w(p+".weight"),stride:.init(stride),padding:.init(padding),groups:groups)
        return weights[p+".bias"].map{value+$0} ?? value
    }
    /// LayerNorm always reduces in FP32; the affine result returns to the
    /// compute dtype. In FP32 mode this is the plain fast call.
    private func norm(_ x:MLXArray,_ p:String,eps:Float=1e-5)->MLXArray {
        let weight=w(p+".weight"),bias=w(p+".bias")
        guard weight.dtype==x.dtype else { return MLXFast.layerNorm(x.asType(.float32),weight:weight,bias:bias,eps:eps).asType(x.dtype) }
        return MLXFast.layerNorm(x,weight:weight,bias:bias,eps:eps)
    }
    private func bn(_ x:MLXArray,_ p:String,eps:Float=1e-5)->MLXArray {
        let mean=w(p+".running_mean")
        guard mean.dtype==x.dtype else {
            let y=(x.asType(.float32)-mean)*rsqrt(w(p+".running_var")+eps)*w(p+".weight")+w(p+".bias")
            return y.asType(x.dtype)
        }
        return (x-mean)*rsqrt(w(p+".running_var")+eps)*w(p+".weight")+w(p+".bias")
    }
    private func gelu(_ x:MLXArray)->MLXArray { x*(1+erf(x/Float(sqrt(2.0))))*0.5 }
    private func noiseBody(_ x:MLXArray)->MLXArray {
        var x=x
        for level in 0..<17 {
            let i=level==0 ? 0 : level==16 ? 47 : 2+(level-1)*3
            x=Self.timed("dncnn.\(i)"){
                let value=conv(x,"dncnn.\(i)",padding:1);eval(value);return value
            }
            if level>0 && level<16 { x=bn(x,"dncnn.\(i+1)") }
            if level<16 { x=maximum(x,MLXArray(Float(0)).asType(x.dtype)) }
            // Bound live convolution/BN intermediates independently of profiling.
            eval(x)
        }
        return x
    }
    /// Isolated backend-comparison hook; the app's normal inference entry point
    /// continues to own preprocessing, full-model execution, and safety policy.
    public func probeNoiseprintTile(_ rgb:MLXArray)->MLXArray { noiseBody(rgb) }
    /// Exactly the 17-layer receptive field, with halo, preserving native input.
    private func noise(_ x:MLXArray)->MLXArray {
        let h=x.dim(1),width=x.dim(2)
        if h<=noiseTile && width<=noiseTile { return noiseBody(x) }
        var rows:[MLXArray]=[]
        for y in stride(from:0,to:h,by:noiseTile) {
            var columns:[MLXArray]=[]
            for a in stride(from:0,to:width,by:noiseTile) {
                let y0=max(0,y-17),x0=max(0,a-17),y1=min(h,y+noiseTile+17),x1=min(width,a+noiseTile+17)
                let cropped=autoreleasepool {
                    let out=noiseBody(x[0...,y0..<y1,x0..<x1,0...])
                    let result=out[0...,(y-y0)..<min(y-y0+noiseTile,h-y0),(a-x0)..<min(a-x0+noiseTile,width-x0),0...].contiguous()
                    eval(result);return result
                }
                columns.append(cropped);TruForMemoryPolicy.releaseUnusedCPU()
            }
            let row=concatenated(columns,axis:2);eval(row);rows.append(row)
        }
        return concatenated(rows,axis:1)
    }
    private func noise(_ input:TruForInput)throws->MLXArray {
        if let coreMLNoiseprint {
            do { return try coreMLNoiseprint.predict(input,tileSide:noiseTile) }
            catch {
                self.coreMLNoiseprint=nil
                if !coreMLFailureReported {
                    FileHandle.standardError.write(Data("TruFor Core ML FP16 noiseprint failed; falling back to native MLX: \(error)\n".utf8))
                    coreMLFailureReported=true
                }
            }
        }
        if let array=input.array { return noise(precision == .fp16 ? array.asType(.float16):array) }
        let h=input.height,width=input.width
        var rows:[MLXArray]=[]
        for y in stride(from:0,to:h,by:noiseTile) {
            var columns:[MLXArray]=[]
            for x in stride(from:0,to:width,by:noiseTile) {
                let y0=max(0,y-17),x0=max(0,x-17),y1=min(h,y+noiseTile+17),x1=min(width,x+noiseTile+17)
                let cropped=try autoreleasepool {
                    let tile=try input.read(y0..<y1,columns:x0..<x1)
                    let out=noiseBody(precision == .fp16 ? tile.asType(.float16):tile)
                    let result=out[0...,(y-y0)..<min(y-y0+noiseTile,h-y0),(x-x0)..<min(x-x0+noiseTile,width-x0),0...].contiguous()
                    eval(result);return result
                }
                columns.append(cropped);TruForMemoryPolicy.releaseUnusedCPU()
            }
            let row=concatenated(columns,axis:2);eval(row);rows.append(row)
        }
        let output=concatenated(rows,axis:1);eval(output);return output
    }
    private func attentionResidual(_ input:MLXArray,_ p:String,heads:Int,sr:Int)->MLXArray {
        let b=input.dim(0),h=input.dim(1),width=input.dim(2),c=input.dim(3),n=h*width,d=c/heads
        let tokens=norm(input,p+".norm1",eps:1e-6).reshaped(b,n,c)
        let q=linear(tokens,p+".attn.q").reshaped(b,n,heads,d).transposed(axes:[0,2,1,3])
        let source:MLXArray
        if sr>1 { source=norm(conv(tokens.reshaped(b,h,width,c),p+".attn.sr",stride:sr).reshaped(b,-1,c),p+".attn.norm") }
        else { source=tokens }
        let kv=linear(source,p+".attn.kv").reshaped(b,-1,2,heads,d).transposed(axes:[2,0,3,1,4])
        eval(q,kv)
        let attended:MLXArray
        if queryRows<=0 || n<=queryRows {
            attended=MLXFast.scaledDotProductAttention(queries:q,keys:kv[0],values:kv[1],scale:1/sqrt(Float(d)),mask:.none)
                .transposed(axes:[0,2,1,3]).reshaped(b,h,width,c)
        } else {
            var pieces:[MLXArray]=[]
            for offset in stride(from:0,to:n,by:queryRows) {
                let piece=MLXFast.scaledDotProductAttention(queries:q[0...,0...,offset..<min(n,offset+queryRows),0...],keys:kv[0],values:kv[1],scale:1/sqrt(Float(d)),mask:.none)
                eval(piece);pieces.append(piece)
            }
            attended=concatenated(pieces,axis:2).transposed(axes:[0,2,1,3]).reshaped(b,h,width,c)
        }
        let residual=input+linear(attended,p+".attn.proj")
        eval(residual)
        return residual
    }
    private func block(_ input:MLXArray,_ p:String,heads:Int,sr:Int)->MLXArray {
        // The helper scope releases Q/K/V before the expanded MLP is allocated.
        let residual=attentionResidual(input,p,heads:heads,sr:sr)
        var hidden=linear(norm(residual,p+".norm2",eps:1e-6),p+".mlp.fc1");eval(hidden)
        hidden=conv(hidden,p+".mlp.dwconv.dwconv",padding:1,groups:hidden.dim(3));eval(hidden)
        hidden=gelu(hidden);eval(hidden)
        let output=residual+linear(hidden,p+".mlp.fc2");eval(output)
        return output
    }
    private func rectify(_ a:MLXArray,_ b:MLXArray,_ p:String)->(MLXArray,MLXArray) {
        let x=concatenated([a,b],axis:-1),batch=a.dim(0),c=a.dim(3)
        let pooled=concatenated([mean(x,axes:[1,2]),max(x,axes:[1,2])],axis:-1)
        let cw=sigmoid(linear(maximum(linear(pooled,p+".channel_weights.mlp.0"),MLXArray(Float(0)).asType(pooled.dtype)),p+".channel_weights.mlp.2")).reshaped(batch,1,1,2,c)
        let sw=sigmoid(conv(maximum(conv(x,p+".spatial_weights.mlp.0"),MLXArray(Float(0)).asType(x.dtype)),p+".spatial_weights.mlp.2"))
        return (a+0.5*cw[0...,0...,0...,1,0...]*b+0.5*sw[0...,0...,0...,1..<2]*b,
                b+0.5*cw[0...,0...,0...,0,0...]*a+0.5*sw[0...,0...,0...,0..<1]*a)
    }
    private func fusion(_ a:MLXArray,_ b:MLXArray,_ p:String,heads:Int)->MLXArray {
        let batch=a.dim(0),h=a.dim(1),width=a.dim(2),c=a.dim(3),n=h*width,d=c/heads
        let one=a.reshaped(batch,n,c),two=b.reshaped(batch,n,c)
        let u=maximum(linear(one,p+".cross.channel_proj1"),MLXArray(Float(0)).asType(one.dtype))
        let v=maximum(linear(two,p+".cross.channel_proj2"),MLXArray(Float(0)).asType(two.dtype))
        func cross(_ x:MLXArray,_ other:MLXArray,_ which:Int)->MLXArray {
            let q=x.reshaped(batch,n,heads,d).transposed(axes:[0,2,1,3])
            let kv=linear(other,p+".cross.cross_attn.kv\(which)").reshaped(batch,n,2,heads,d).transposed(axes:[2,0,3,1,4])
            let context=softmax(matmul(kv[0].swappedAxes(-2,-1),kv[1])/sqrt(Float(d)),axis:-2)
            return matmul(q,context).transposed(axes:[0,2,1,3]).reshaped(batch,n,c)
        }
        let oneOut=norm(one+linear(concatenated([u[0...,0...,0..<c],cross(u[0...,0...,c..<(2*c)],v[0...,0...,c..<(2*c)],2)],axis:-1),p+".cross.end_proj1"),p+".cross.norm1")
        eval(oneOut)
        let twoOut=norm(two+linear(concatenated([v[0...,0...,0..<c],cross(v[0...,0...,c..<(2*c)],u[0...,0...,c..<(2*c)],1)],axis:-1),p+".cross.end_proj2"),p+".cross.norm2")
        eval(twoOut)
        let x=concatenated([oneOut,twoOut],axis:-1).reshaped(batch,h,width,2*c)
        let residual=conv(x,p+".channel_emb.residual")
        let local=conv(maximum(conv(conv(x,p+".channel_emb.channel_embed.0"),p+".channel_emb.channel_embed.1",padding:1,groups:c),MLXArray(Float(0)).asType(x.dtype)),p+".channel_emb.channel_embed.3")
        return bn(residual+bn(local,p+".channel_emb.channel_embed.4"),p+".channel_emb.norm")
    }
    private func encoder(_ rgb:MLXArray,_ np:MLXArray,trace:((String,MLXArray)throws->Void)?) throws -> [MLXArray] {
        var a=(rgb-MLXArray([Float(0.485),0.456,0.406]).asType(rgb.dtype))/MLXArray([Float(0.229),0.224,0.225]).asType(rgb.dtype)
        var b=concatenated([np,np,np],axis:-1),outputs:[MLXArray]=[]
        let depths=[3,4,6,3],heads=[1,2,5,8],sr=[8,4,2,1]
        for i in 0..<4 {
            let stage=i+1
            a=norm(conv(a,"backbone.patch_embed\(stage).proj",stride:i==0 ? 4:2,padding:i==0 ? 3:1),"backbone.patch_embed\(stage).norm");eval(a)
            b=norm(conv(b,"backbone.extra_patch_embed\(stage).proj",stride:i==0 ? 4:2,padding:i==0 ? 3:1),"backbone.extra_patch_embed\(stage).norm");eval(b)
            for j in 0..<depths[i] {
                a=Self.timed("block\(stage).\(j)"){ block(a,"backbone.block\(stage).\(j)",heads:heads[i],sr:sr[i]) }
                b=Self.timed("xblock\(stage).\(j)"){ block(b,"backbone.extra_block\(stage).\(j)",heads:heads[i],sr:sr[i]) }
            }
            a=norm(a,"backbone.norm\(stage)",eps:1e-6);b=norm(b,"backbone.extra_norm\(stage)",eps:1e-6)
            let (x,y)=rectify(a,b,"backbone.FRMs.\(i)");eval(x,y)
            let fused=Self.timed("fusion\(stage)"){ fusion(x,y,"backbone.FFMs.\(i)",heads:heads[i]) };eval(fused)
            a=x;b=y;outputs.append(fused)
            try trace?("stage_\(stage)",fused.asType(.float32))
        }
        return outputs
    }
    public static func resize(_ x:MLXArray,height:Int,width:Int)->MLXArray {
        let h=x.dim(1),w=x.dim(2)
        if h==height && w==width { return x }
        let ys=(0..<height).map{max(0,min(Float(h-1),(Float($0)+0.5)*Float(h)/Float(height)-0.5))}
        let xs=(0..<width).map{max(0,min(Float(w-1),(Float($0)+0.5)*Float(w)/Float(width)-0.5))}
        let y0=MLXArray(ys.map{Int32($0)}),x0=MLXArray(xs.map{Int32($0)})
        let fy=MLXArray(ys.map{$0-floor($0)},[1,height,1,1]).asType(x.dtype),fx=MLXArray(xs.map{$0-floor($0)},[1,1,width,1]).asType(x.dtype)
        let rows=take(x,y0,axis:1)*(1-fy)+take(x,minimum(y0+1,MLXArray(Int32(h-1))),axis:1)*fy
        return take(rows,x0,axis:2)*(1-fx)+take(rows,minimum(x0+1,MLXArray(Int32(w-1))),axis:2)*fx
    }
    /// Project only the source rows needed by one output strip. The global
    /// half-pixel coordinates and projection-before-interpolation order match
    /// the full decoder, including odd geometries and clamped image edges.
    private func projectedStrip(_ feature:MLXArray,_ p:String,height:Int,width:Int,rows:Range<Int>)->MLXArray {
        let h=feature.dim(1),w=feature.dim(2)
        if h==height && w==width { return linear(feature[0...,rows,0...,0...],p) }
        let ys=rows.map{max(0,min(Float(h-1),(Float($0)+0.5)*Float(h)/Float(height)-0.5))}
        let first=Int(ys.first!),last=min(h-1,Int(ys.last!)+1)
        let projected=linear(feature[0...,first..<(last+1),0...,0...],p)
        let xs=(0..<width).map{max(0,min(Float(w-1),(Float($0)+0.5)*Float(w)/Float(width)-0.5))}
        let y0=MLXArray(ys.map{Int32($0)-Int32(first)}),x0=MLXArray(xs.map{Int32($0)})
        let fy=MLXArray(ys.map{$0-floor($0)},[1,rows.count,1,1]).asType(feature.dtype)
        let fx=MLXArray(xs.map{$0-floor($0)},[1,1,width,1]).asType(feature.dtype)
        let interpolated=take(projected,y0,axis:1)*(1-fy)+take(projected,minimum(y0+1,MLXArray(Int32(last-first))),axis:1)*fy
        return take(interpolated,x0,axis:2)*(1-fx)+take(interpolated,minimum(x0+1,MLXArray(Int32(w-1))),axis:2)*fx
    }
    private func decoderStrip(_ features:[MLXArray],_ p:String,rows:Range<Int>)->MLXArray {
        let h=features[0].dim(1),width=features[0].dim(2),parts=fuseParts[p]!
        var fused:MLXArray?
        for (part,i) in (0..<4).reversed().enumerated() {
            let resized=projectedStrip(features[i],p+".linear_c\(i+1).proj",height:h,width:width,rows:rows)
            let contribution=conv2d(resized,parts[part])
            fused=fused.map{$0+contribution} ?? contribution
            eval(fused!)
        }
        let x=fused!+w(p+".linear_fuse.0.bias")
        let result=conv(maximum(bn(x,p+".linear_fuse.1",eps:0.001),MLXArray(Float(0)).asType(x.dtype)),p+".linear_pred")
        eval(result)
        return result
    }
    private func rowDecoderStrip(_ features:[TruForRows],_ p:String,rows:Range<Int>)throws->MLXArray {
        let h=features[0].height,width=features[0].width,y=rows.lowerBound,end=rows.upperBound
        var fused:MLXArray?
        for (part,i) in (0..<4).reversed().enumerated() {
            let feature=features[i],ys=(y..<end).map{max(0,min(Float(feature.height-1),(Float($0)+0.5)*Float(feature.height)/Float(h)-0.5))}
            let first=feature.height==h ? y:Int(ys.first!),last=feature.height==h ? end-1:min(feature.height-1,Int(ys.last!)+1)
            let projected=linear(try feature.read(first..<(last+1)),p+".linear_c\(i+1).proj")
            let resized:MLXArray
            if feature.height==h && feature.width==width { resized=projected }
            else {
                let xs=(0..<width).map{max(0,min(Float(feature.width-1),(Float($0)+0.5)*Float(feature.width)/Float(width)-0.5))}
                let y0=MLXArray(ys.map{Int32($0)-Int32(first)}),x0=MLXArray(xs.map{Int32($0)})
                let fy=MLXArray(ys.map{$0-floor($0)},[1,end-y,1,1]),fx=MLXArray(xs.map{$0-floor($0)},[1,1,width,1])
                let sourceRows=take(projected,y0,axis:1)*(1-fy)+take(projected,minimum(y0+1,MLXArray(Int32(last-first))),axis:1)*fy
                resized=take(sourceRows,x0,axis:2)*(1-fx)+take(sourceRows,minimum(x0+1,MLXArray(Int32(feature.width-1))),axis:2)*fx
            }
            let contribution=conv2d(resized,fuseParts[p]![part])
            fused=fused.map{$0+contribution} ?? contribution;eval(fused!)
        }
        let x=fused!+w(p+".linear_fuse.0.bias")
        let output=conv(maximum(bn(x,p+".linear_fuse.1",eps:0.001),MLXArray(Float(0)).asType(x.dtype)),p+".linear_pred");eval(output);return output
    }
    private func decoder(_ features:[MLXArray],_ p:String)->MLXArray {
        let h=features[0].dim(1),width=features[0].dim(2)
        // Each 512-channel FP32 strip is at most 16 MiB for ordinary geometry.
        // Only the final one/two-channel logits survive the helper's scope.
        let stripRows=max(1,8192/width)
        var outputs:[MLXArray]=[]
        for y in stride(from:0,to:h,by:stripRows) {
            outputs.append(decoderStrip(features,p,rows:y..<min(h,y+stripRows)))
        }
        let result=concatenated(outputs,axis:1);eval(result)
        return result
    }
    private func statistics(_ x:MLXArray,logWeight:MLXArray?=nil)->MLXArray {
        let values=x.asType(.float32).reshaped(x.dim(0),-1)
        let raw=logWeight?.asType(.float32).reshaped(x.dim(0),-1) ?? zeros(like:values)
        let lw=raw-logSumExp(raw,axis:-1,keepDims:true),weight=exp(lw)
        return stacked([-logSumExp(lw-values,axis:-1),logSumExp(lw+values,axis:-1),sum(weight*values,axis:-1),sum(weight*values*values,axis:-1)],axis:1)
    }
    /// Features leave scope before the full-resolution statistics are computed.
    private func logits(_ input:TruForInput,_ np:MLXArray,trace:((String,MLXArray)throws->Void)?) throws -> (MLXArray,MLXArray) {
        if input.usesRows {
            let (smallOut,smallConf)=try rowLogits(input,np,trace:trace)
            let out=Self.resize(smallOut,height:input.height,width:input.width);eval(out)
            let conf=Self.resize(smallConf,height:input.height,width:input.width);eval(conf)
            return (out,conf)
        }
        let rgb=precision == .fp16 ? input.array!.asType(.float16):input.array!
        let features=try Self.timed("encoder"){ try encoder(rgb,np,trace:trace) }
        let out=Self.timed("decode_head"){ Self.resize(decoder(features,"decode_head"),height:input.height,width:input.width) };eval(out)
        let conf=Self.timed("decode_head_conf"){ Self.resize(decoder(features,"decode_head_conf"),height:input.height,width:input.width) };eval(conf)
        return (out,conf)
    }
    /// Profiling observes the same bounded execution used in production.
    static let profiling=ProcessInfo.processInfo.environment["WAIFU_TRUFOR_PROFILE"] != nil
    private static let clock=ContinuousClock()
    @discardableResult
    private static func timed<T>(_ name:String,_ body:() throws->T)rethrows->T {
        if !profiling { return try body() }
        let start=clock.now
        let value=try body()
        if let tensor=value as? MLXArray { eval(tensor) }
        else if let tensors=value as? (MLXArray,MLXArray) { eval(tensors.0,tensors.1) }
        else if let tensors=value as? [MLXArray] { eval(tensors) }
        let elapsed=clock.now-start
        FileHandle.standardError.write(Data(String(format:"{\"phase\":\"%@\",\"ms\":%.1f}\n",name,Double(elapsed.components.seconds)*1000+Double(elapsed.components.attoseconds)/1e15).utf8))
        return value
    }
    /// Scores and evidence share the original FP32 math and native geometry.
    /// Deliberate evaluation boundaries prevent whole-forward graph retention.
    public func predict(_ rgb:MLXArray,evidence:Bool,trace:((String,MLXArray)throws->Void)?=nil) throws -> TruForOutput {
        try predict(TruForInput(rgb),evidence:evidence,trace:trace)
    }
    public func predict(_ input:TruForInput,evidence:Bool,trace:((String,MLXArray)throws->Void)?=nil) throws -> TruForOutput {
        let np=try Self.timed("noise"){ try noise(input) };eval(np)
        Memory.clearCache();TruForMemoryPolicy.releaseUnusedCPU()
        if !evidence && trace == nil && input.usesRows &&
            ProcessInfo.processInfo.environment["WAIFU_TRUFOR_SCORE_ONLY_STREAM"] != "0" {
            let score=try Self.timed("score-only"){try scoreOnly(input,np)}
            return .init(score:score,anomaly:nil,reliability:nil,noiseprint:nil)
        }
        try trace?("noiseprint",np.asType(.float32))
        let (out,conf)=try logits(input,np,trace:trace)
        let stats=detectionStatistics(out,conf)
        let score=sigmoid(linear(maximum(linear(stats,"detection.0"),MLXArray(Float(0))),"detection.3")).reshaped(-1);eval(score)
        try trace?("localization_logits",out.asType(.float32));try trace?("confidence_logits",conf.asType(.float32))
        try trace?("detection_statistics",stats);try trace?("score",score.asType(.float32))
        let conf32=conf.asType(.float32)
        let anomaly=evidence ? softmax(out.asType(.float32),axis:-1)[0...,0...,0...,1].contiguous() : nil
        let reliability=evidence ? sigmoid(conf32).squeezed(axis:3) : nil
        let noiseprint=evidence ? np.asType(.float32).squeezed(axis:3) : nil
        eval([score]+[anomaly,reliability,noiseprint].compactMap{$0})
        return .init(score:score.asType(.float32),anomaly:anomaly,reliability:reliability,noiseprint:noiseprint)
    }
}

// The large-image path advances through owned FP32 chunks held only in RAM. Attention still sees global K/V and fusion still uses global
// channel statistics; this is not independent-image tiling or resized inference.
extension TruForModel {
    private func rowMap(_ source:TruForRows,consume:Bool=false,_ body:(MLXArray)->MLXArray)throws->TruForRows {
        let output=try TruForRows(height:source.height,width:source.width,channels:source.channels)
        for y in stride(from:0,to:source.height,by:source.stripRows) {
            try autoreleasepool {
                let end=min(source.height,y+source.stripRows),input=try source.read(y..<end)
                if consume { source.discard(before:end) }
                try output.append(body(input))
            }
        }
        return output
    }
    private func patchRows(height:Int,width:Int,_ p:String,stride step:Int,padding:Int,
                           read:(Range<Int>)throws->MLXArray,normalize:String?)throws->TruForRows {
        let kernel=w(p+".weight"),kh=kernel.dim(1),kw=kernel.dim(2)
        let h=(height+2*padding-kh)/step+1,wide=(width+2*padding-kw)/step+1
        let output=try TruForRows(height:h,width:wide,channels:kernel.dim(0))
        // Strided attention reduction can consume 64 source pixels per output
        // pixel. Bound the input strip too, avoiding a 267 MB source read at SR8.
        let stripRows=min(output.stripRows,max(1,4_194_304/(width*kernel.dim(3)*step)))
        for y in stride(from:0,to:h,by:stripRows) {
            try autoreleasepool {
                let end=min(h,y+stripRows),lo=y*step-padding,hi=(end-1)*step-padding+kh
                var x=try read(max(0,lo)..<min(height,hi))
                x=padded(x,widths:[0,.init((max(0,-lo),max(0,hi-height))),0,0])
                var result=conv2d(x,kernel,stride:.init(step),padding:.init((0,padding)))
                if let bias=weights[p+".bias"] { result=result+bias }
                if let normalize { result=norm(result,normalize) }
                try output.append(result)
            }
        }
        return output
    }
    private func rowAttention(_ x:MLXArray,_ kv:MLXArray,_ p:String,heads:Int)->MLXArray {
        let h=x.dim(1),width=x.dim(2),c=x.dim(3),n=h*width,d=c/heads
        let q=linear(norm(x,p+".norm1",eps:1e-6),p+".attn.q").reshaped(1,n,heads,d).transposed(axes:[0,2,1,3]);eval(q)
        var pieces:[MLXArray]=[]
        for start in stride(from:0,to:n,by:queryRows) {
            let piece=MLXFast.scaledDotProductAttention(queries:q[0...,0...,start..<min(n,start+queryRows),0...],keys:kv[0],values:kv[1],scale:1/sqrt(Float(d)),mask:.none)
            eval(piece);pieces.append(piece)
        }
        let attended=concatenated(pieces,axis:2).transposed(axes:[0,2,1,3]).reshaped(1,h,width,c)
        return x+linear(attended,p+".attn.proj")
    }
    private func rowBlock(_ source:TruForRows,_ p:String,heads:Int,sr:Int)throws->TruForRows {
        // Spatial-reduction K/V are global and small; queries stream over rows.
        let reduced:TruForRows
        if sr>1 {
            reduced=try patchRows(height:source.height,width:source.width,p+".attn.sr",stride:sr,padding:0,
                read:{ self.norm(try source.read($0),p+".norm1",eps:1e-6) },normalize:p+".attn.norm")
        } else { reduced=try rowMap(source){ self.norm($0,p+".norm1",eps:1e-6) } }
        var keys:[MLXArray]=[]
        for y in stride(from:0,to:reduced.height,by:reduced.stripRows) {
            let part=linear(try reduced.read(y..<min(reduced.height,y+reduced.stripRows)),p+".attn.kv").reshaped(1,-1,2,heads,source.channels/heads)
            eval(part);keys.append(part)
        }
        let kv=concatenated(keys,axis:1).transposed(axes:[2,0,3,1,4]);eval(kv);keys.removeAll()
        let residual=try rowMap(source,consume:true){ self.rowAttention($0,kv,p,heads:heads) }
        let output=try TruForRows(height:source.height,width:source.width,channels:source.channels)
        for y in stride(from:0,to:source.height,by:source.stripRows) {
            try autoreleasepool {
                let end=min(source.height,y+source.stripRows),first=max(0,y-1),last=min(source.height,end+1)
                let x=try residual.read(first..<last)
                residual.discard(before:max(0,end-1))
                let hidden=linear(norm(x,p+".norm2",eps:1e-6),p+".mlp.fc1")
                let spatial=conv(hidden,p+".mlp.dwconv.dwconv",padding:1,groups:hidden.dim(3))[0...,(y-first)..<(end-first),0...,0...]
                try output.append(x[0...,(y-first)..<(end-first),0...,0...]+linear(gelu(spatial),p+".mlp.fc2"))
            }
        }
        return output
    }
    private func rowRectify(_ a:TruForRows,_ b:TruForRows,_ p:String)throws->(TruForRows,TruForRows) {
        let c=a.channels
        var sums=[Double](repeating:0,count:2*c),maxima=[Float](repeating:-Float.infinity,count:2*c)
        for y in stride(from:0,to:a.height,by:a.stripRows) {
            try autoreleasepool {
                let end=min(a.height,y+a.stripRows),x=concatenated([try a.read(y..<end),try b.read(y..<end)],axis:-1)
                let totals=sum(x,axes:[1,2]).asArray(Float.self),tops=max(x,axes:[1,2]).asArray(Float.self)
                for i in sums.indices { sums[i]+=Double(totals[i]);maxima[i]=Swift.max(maxima[i],tops[i]) }
            }
        }
        let pooled=MLXArray(sums.map{Float($0/Double(a.height*a.width))}+maxima,[1,4*c])
        let cw=sigmoid(linear(maximum(linear(pooled,p+".channel_weights.mlp.0"),MLXArray(Float(0))),p+".channel_weights.mlp.2")).reshaped(1,1,1,2,c);eval(cw)
        let one=try TruForRows(height:a.height,width:a.width,channels:c),two=try TruForRows(height:b.height,width:b.width,channels:c)
        for y in stride(from:0,to:a.height,by:a.stripRows) {
            try autoreleasepool {
                let end=min(a.height,y+a.stripRows),x=try a.read(y..<end),z=try b.read(y..<end)
                a.discard(before:end);b.discard(before:end)
                let sw=sigmoid(conv(maximum(conv(concatenated([x,z],axis:-1),p+".spatial_weights.mlp.0"),MLXArray(Float(0))),p+".spatial_weights.mlp.2"));eval(sw)
                try one.append(x+0.5*cw[0...,0...,0...,1,0...]*z+0.5*sw[0...,0...,0...,1..<2]*z)
                try two.append(z+0.5*cw[0...,0...,0...,0,0...]*x+0.5*sw[0...,0...,0...,0..<1]*x)
            }
        }
        return (one,two)
    }
    private func rowFusion(_ a:TruForRows,_ b:TruForRows,_ p:String,heads:Int)throws->TruForRows {
        let c=a.channels,d=c/heads
        var contexts:[MLXArray?]=[nil,nil]
        // Cross attention is channel attention. Accumulate its global K^T V
        // matrices before applying any query strip, preserving global context.
        for y in stride(from:0,to:a.height,by:a.stripRows) {
            try autoreleasepool {
                let end=min(a.height,y+a.stripRows),n=(end-y)*a.width
                let u=maximum(linear(try a.read(y..<end),p+".cross.channel_proj1"),MLXArray(Float(0))).reshaped(1,n,2*c)
                let v=maximum(linear(try b.read(y..<end),p+".cross.channel_proj2"),MLXArray(Float(0))).reshaped(1,n,2*c)
                for (index,other,which) in [(0,v,2),(1,u,1)] {
                    let kv=linear(other[0...,0...,0..<c],p+".cross.cross_attn.kv\(which)").reshaped(1,n,2,heads,d).transposed(axes:[2,0,3,1,4])
                    let part=matmul(kv[0].swappedAxes(-2,-1),kv[1])
                    contexts[index]=contexts[index].map{$0+part} ?? part;eval(contexts[index]!)
                }
            }
        }
        let ctx=contexts.map{softmax($0!/sqrt(Float(d)),axis:-2)};eval(ctx)
        let merged=try TruForRows(height:a.height,width:a.width,channels:2*c)
        for y in stride(from:0,to:a.height,by:a.stripRows) {
            try autoreleasepool {
                let end=min(a.height,y+a.stripRows),n=(end-y)*a.width
                let one=try a.read(y..<end).reshaped(1,n,c),two=try b.read(y..<end).reshaped(1,n,c)
                a.discard(before:end);b.discard(before:end)
                let u=maximum(linear(one,p+".cross.channel_proj1"),MLXArray(Float(0)))
                let v=maximum(linear(two,p+".cross.channel_proj2"),MLXArray(Float(0)))
                func cross(_ x:MLXArray,_ context:MLXArray)->MLXArray {
                    matmul(x.reshaped(1,n,heads,d).transposed(axes:[0,2,1,3]),context).transposed(axes:[0,2,1,3]).reshaped(1,n,c)
                }
                let x=norm(one+linear(concatenated([u[0...,0...,0..<c],cross(u[0...,0...,c..<(2*c)],ctx[0])],axis:-1),p+".cross.end_proj1"),p+".cross.norm1")
                let z=norm(two+linear(concatenated([v[0...,0...,0..<c],cross(v[0...,0...,c..<(2*c)],ctx[1])],axis:-1),p+".cross.end_proj2"),p+".cross.norm2")
                try merged.append(concatenated([x,z],axis:-1).reshaped(1,end-y,a.width,2*c))
            }
        }
        let output=try TruForRows(height:a.height,width:a.width,channels:c)
        for y in stride(from:0,to:a.height,by:a.stripRows) {
            try autoreleasepool {
                let end=min(a.height,y+a.stripRows),first=max(0,y-1),last=min(a.height,end+1)
                let x=try merged.read(first..<last)
                merged.discard(before:max(0,end-1))
                let residual=conv(x,p+".channel_emb.residual")
                let local=conv(maximum(conv(conv(x,p+".channel_emb.channel_embed.0"),p+".channel_emb.channel_embed.1",padding:1,groups:c),MLXArray(Float(0))),p+".channel_emb.channel_embed.3")
                let fused=bn(residual+bn(local,p+".channel_emb.channel_embed.4"),p+".channel_emb.norm")
                try output.append(fused[0...,(y-first)..<(end-first),0...,0...])
            }
        }
        return output
    }
    private func rowEncoder(_ rgb:TruForInput,_ np:MLXArray,trace:((String,MLXArray)throws->Void)?)throws->[TruForRows] {
        let depths=[3,4,6,3],heads=[1,2,5,8],sr=[8,4,2,1]
        var a=try patchRows(height:rgb.height,width:rgb.width,"backbone.patch_embed1.proj",stride:4,padding:3,
            read:{ (try rgb.read($0)-MLXArray([Float(0.485),0.456,0.406]))/MLXArray([Float(0.229),0.224,0.225]) },normalize:"backbone.patch_embed1.norm")
        rgb.releasePixels()
        var b=try patchRows(height:np.dim(1),width:np.dim(2),"backbone.extra_patch_embed1.proj",stride:4,padding:3,
            read:{let x=np[0...,$0,0...,0...];return concatenated([x,x,x],axis:-1)},normalize:"backbone.extra_patch_embed1.norm")
        var outputs:[TruForRows]=[]
        for i in 0..<4 {
            let stage=i+1
            for j in 0..<depths[i] {
                a=try rowBlock(a,"backbone.block\(stage).\(j)",heads:heads[i],sr:sr[i])
                b=try rowBlock(b,"backbone.extra_block\(stage).\(j)",heads:heads[i],sr:sr[i])
            }
            a=try rowMap(a,consume:true){self.norm($0,"backbone.norm\(stage)",eps:1e-6)}
            b=try rowMap(b,consume:true){self.norm($0,"backbone.extra_norm\(stage)",eps:1e-6)}
            (a,b)=try rowRectify(a,b,"backbone.FRMs.\(i)")
            // The following patch embeddings depend only on the rectified
            // streams. Prepare them before fusion consumes those source rows.
            let nextA:TruForRows?,nextB:TruForRows?
            if stage<4 {
                nextA=try patchRows(height:a.height,width:a.width,"backbone.patch_embed\(stage+1).proj",stride:2,padding:1,read:a.read,normalize:"backbone.patch_embed\(stage+1).norm")
                nextB=try patchRows(height:b.height,width:b.width,"backbone.extra_patch_embed\(stage+1).proj",stride:2,padding:1,read:b.read,normalize:"backbone.extra_patch_embed\(stage+1).norm")
            } else {nextA=nil;nextB=nil}
            let feature=try rowFusion(a,b,"backbone.FFMs.\(i)",heads:heads[i]);outputs.append(feature)
            if let trace {try trace("stage_\(stage)",feature.read(0..<feature.height))}
            if let nextA,let nextB {a=nextA;b=nextB}
        }
        return outputs
    }
    private func rowLogits(_ input:TruForInput,_ np:MLXArray,trace:((String,MLXArray)throws->Void)?)throws->(MLXArray,MLXArray) {
        let features=try Self.timed("encoder.rows"){try rowEncoder(input,np,trace:trace)}
        return (try rowDecoder(features,"decode_head"),try rowDecoder(features,"decode_head_conf"))
    }
    private func rowDecoder(_ features:[TruForRows],_ p:String)throws->MLXArray {
        let h=features[0].height,width=features[0].width,stripRows=max(1,8192/width)
        var outputs:[MLXArray]=[]
        for y in stride(from:0,to:h,by:stripRows) {
            try autoreleasepool {
                let end=min(h,y+stripRows)
                outputs.append(try rowDecoderStrip(features,p,rows:y..<end))
            }
        }
        let output=concatenated(outputs,axis:1);eval(output);return output
    }
}

extension TruForModel {
    private struct DetectionPool {
            var normalizer = -Double.infinity
            var negative = -Double.infinity
            var positive = -Double.infinity
            var scale = -Double.infinity
            var mass=0.0,first=0.0,second=0.0
            mutating func add(_ x:MLXArray,logWeight:MLXArray) {
                let maximum=max(logWeight),weights=exp(logWeight-maximum)
                let summary=stacked([logSumExp(logWeight),logSumExp(logWeight-x),logSumExp(logWeight+x),
                                     maximum,sum(weights),sum(weights*x),sum(weights*x*x)]).asArray(Float.self)
                func logAdd(_ a:Double,_ b:Double)->Double {
                    if a == -Double.infinity { return b }
                    let m=Swift.max(a,b);return m+log(exp(a-m)+exp(b-m))
                }
                normalizer=logAdd(normalizer,Double(summary[0]))
                negative=logAdd(negative,Double(summary[1]));positive=logAdd(positive,Double(summary[2]))
                let nextScale=Swift.max(scale,Double(summary[3]))
                let oldFactor=scale == -Double.infinity ? 0:exp(scale-nextScale)
                let newFactor=exp(Double(summary[3])-nextScale)
                mass=mass*oldFactor+Double(summary[4])*newFactor
                first=first*oldFactor+Double(summary[5])*newFactor
                second=second*oldFactor+Double(summary[6])*newFactor
                scale=nextScale
            }
            var values:[Float] { [Float(normalizer-negative),Float(positive-normalizer),Float(first/mass),Float(second/mass)] }
    }
    private struct DetectionAccumulator {
        var confidence=DetectionPool(),localization=DetectionPool()
        mutating func add(_ out:MLXArray,_ conf:MLXArray) {
            let rows=max(1,65_536/out.dim(2))
            for y in stride(from:0,to:out.dim(1),by:rows) {
                autoreleasepool {
                    let end=min(out.dim(1),y+rows)
                    let c=conf[0...,y..<end,0...,0...].asType(.float32)
                    let diff=out[0...,y..<end,0...,1..<2]-out[0...,y..<end,0...,0..<1]
                    let logConf=minimum(c,MLXArray(Float(0)))-log(1+exp(-abs(c)))
                    confidence.add(c,logWeight:zeros(like:c));localization.add(diff,logWeight:logConf)
                }
            }
        }
        var statistics:MLXArray { MLXArray(confidence.values+localization.values,[1,8]) }
    }
    private func sourceRows(for targetRows:Range<Int>,sourceHeight:Int,targetHeight:Int)->Range<Int> {
        func mapped(_ y:Int)->Float { max(0,min(Float(sourceHeight-1),(Float(y)+0.5)*Float(sourceHeight)/Float(targetHeight)-0.5)) }
        let first=Int(mapped(targetRows.lowerBound)),last=Int(mapped(targetRows.upperBound-1))
        return first..<min(sourceHeight,last+2)
    }
    private func resizeStrip(_ x:MLXArray,sourceFirst:Int,sourceHeight:Int,targetHeight:Int,targetWidth:Int,rows:Range<Int>)->MLXArray {
        let width=x.dim(2)
        if sourceHeight==targetHeight && width==targetWidth { return x }
        let ys=rows.map{max(0,min(Float(sourceHeight-1),(Float($0)+0.5)*Float(sourceHeight)/Float(targetHeight)-0.5))}
        let last=min(sourceHeight-1,Int(ys.last!)+1)
        let y0=MLXArray(ys.map{Int32($0)-Int32(sourceFirst)}),xs=(0..<targetWidth).map{max(0,min(Float(width-1),(Float($0)+0.5)*Float(width)/Float(targetWidth)-0.5))}
        let x0=MLXArray(xs.map{Int32($0)}),fy=MLXArray(ys.map{$0-floor($0)},[1,rows.count,1,1]),fx=MLXArray(xs.map{$0-floor($0)},[1,1,targetWidth,1])
        let interpolated=take(x,y0,axis:1)*(1-fy)+take(x,minimum(y0+1,MLXArray(Int32(last-sourceFirst))),axis:1)*fy
        let resized=take(interpolated,x0,axis:2)*(1-fx)+take(interpolated,minimum(x0+1,MLXArray(Int32(width-1))),axis:2)*fx
        return resized
    }
    private func scoreOnly(_ input:TruForInput,_ np:MLXArray)throws->MLXArray {
        let out:MLXArray,conf:MLXArray
        if input.usesRows {
            let features=try Self.timed("encoder.rows"){try rowEncoder(input,np,trace:nil)}
            out=try Self.timed("decode_head"){try rowDecoder(features,"decode_head")}
            conf=try Self.timed("decode_head_conf"){try rowDecoder(features,"decode_head_conf")}
        } else {
            let rgb=precision == .fp16 ? input.array!.asType(.float16):input.array!
            let features=try Self.timed("encoder"){try encoder(rgb,np,trace:nil)}
            out=Self.timed("decode_head"){decoder(features,"decode_head")}
            conf=Self.timed("decode_head_conf"){decoder(features,"decode_head_conf")}
        }
        let sourceHeight=out.dim(1)
        var accumulator=DetectionAccumulator()
        // Keep the small decoder logits, but never allocate image-sized upscaled
        // logits for a score-only request. One-megapixel output strips amortize
        // resize dispatch while staying small beside the encoder features.
        let outputRows=max(1,1_048_576/input.width)
        for y in stride(from:0,to:input.height,by:outputRows) {
            autoreleasepool {
                let rows=y..<min(input.height,y+outputRows)
                let source=sourceRows(for:rows,sourceHeight:sourceHeight,targetHeight:input.height)
                let outStrip=resizeStrip(out[0...,source,0...,0...],sourceFirst:source.lowerBound,sourceHeight:sourceHeight,targetHeight:input.height,targetWidth:input.width,rows:rows)
                let confStrip=resizeStrip(conf[0...,source,0...,0...],sourceFirst:source.lowerBound,sourceHeight:sourceHeight,targetHeight:input.height,targetWidth:input.width,rows:rows)
                eval(outStrip,confStrip);accumulator.add(outStrip,confStrip)
            }
        }
        let score=sigmoid(linear(maximum(linear(accumulator.statistics,"detection.0"),MLXArray(Float(0))),"detection.3")).reshaped(-1)
        eval(score);return score.asType(.float32)
    }
    private func detectionStatistics(_ out:MLXArray,_ conf:MLXArray)->MLXArray {
        var accumulator=DetectionAccumulator();accumulator.add(out,conf);return accumulator.statistics
    }
}
