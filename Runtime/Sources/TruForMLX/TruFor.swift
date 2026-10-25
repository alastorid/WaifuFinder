// Native adaptation of GRIP-UNINA TruFor (CVPR 2023), informational/nonprofit only.
// Copyright (c) 2023 GRIP-UNINA. CMX Copyright (c) 2022 Huayao Liu (MIT).
// Retain Runtime/TRUFOR-LICENSE.txt and TRUFOR-CMX-LICENSE.txt with adaptations.
import Foundation
import CryptoKit
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
    private var compiled:[String:@Sendable ([MLXArray])->[MLXArray]]=[:]
    private var compiledOrder:[String]=[]
    private let compiledLimit=4
    public let device:String
    public let queryRows:Int
    public let noiseTile:Int
    public init(directory:URL) throws {
        guard let gpu=MTLCreateSystemDefaultDevice(),gpu.supportsFamily(.apple7),gpu.maxThreadsPerThreadgroup.width>=256 else { throw TruForError.invalid("Unsupported GPU") }
        device=gpu.name
        queryRows=Int(ProcessInfo.processInfo.environment["WAIFU_TRUFOR_QUERY_ROWS"] ?? "512") ?? 512
        noiseTile=Int(ProcessInfo.processInfo.environment["WAIFU_TRUFOR_NOISE_TILE"] ?? "1024") ?? 1024
        guard [0,128,256,512,1024].contains(queryRows),[128,256,512,1024].contains(noiseTile) else { throw TruForError.invalid("Unsupported profile override") }
        precision=Precision(rawValue:ProcessInfo.processInfo.environment["WAIFU_TRUFOR_PRECISION"] ?? "") ?? .fp32
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
            x=Self.profiling ? Self.timed("dncnn.\(i)"){ conv(x,"dncnn.\(i)",padding:1) } : conv(x,"dncnn.\(i)",padding:1)
            if level>0 && level<16 { x=bn(x,"dncnn.\(i+1)") }
            if level<16 { x=maximum(x,MLXArray(Float(0)).asType(x.dtype)) }
            if Self.profiling && level<16 { eval(x) }
        }
        return x
    }
    /// Exactly the 17-layer receptive field, with halo, preserving native input.
    private func noise(_ x:MLXArray)->MLXArray {
        let h=x.dim(1),width=x.dim(2)
        if h<=noiseTile && width<=noiseTile { return noiseBody(x) }
        var rows:[MLXArray]=[]
        for y in stride(from:0,to:h,by:noiseTile) {
            var columns:[MLXArray]=[]
            for a in stride(from:0,to:width,by:noiseTile) {
                let y0=max(0,y-17),x0=max(0,a-17),y1=min(h,y+noiseTile+17),x1=min(width,a+noiseTile+17)
                let out=noiseBody(x[0...,y0..<y1,x0..<x1,0...])
                let cropped=out[0...,(y-y0)..<min(y-y0+noiseTile,h-y0),(a-x0)..<min(a-x0+noiseTile,width-x0),0...].contiguous()
                eval(cropped);columns.append(cropped)
            }
            rows.append(concatenated(columns,axis:2))
        }
        return concatenated(rows,axis:1)
    }
    private func block(_ input:MLXArray,_ p:String,heads:Int,sr:Int)->MLXArray {
        let b=input.dim(0),h=input.dim(1),width=input.dim(2),c=input.dim(3),n=h*width,d=c/heads
        let tokens=norm(input,p+".norm1",eps:1e-6).reshaped(b,n,c)
        let q=linear(tokens,p+".attn.q").reshaped(b,n,heads,d).transposed(axes:[0,2,1,3])
        let source:MLXArray
        if sr>1 { source=norm(conv(tokens.reshaped(b,h,width,c),p+".attn.sr",stride:sr).reshaped(b,-1,c),p+".attn.norm") }
        else { source=tokens }
        let kv=linear(source,p+".attn.kv").reshaped(b,-1,2,heads,d).transposed(axes:[2,0,3,1,4])
        let attended:MLXArray
        if queryRows<=0 || n<=queryRows {
            attended=MLXFast.scaledDotProductAttention(queries:q,keys:kv[0],values:kv[1],scale:1/sqrt(Float(d)),mask:.none)
                .transposed(axes:[0,2,1,3]).reshaped(b,h,width,c)
        } else {
            var pieces:[MLXArray]=[]
            for offset in stride(from:0,to:n,by:queryRows) {
                pieces.append(MLXFast.scaledDotProductAttention(queries:q[0...,0...,offset..<min(n,offset+queryRows),0...],keys:kv[0],values:kv[1],scale:1/sqrt(Float(d)),mask:.none))
            }
            attended=concatenated(pieces,axis:2).transposed(axes:[0,2,1,3]).reshaped(b,h,width,c)
        }
        let residual=input+linear(attended,p+".attn.proj")
        let hidden=linear(norm(residual,p+".norm2",eps:1e-6),p+".mlp.fc1")
        return residual+linear(gelu(conv(hidden,p+".mlp.dwconv.dwconv",padding:1,groups:hidden.dim(3))),p+".mlp.fc2")
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
        let twoOut=norm(two+linear(concatenated([v[0...,0...,0..<c],cross(v[0...,0...,c..<(2*c)],u[0...,0...,c..<(2*c)],1)],axis:-1),p+".cross.end_proj2"),p+".cross.norm2")
        let x=concatenated([oneOut,twoOut],axis:-1).reshaped(batch,h,width,2*c)
        let residual=conv(x,p+".channel_emb.residual")
        let local=conv(maximum(conv(conv(x,p+".channel_emb.channel_embed.0"),p+".channel_emb.channel_embed.1",padding:1,groups:c),MLXArray(Float(0)).asType(x.dtype)),p+".channel_emb.channel_embed.3")
        return bn(residual+bn(local,p+".channel_emb.channel_embed.4"),p+".channel_emb.norm")
    }
    private func encoder(_ rgb:MLXArray,_ np:MLXArray,phased:Bool,trace:((String,MLXArray)throws->Void)?)->[MLXArray] {
        var a=(rgb-MLXArray([Float(0.485),0.456,0.406]).asType(rgb.dtype))/MLXArray([Float(0.229),0.224,0.225]).asType(rgb.dtype)
        var b=concatenated([np,np,np],axis:-1),outputs:[MLXArray]=[]
        let depths=[3,4,6,3],heads=[1,2,5,8],sr=[8,4,2,1]
        for i in 0..<4 {
            let stage=i+1
            let next:(MLXArray,MLXArray,MLXArray)
            if phased {
                next=Self.timed("stage_\(stage)"){
                    var a=norm(conv(a,"backbone.patch_embed\(stage).proj",stride:i==0 ? 4:2,padding:i==0 ? 3:1),"backbone.patch_embed\(stage).norm")
                    var b=norm(conv(b,"backbone.extra_patch_embed\(stage).proj",stride:i==0 ? 4:2,padding:i==0 ? 3:1),"backbone.extra_patch_embed\(stage).norm")
                    for j in 0..<depths[i] {
                        a=Self.timed("block\(stage).\(j)"){ block(a,"backbone.block\(stage).\(j)",heads:heads[i],sr:sr[i]) }
                        b=Self.timed("xblock\(stage).\(j)"){ block(b,"backbone.extra_block\(stage).\(j)",heads:heads[i],sr:sr[i]) }
                    }
                    a=norm(a,"backbone.norm\(stage)",eps:1e-6);b=norm(b,"backbone.extra_norm\(stage)",eps:1e-6)
                    let (x,y)=rectify(a,b,"backbone.FRMs.\(i)")
                    let fused=Self.timed("fusion\(stage)"){ fusion(x,y,"backbone.FFMs.\(i)",heads:heads[i]) };eval(x,y,fused)
                    return (x,y,fused)
                }
            } else {
                var a=norm(conv(a,"backbone.patch_embed\(stage).proj",stride:i==0 ? 4:2,padding:i==0 ? 3:1),"backbone.patch_embed\(stage).norm")
                var b=norm(conv(b,"backbone.extra_patch_embed\(stage).proj",stride:i==0 ? 4:2,padding:i==0 ? 3:1),"backbone.extra_patch_embed\(stage).norm")
                for j in 0..<depths[i] { a=block(a,"backbone.block\(stage).\(j)",heads:heads[i],sr:sr[i]);b=block(b,"backbone.extra_block\(stage).\(j)",heads:heads[i],sr:sr[i]) }
                a=norm(a,"backbone.norm\(stage)",eps:1e-6);b=norm(b,"backbone.extra_norm\(stage)",eps:1e-6)
                let (x,y)=rectify(a,b,"backbone.FRMs.\(i)")
                next=(x,y,fusion(x,y,"backbone.FFMs.\(i)",heads:heads[i]))
            }
            a=next.0;b=next.1;outputs.append(next.2);try? trace?("stage_\(stage)",next.2.asType(.float32))
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
    private func decoder(_ features:[MLXArray],_ p:String)->MLXArray {
        let h=features[0].dim(1),width=features[0].dim(2)
        let parts=fuseParts[p]!
        var fused:MLXArray?
        for (part,i) in (0..<4).reversed().enumerated() {
            let projected=linear(features[i],p+".linear_c\(i+1).proj")
            let resized=Self.resize(projected,height:h,width:width)
            let contribution=conv2d(resized,parts[part])
            fused=fused.map{$0+contribution} ?? contribution
            if phasedWork { eval(fused!) }
        }
        let x=fused!+w(p+".linear_fuse.0.bias")
        return conv(maximum(bn(x,p+".linear_fuse.1",eps:0.001),MLXArray(Float(0)).asType(x.dtype)),p+".linear_pred")
    }
    /// Set only on the validation path, where per-part syncs bound memory while
    /// the trace runs; the compiled production graph never checks this.
    private var phasedWork:Bool=false
    private func statistics(_ x:MLXArray,logWeight:MLXArray?=nil)->MLXArray {
        let values=x.asType(.float32).reshaped(x.dim(0),-1)
        let raw=logWeight?.asType(.float32).reshaped(x.dim(0),-1) ?? zeros(like:values)
        let lw=raw-logSumExp(raw,axis:-1,keepDims:true),weight=exp(lw)
        return stacked([-logSumExp(lw-values,axis:-1),logSumExp(lw+values,axis:-1),sum(weight*values,axis:-1),sum(weight*values*values,axis:-1)],axis:1)
    }
    /// Score, noiseprint, localization and confidence logits in one graph. The
    /// detection head runs in FP32 on FP32 statistics in both precision modes.
    private func forwardCore(_ rgb:MLXArray,_ np:MLXArray,phased:Bool)->[MLXArray] {
        let features=encoder(rgb,np,phased:phased,trace:nil)
        let out=Self.resize(decoder(features,"decode_head"),height:rgb.dim(1),width:rgb.dim(2))
        let conf=Self.resize(decoder(features,"decode_head_conf"),height:rgb.dim(1),width:rgb.dim(2))
        let diff=out[0...,0...,0...,1..<2]-out[0...,0...,0...,0..<1]
        let logConf=minimum(conf.asType(.float32),MLXArray(Float(0)))-log(1+exp(-abs(conf.asType(.float32))))
        let stats=concatenated([statistics(conf),statistics(diff,logWeight:logConf)],axis:1)
        let score=sigmoid(linear(maximum(linear(stats,"detection.0"),MLXArray(Float(0))),"detection.3")).reshaped(-1)
        return [score,np,out,conf]
    }
    /// `WAIFU_TRUFOR_PROFILE=1` prints per-phase wall time on the validation
    /// path; profiling evals the phase result so the measurement includes the
    /// sync, and the compiled production path never carries this instrumentation.
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
        FileHandle.standardError.write(Data(String(format:"{\"phase\":\"%@\",\"ms\":%.1f}\n",name,Double(elapsed.components.attoseconds)/1e15).utf8))
        return value
    }
    /// Both paths execute the identical trained detection head. Score-only never
    /// reads back or persists spatial maps. Full evidence exposes genuine maps.
    /// The trace path stays uncompiled so validation sees real intermediates.
    public func predict(_ rgb:MLXArray,evidence:Bool,trace:((String,MLXArray)throws->Void)?=nil) throws -> TruForOutput {
        guard rgb.ndim==4,rgb.dim(3)==3,rgb.dim(0)<=4,rgb.dim(1)>=32,rgb.dim(2)>=32,rgb.dim(1)*rgb.dim(2)*rgb.dim(0)<=24_000_000 else { throw TruForError.invalid("Unsupported native image geometry; images are never silently resized") }
        let rgbC=precision == .fp16 ? rgb.asType(.float16) : rgb
        let np=Self.timed("noise"){ noise(rgbC) };Self.timed("noise-sync"){ eval(np) }
        try trace?("noiseprint",np.asType(.float32))
        let score:MLXArray,out:MLXArray,conf:MLXArray
        if let trace {
            phasedWork=true
            let features=Self.timed("encoder"){ encoder(rgbC,np,phased:true,trace:trace) }
            let outT=Self.timed("decode_head"){ Self.resize(decoder(features,"decode_head"),height:rgbC.dim(1),width:rgbC.dim(2)) }
            let confT=Self.timed("decode_head_conf"){ Self.resize(decoder(features,"decode_head_conf"),height:rgbC.dim(1),width:rgbC.dim(2)) }
            phasedWork=false
            let (stats,scoreT):(MLXArray,MLXArray)=Self.timed("detection"){
                let diff=outT[0...,0...,0...,1..<2]-outT[0...,0...,0...,0..<1]
                let conf32=confT.asType(.float32)
                let logConf=minimum(conf32,MLXArray(Float(0)))-log(1+exp(-abs(conf32)))
                let stats=concatenated([statistics(confT),statistics(diff,logWeight:logConf)],axis:1)
                return (stats,sigmoid(linear(maximum(linear(stats,"detection.0"),MLXArray(Float(0))),"detection.3")).reshaped(-1))
            }
            score=scoreT;out=outT;conf=confT
            try trace("localization_logits",out.asType(.float32));try trace("confidence_logits",conf.asType(.float32));try trace("detection_statistics",stats);try trace("score",score.asType(.float32))
        } else {
            let key="\(rgbC.dim(0))x\(rgbC.dim(1))x\(rgbC.dim(2))"
            let graph: @Sendable ([MLXArray])->[MLXArray]
            if let cached=compiled[key] {
                graph=cached;compiledOrder.removeAll{$0==key};compiledOrder.append(key)
            } else {
                while compiledOrder.count>=compiledLimit { compiled.removeValue(forKey:compiledOrder.removeFirst()) }
                graph=compile { [unowned self] arrays in forwardCore(arrays[0],arrays[1],phased:false) }
                compiled[key]=graph;compiledOrder.append(key)
            }
            let outputs=graph([rgbC,np]);eval(outputs)
            score=outputs[0];out=outputs[2];conf=outputs[3]
        }
        return .init(score:score.asType(.float32),anomaly:evidence ? softmax(out.asType(.float32),axis:-1)[0...,0...,0...,1] : nil,reliability:evidence ? sigmoid(conf.asType(.float32)).squeezed(axis:3) : nil,noiseprint:evidence ? np.asType(.float32).squeezed(axis:3) : nil)
    }
}
