import Foundation
import ImageIO
import CoreGraphics
import MLX
import SAFIREMLX

extension Array {
    func chunked(_ size:Int) -> [[Element]] {
        stride(from:0,to:count,by:size).map { Array(self[$0..<Swift.min($0+size,count)]) }
    }
}

func option(_ key:String) -> String? { CommandLine.arguments.firstIndex(of:key).flatMap { $0+1 < CommandLine.arguments.count ? CommandLine.arguments[$0+1] : nil } }
import Darwin
func footprint() -> Int {
    var info=task_vm_info_data_t();var count=mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size/4)
    let result=withUnsafeMutablePointer(to:&info) { pointer in
        pointer.withMemoryRebound(to:integer_t.self,capacity:Int(count)) { intPointer in
            task_info(mach_task_self_,task_flavor_t(TASK_VM_INFO),intPointer,&count)
        }
    }
    return result==KERN_SUCCESS ? Int(info.phys_footprint) : 0
}
func memProbe(_ label:String) {
    guard ProcessInfo.processInfo.environment["WAIFU_SAFIRE_DEBUG"] != nil else { return }
    let active=Double(Memory.activeMemory)/1048576,cache=Double(Memory.cacheMemory)/1048576
    let peak=Double(Memory.peakMemory)/1048576,foot=Double(footprint())/1048576
    let line="FOOTPRINT \(label) foot \(Int(foot)) MB  mlx-active \(Int(active)) cache \(Int(cache)) peak \(Int(peak))\n"
    FileHandle.standardError.write(Data(line.utf8))
}
func fail(_ message:String) -> Never { FileHandle.standardError.write(Data((message+"\n").utf8)); exit(1) }

do {
    guard let path=option("--model") else { fail("Usage: waifu-safire --model DIRECTORY --activations INPUT.safetensors --output DIRECTORY | --image IMAGE --output RESULT.json | --serve") }
    // The buffer cache only needs to cover one stage's reuse; a 128 MiB cap
    // simply raised the allocator high-water mark.
    Memory.cacheLimit=64*1024*1024
    Memory.memoryLimit=2*1024*1024*1024
    memProbe("pre-init")
    let model=try SAFIREModel(directory:URL(fileURLWithPath:path))
    memProbe("post-init")
    if ProcessInfo.processInfo.environment["WAIFU_SAFIRE_STAGE_PROFILE"] != nil {
        model.stageRecorder={ name,seconds,active,peak in
            FileHandle.standardError.write(Data(String(format:"STAGE {\"name\":\"%@\",\"seconds\":%.4f,\"activeBytes\":%d,\"peakBytes\":%d}\n",name,seconds,active,peak).utf8))
        }
    }
    if let input=option("--activations"), let output=option("--output") {
        let tensors=try loadArrays(url:URL(fileURLWithPath:input))
        guard let x=tensors["input"] else { fail("Missing input tensor") }
        let root=URL(fileURLWithPath:output,isDirectory:true)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        let start=Date()
        let embedding=try model.encode(x) { name,value in
            try save(arrays:[name:value],url:root.appendingPathComponent(name+".safetensors"))
        }
        let points=MLXArray([Float(32),32,512,512],[2,2])
        let decoded=try model.decode(embedding:embedding,points:points)
        try save(arrays:["masks":decoded.masks,"quality":decoded.quality],url:root.appendingPathComponent("decoder.safetensors"))
        if CommandLine.arguments.contains("--automatic") {
            let prediction=try model.predict(x)
            try save(arrays:["probabilities":MLXArray(prediction.probabilities,[1024,1024]),
                             "partitions":MLXArray(prediction.partitions.map(Float.init),[1024,1024])],url:root.appendingPathComponent("automatic.safetensors"))
        }
        var report:[String:Any]=["seconds":Date().timeIntervalSince(start),"status":"native execution only; parity requires oracle comparison",
            "peakMLXBytes":Memory.peakMemory,"activeMLXBytes":Memory.activeMemory]
        report["profile"]=try JSONSerialization.jsonObject(with:JSONEncoder().encode(model.diagnostics))
        try JSONSerialization.data(withJSONObject:report,options:[.prettyPrinted,.sortedKeys]).write(to:root.appendingPathComponent("execution.json"))
    } else if CommandLine.arguments.contains("--check-preprocessing") {
        for (w,h) in [(1,1),(7,11),(257,189),(1024,1024),(2051,1025)] {
            let bytes=(0..<(w*h*4)).map { UInt8(($0*37+11)%256) }
            let actual=try SAFIREPreprocessing.rgba(bytes,width:w,height:h)
            let expected=MLXArray(SAFIREPreprocessing.reference(bytes,width:w,height:h),[1,1024,1024,3])
            let error=max(abs(actual-expected)).item(Float.self)
            guard error<1e-6 else { fail("Preprocessing parity failed: \(w)x\(h), \(error)") }
        }
        print("Fused preprocessing matches scalar fixed-point reference on five geometries")
    } else if CommandLine.arguments.contains("--check-fused") {
        for channels in [64,256,768] {
            for geluFlag in [false,true] {
                for residualFlag in [false,true] {
                    var row=[Float]((0..<channels).map { Float(($0*37)%91)/31-1.5 })
                    var res=[Float]((0..<channels).map { Float(($0*53)%77)/29-1.3 })
                    let weight=[Float]((0..<channels).map { Float(($0%13)-6)/7 })
                    let bias=[Float]((0..<channels).map { Float(($0%9)-4)/11 })
                    let x=MLXArray(row,[1,channels])
                    let out=SAFIRENorm.norm(x,residual:residualFlag ? MLXArray(res,[1,channels]) : nil,
                        weight:MLXArray(weight),bias:MLXArray(bias),eps:1e-6,gelu:geluFlag)
                    eval(out)
                    if residualFlag { for i in 0..<channels { row[i]+=res[i] } }
                    let expected=SAFIRENorm.referenceNorm(row,weight:weight,bias:bias,eps:1e-6,gelu:geluFlag)
                    let actual=out.asArray(Float.self)
                    for i in 0..<channels {
                        guard abs(actual[i]-expected[i])<2e-6 else {
                            fail("Fused norm mismatch C=\(channels) gelu=\(geluFlag) res=\(residualFlag) i=\(i): \(actual[i]) vs \(expected[i])")
                        }
                    }
                }
            }
        }
        print("Fused norm matches scalar reference on three channel widths, with and without residual/GELU")
    } else if CommandLine.arguments.contains("--check-relbias") {
        // Oracle for the fused relative-position bias outer-sum: the kernel's
        // dense tile must match the MLX broadcast-add construction bit-near on
        // random gathered tables, and SDPA fed by either must agree.
        func lcg(_ n:Int,_ seed:UInt64) -> [Float] {
            var state=seed
            return (0..<n).map { _ in
                state=state&*6364136223846793005&+1442695040888963407
                return Float(Int64(bitPattern:state >> 11 % 2048)%2048)/1024-1
            }
        }
        for (heads,rows) in [(4,64),(12,64),(12,256)] {
            let groups=rows/64,keys=4096
            let h=lcg(heads*groups*4096,0x71),w=lcg(heads*groups*4096,0x72)
            let height=MLXArray(h,[1,heads,groups,64,64]),width=MLXArray(w,[1,heads,groups,64,64])
            let expected=(height.expandedDimensions(axis:-1)+width.expandedDimensions(axis:-2))
                .reshaped(1,heads,rows,keys)
            let actual=SAFIREBias.outerSum(height:height,width:width,rows:rows,heads:heads)
            eval(expected,actual)
            let e=expected.asArray(Float.self),a=actual.asArray(Float.self)
            var worst:Float=0
            for i in 0..<e.count { worst=max(worst,abs(e[i]-a[i])) }
            guard worst<1e-6 else { fail("Rel-bias kernel mismatch heads=\(heads) rows=\(rows): \(worst)") }
            print(String(format:"Rel-bias kernel matches broadcast-add heads=%d rows=%d: max diff %.2e",heads,rows,worst))
            if heads==12 && rows==256 {
                let q=lcg(heads*rows*64,0x73),k=lcg(heads*keys*64,0x74),v=lcg(heads*keys*64,0x75)
                let qh=MLXArray(q,[1,heads,rows,64]),kh=MLXArray(k,[1,heads,keys,64]),vh=MLXArray(v,[1,heads,keys,64])
                let a1=MLXFast.scaledDotProductAttention(queries:qh,keys:kh,values:vh,scale:0.125,mask:actual)
                let a2=MLXFast.scaledDotProductAttention(queries:qh,keys:kh,values:vh,scale:0.125,mask:expected)
                eval(a1,a2)
                let x=a1.asArray(Float.self),y=a2.asArray(Float.self)
                var worst2:Float=0
                for i in 0..<x.count { worst2=max(worst2,abs(x[i]-y[i])) }
                guard worst2<1e-6 else { fail("Rel-bias SDPA mismatch: \(worst2)") }
                print(String(format:"SDPA with fused rel-bias matches broadcast-add bias: max diff %.2e",worst2))
            }
        }
        print("Fused rel-bias outer-sum matches the broadcast-add reference")
    } else if CommandLine.arguments.contains("--check-maskpost") {
        // Random signed logits exercise every branch of the fused resize/
        // threshold/sample/area kernel against the scalar reference.
        for batch in [1,3,4] {
            var values:[Float]=[]
            for i in 0..<(batch*256*256) { values.append(Float((i*2654435761)%1000)/500-1) }
            let masks=MLXArray(values,[batch,256,256])
            let post=SAFIREMaskPost.apply(masks,batch:batch)
            let areas=sum(post.partials,axes:[1,2])
            eval(post.samples,areas)
            let reference=SAFIREMaskPost.reference((0..<batch).map { b in Array(values[(b*65536)..<((b+1)*65536)]) },batch:batch)
            for b in 0..<batch {
                let sampled=post.samples[b].asArray(Float.self)
                for i in 0..<4096 where sampled[i] != reference.samples[b][i] {
                    fail("Mask-post sample mismatch: batch \(b) sample \(i): \(sampled[i]) vs \(reference.samples[b][i])")
                }
                guard Int(areas[b].item(Int32.self))==reference.areas[b] else {
                    fail("Mask-post area mismatch: batch \(b): \(areas[b].item(Int32.self)) vs \(reference.areas[b])")
                }
            }
        }
        print("Fused mask post-processing matches scalar resize/threshold reference on three batches")
    } else if CommandLine.arguments.contains("--check-upscale") {
        // Scalar oracle for the fused upscaling GEMMs on random inputs: the
        // kernel output must match the naive GEMM+bias+pixel-shuffle
        // (+LayerNorm+GELU) reference, and the MLX pipeline it replaces.
        func random(_ count:Int,_ scale:Float=1) -> [Float] {
            (0..<count).map { Float(($0*2654435761)%1000)/1000*scale - scale/2 }
        }
        func flatten(_ v:[[[Float]]]) -> [Float] { v.flatMap { $0.flatMap { $0 } } }
        func compare(_ actual:[[Float]],_ expected:[[Float]],_ tolerance:Float,_ label:String) {
            var worst=Float(0)
            for (row,(a,e)) in zip(actual,expected).enumerated() {
                for (i,(av,ev)) in zip(a,e).enumerated() {
                    worst=max(worst,abs(av-ev))
                    guard abs(av-ev) <= tolerance + 1e-4*abs(ev) else {
                        fail("\(label) mismatch row \(row) element \(i): \(av) vs \(ev)")
                    }
                }
            }
            print(String(format:"%@: max deviation %.2e",label,Double(worst)))
        }
        for batch in [1,2] {
            // Stage 1: GEMM [batch*4096,256]x[256,256] + bias + shuffle + LN(64)+GELU.
            let x1=random(batch*4096*256,0.5), w1=random(256*256,0.05), b1t=random(256,0.1)
            let lnw=random(64,0.5).map { $0+1 }, lnb=random(64,0.1)
            let xr1=MLXArray(x1,[batch,64,64,256]), w1m=MLXArray(w1,[256,256]), b1m=MLXArray(b1t,[256])
            let fused1=SAFIREUpscale.upscale1(xr1,weight:w1m,bias:b1m,lnWeight:MLXArray(lnw),lnBias:MLXArray(lnb),eps:1e-6)
            eval(fused1)
            let ref1=SAFIREUpscale.reference1((0..<batch).map { Array(x1[($0*4096*256)..<((($0+1)*4096)*256)]).chunked(256) },
                weight:w1.chunked(256),bias:b1t,lnWeight:lnw,lnBias:lnb,eps:1e-6)
            for b in 0..<batch { compare(fused1[b].asArray(Float.self).chunked(64),flatten(ref1[b]).chunked(64),Float(2e-3),"Fused upscale1 batch \(batch) vs scalar oracle") }
            // Same stage against the exact MLX pipeline being replaced.
            let block1=xr1.reshaped(-1,256).matmul(w1m)+b1m
            let mlx1=SAFIRENorm.norm(block1.reshaped(batch,64,64,2,2,64)
                .transposed(axes:[0,1,3,2,4,5]).reshaped(batch,128,128,64),
                residual:nil,weight:MLXArray(lnw),bias:MLXArray(lnb),eps:1e-6,gelu:true)
            eval(mlx1)
            compare(fused1.asArray(Float.self).chunked(64),mlx1.asArray(Float.self).chunked(64),Float(1e-3),"Fused upscale1 batch \(batch) vs MLX pipeline")
            // The threadgroup-staged-W variant must hold the same oracle.
            let staged1=SAFIREUpscale.upscale1(xr1,weight:w1m,bias:b1m,lnWeight:MLXArray(lnw),lnBias:MLXArray(lnb),eps:1e-6,bm:32,tg:256,kunroll:16)
            eval(staged1)
            compare(staged1.asArray(Float.self).chunked(64),mlx1.asArray(Float.self).chunked(64),Float(1e-3),"Staged-W upscale1 batch \(batch) vs MLX pipeline")
            let staged1b=SAFIREUpscale.upscale1(xr1,weight:w1m,bias:b1m,lnWeight:MLXArray(lnw),lnBias:MLXArray(lnb),eps:1e-6,bm:64,tg:256,kunroll:16)
            eval(staged1b)
            compare(staged1b.asArray(Float.self).chunked(64),mlx1.asArray(Float.self).chunked(64),Float(1e-3),"Staged-W upscale1 bm64 batch \(batch) vs MLX pipeline")
            // Stage 2: GEMM [batch*16384,64]x[64,128] + bias + shuffle + GELU.
            let x2=random(batch*16384*64,0.5), w2=random(64*128,0.08), b2t=random(128,0.1)
            let xr2=MLXArray(x2,[batch,128,128,64]), w2m=MLXArray(w2,[64,128]), b2m=MLXArray(b2t,[128])
            let fused2=SAFIREUpscale.upscale2(xr2,weight:w2m,bias:b2m)
            eval(fused2)
            let ref2=SAFIREUpscale.reference2((0..<batch).map { Array(x2[($0*16384*64)..<((($0+1)*16384)*64)]).chunked(64) },
                weight:w2.chunked(128),bias:b2t)
            for b in 0..<batch { compare(fused2[b].asArray(Float.self).chunked(32),flatten(ref2[b]).chunked(32),Float(2e-3),"Fused upscale2 batch \(batch) vs scalar oracle") }
            let block2=xr2.reshaped(-1,64).matmul(w2m)+b2m
            let mlx2=SAFIRENorm.gelu(block2.reshaped(batch,128,128,2,2,32)
                .transposed(axes:[0,1,3,2,4,5]).reshaped(batch,256,256,32))
            eval(mlx2)
            compare(fused2.asArray(Float.self).chunked(32),mlx2.asArray(Float.self).chunked(32),Float(1e-3),"Fused upscale2 batch \(batch) vs MLX pipeline")
            let staged2=SAFIREUpscale.upscale2(xr2,weight:w2m,bias:b2m,bm:32,tg:128,kunroll:16)
            eval(staged2)
            compare(staged2.asArray(Float.self).chunked(32),mlx2.asArray(Float.self).chunked(32),Float(1e-3),"Staged-W upscale2 batch \(batch) vs MLX pipeline")
            let staged2b=SAFIREUpscale.upscale2(xr2,weight:w2m,bias:b2m,bm:64,tg:256,kunroll:16)
            eval(staged2b)
            compare(staged2b.asArray(Float.self).chunked(32),mlx2.asArray(Float.self).chunked(32),Float(1e-3),"Staged-W upscale2 bm64 batch \(batch) vs MLX pipeline")
        }
        print("Fused upscaling GEMMs match scalar and MLX-pipeline references on two batch sizes")
    } else if CommandLine.arguments.contains("--gemm-bench") {
        // Developer micro-benchmark: raw MLX GEMM rates for the decoder shapes.
        func bench(_ m:Int,_ k:Int,_ n:Int,_ dtype:DType,repeats:Int=20) {
            let a=MLXArray([Float](repeating:0.01,count:m*k),[m,k]).asType(dtype)
            let b=MLXArray([Float](repeating:0.02,count:k*n),[k,n]).asType(dtype)
            var c=a.matmul(b); eval(c)
            let start=Date()
            for _ in 0..<repeats { c=a.matmul(b); eval(c) }
            let seconds=Date().timeIntervalSince(start)/Double(repeats)
            let label=dtype == .float16 ? "fp16" : "fp32"
            let flops=Double(m)*Double(k)*Double(n)*2/seconds/1e12
            print(String(format:"%dx%dx%d %@: %.2f ms  %.2f TFLOP/s",m,k,n,label,seconds*1000,flops))
        }
        bench(65536,256,256,.float32); bench(65536,256,256,.float16)
        bench(16384,32,128,.float32); bench(16384,32,128,.float16)
        bench(65536,32,1,.float32); bench(65536,32,1,.float16)
        let batchedA=MLXArray([Float](repeating:0.01,count:16*4096*64),[16,4096,64]).asType(.float32)
        let batchedB=MLXArray([Float](repeating:0.02,count:16*64*7),[16,64,7]).asType(.float32)
        var batched=batchedA.matmul(batchedB); eval(batched)
        let batchedStart=Date()
        for _ in 0..<20 { batched=batchedA.matmul(batchedB); eval(batched) }
        print(String(format:"batched 16x4096x64 @ 16x64x7: %.2f ms",(Date().timeIntervalSince(batchedStart)/20)*1000))
        func elementwise(_ label:String,_ size:[Int],repeats:Int=20,_ body:(MLXArray)->MLXArray) {
            let x=MLXArray([Float](repeating:0.3,count:size.reduce(1,*)),size)
            var y=body(x); eval(y)
            let start=Date()
            for _ in 0..<repeats { y=body(x); eval(y) }
            print(String(format:"%@: %.2f ms",label,(Date().timeIntervalSince(start)/Double(repeats))*1000))
        }
        let block=MLXArray([Float](repeating:0.3,count:16*64*64*2*2*64),[16,64,64,2,2,64])
        var shuffled=block.transposed(axes:[0,1,3,2,4,5]).reshaped(16,128,128,64); eval(shuffled)
        let shuffleStart=Date()
        for _ in 0..<20 { shuffled=block.transposed(axes:[0,1,3,2,4,5]).reshaped(16,128,128,64); eval(shuffled) }
        print(String(format:"pixel-shuffle 16x64->128: %.2f ms",(Date().timeIntervalSince(shuffleStart)/20)*1000))
        elementwise("layerNorm [16,128,128,64]",[16,128,128,64]) { MLXFast.layerNorm($0,weight:MLXArray([Float](repeating:1,count:64)),bias:MLXArray([Float](repeating:0,count:64)),eps:1e-5) }
        elementwise("gelu [16,128,128,64]",[16,128,128,64]) { $0*0.5*(1+erf($0/1.41421356)) }
        elementwise("add [16,128,128,64]",[16,128,128,64]) { $0+1 }
        elementwise("gelu [16,256,256,32]",[16,256,256,32]) { $0*0.5*(1+erf($0/1.41421356)) }
        elementwise("sigmoid resize-ish [1,1024,1024,1]",[1,1024,1024,1]) { sigmoid($0) }
        // Fused decoder-tail upscale GEMMs vs the MLX pipeline they replace,
        // per prompt-chunk size (the automatic profile runs chunk 4 on M3).
        func benchUpscale(_ chunk:Int,_ stage:Int,repeats:Int=30) {
            func timed(_ label:String,_ body:()->MLXArray) {
                var out=body(); eval(out)
                let start=Date()
                for _ in 0..<repeats { out=body(); eval(out) }
                print(String(format:"  %@: %.3f ms",label,(Date().timeIntervalSince(start)/Double(repeats))*1000))
            }
            let lnw=MLXArray([Float](repeating:1,count:64)),lnb=MLXArray([Float](repeating:0,count:64))
            if stage == 1 {
                let x=MLXArray([Float](repeating:0.01,count:chunk*4096*256),[chunk,64,64,256])
                let w=MLXArray([Float](repeating:0.02,count:256*256),[256,256])
                let bias=MLXArray([Float](repeating:0.01,count:256),[256])
                let rows=chunk*4096
                timed("raw MLX matmul [\(rows),256]x[256,256]") { x.reshaped(-1,256).matmul(w) }
                timed("MLX pipeline (matmul+bias+shuffle+LN+GELU)") {
                    let block=x.reshaped(-1,256).matmul(w)+bias
                    return SAFIRENorm.norm(block.reshaped(chunk,64,64,2,2,64)
                        .transposed(axes:[0,1,3,2,4,5]).reshaped(chunk,128,128,64),
                        residual:nil,weight:lnw,bias:lnb,eps:1e-6,gelu:true)
                }
                for (bm,tg,ku) in [(32,128,1),(64,256,1),(32,256,16),(64,256,16),(64,512,16)] {
                    if (chunk*4096)%bm != 0 { continue }
                    timed("fused kernel u1 bm\(bm) tg\(tg) ku\(ku)") {
                        SAFIREUpscale.upscale1(x,weight:w,bias:bias,lnWeight:lnw,lnBias:lnb,eps:1e-6,bm:bm,tg:tg,kunroll:ku)
                    }
                }
            } else {
                let x=MLXArray([Float](repeating:0.01,count:chunk*16384*64),[chunk,128,128,64])
                let w=MLXArray([Float](repeating:0.02,count:64*128),[64,128])
                let bias=MLXArray([Float](repeating:0.01,count:128),[128])
                let rows=chunk*16384
                timed("raw MLX matmul [\(rows),64]x[64,128]") { x.reshaped(-1,64).matmul(w) }
                timed("MLX pipeline (matmul+bias+shuffle+GELU)") {
                    let block=x.reshaped(-1,64).matmul(w)+bias
                    return SAFIRENorm.gelu(block.reshaped(chunk,128,128,2,2,32)
                        .transposed(axes:[0,1,3,2,4,5]).reshaped(chunk,256,256,32))
                }
                for (bm,tg,ku) in [(32,128,1),(64,256,1),(32,128,16),(64,256,16),(64,128,16)] {
                    if (chunk*16384)%bm != 0 { continue }
                    timed("fused kernel u2 bm\(bm) tg\(tg) ku\(ku)") {
                        SAFIREUpscale.upscale2(x,weight:w,bias:bias,bm:bm,tg:tg,kunroll:ku)
                    }
                }
            }
        }
        for chunk in [1,4,16] {
            print("upscale stage 1, chunk \(chunk):")
            benchUpscale(chunk,1)
            print("upscale stage 2, chunk \(chunk):")
            benchUpscale(chunk,2)
        }
    } else if let destination=option("--prepare") {
        try SAFIREModel.prepare(source:URL(fileURLWithPath:path),destination:URL(fileURLWithPath:destination,isDirectory:true))
        print("Prepared FP16 SAFIRE checkpoint at \(destination)")
    } else if CommandLine.arguments.contains("--validate") {
        let input=MLXArray.zeros([1,1024,1024,3])
        let embedding=try model.encode(input)
        let output=try model.decode(embedding:embedding,points:MLXArray([Float(512),512],[1,2]))
        guard all(isFinite(output.masks)).item(Bool.self),all(isFinite(output.quality)).item(Bool.self) else { fail("Nonfinite SAFIRE output") }
        print("SAFIRE native load/inference check passed")
    } else if let image=option("--image"),let output=option("--output") {
        try infer(model,image:image,output:output)
    } else if CommandLine.arguments.contains("--serve") {
        print("SAF1"); fflush(stdout)
        // Explicitly versioned interactive ops beside the one-shot infer path:
        // {"op":"encode","input":PNG,"key":"ID"}            → cache embedding
        // {"op":"decode","key":"ID","points":[[x,y],…],
        //  "output":FILE}                                    → masks/quality safetensors
        // Requests without "op" keep the original infer contract.
        var cache=EmbeddingCache(limit:2)
        while let line=readLine() {
            do {
                guard let bytes=line.data(using:.utf8),
                      let request=try JSONSerialization.jsonObject(with:bytes) as? [String:Any] else {
                    throw SAFIREError.invalid("Invalid worker request")
                }
                switch request["op"] as? String {
                case "encode":
                    guard let image=request["input"] as? String, let key=request["key"] as? String else {
                        throw SAFIREError.invalid("Encode op requires input and key")
                    }
                    let seconds=try cache.encode(model:model,image:image,key:key)
                    print("OK {\"seconds\":\(seconds)}")
                case "decode":
                    guard let key=request["key"] as? String, let output=request["output"] as? String,
                          let points=request["points"] as? [[Double]], !points.isEmpty, points.count<=16,
                          points.allSatisfy({ $0.count==2 }) else {
                        throw SAFIREError.invalid("Decode op requires key, points (≤16) and output")
                    }
                    let flat=points.flatMap { [Float($0[0]),Float($0[1])] }
                    let start=Date()
                    let decoded=try model.decode(embedding:cache.embedding(key:key),
                        points:MLXArray(flat,[points.count,2]))
                    try save(arrays:["masks":decoded.masks,"quality":decoded.quality],
                             url:URL(fileURLWithPath:output))
                    print("OK {\"seconds\":\(Date().timeIntervalSince(start)),\"points\":\(points.count)}")
                default:
                    guard let image=request["input"] as? String, let output=request["output"] as? String else {
                        throw SAFIREError.invalid("Invalid worker request")
                    }
                    try infer(model,image:image,output:output,binary:request["format"] as? String == "sfm1")
                    print("OK")
                }
            } catch { print("ERROR \(error)") }
            fflush(stdout)
        }
    } else { fail("Missing inference command") }
} catch { fail("SAFIRE: \(error)") }

/// Bounded GPU-resident embedding cache for interactive investigation: one
/// encode per photograph, then repeated decodes against the held embedding.
/// Least-recently-used eviction keeps at most `limit` embeddings resident.
struct EmbeddingCache {
    private let limit:Int
    private var entries:[String:MLXArray]=[:]
    private var order:[String]=[]
    init(limit:Int) { self.limit=max(1,limit) }
    mutating func encode(model:SAFIREModel,image:String,key:String) throws -> Double {
        let input=try rgbaImage(image)
        let normalized=try SAFIREPreprocessing.rgba(input.bytes,width:input.width,height:input.height)
        let start=Date()
        let embedding=try model.encode(normalized)
        eval(embedding)
        order.removeAll { $0==key }
        entries[key]=embedding; order.append(key)
        while order.count>limit {
            let evicted=order.removeFirst(); entries[evicted]=nil
        }
        return Date().timeIntervalSince(start)
    }
    mutating func embedding(key:String) throws -> MLXArray {
        guard let embedding=entries[key] else { throw SAFIREError.invalid("Unknown embedding key; encode first") }
        order.removeAll { $0==key }; order.append(key)
        return embedding
    }
}

func infer(_ model:SAFIREModel,image:String,output:String,binary:Bool=false) throws {
    let start=Date()
    let input=try rgbaImage(image)
    let normalized=try SAFIREPreprocessing.rgba(input.bytes,width:input.width,height:input.height)
    if let dump=option("--dump-preprocessing") { try save(arrays:["input":normalized],url:URL(fileURLWithPath:dump)) }
    memProbe("pre-predict")
    let prediction=try model.predict(normalized)
    memProbe("post-predict")
    if binary {
        // SFM1 is shared with the app's SafireRasterCodec. Keep dense GPU output
        // as contiguous bytes rather than formatting a million JSON numbers.
        var data=Data("SFM1".utf8)
        for value in [UInt32(1024),UInt32(1024),UInt32(1)] {
            var value=value.littleEndian;withUnsafeBytes(of:&value) { data.append(contentsOf:$0) }
        }
        prediction.probabilities.withUnsafeBytes { data.append(contentsOf:$0) }
        prediction.partitions.withUnsafeBytes { data.append(contentsOf:$0) }
        try data.write(to:URL(fileURLWithPath:output),options:.atomic);return
    }
    let result:[String:Any]=["width":1024,"height":1024,"probabilities":prediction.probabilities,
        "partitions":prediction.partitions.map(Int.init),"selectedPrompts":prediction.selectedPrompts,
        "runtimeRevision":SAFIREModel.revision,"seconds":Date().timeIntervalSince(start),
        "peakMLXBytes":Memory.peakMemory,"device":model.diagnostics.device]
    try JSONSerialization.data(withJSONObject:result,options:[.sortedKeys]).write(to:URL(fileURLWithPath:output),options:.atomic)
}
func rgbaImage(_ path:String) throws -> (bytes:[UInt8],width:Int,height:Int) {
    guard let source=CGImageSourceCreateWithURL(URL(fileURLWithPath:path) as CFURL,nil),
          let image=CGImageSourceCreateImageAtIndex(source,0,nil),image.width<=32768,image.height<=32768,
          image.width*image.height<=160_000_000 else { throw SAFIREError.invalid("Invalid oriented RGBA image") }
    var bytes=[UInt8](repeating:0,count:image.width*image.height*4)
    let ok=bytes.withUnsafeMutableBytes { pointer -> Bool in
        guard let ctx=CGContext(data:pointer.baseAddress,width:image.width,height:image.height,bitsPerComponent:8,bytesPerRow:image.width*4,
                               space:CGColorSpace(name:CGColorSpace.sRGB)!,bitmapInfo:CGImageAlphaInfo.premultipliedLast.rawValue) else { return false }
        ctx.draw(image,in:CGRect(x:0,y:0,width:image.width,height:image.height)); return true
    }
    guard ok else { throw SAFIREError.invalid("Unable to decode pixels") }
    return (bytes,image.width,image.height)
}
