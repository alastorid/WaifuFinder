import Foundation
import ImageIO
import CoreGraphics
import MLX
import SAFIREMLX

func option(_ key:String) -> String? { CommandLine.arguments.firstIndex(of:key).flatMap { $0+1 < CommandLine.arguments.count ? CommandLine.arguments[$0+1] : nil } }
func fail(_ message:String) -> Never { FileHandle.standardError.write(Data((message+"\n").utf8)); exit(1) }

do {
    guard let path=option("--model") else { fail("Usage: waifu-safire --model DIRECTORY --activations INPUT.safetensors --output DIRECTORY | --image IMAGE --output RESULT.json | --serve") }
    Memory.cacheLimit=128*1024*1024
    Memory.memoryLimit=2*1024*1024*1024
    let model=try SAFIREModel(directory:URL(fileURLWithPath:path))
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
        var report:[String:Any]=["seconds":Date().timeIntervalSince(start),"status":"native execution only; parity requires oracle comparison"]
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
        while let line=readLine() {
            do {
                guard let bytes=line.data(using:.utf8), let request=try JSONSerialization.jsonObject(with:bytes) as? [String:String],
                      let image=request["input"],let output=request["output"] else { throw SAFIREError.invalid("Invalid worker request") }
                try infer(model,image:image,output:output,binary:request["format"] == "sfm1"); print("OK")
            } catch { print("ERROR \(error)") }
            fflush(stdout)
        }
    } else { fail("Missing inference command") }
} catch { fail("SAFIRE: \(error)") }

func infer(_ model:SAFIREModel,image:String,output:String,binary:Bool=false) throws {
    let start=Date()
    let input=try rgbaImage(image)
    let normalized=try SAFIREPreprocessing.rgba(input.bytes,width:input.width,height:input.height)
    if let dump=option("--dump-preprocessing") { try save(arrays:["input":normalized],url:URL(fileURLWithPath:dump)) }
    let prediction=try model.predict(normalized)
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
