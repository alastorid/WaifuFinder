// GRIP-UNINA TruFor native worker, informational/nonprofit use only.
import Foundation
import Darwin
import ImageIO
import CoreGraphics
import MLX
import TruForMLX

func option(_ key:String)->String? { CommandLine.arguments.firstIndex(of:key).flatMap{$0+1<CommandLine.arguments.count ? CommandLine.arguments[$0+1] : nil} }
func fail(_ text:String)->Never { FileHandle.standardError.write(Data((text+"\n").utf8));exit(1) }
func image(_ path:String)throws->(TruForInput,Int,Int) {
    guard let source=CGImageSourceCreateWithURL(URL(fileURLWithPath:path) as CFURL,[kCGImageSourceShouldCache:false] as CFDictionary),
          let properties=CGImageSourceCopyPropertiesAtIndex(source,0,nil) as? [CFString:Any],
          let width=properties[kCGImagePropertyPixelWidth] as? Int,let height=properties[kCGImagePropertyPixelHeight] as? Int else { throw TruForError.invalid("Invalid native image") }
    // Reject unsafe geometry before ImageIO allocates the decoded bitmap.
    try TruForMemoryPolicy.validate(width:width,height:height)
    guard let image=CGImageSourceCreateImageAtIndex(source,0,nil),image.width==width,image.height==height else { throw TruForError.invalid("Image decode failed") }
    var rgba=[UInt8](repeating:0,count:image.width*image.height*4)
    let success=rgba.withUnsafeMutableBytes { data in
        guard let context=CGContext(data:data.baseAddress,width:image.width,height:image.height,bitsPerComponent:8,bytesPerRow:image.width*4,space:CGColorSpace(name:CGColorSpace.sRGB)!,bitmapInfo:CGImageAlphaInfo.premultipliedLast.rawValue) else { return false }
        context.draw(image,in:CGRect(x:0,y:0,width:image.width,height:image.height));return true
    }
    guard success else { throw TruForError.invalid("Image decode failed") }
    // Avoid inventing RGB underneath transparent pixels, unlike silently flattening.
    guard stride(from:3,to:rgba.count,by:4).allSatisfy({rgba[$0]==255}) else { throw TruForError.invalid("TruFor requires an opaque Photos rendition") }
    return (try TruForInput(rgba:rgba,width:image.width,height:image.height),image.width,image.height)
}
func run(_ model:TruForModel,paths:[String],output:String,evidence:Bool)throws {
    guard !paths.isEmpty,paths.count<=4,(!evidence || paths.count==1) else { throw TruForError.invalid("Invalid batch") }
    let profiling=ProcessInfo.processInfo.environment["WAIFU_TRUFOR_PROFILE"] != nil
    func mark(_ name:String,_ start:ContinuousClock.Instant)->ContinuousClock.Instant {
        if profiling {
            let elapsed=ContinuousClock.now-start
            FileHandle.standardError.write(Data(String(format:"{\"phase\":\"%@\",\"ms\":%.1f}\n",name,Double(elapsed.components.seconds)*1000+Double(elapsed.components.attoseconds)/1e15).utf8))
        }
        return ContinuousClock.now
    }
    var start=ContinuousClock.now
    if evidence {
        let decoded=try autoreleasepool { try image(paths[0]) };TruForMemoryPolicy.releaseUnusedCPU();start=mark("decode",start)
        let result=try model.predict(decoded.0,evidence:true);start=mark("predict",start)
        let score=result.score.item(Float.self),maps=[result.anomaly!,result.reliability!,result.noiseprint!]
        eval(maps);start=mark("maps-sync",start)
        var header=Data("TFM1".utf8)
        for value in [UInt32(decoded.1),UInt32(decoded.2),UInt32(3)] { var v=value.littleEndian;withUnsafeBytes(of:&v){header.append(contentsOf:$0)} }
        var value=score;withUnsafeBytes(of:&value){header.append(contentsOf:$0)}
        let destination=URL(fileURLWithPath:output),temporary=destination.deletingLastPathComponent().appendingPathComponent(".trufor-\(UUID().uuidString).tmp")
        guard FileManager.default.createFile(atPath:temporary.path,contents:nil,attributes:[.posixPermissions:0o600]) else { throw TruForError.invalid("Unable to write TruFor evidence") }
        defer { try? FileManager.default.removeItem(at:temporary) }
        let handle=try FileHandle(forWritingTo:temporary)
        defer { try? handle.close() }
        try handle.write(contentsOf:header)
        let rows=max(1,65_536/decoded.1)
        for map in maps {
            for y in stride(from:0,to:decoded.2,by:rows) {
                try autoreleasepool {
                    let values=map[0...,y..<min(decoded.2,y+rows),0...].asArray(Float.self)
                    try values.withUnsafeBytes { try handle.write(contentsOf:Data($0)) }
                }
            }
        }
        try handle.close()
        guard rename(temporary.path,destination.path)==0 else { throw TruForError.invalid("Unable to finalize TruFor evidence") }
        start=mark("serialize",start)
    } else {
        // Bound the request to one decoded image and one model forward at a time.
        let scores=try paths.map { path in
            try autoreleasepool { () throws -> Float in
                let decoded=try autoreleasepool { try image(path) }
                TruForMemoryPolicy.releaseUnusedCPU()
                return try model.predict(decoded.0,evidence:false).score.item(Float.self)
            }
        }
        start=mark("predict+\(paths.count)",start)
        try JSONSerialization.data(withJSONObject:["scores":scores],options:[.sortedKeys]).write(to:URL(fileURLWithPath:output),options:.atomic);start=mark("serialize",start)
    }
}
let memoryMonitor=TruForMemoryMonitor()
func perform(_ model:TruForModel,paths:[String],output:String,evidence:Bool)throws {
    Memory.clearCache();TruForMemoryPolicy.releaseUnusedCPU();memoryMonitor.reset()
    defer { Memory.clearCache();TruForMemoryPolicy.releaseUnusedCPU();memoryMonitor.report("request-memory") }
    try autoreleasepool { try run(model,paths:paths,output:output,evidence:evidence) }
}
do {
    guard let path=option("--model") else { fail("Missing native model directory") }
    Memory.memoryLimit=1_200_000_000;Memory.cacheLimit=0
    let model=try autoreleasepool {try TruForModel(directory:URL(fileURLWithPath:path))}
    Memory.clearCache();memoryMonitor.report("model-memory")
    if let input=option("--dump-input"),let output=option("--output") {
        let decoded=try image(input);try save(arrays:["input":try decoded.0.materialize()],url:URL(fileURLWithPath:output))
    }
    else if CommandLine.arguments.contains("--validate") { print("TruFor native weights verified · \(model.device)") }
    else if let input=option("--activations"),let output=option("--output") {
        let inputURL=URL(fileURLWithPath:input)
        let fileSize=try inputURL.resourceValues(forKeys:[.fileSizeKey]).fileSize ?? Int.max
        guard fileSize<=TruForMemoryPolicy.maximumPixels*12+1_048_576 else { throw TruForError.invalid("Activation input exceeds TruFor's 2 GB memory budget") }
        memoryMonitor.reset()
        let tensors=try loadArrays(url:inputURL);guard let x=tensors["input"] else { throw TruForError.invalid("Missing input") }
        let root=URL(fileURLWithPath:output,isDirectory:true);try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        let result=try model.predict(x,evidence:true) { name,value in try save(arrays:[name:value],url:root.appendingPathComponent(name+".safetensors")) }
        try save(arrays:["anomaly":result.anomaly!,"reliability":result.reliability!,"noiseprint":result.noiseprint!,"score":result.score],url:root.appendingPathComponent("outputs.safetensors"))
        Memory.clearCache();memoryMonitor.report("activation-memory")
        print("Native TruFor activations written")
    } else if CommandLine.arguments.contains("--serve") {
        print("TRF1");fflush(stdout)
        while let line=readLine() {
            do {
                guard let bytes=line.data(using:.utf8),let request=try JSONSerialization.jsonObject(with:bytes) as? [String:Any],let paths=request["inputs"] as? [String],let output=request["output"] as? String else { throw TruForError.invalid("Invalid worker request") }
                try perform(model,paths:paths,output:output,evidence:request["mode"] as? String == "evidence");print("OK")
            } catch { print("ERROR \(error)") };fflush(stdout)
        }
    } else if let input=option("--image"),let output=option("--output") { try perform(model,paths:[input],output:output,evidence:CommandLine.arguments.contains("--evidence")) }
    else { fail("Missing inference command") }
} catch { fail("TruFor: \(error)") }
