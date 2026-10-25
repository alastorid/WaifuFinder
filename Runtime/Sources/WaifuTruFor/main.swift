// GRIP-UNINA TruFor native worker, informational/nonprofit use only.
import Foundation
import ImageIO
import CoreGraphics
import MLX
import TruForMLX

func option(_ key:String)->String? { CommandLine.arguments.firstIndex(of:key).flatMap{$0+1<CommandLine.arguments.count ? CommandLine.arguments[$0+1] : nil} }
func fail(_ text:String)->Never { FileHandle.standardError.write(Data((text+"\n").utf8));exit(1) }
func image(_ path:String)throws->(MLXArray,Int,Int) {
    guard let source=CGImageSourceCreateWithURL(URL(fileURLWithPath:path) as CFURL,nil),let image=CGImageSourceCreateImageAtIndex(source,0,nil),image.width>=32,image.height>=32,image.width*image.height<=24_000_000 else { throw TruForError.invalid("Invalid native image") }
    var rgba=[UInt8](repeating:0,count:image.width*image.height*4)
    let success=rgba.withUnsafeMutableBytes { data in
        guard let context=CGContext(data:data.baseAddress,width:image.width,height:image.height,bitsPerComponent:8,bytesPerRow:image.width*4,space:CGColorSpace(name:CGColorSpace.sRGB)!,bitmapInfo:CGImageAlphaInfo.premultipliedLast.rawValue) else { return false }
        context.draw(image,in:CGRect(x:0,y:0,width:image.width,height:image.height));return true
    }
    guard success else { throw TruForError.invalid("Image decode failed") }
    // Avoid inventing RGB underneath transparent pixels, unlike silently flattening.
    guard stride(from:3,to:rgba.count,by:4).allSatisfy({rgba[$0]==255}) else { throw TruForError.invalid("TruFor requires an opaque Photos rendition") }
    return (try TruForPreprocessing.rgba(rgba,width:image.width,height:image.height),image.width,image.height)
}
func run(_ model:TruForModel,paths:[String],output:String,evidence:Bool)throws {
    guard !paths.isEmpty,paths.count<=4,(!evidence || paths.count==1) else { throw TruForError.invalid("Invalid batch") }
    let profiling=ProcessInfo.processInfo.environment["WAIFU_TRUFOR_PROFILE"] != nil
    func mark(_ name:String,_ start:ContinuousClock.Instant)->ContinuousClock.Instant {
        if profiling {
            let elapsed=ContinuousClock.now-start
            FileHandle.standardError.write(Data(String(format:"{\"phase\":\"%@\",\"ms\":%.1f}\n",name,Double(elapsed.components.attoseconds)/1e15).utf8))
        }
        return ContinuousClock.now
    }
    var start=ContinuousClock.now
    let images=try paths.map(image);start=mark("decode+\(paths.count)",start)
    if evidence {
        let result=try model.predict(images[0].0,evidence:true);start=mark("predict",start)
        let score=result.score.item(Float.self),maps=[result.anomaly!,result.reliability!,result.noiseprint!]
        eval(maps);start=mark("maps-sync",start)
        var data=Data("TFM1".utf8)
        for value in [UInt32(images[0].1),UInt32(images[0].2),UInt32(3)] { var v=value.littleEndian;withUnsafeBytes(of:&v){data.append(contentsOf:$0)} }
        var value=score;withUnsafeBytes(of:&value){data.append(contentsOf:$0)}
        for map in maps { map.asArray(Float.self).withUnsafeBytes{data.append(contentsOf:$0)} }
        try data.write(to:URL(fileURLWithPath:output),options:.atomic);start=mark("serialize",start)
    } else {
        let scores:[Float]
        if images.allSatisfy({$0.1==images[0].1 && $0.2==images[0].2}),images.count*images[0].1*images[0].2<=1_048_576 {
            scores=try model.predict(concatenated(images.map{$0.0},axis:0),evidence:false).score.asArray(Float.self)
        } else { scores=try images.map{try model.predict($0.0,evidence:false).score.item(Float.self)} }
        start=mark("predict+\(paths.count)",start)
        try JSONSerialization.data(withJSONObject:["scores":scores],options:[.sortedKeys]).write(to:URL(fileURLWithPath:output),options:.atomic);start=mark("serialize",start)
    }
}
do {
    guard let path=option("--model") else { fail("Missing native model directory") }
    Memory.memoryLimit=6*1024*1024*1024;Memory.cacheLimit=128*1024*1024
    let model=try TruForModel(directory:URL(fileURLWithPath:path))
    if let input=option("--dump-input"),let output=option("--output") {
        let decoded=try image(input);try save(arrays:["input":decoded.0],url:URL(fileURLWithPath:output))
    }
    else if CommandLine.arguments.contains("--validate") { print("TruFor native weights verified · \(model.device)") }
    else if let input=option("--activations"),let output=option("--output") {
        let tensors=try loadArrays(url:URL(fileURLWithPath:input));guard let x=tensors["input"] else { throw TruForError.invalid("Missing input") }
        let root=URL(fileURLWithPath:output,isDirectory:true);try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        let result=try model.predict(x,evidence:true) { name,value in try save(arrays:[name:value],url:root.appendingPathComponent(name+".safetensors")) }
        try save(arrays:["anomaly":result.anomaly!,"reliability":result.reliability!,"noiseprint":result.noiseprint!,"score":result.score],url:root.appendingPathComponent("outputs.safetensors"))
        print("Native TruFor activations written")
    } else if CommandLine.arguments.contains("--serve") {
        print("TRF1");fflush(stdout)
        while let line=readLine() {
            do {
                guard let bytes=line.data(using:.utf8),let request=try JSONSerialization.jsonObject(with:bytes) as? [String:Any],let paths=request["inputs"] as? [String],let output=request["output"] as? String else { throw TruForError.invalid("Invalid worker request") }
                try run(model,paths:paths,output:output,evidence:request["mode"] as? String == "evidence");print("OK")
            } catch { print("ERROR \(error)") };fflush(stdout)
        }
    } else if let input=option("--image"),let output=option("--output") { try run(model,paths:[input],output:output,evidence:CommandLine.arguments.contains("--evidence")) }
    else { fail("Missing inference command") }
} catch { fail("TruFor: \(error)") }
