import Foundation
import CoreGraphics
import CoreImage
import ImageIO
import UniformTypeIdentifiers
import CryptoKit

// This error adapter lets this offline generator compile the unchanged production
// NativeSpatialUpscaler without importing the unrelated video workspace types.
enum UncoverError: Error { case backend(String) }

@main struct MakeLineageFixtures {
    static let context=CIContext(options:[.cacheIntermediates:false])
    static func image(_ ci:CIImage) throws -> CGImage {
        guard let result=context.createCGImage(ci,from:ci.extent,format:.RGBA8,colorSpace:CGColorSpace(name:CGColorSpace.sRGB)) else { throw UncoverError.backend("Unable to render fixture") }
        return result
    }
    static func resize(_ source:CGImage,_ side:Int) throws -> CGImage {
        let factor=CGFloat(side)/CGFloat(max(source.width,source.height))
        return try image(CIImage(cgImage:source).transformed(by:CGAffineTransform(scaleX:factor,y:factor)))
    }
    static func save(_ image:CGImage,to url:URL,jpeg:Double?=nil) throws {
        let type=jpeg==nil ? UTType.png.identifier : UTType.jpeg.identifier
        guard let out=CGImageDestinationCreateWithURL(url as CFURL,type as CFString,1,nil) else { throw UncoverError.backend("Unable to write fixture") }
        CGImageDestinationAddImage(out,image,jpeg.map{[kCGImageDestinationLossyCompressionQuality:$0] as CFDictionary})
        guard CGImageDestinationFinalize(out) else { throw UncoverError.backend("Unable to encode fixture") }
    }
    static func watermark(_ source:CGImage) throws -> CGImage {
        let w=CGFloat(source.width),h=CGFloat(source.height),unit=w/512
        var result=CIImage(cgImage:source)
        // A small visible WF watermark, made from deterministic vector pixels.
        let letters=[["10001","10001","10101","10101","11011","10001","10001"],
                     ["11111","10000","10000","11110","10000","10000","10000"]]
        for (letter,rows) in letters.enumerated() { for (row,bits) in rows.enumerated() { for (col,bit) in bits.enumerated() where bit=="1" {
            let rect=CGRect(x:(350+CGFloat(letter*42+col*6))*unit,y:h-(430+CGFloat(row*6))*unit,width:5*unit,height:5*unit)
            result=CIImage(color:CIColor(red:1,green:1,blue:1,alpha:0.7)).cropped(to:rect).composited(over:result)
        } } }
        return try image(result.cropped(to:CGRect(x:0,y:0,width:w,height:h)))
    }
    static func inpaint(_ source:CGImage) throws -> CGImage {
        let w=source.width,h=source.height
        var rgb=try NativeSpatialUpscaler.straightRGBPixels(source)
        let x0=w*43/100,y0=h*36/100,size=max(8,w/10)
        for y in y0..<(y0+size) { for x in x0..<(x0+size) { for c in 0..<3 {
            let top=Float(rgb[((y0-1)*w+x)*3+c]),bottom=Float(rgb[((y0+size)*w+x)*3+c])
            let t=Float(y-y0+1)/Float(size+1)
            rgb[(y*w+x)*3+c]=UInt8(clamping:Int((top*(1-t)+bottom*t).rounded()))
        } } }
        var rgba=[UInt8](repeating:255,count:w*h*4)
        for i in 0..<(w*h) { for c in 0..<3 { rgba[i*4+c]=rgb[i*3+c] } }
        guard let provider=CGDataProvider(data:Data(rgba) as CFData),let result=CGImage(width:w,height:h,bitsPerComponent:8,bitsPerPixel:32,bytesPerRow:w*4,space:CGColorSpace(name:CGColorSpace.sRGB)!,bitmapInfo:CGBitmapInfo(rawValue:CGImageAlphaInfo.premultipliedLast.rawValue),provider:provider,decode:nil,shouldInterpolate:false,intent:.defaultIntent) else { throw UncoverError.backend("Unable to create inpaint fixture") }
        return result
    }
    static func main() throws {
        guard CommandLine.arguments.count>=3 else { throw UncoverError.backend("Usage: maker SOURCE_IMAGE NEW_DIRECTORY [REALESRGAN_MODEL] [DISTINCT_NASA_CAPTURE]") }
        let sourceURL=URL(fileURLWithPath:CommandLine.arguments[1]),root=URL(fileURLWithPath:CommandLine.arguments[2],isDirectory:true)
        guard !FileManager.default.fileExists(atPath:root.path) else { throw UncoverError.backend("Fixture directory already exists; use a new version") }
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        guard let source=CGImageSourceCreateWithURL(sourceURL as CFURL,nil),let raw=CGImageSourceCreateThumbnailAtIndex(source,0,[kCGImageSourceCreateThumbnailFromImageAlways:true,kCGImageSourceCreateThumbnailWithTransform:true,kCGImageSourceThumbnailMaxPixelSize:512] as CFDictionary) else { throw UncoverError.backend("Unreadable fixture source") }
        var records:[[String:Any]]=[]
        func add(_ name:String,_ image:CGImage,_ parents:[String],_ operation:String,jpeg:Double?=nil) throws {
            let url=root.appendingPathComponent(name);try save(image,to:url,jpeg:jpeg)
            let digest=SHA256.hash(data:try Data(contentsOf:url)).map{String(format:"%02x",$0)}.joined()
            records.append(["name":name,"parents":parents,"operation":operation,"width":image.width,"height":image.height,"sha256":digest])
        }
        try add("baseline.png",raw,[],"Apple ImageIO 512px rendition of the attributed source; not a claim of an untouched camera original")
        try add("recompressed.jpg",raw,["baseline.png"],"JPEG recompression quality 0.35",jpeg:0.35)
        let down=try resize(raw,256)
        try add("downsampled.png",down,["baseline.png"],"Downsample 0.5x")
        try add("upscaled.png",resize(down,512),["downsampled.png"],"Classical interpolation 2x")
        try add("watermarked.png",watermark(raw),["baseline.png"],"Semitransparent WF watermark")
        try add("watermark-removed.png",raw,["watermarked.png"],"Exact restoration of the known baseline pixels under the watermark")
        let retouched=try inpaint(raw)
        try add("locally-inpainted.png",retouched,["baseline.png"],"Classical boundary-interpolated patch fill, top-left x=43%, y=36%, width=height=10%")
        try add("generation-3.jpg",watermark(down),["downsampled.png"],"Downsample, then watermark, then JPEG recompression",jpeg:0.4)
        let tone=try image(CIImage(cgImage:raw).applyingFilter("CIColorControls",parameters:[kCIInputBrightnessKey:0.035,kCIInputSaturationKey:1.05]))
        try add("sibling-tone.png",tone,["baseline.png"],"Independent tone adjustment")
        if let crop=raw.cropping(to:CGRect(x:64,y:64,width:raw.width-128,height:raw.height-128)) { try add("sibling-crop.png",crop,["baseline.png"],"Independent crop") }
        if CommandLine.arguments.count>=4 {
            let restoration=try NativeSpatialUpscaler(modelURL:URL(fileURLWithPath:CommandLine.arguments[3]))
            try add("ai-restored.png",restoration.upscale(down),["downsampled.png"],"Actual native Core ML RealESRGAN x2 restoration; no simulated AI output")
        }
        if CommandLine.arguments.count>=5 {
            let distinctURL=URL(fileURLWithPath:CommandLine.arguments[4])
            guard let other=CGImageSourceCreateWithURL(distinctURL as CFURL,nil),let image=CGImageSourceCreateThumbnailAtIndex(other,0,[kCGImageSourceCreateThumbnailFromImageAlways:true,kCGImageSourceCreateThumbnailWithTransform:true,kCGImageSourceThumbnailMaxPixelSize:512] as CFDictionary) else { throw UncoverError.backend("Unreadable distinct capture") }
            try add("distinct-capture.jpg",image,[],"Separate NASA Apollo 17 photograph AS17-148-22726; not derived from baseline",jpeg:0.95)
        }
        let missing=root.appendingPathComponent("missing-baseline",isDirectory:true)
        try FileManager.default.createDirectory(at:missing,withIntermediateDirectories:true)
        for name in ["downsampled.png","watermarked.png","sibling-tone.png"] { try FileManager.default.copyItem(at:root.appendingPathComponent(name),to:missing.appendingPathComponent(name)) }
        let report:[String:Any]=["source":"NASA Johnson Space Center, Apollo 17 photograph AS17-148-22727","sourceURL":"https://science.nasa.gov/resource/the-blue-marble/","mediaTerms":"https://www.nasa.gov/nasa-brand-center/images-and-media/","generation":"Apple ImageIO/CoreImage and production NativeSpatialUpscaler; ground truth describes fixture edits only","images":records]
        try JSONSerialization.data(withJSONObject:report,options:[.prettyPrinted,.sortedKeys]).write(to:root.appendingPathComponent("history.json"))
        print("Generated",records.count,"known-history images at",root.path)
    }
}
