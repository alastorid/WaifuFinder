// Offline, Apple-native RGB calibration packing; no Core Image/Metal context.
import Foundation
import CoreGraphics
import ImageIO

guard CommandLine.arguments.count == 4, let size=Int(CommandLine.arguments[3]), size > 0,
      let source=CGImageSourceCreateWithURL(URL(fileURLWithPath:CommandLine.arguments[1]) as CFURL,nil),
      let image=CGImageSourceCreateImageAtIndex(source,0,nil) else { fatalError("Supply IMAGE FIXTURES SIZE") }
let directory=URL(fileURLWithPath:CommandLine.arguments[2])
try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
var cases=(try? JSONSerialization.jsonObject(with:Data(contentsOf:directory.appendingPathComponent("cases.json"))) as? [[String:Any]]) ?? []
for (index,rect) in [CGRect(x:0,y:0,width:image.width,height:image.height),
                      CGRect(x:image.width/4,y:image.height/8,width:image.width/2,height:image.height/2),
                      CGRect(x:image.width/4,y:image.height/2,width:image.width/2,height:image.height/2)].enumerated() {
    guard let crop=image.cropping(to:rect) else { fatalError("Invalid crop") }
    var rgb=[UInt8](repeating:0,count:size*size*4)
    rgb.withUnsafeMutableBytes { raw in
        let context=CGContext(data:raw.baseAddress,width:size,height:size,bitsPerComponent:8,bytesPerRow:size*4,
            space:CGColorSpace(name:CGColorSpace.sRGB)!,bitmapInfo:CGImageAlphaInfo.noneSkipLast.rawValue)!
        context.interpolationQuality = .high
        context.draw(crop,in:CGRect(x:0,y:0,width:size,height:size))
    }
    var planar=[Float](repeating:0,count:size*size*3)
    for c in 0..<3 { for pixel in 0..<size*size { planar[c*size*size+pixel]=Float(rgb[pixel*4+c])/255 } }
    let name="nasa-\(index)-\(size)"
    try planar.withUnsafeBytes { try Data($0).write(to:directory.appendingPathComponent(name+"-input.bin")) }
    cases.removeAll { $0["name"] as? String == name }
    cases.append(["name":name,"size":size,"width":size,"height":size,"calibrationOnly":true])
}
try JSONSerialization.data(withJSONObject:cases,options:[.prettyPrinted,.sortedKeys]).write(to:directory.appendingPathComponent("cases.json"))
