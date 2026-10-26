// Deterministic opaque PNG fixtures using Apple frameworks only. No UI access.
import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
let root=URL(fileURLWithPath:CommandLine.arguments[1],isDirectory:true)
try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
for (name,width,height) in [("odd",644,516),("landscape",1025,769),("portrait",769,1025),
                            ("maximum",2000,2000),("wide",4000,1000),("tall",1000,4000),("full-native",6000,4000),("oversize",6001,4000)] {
    var bytes=[UInt8](repeating:255,count:width*height*4)
    for y in 0..<height { for x in 0..<width {
        let i=(y*width+x)*4
        bytes[i]=UInt8((x*17+y*31)%256);bytes[i+1]=UInt8((x*37+y*11)%256);bytes[i+2]=UInt8((x*7+y*43)%256)
    } }
    let provider=CGDataProvider(data:Data(bytes) as CFData)!
    let image=CGImage(width:width,height:height,bitsPerComponent:8,bitsPerPixel:32,bytesPerRow:width*4,
                      space:CGColorSpace(name:CGColorSpace.sRGB)!,bitmapInfo:CGBitmapInfo(rawValue:CGImageAlphaInfo.premultipliedLast.rawValue),
                      provider:provider,decode:nil,shouldInterpolate:false,intent:.defaultIntent)!
    let output=CGImageDestinationCreateWithURL(root.appendingPathComponent(name+".png") as CFURL,UTType.png.identifier as CFString,1,nil)!
    CGImageDestinationAddImage(output,image,nil)
    guard CGImageDestinationFinalize(output) else { fatalError("PNG encoding failed") }
}
