// Apple-native, deterministic image-only fixtures. No UI or external media tools.
import AppKit
import CoreGraphics
import CoreText
import ImageIO
import UniformTypeIdentifiers
let args = CommandLine.arguments
let input = URL(fileURLWithPath: args[1]), output = URL(fileURLWithPath: args[2])
try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
let source = CGImageSourceCreateWithURL(input as CFURL, nil)!
let image = CGImageSourceCreateImageAtIndex(source, 0, nil)!
let w = image.width, h = image.height
let context = CGContext(data:nil,width:w,height:h,bitsPerComponent:8,bytesPerRow:w*4,
    space:CGColorSpace(name:CGColorSpace.sRGB)!,bitmapInfo:CGImageAlphaInfo.premultipliedLast.rawValue)!
context.draw(image,in:CGRect(x:0,y:0,width:w,height:h))
context.setFillColor(CGColor(red:1,green:1,blue:1,alpha:0.78))
context.fill(CGRect(x:0,y:h/2-65,width:w,height:130))
let text = NSAttributedString(string:"SAMPLE WATERMARK",attributes:[.font:NSFont.boldSystemFont(ofSize:65),.foregroundColor:NSColor.red])
context.textPosition=CGPoint(x:40,y:h/2-20)
CTLineDraw(CTLineCreateWithAttributedString(text),context)
let destination = CGImageDestinationCreateWithURL(output.appendingPathComponent("watermarked.png") as CFURL,UTType.png.identifier as CFString,1,nil)!
CGImageDestinationAddImage(destination,context.makeImage()!,nil)
precondition(CGImageDestinationFinalize(destination))
