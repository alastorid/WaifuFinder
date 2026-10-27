// Offline native image comparison. SSIM uses nonoverlapping 8x8 luma windows.
import Foundation
import CoreGraphics
import ImageIO

func read(_ path:String) throws -> (Int,Int,[UInt8]) {
    guard let source=CGImageSourceCreateWithURL(URL(fileURLWithPath:path) as CFURL,nil),
          let image=CGImageSourceCreateImageAtIndex(source,0,nil) else { throw NSError(domain:"Image",code:1) }
    var pixels=[UInt8](repeating:0,count:image.width*image.height*4)
    pixels.withUnsafeMutableBytes { raw in
        let context=CGContext(data:raw.baseAddress,width:image.width,height:image.height,bitsPerComponent:8,bytesPerRow:image.width*4,
            space:CGColorSpace(name:CGColorSpace.sRGB)!,bitmapInfo:CGImageAlphaInfo.noneSkipLast.rawValue)!
        context.draw(image,in:CGRect(x:0,y:0,width:image.width,height:image.height))
    }
    return (image.width,image.height,pixels)
}
guard CommandLine.arguments.count == 4 else { fatalError("Supply REFERENCE ACTUAL REPORT") }
let (w,h,a)=try read(CommandLine.arguments[1]),(bw,bh,b)=try read(CommandLine.arguments[2])
guard w == bw && h == bh else { fatalError("Output dimensions differ") }
var square=0.0,absolute=0.0,maximum=0.0,edge=0.0,bias=[Double](repeating:0,count:3)
for y in 0..<h { for x in 0..<w { for c in 0..<3 {
    let i=(y*w+x)*4+c,d=(Double(b[i])-Double(a[i]))/255
    square += d*d;absolute += abs(d);maximum=max(maximum,abs(d));bias[c] += d
    if x > 0 { edge += abs((Double(b[i])-Double(b[i-4])-Double(a[i])+Double(a[i-4]))/255) }
    if y > 0 { edge += abs((Double(b[i])-Double(b[i-w*4])-Double(a[i])+Double(a[i-w*4]))/255) }
} } }
var ssim=0.0,windows=0
for y in stride(from:0,to:h,by:8) { for x in stride(from:0,to:w,by:8) {
    var sx=0.0,sy=0.0,sxx=0.0,syy=0.0,sxy=0.0,n=0.0
    for yy in y..<min(h,y+8) { for xx in x..<min(w,x+8) {
        let i=(yy*w+xx)*4
        let aa=(0.2126*Double(a[i])+0.7152*Double(a[i+1])+0.0722*Double(a[i+2]))/255
        let bb=(0.2126*Double(b[i])+0.7152*Double(b[i+1])+0.0722*Double(b[i+2]))/255
        sx += aa;sy += bb;sxx += aa*aa;syy += bb*bb;sxy += aa*bb;n += 1
    } }
    let mx=sx/n,my=sy/n,vx=max(0,sxx/n-mx*mx),vy=max(0,syy/n-my*my),cov=sxy/n-mx*my
    ssim += ((2*mx*my+0.0001)*(2*cov+0.0009))/((mx*mx+my*my+0.0001)*(vx+vy+0.0009));windows += 1
} }
// Each FP16 tensor is separately gated against FP32 at 0.01. Comparing two
// rounded 8-bit FP16 outputs allows their combined error: at most five code
// values, with much tighter whole-image PSNR/SSIM and mean-error gates.
let count=Double(w*h*3),mse=square/count,psnr=mse == 0 ? 999 : -10*log10(mse),similarity=ssim/Double(windows)
let report:[String:Any]=["width":w,"height":h,"psnrDB":psnr,"ssim8x8Luma":similarity,"maximumAbsoluteError":maximum,"meanAbsoluteError":absolute/count,
    "edgeGradientMeanAbsoluteError":edge/Double(((w-1)*h+w*(h-1))*3),"meanRGBBias":bias.map { $0/Double(w*h) },
    "maximumAllowedByteDifference":5,"passed":maximum <= 5.0/255 && absolute/count <= 0.002 && psnr >= 45 && similarity >= 0.995]
try JSONSerialization.data(withJSONObject:report,options:[.prettyPrinted,.sortedKeys]).write(to:URL(fileURLWithPath:CommandLine.arguments[3]))
print("PSNR \(psnr) dB, SSIM \(similarity), max \(maximum)")
