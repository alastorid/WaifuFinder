import SwiftUI
import AppKit

struct LineageOverlayMaterial: Sendable {
    let result: SafireSpatialResult
    let regions: [LineageRegion]
    let heatmap: CGImage
    let partitions: CGImage?
    static func make(_ result:SafireSpatialResult) throws -> Self {
        try result.validate()
        func image(partitions:Bool) throws -> CGImage {
            var bytes=[UInt8](repeating:0,count:result.width*result.height*4)
            let palette:[[UInt8]]=[[75,160,255],[255,150,45],[150,100,235],[40,210,150],[255,90,130],[210,200,30],[40,200,210],[190,140,90]]
            for i in result.probabilities.indices {
                let alpha:UInt8
                let color:[UInt8]
                if partitions,let labels=result.partitions {
                    alpha=110;color=palette[Int(labels[i])%palette.count]
                } else {
                    alpha=UInt8(min(180,max(0,Int(result.probabilities[i]*180))));color=[255,45,0]
                }
                for c in 0..<3 { bytes[i*4+c]=UInt8(Int(color[c])*Int(alpha)/255) }
                bytes[i*4+3]=alpha
            }
            guard let provider=CGDataProvider(data:Data(bytes) as CFData),let image=CGImage(width:result.width,height:result.height,bitsPerComponent:8,bitsPerPixel:32,bytesPerRow:result.width*4,
                space:CGColorSpace(name:CGColorSpace.sRGB)!,bitmapInfo:CGBitmapInfo(rawValue:CGImageAlphaInfo.premultipliedLast.rawValue),provider:provider,decode:nil,shouldInterpolate:false,intent:.defaultIntent) else { throw CrystalOrbImageSource.failure }
            return image
        }
        return try .init(result:result,regions:result.regions(),heatmap:image(partitions:false),partitions:result.partitions==nil ? nil : image(partitions:true))
    }
}
struct LineageOverlayConfiguration {
    let material:LineageOverlayMaterial
    let outlines:Bool
    let masks:Bool
    let contours:Bool
    let partitions:Bool
    let selected:Int?
    let select:(Int)->Void
}
struct LineageEvidenceOverlay: View {
    let configuration:LineageOverlayConfiguration
    let viewport:CrystalOrbViewport
    let alignment:CrystalOrbAlignment
    private var mapping:CGAffineTransform {
        LineagePixelMapping.transform(width:configuration.material.result.width,height:configuration.material.result.height,alignment:alignment,viewport:viewport)
    }
    var body: some View {
        ZStack(alignment:.topLeading) {
            Canvas { context,_ in
                let material=configuration.material
                if configuration.masks || configuration.partitions {
                    let image=configuration.partitions ? material.partitions ?? material.heatmap : material.heatmap
                    var pixels=context; pixels.concatenate(mapping)
                    pixels.draw(Image(decorative:image,scale:1),in:CGRect(x:0,y:0,width:material.result.width,height:material.result.height))
                }
                for region in material.regions {
                    let selected=configuration.selected==region.id
                    if configuration.contours || selected {
                        var path=Path()
                        for line in region.contour { path.move(to:line.a.applying(mapping));path.addLine(to:line.b.applying(mapping)) }
                        context.stroke(path,with:.color(selected ? .yellow : .red),lineWidth:selected ? 2 : 1)
                    }
                    if configuration.outlines || selected {
                        let corners=[CGPoint(x:region.bounds.minX,y:region.bounds.minY),CGPoint(x:region.bounds.maxX,y:region.bounds.minY),CGPoint(x:region.bounds.maxX,y:region.bounds.maxY),CGPoint(x:region.bounds.minX,y:region.bounds.maxY)].map { $0.applying(mapping) }
                        var path=Path();path.addLines(corners);path.closeSubpath()
                        context.stroke(path,with:.color(selected ? .yellow : .red),lineWidth:selected ? 3 : 2)
                    }
                }
            }.allowsHitTesting(false)
            if configuration.outlines {
                ForEach(configuration.material.regions) { region in
                    let rect=region.bounds.applying(mapping)
                    Button { configuration.select(region.id) } label: {
                        Rectangle().fill(Color.clear).contentShape(Rectangle())
                            .overlay(alignment:.topLeading) {
                                Text("Region \(region.id+1)").font(.caption2).padding(4).background(.regularMaterial,in:Capsule())
                            }
                    }.buttonStyle(.plain).frame(width:max(12,rect.width),height:max(12,rect.height)).position(x:rect.midX,y:rect.midY)
                        .accessibilityLabel("Inspect SAFIRE suspicious region \(region.id+1)")
                }
            }
        }
    }
}

/// One mapping serves raster masks, contours, hit targets and region bounds.
/// Evidence starts top-left; comparison pixels are bottom-left; screen is top-left.
enum LineagePixelMapping {
    static func transform(width:Int,height:Int,alignment:CrystalOrbAlignment,viewport:CrystalOrbViewport) -> CGAffineTransform {
        let native=alignment.candidateSize
        let source=CGAffineTransform(a:native.width/CGFloat(width),b:0,c:0,d:-native.height/CGFloat(height),tx:0,ty:native.height)
        let screen=CGAffineTransform(a:viewport.scale,b:0,c:0,d:-viewport.scale,
            tx:viewport.size.width/2-viewport.center.x*viewport.scale,ty:viewport.size.height/2+viewport.center.y*viewport.scale)
        return source.concatenating(alignment.transform(candidate:true)).concatenating(screen)
    }
}
