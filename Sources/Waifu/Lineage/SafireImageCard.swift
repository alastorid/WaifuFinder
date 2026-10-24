import SwiftUI
import CoreImage

struct SafireCardMaterial:Sendable {
    let highlighted:CGImage
    let mask:CGImage
}

/// Bounded-size previews. Threshold bytes are prepared off the main actor;
/// scaling, mask compositing and the bright boundary run through Core Image.
actor SafireCardRenderer {
    static let shared=SafireCardRenderer()
    private let context=CIContext(options:[.cacheIntermediates:false])
    func make(image:CGImage,result:SafireSpatialResult,maximumSide:Int=560) throws -> SafireCardMaterial {
        try Task.checkCancellation()
        let binary=result.probabilities.map { $0>=0.5 ? UInt8(255) : UInt8(0) }
        guard let provider=CGDataProvider(data:Data(binary) as CFData),let gray=CGImage(width:result.width,height:result.height,bitsPerComponent:8,bitsPerPixel:8,bytesPerRow:result.width,space:CGColorSpaceCreateDeviceGray(),bitmapInfo:CGBitmapInfo(rawValue:CGImageAlphaInfo.none.rawValue),provider:provider,decode:nil,shouldInterpolate:false,intent:.defaultIntent) else { throw PhotosMedia.failure("Unable to render SAFIRE mask") }
        let factor=min(1,CGFloat(maximumSide)/CGFloat(max(image.width,image.height)))
        let width=max(1,Int(CGFloat(image.width)*factor)),height=max(1,Int(CGFloat(image.height)*factor))
        let rect=CGRect(x:0,y:0,width:width,height:height)
        let photo=CIImage(cgImage:image).transformed(by:CGAffineTransform(scaleX:CGFloat(width)/CGFloat(image.width),y:CGFloat(height)/CGFloat(image.height))).cropped(to:rect)
        let mask=CIImage(cgImage:gray).transformed(by:CGAffineTransform(scaleX:CGFloat(width)/CGFloat(result.width),y:CGFloat(height)/CGFloat(result.height))).cropped(to:rect)
        let red=CIImage(color:CIColor(red:1,green:0.08,blue:0,alpha:0.68)).cropped(to:rect).composited(over:photo)
        let highlighted=red.applyingFilter("CIBlendWithMask",parameters:[kCIInputBackgroundImageKey:photo,kCIInputMaskImageKey:mask])
        let boundary=mask.applyingFilter("CIMorphologyGradient",parameters:[kCIInputRadiusKey:1.5])
        let outlined=CIImage(color:CIColor(red:1,green:0.95,blue:0.1)).cropped(to:rect)
            .applyingFilter("CIBlendWithMask",parameters:[kCIInputBackgroundImageKey:highlighted,kCIInputMaskImageKey:boundary])
        guard let overlay=context.createCGImage(outlined,from:rect,format:.RGBA8,colorSpace:CrystalOrbDifferenceRenderer.color),let raster=context.createCGImage(mask,from:rect,format:.RGBA8,colorSpace:CrystalOrbDifferenceRenderer.color) else { throw PhotosMedia.failure("Unable to display SAFIRE evidence") }
        return .init(highlighted:overlay,mask:raster)
    }
}

struct SafireImageCard:View {
    let photo:LineagePhoto
    let summary:LineageSafireSummary?
    let first:Bool
    let complete:Bool
    let selected:Bool
    let select:()->Void
    @State private var material:SafireCardMaterial?
    @State private var error:String?
    private var taskID:String { photo.contentIdentity+"|"+(summary?.modelFingerprint ?? "pending") }
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            HStack {
                Text(photo.name).font(.headline).lineLimit(1)
                if first,summary != nil {
                    Text(complete ? "Original candidate" : "Least masked so far").font(.caption.weight(.semibold)).padding(.horizontal,8).padding(.vertical,4).background(.green.opacity(0.12),in:Capsule())
                }
                Spacer()
                if let summary { Text(summary.ratio,format:.percent.precision(.fractionLength(2))).monospacedDigit().font(.headline) }
            }
            if let material {
                HStack(alignment:.top,spacing:12) {
                    picture(material.highlighted,title:"Highlighted area")
                    picture(material.mask,title:"Forged mask · white = flagged")
                }
            } else if let error {
                Text(error).font(.callout).foregroundStyle(.secondary).frame(height:100)
            } else if summary != nil {
                ProgressView("Loading cached mask…").controlSize(.small).frame(height:100)
            } else {
                Text("Waiting for SAFIRE findings…").foregroundStyle(.secondary).frame(height:90)
            }
            if let summary {
                Text(summary.flaggedPixels==0 ? "SAFIRE’s mask does not flag any area at the current threshold." : "The red overlay and bright boundary show the area SAFIRE flags; the separate mask preserves the spatial shape.")
                    .font(.callout).foregroundStyle(.secondary)
                if first {
                    Text("A source candidate because it has the lowest SAFIRE mask-area ratio\(complete ? " in this selection" : " among results received so far"). This ranking rule suggests a source candidate; it does not establish capture or edit order.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Button(selected ? "Evidence selected" : "Read region evidence",action:select).controlSize(.small)
            }
        }.padding(16).background(.background,in:RoundedRectangle(cornerRadius:12))
            .overlay(RoundedRectangle(cornerRadius:12).stroke(selected ? Color.accentColor : .secondary.opacity(0.18),lineWidth:selected ? 2 : 1))
            .task(id:taskID) {
                guard let summary else { return }
                do {
                    guard let result=await SafireEvidenceCache.shared.read(photo:photo,fingerprint:summary.modelFingerprint) else { error="This mask is no longer cached. Refresh findings to regenerate it.";return }
                    let image=try await PhotosMedia.shared.analysisImage(photo.url,seconds:0)
                    let value=try await SafireCardRenderer.shared.make(image:image,result:result)
                    guard !Task.isCancelled else { return };material=value
                } catch { if !Task.isCancelled { self.error=error.localizedDescription } }
            }
            .onDisappear { material=nil }
    }
    private func picture(_ image:CGImage,title:String)->some View {
        VStack(alignment:.leading,spacing:6) {
            Text(title).font(.caption.weight(.medium))
            Image(decorative:image,scale:1).resizable().interpolation(.none).scaledToFit().frame(maxWidth:.infinity,maxHeight:280)
                .background(.black.opacity(0.06)).clipShape(RoundedRectangle(cornerRadius:7))
        }.frame(maxWidth:.infinity)
    }
}

enum LineageMaskRanking {
    static func ordered(_ photos:[LineagePhoto],findings:[String:LineageSafireSummary]) -> [LineagePhoto] {
        photos.enumerated().sorted { a,b in
            let left=findings[a.element.id].flatMap{$0.contentIdentity==a.element.contentIdentity ? $0 : nil}
            let right=findings[b.element.id].flatMap{$0.contentIdentity==b.element.contentIdentity ? $0 : nil}
            switch (left,right) {
            case let (.some(l),.some(r)):return l.ratio==r.ratio ? a.offset<b.offset : l.ratio<r.ratio
            case (.some,.none):return true
            case (.none,.some):return false
            case (.none,.none):return a.offset<b.offset
            }
        }.map(\.element)
    }
}
