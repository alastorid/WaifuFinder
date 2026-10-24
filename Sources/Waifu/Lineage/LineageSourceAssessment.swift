import Foundation
import CoreGraphics
import CoreImage

/// A reproducible, explicitly tentative chronology hypothesis. The dates are
/// supporting metadata; native pixel correspondence is required independently.
struct LineageSourceEvidence: Codable, Equatable, Sendable {
    let parentID: String
    let childID: String
    let parentDate: Date
    let childDate: Date
    let meanNativeDifference: Double
    let correlation: Double
    let dimensionsChanged: Bool
    let method: String
    var explanation: String {
        let observation = method == "contained crop correspondence"
            ? "The later-dated version matches a contained crop of the earlier-dated version."
            : dimensionsChanged
            ? "Resampling the earlier-dated version reproduces the other version’s native pixel grid."
            : "The earlier-dated version reproduces the other version’s native pixel grid."
        return observation + " The stored Photos dates support this direction, but dates can be changed or inherited. This is a possible derivation, not proof of chronology or editing software. SAFIRE does not predict this order."
    }
}

/// Bounded, serial native-pixel checks after inexpensive family discovery.
/// A smaller image may be the parent of a larger interpolated derivative.
actor LineageSourceAssessor {
    private var cache: [(String,CGImage)] = []
    private let context=CIContext(options:[.cacheIntermediates:false,
        .workingColorSpace:CGColorSpace(name:CGColorSpace.sRGB)!,
        .outputColorSpace:CGColorSpace(name:CGColorSpace.sRGB)!])

    private func original(_ photo:LineagePhoto) async throws -> CGImage {
        if let value=cache.first(where:{$0.0==photo.contentIdentity}) { return value.1 }
        let value=try await PhotosMedia.shared.image(photo.url,seconds:0)
        try Task.checkCancellation()
        cache.append((photo.contentIdentity,value));if cache.count>2 { cache.removeFirst() }
        return value
    }
    func assess(_ relation:LineageRelation,photos:[LineagePhoto]) async throws -> LineageSourceEvidence? {
        guard let a=photos.first(where:{$0.id==relation.a}),let b=photos.first(where:{$0.id==relation.b}),
              let ad=a.photosDate,let bd=b.photosDate,ad != bd else { return nil }
        if let prior=relation.sourceEvidence { return prior }
        let parent=ad<bd ? a : b,child=ad<bd ? b : a
        let parentDate=min(ad,bd),childDate=max(ad,bd)
        let source=try await original(parent),destination=try await original(child)
        return try await assessPixels(parent:source,child:destination,parentID:parent.id,childID:child.id,
                                parentDate:parentDate,childDate:childDate)
    }
    func assessPixels(parent:CGImage,child:CGImage,parentID:String,childID:String,parentDate:Date,childDate:Date) async throws -> LineageSourceEvidence? {
        try Task.checkCancellation()
        guard parentDate<childDate else { return nil }
        let aspectA=Double(parent.width)/Double(parent.height),aspectB=Double(child.width)/Double(child.height)
        let sameAspect=abs(log(aspectA/aspectB))<0.001
        // Never use a thumbnail's extra interpolation to claim native detail.
        // Compare in the child's actual grid with explicit resampling operations.
        let source=CIImage(cgImage:parent),target=CIImage(cgImage:child)
        let rect=CGRect(x:0,y:0,width:child.width,height:child.height)
        let factor=CGFloat(child.width)/CGFloat(parent.width)
        let methods: [(String,CIImage)]
        if parent.width==child.width && parent.height==child.height { methods=[("native pixel equivalence",source)] }
        else {
            let lanczos=source.applyingFilter("CILanczosScaleTransform",parameters:[kCIInputScaleKey:factor,kCIInputAspectRatioKey:1])
            let linear=source.transformed(by:CGAffineTransform(scaleX:factor,y:factor))
            methods=[("Lanczos resampling",lanczos),("linear resampling",linear)]
        }
        for (method,resampled) in methods where sameAspect {
            try Task.checkCancellation()
            let difference=resampled.cropped(to:rect).applyingFilter("CIDifferenceBlendMode",parameters:[kCIInputBackgroundImageKey:target])
            let average=difference.applyingFilter("CIAreaAverage",parameters:[kCIInputExtentKey:CIVector(cgRect:rect)])
            var pixel=[Float](repeating:0,count:4)
            pixel.withUnsafeMutableBytes { context.render(average,toBitmap:$0.baseAddress!,rowBytes:16,bounds:CGRect(x:0,y:0,width:1,height:1),format:.RGBAf,colorSpace:CrystalOrbDifferenceRenderer.color) }
            let error=Double(max(pixel[0],pixel[1],pixel[2]))
            guard error.isFinite,error<=0.008 else { continue }
            let scale=min(1,CGFloat(256)/CGFloat(max(child.width,child.height)))
            let extent=CGRect(x:0,y:0,width:max(1,Int(CGFloat(child.width)*scale)),height:max(1,Int(CGFloat(child.height)*scale)))
            let transform=CGAffineTransform(scaleX:scale,y:scale)
            let a=CrystalOrbAnalysis.bytes(resampled.transformed(by:transform),rect:extent)
            let b=CrystalOrbAnalysis.bytes(target.transformed(by:transform),rect:extent)
            var sumA=0.0,sumB=0.0,aa=0.0,bb=0.0,ab=0.0,n=0.0
            for i in stride(from:0,to:a.count,by:4) {
                let av=Double(a[i])*0.2126+Double(a[i+1])*0.7152+Double(a[i+2])*0.0722
                let bv=Double(b[i])*0.2126+Double(b[i+1])*0.7152+Double(b[i+2])*0.0722
                sumA+=av;sumB+=bv;aa+=av*av;bb+=bv*bv;ab+=av*bv;n+=1
            }
            let va=aa-sumA*sumA/n,vb=bb-sumB*sumB/n
            guard va/n>4,vb/n>4 else { continue } // flat fields cannot establish correspondence
            let correlation=(ab-sumA*sumB/n)/sqrt(va*vb)
            guard correlation.isFinite,correlation>=0.999 else { continue }
            return .init(parentID:parentID,childID:childID,parentDate:parentDate,childDate:childDate,
                meanNativeDifference:error,correlation:correlation,
                dimensionsChanged:parent.width != child.width || parent.height != child.height,method:method)
        }
        // A crop is directional only when the candidate's field of view is
        // contained in the earlier version and native overlap pixels agree.
        let geometry=CrystalOrbAlignment(target:CGSize(width:parent.width,height:parent.height),candidate:CGSize(width:child.width,height:child.height))
        guard let result=try await LineageCropRegistrar.shared.align(target:parent,candidate:child,geometry:geometry),result.correlation>=0.999 else { return nil }
        let registered=CrystalOrbAlignment(target:geometry.targetSize,candidate:geometry.candidateSize,registration:result.registration)
        let childRect=CGRect(origin:.zero,size:registered.candidateSize).applying(registered.transform(candidate:true))
        let overlap=childRect.intersection(registered.target)
        let childArea=childRect.width*childRect.height,parentArea=registered.target.width*registered.target.height
        guard childArea/parentArea>=0.35,childArea/parentArea<=0.92,
              overlap.width*overlap.height/childArea>=0.98 else { return nil }
        let pixelsA=CrystalOrbDifferenceRenderer.source(parent,alignment:registered,candidate:false).0
        let pixelsB=CrystalOrbDifferenceRenderer.source(child,alignment:registered,candidate:true).0
        let valid=overlap.insetBy(dx:1,dy:1)
        guard !valid.isEmpty else { return nil }
        let average=pixelsA.applyingFilter("CIDifferenceBlendMode",parameters:[kCIInputBackgroundImageKey:pixelsB])
            .applyingFilter("CIAreaAverage",parameters:[kCIInputExtentKey:CIVector(cgRect:valid)])
        var pixel=[Float](repeating:0,count:4)
        pixel.withUnsafeMutableBytes { context.render(average,toBitmap:$0.baseAddress!,rowBytes:16,bounds:CGRect(x:0,y:0,width:1,height:1),format:.RGBAf,colorSpace:CrystalOrbDifferenceRenderer.color) }
        let error=Double(max(pixel[0],pixel[1],pixel[2]))
        guard error.isFinite,error<=0.008 else { return nil }
        return .init(parentID:parentID,childID:childID,parentDate:parentDate,childDate:childDate,
            meanNativeDifference:error,correlation:result.correlation,dimensionsChanged:true,method:"contained crop correspondence")
    }
}
