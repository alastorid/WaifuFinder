import Foundation
import CoreGraphics
import CoreImage

/// Classical scale/translation correspondence, used only after indexed retrieval.
/// Scores shared textured pixels and checks their overlap; no semantic model or
/// chronology assumption participates. Both crop directions are searched.
actor LineageCropRegistrar {
    static let shared=LineageCropRegistrar()
    private var cached:[String:CrystalOrbRegistrationResult]=[:]
    private var order:[String]=[]
    private struct Plane {
        let pixels:[Double]
        let side:Int
        let target:CGRect
        let candidate:CGRect
        let other:[Double]
    }
    private struct Candidate {
        let x:Double,y:Double,scale:Double,quality:Double
    }
    private func plane(_ a:CGImage,_ b:CGImage,geometry:CrystalOrbAlignment,side:Int) -> Plane {
        let sx=CGFloat(side)/geometry.size.width,sy=CGFloat(side)/geometry.size.height
        let transform=CGAffineTransform(scaleX:sx,y:sy)
        let rect=CGRect(x:0,y:0,width:side,height:side)
        func gray(_ image:CIImage) -> [Double] {
            let bytes=CrystalOrbAnalysis.bytes(image.transformed(by:transform),rect:rect)
            return stride(from:0,to:bytes.count,by:4).map{Double(bytes[$0])*0.2126+Double(bytes[$0+1])*0.7152+Double(bytes[$0+2])*0.0722}
        }
        return .init(pixels:gray(CrystalOrbDifferenceRenderer.source(a,alignment:geometry,candidate:false).0),side:side,
                     target:geometry.target.applying(transform),candidate:geometry.candidate.applying(transform),
                     other:gray(CrystalOrbDifferenceRenderer.source(b,alignment:geometry,candidate:true).0))
    }
    private func score(_ plane:Plane,x:Double,y:Double,scale:Double,step:Int) -> Double {
        let side=plane.side,rect=plane.candidate
        let centerX=Double(rect.midX),centerY=Double(rect.midY)
        var a=0.0,b=0.0,aa=0.0,bb=0.0,ab=0.0,n=0.0
        for py in stride(from:0,to:side,by:step) { for px in stride(from:0,to:side,by:step) {
            let ax=Double(px)+0.5,ay=Double(py)+0.5
            guard plane.target.contains(CGPoint(x:ax,y:ay)) else { continue }
            let bx=(ax-centerX-x)/scale+centerX,by=(ay-centerY-y)/scale+centerY
            guard rect.contains(CGPoint(x:bx,y:by)) else { continue }
            let u=max(0,min(Double(side-1),bx-0.5)),v=max(0,min(Double(side-1),by-0.5))
            let x0=Int(u),y0=Int(v),x1=min(side-1,x0+1),y1=min(side-1,y0+1)
            let fx=u-Double(x0),fy=v-Double(y0)
            let av=plane.pixels[py*side+px]
            let top=plane.other[y0*side+x0]*(1-fx)+plane.other[y0*side+x1]*fx
            let bottom=plane.other[y1*side+x0]*(1-fx)+plane.other[y1*side+x1]*fx
            let bv=top*(1-fy)+bottom*fy
            a+=av;b+=bv;aa+=av*av;bb+=bv*bv;ab+=av*bv;n+=1
        } }
        let areaA=Double(plane.target.width*plane.target.height),areaB=Double(rect.width*rect.height)*scale*scale
        let overlap=n*Double(step*step)
        guard n>=100,overlap/min(areaA,areaB)>=0.87,overlap/max(areaA,areaB)>=0.35 else { return -1 }
        let va=aa-a*a/n,vb=bb-b*b/n
        guard va/n>4,vb/n>4 else { return -1 }
        return (ab-a*b/n)/sqrt(va*vb)
    }
    func align(target:CGImage,candidate:CGImage,geometry:CrystalOrbAlignment,identity:String?=nil) throws -> CrystalOrbRegistrationResult? {
        try Task.checkCancellation()
        let key=identity.map{$0+"|\(geometry.targetSize)|\(geometry.candidateSize)|\(target.width)|\(candidate.width)"}
        if let key,let cached=cached[key] { return cached }
        let base=CrystalOrbAlignment(target:geometry.targetSize,candidate:geometry.candidateSize)
        let coarse=plane(target,candidate,geometry:base,side:64)
        var best=Candidate(x:0,y:0,scale:1,quality:-1)
        for scale in stride(from:0.5,through:1.75,by:0.0625) {
            try Task.checkCancellation()
            for y in stride(from:-24.0,through:24,by:4) { for x in stride(from:-24.0,through:24,by:4) {
                let quality=score(coarse,x:x,y:y,scale:scale,step:2)
                if quality>best.quality { best = .init(x:x,y:y,scale:scale,quality:quality) }
            } }
        }
        guard best.quality>=0.60 else { return nil }
        for side in [128,256] {
            let plane=plane(target,candidate,geometry:base,side:side)
            let ratio=Double(side)/Double(side==128 ? 64 : 128)
            let initial=Candidate(x:best.x*ratio,y:best.y*ratio,scale:best.scale,quality:best.quality)
            best = .init(x:initial.x,y:initial.y,scale:initial.scale,quality:-1)
            let scaleStep=side==128 ? 0.01 : 0.0025
            let offsetStep=side==128 ? 1.0 : 0.5
            for ds in -4...4 {
                try Task.checkCancellation()
                let scale=initial.scale+Double(ds)*scaleStep
                for dy in -4...4 { for dx in -4...4 {
                    let x=initial.x+Double(dx)*offsetStep,y=initial.y+Double(dy)*offsetStep
                    let quality=score(plane,x:x,y:y,scale:scale,step:side==128 ? 3 : 5)
                    if quality>best.quality { best = .init(x:x,y:y,scale:scale,quality:quality) }
                } }
            }
        }
        guard best.quality>=0.985 else { return nil }
        let result=CrystalOrbRegistrationResult(registration:.init(x:best.x*geometry.size.width/256,y:best.y*geometry.size.height/256,scale:best.scale),correlation:best.quality)
        if let key {
            cached[key]=result;order.append(key)
            while order.count>128 { cached[order.removeFirst()]=nil }
        }
        return result
    }
}
