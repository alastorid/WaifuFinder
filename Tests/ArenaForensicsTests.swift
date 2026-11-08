import CoreGraphics
import CoreImage
import Foundation
import ImageIO
import UniformTypeIdentifiers

private func arenaForensicFixture(_ url: URL, variant: Int = 0, shift: Int = 0) throws {
    let width = 256, height = 256
    var pixels = [UInt8](repeating: 255, count: width*height*4)
    for y in 0..<height {
        for x in 0..<width {
            let sourceX = x-shift
            let k = (y*width+x)*4
            for c in 0..<3 {
                let value = (sourceX*sourceX/19+y*y/23+sourceX*y/41+c*23+(sourceX/13+y/17)%2*70)%180+30
                pixels[k+c] = UInt8(max(0,min(255,value)))
            }
            if variant == 1 && (64..<96).contains(x) && (64..<96).contains(y) { pixels[k] = 255; pixels[k+1] = 0; pixels[k+2] = 0 }
            if variant == 2 { pixels[k] = UInt8((x*19+y*31)%256); pixels[k+1] = UInt8((x*47+y*13)%256); pixels[k+2] = 0 }
        }
    }
    let data = Data(pixels)
    let provider = CGDataProvider(data: data as CFData)!
    let image = CGImage(width: width,height: height,bitsPerComponent: 8,bitsPerPixel: 32,bytesPerRow: width*4,
        space: CGColorSpace(name: CGColorSpace.sRGB)!,bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),provider: provider,decode: nil,shouldInterpolate: false,intent: .defaultIntent)!
    let destination = CGImageDestinationCreateWithURL(url as CFURL,UTType.png.identifier as CFString,1,nil)!
    CGImageDestinationAddImage(destination,image,nil)
    guard CGImageDestinationFinalize(destination) else { throw CocoaError(.fileWriteUnknown) }
}
func registerArenaForensicsTests() {
    Registry.shared.add(suite: "PhotoArena Forensics", TestCase(name: "native tile differences localize patch and retain uncertainty", knownBug: nil) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("arena-forensic-\(UUID())")
        try FileManager.default.createDirectory(at: root,withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let a = root.appendingPathComponent("a.png"), b = root.appendingPathComponent("b.png")
        try arenaForensicFixture(a); try arenaForensicFixture(b,variant: 1)
        let analysis = try ArenaForensics.analyzeFiles([ArenaSource(id: "a",url: a,seconds: 0),ArenaSource(id: "b",url: b,seconds: 0)],tileSize: 32)
        checkEqual(analysis.width,256,"native dimensions retained")
        checkEqual(analysis.sources.filter(\.accepted).count,2,"same scene with small edit aligns")
        let changed = analysis.regions.filter { $0.measurements.contains(where: { $0.difference > 0.03 }) }
        checkEqual(changed.count,1,"one modified analysis tile localized")
        checkEqual(changed.first?.rect,ArenaPixelRect(CGRect(x:64,y:160,width:32,height:32)),"top-down fixture pixels map to bottom-left original coordinates")
        let mask=try ArenaForensics.differenceMask(analysis.sources[0],other:analysis.sources[1],rect:CGRect(x:0,y:0,width:256,height:256))
        let maskBytes=Array((mask.dataProvider!.data! as Data))
        let detected=stride(from:0,to:maskBytes.count,by:4).filter { maskBytes[$0] > 127 }.count
        checkEqual(detected,1024,"exact binary difference mask matches 32×32 ground-truth patch (IoU 1)")
        check(analysis.regions.allSatisfy(\.unresolved),"appearance cannot prove provenance")
        checkEqual(analysis.regions.count,64,"all native pixels covered")
        let crop = try ArenaForensics.crop(analysis.sources[0],rect: CGRect(x: 32,y: 32,width: 64,height: 64))
        checkEqual(crop.width,64,"crop has native reference sampling")
        let decoded = try JSONDecoder().decode(ArenaAnalysis.self,from: JSONEncoder().encode(analysis))
        checkEqual(decoded.regions.map(\.rect),analysis.regions.map(\.rect),"coordinates survive JSON export")
    })
    Registry.shared.add(suite: "PhotoArena Forensics", TestCase(name: "unrelated scene is rejected instead of forced into reconstruction", knownBug: nil) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("arena-unrelated-\(UUID())")
        try FileManager.default.createDirectory(at: root,withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let a=root.appendingPathComponent("a.png"), b=root.appendingPathComponent("b.png")
        try arenaForensicFixture(a); try arenaForensicFixture(b,variant: 2)
        let result=try ArenaForensics.analyzeFiles([ArenaSource(id: "a",url: a,seconds: 0),ArenaSource(id: "b",url: b,seconds: 0)])
        checkEqual(result.sources.filter(\.accepted).count,1,"unrelated photograph rejected")
        check(result.regions.allSatisfy(\.unresolved),"missing trustworthy replacement remains unresolved")
    })
    Registry.shared.add(suite: "PhotoArena Forensics", TestCase(name: "translated source registers in native pixel coordinates", knownBug: nil) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("arena-shift-\(UUID())")
        try FileManager.default.createDirectory(at: root,withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let a=root.appendingPathComponent("a.png"), b=root.appendingPathComponent("b.png")
        try arenaForensicFixture(a); try arenaForensicFixture(b,shift: 7)
        let result=try ArenaForensics.analyzeFiles([ArenaSource(id: "a",url: a,seconds: 0),ArenaSource(id: "b",url: b,seconds: 0)])
        checkEqual(result.sources.filter(\.accepted).count,2,"translated scene remains usable")
        let moving=result.sources.first { $0.id != result.referenceID }!
        checkClose(abs(moving.transform.point(CGPoint(x: 128,y: 128)).x-128),7,tolerance: 0.8,"translation maps native pixels")
        let mask=try ArenaForensics.validityMask(moving,rect: CGRect(x: 0,y: 0,width: 256,height: 256))
        checkEqual(mask.width,256,"validity mask reference geometry")
    })
    Registry.shared.add(suite: "PhotoArena Forensics", TestCase(name: "native crop chooses original sampling instead of enlargement", knownBug: nil) {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent("arena-crop-\(UUID())")
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer { try? FileManager.default.removeItem(at:root) }
        let a=root.appendingPathComponent("a.png"), b=root.appendingPathComponent("crop.png")
        try arenaForensicFixture(a)
        let input=CIImage(contentsOf:a)!
        let cropped=input.cropped(to:CGRect(x:48,y:32,width:160,height:192)).transformed(by:.init(translationX:-48,y:-32))
        try CIContext().writePNGRepresentation(of:cropped,to:b,format:.RGBA8,colorSpace:CGColorSpace(name:CGColorSpace.sRGB)!)
        let analysis=try ArenaForensics.analyzeFiles([ArenaSource(id:"a",url:a,seconds:0),ArenaSource(id:"crop",url:b,seconds:0)])
        let source=analysis.sources.first { $0.id == "crop" }!
        check(source.accepted,"native crop registers")
        checkClose(source.transform.a,1,tolerance:0.001,"native crop not arbitrarily enlarged")
        checkClose(source.transform.tx,48,tolerance:0.1,"crop x maps to original pixels")
        checkClose(source.transform.ty,32,tolerance:0.1,"crop y maps to original pixels")
    })
    Registry.shared.add(suite: "PhotoArena Forensics", TestCase(name: "projective coordinates preserve source native mapping", knownBug: nil) {
        var transform=ArenaTransform(CGAffineTransform(a: 1.1,b: 0.02,c: -0.03,d: 1.1,tx: 12,ty: 9))
        transform.perspectiveX=0.0001
        let point=transform.point(CGPoint(x: 100,y: 50))
        checkClose(point.x,(110-1.5+12)/1.01,tolerance: 0.00001,"homogeneous source mapping x")
        checkClose(point.y,(2+55+9)/1.01,tolerance: 0.00001,"homogeneous source mapping y")
    })
}
