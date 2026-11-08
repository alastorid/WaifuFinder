// Build with PhotoArena.swift, ArenaAnalysisTypes.swift and ArenaForensics.swift.
// Real-image geometric registration tests using Apple-native fixture generation.
import CoreImage
import Foundation

@main struct ArenaRegistrationBenchmark {
    static func main() throws {
        let args=CommandLine.arguments
        let input=URL(fileURLWithPath: args.count > 1 ? args[1] : "Tests/Fixtures/RealESRGAN/nasa-collins-896x1134.jpg")
        let report=URL(fileURLWithPath: args.count > 2 ? args[2] : "docs/photoarena-registration-benchmark.json")
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent("arena-registration-\(UUID())")
        try FileManager.default.createDirectory(at: directory,withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let context=CIContext(options: [.cacheIntermediates:false])
        guard let image=CIImage(contentsOf: input,options:[.applyOrientationProperty:true]) else { throw CocoaError(.fileReadCorruptFile) }
        let r=image.extent, w=r.width, h=r.height
        let rotation=CGAffineTransform(translationX:-w/2,y:-h/2).concatenating(.init(rotationAngle:2 * .pi/180)).concatenating(.init(translationX:w/2,y:h/2))
        let cropRect=CGRect(x:70,y:95,width:w-140,height:h-190)
        let offCenterCrop=CGRect(x:35,y:127,width:w-140,height:h-190)
        let variants:[(String,CIImage)]=[
            ("rotation-2-degrees",image.transformed(by:rotation).cropped(to:r)),
            ("crop-inset",image.cropped(to:cropRect).transformed(by:.init(translationX:-cropRect.minX,y:-cropRect.minY))),
            ("crop-off-center",image.cropped(to:offCenterCrop).transformed(by:.init(translationX:-offCenterCrop.minX,y:-offCenterCrop.minY))),
            ("mild-perspective",image.applyingFilter("CIPerspectiveTransform",parameters:["inputBottomLeft":CIVector(x:12,y:8),"inputBottomRight":CIVector(x:w-8,y:0),"inputTopLeft":CIVector(x:0,y:h-13),"inputTopRight":CIVector(x:w-17,y:h-3)]).cropped(to:r))]
        var rows:[[String:Any]]=[]
        for (name,variant) in variants {
            let url=directory.appendingPathComponent(name+".png")
            try context.writePNGRepresentation(of:variant,to:url,format:.RGBA8,colorSpace:CGColorSpace(name:CGColorSpace.sRGB)!)
            let start=Date()
            let result=try ArenaForensics.analyzeFiles([ArenaSource(id:"reference",url:input,seconds:0),ArenaSource(id:name,url:url,seconds:0)])
            let candidate=result.sources.first { $0.id != result.referenceID }!
            let coverage=result.regions.flatMap(\.measurements).filter { $0.sourceID == candidate.id }.reduce(0.0) { $0+$1.validFraction }/Double(result.regions.count)
            rows.append(["case":name,"accepted":candidate.accepted,"correlation":candidate.alignmentConfidence,"meanTileCoverage":coverage,"analysisSeconds":Date().timeIntervalSince(start),"transform":try JSONSerialization.jsonObject(with:JSONEncoder().encode(candidate.transform)),"issues":candidate.issues])
        }
        let object:[String:Any]=["fixture":input.lastPathComponent,"generator":"Apple Core Image; no learned modification","cases":rows,"limitations":["Single real photograph; global correlation is not proof of local pixel alignment or robustness across scenes.","Perspective mapping is estimated; no feature-level ground-truth reprojection score is claimed."]]
        try JSONSerialization.data(withJSONObject:object,options:[.prettyPrinted,.sortedKeys]).write(to:report,options:.atomic)
        for row in rows { print(row) }
    }
}
