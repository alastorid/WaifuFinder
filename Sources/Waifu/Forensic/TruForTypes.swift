import Foundation
import ImageIO

struct TruForConfiguration:Sendable {
    static let sourceSHA="ac1d90e329a72e0d66e8665e123a19e94bfae3209c3ef8a4f9ca3b91578c7844"
    static let weightsSHA="3b2c7146303e984c43b51f26e38d093c8b2ebba9a13e41bb029a98aa5b29783c"
    static let runtime="trufor-native-v1"
    // Core ML FP16 is used for the DnCNN body when the compiled companion is
    // present; the CMX encoder and detection heads stay on native FP32.
    private static let nativeExecution="trufor-fp32-v2"
    private static let coreMLExecution="trufor-coreml-dncnn-fp16-v1"
    static var execution:String {
        guard let directory=availableDirectory,
              FileManager.default.fileExists(atPath:directory.appendingPathComponent("CoreMLNoiseprintFP16.mlmodelc").path) else {
            return nativeExecution
        }
        return coreMLExecution
    }
    static var scoreFingerprint:String { LineageIdentity.hash(weightsSHA+"|"+execution) }
    static let preprocessing="trufor-native-rgb-div256-v1"
    let directory:URL
    var fingerprint:String { Self.weightsSHA + "|" + Self.execution }
    static var pointerURL:URL { SeedVR2Configuration.configurationURL.deletingLastPathComponent().appendingPathComponent("trufor-model.txt") }
    static var executable:URL? { Bundle.main.url(forResource:"waifu-trufor",withExtension:nil,subdirectory:"Runtime") }
    static var availableDirectory:URL? {
        guard let data=try? Data(contentsOf:pointerURL),let text=String(data:data,encoding:.utf8)?.trimmingCharacters(in:.whitespacesAndNewlines),!text.isEmpty else { return nil }
        return URL(fileURLWithPath:text)
    }
    static func load()->Self? {
        guard WaifuDistribution.isNonCommercialOnly,let directory=availableDirectory,let executable,FileManager.default.isExecutableFile(atPath:executable.path) else { return nil }
        return try? validate(directory)
    }
    static func validate(_ directory:URL)throws->Self {
        let meta=try JSONSerialization.jsonObject(with:Data(contentsOf:directory.appendingPathComponent("architecture.json"))) as? [String:Any]
        let validation=try JSONSerialization.jsonObject(with:Data(contentsOf:directory.appendingPathComponent("validation.json"))) as? [String:Any]
        let hashes=try ModelInstaller.hashes(directory)
        guard meta?["sourceSHA256"] as? String == sourceSHA,meta?["runtimeRevision"] as? String == runtime,
              meta?["preprocessingVersion"] as? String == preprocessing,hashes["model.safetensors"]==weightsSHA,
              validation?["status"] as? String == "passed",validation?["sourceSHA256"] as? String == sourceSHA,
              validation?["scoreParityPassed"] as? Bool == true,validation?["spatialParityPassed"] as? Bool == true else { throw ModelInstaller.fail("This TruFor folder has not passed official score and spatial validation.") }
        return .init(directory:directory)
    }
}
struct TruForPhoto:Sendable {
    let url:URL
    let assetKey:PhotosMedia.AssetCacheKey
    var contentIdentity:String { LineageIdentity.hash("\(assetKey.identifier)|\(assetKey.modifiedMilliseconds)|\(assetKey.pixelWidth)x\(assetKey.pixelHeight)") }
}
struct TruForSpatial:Sendable {
    let width:Int,height:Int
    let score:Float
    let anomaly:[Float]
    let reliability:[Float]
    let noiseprint:[Float]
    var byteCount:Int { width*height*12+20 }
    func validate()throws {
        guard width>=32,height>=32,width<=32768,height<=32768,width*height<=24_000_000,score.isFinite,(0...1).contains(score),
              anomaly.count==width*height,reliability.count==anomaly.count,noiseprint.count==anomaly.count,
              anomaly.allSatisfy({$0.isFinite && (0...1).contains($0)}),reliability.allSatisfy({$0.isFinite && (0...1).contains($0)}),noiseprint.allSatisfy(\.isFinite) else { throw ModelInstaller.fail("Invalid TruFor evidence") }
    }
    static func decode(_ data:Data)throws->Self {
        guard data.count>=20,data.prefix(4)==Data("TFM1".utf8) else { throw ModelInstaller.fail("Invalid TruFor evidence header") }
        func word(_ p:Int)->UInt32 { data.withUnsafeBytes{UInt32(littleEndian:$0.loadUnaligned(fromByteOffset:p,as:UInt32.self))} }
        let w=Int(word(4)),h=Int(word(8))
        guard w>=32,h>=32,w<=32768,h<=32768,w*h<=24_000_000,word(12)==3,data.count==20+w*h*12 else { throw ModelInstaller.fail("Invalid TruFor evidence geometry") }
        let score=data.withUnsafeBytes{$0.loadUnaligned(fromByteOffset:16,as:Float.self)}
        func array(_ i:Int)->[Float] {
            var values=[Float](repeating:0,count:w*h);let offset=20+i*w*h*4
            values.withUnsafeMutableBytes { out in data.withUnsafeBytes { input in out.copyMemory(from:UnsafeRawBufferPointer(rebasing:input[offset..<(offset+w*h*4)])) } };return values
        }
        let value=Self(width:w,height:h,score:score,anomaly:array(0),reliability:array(1),noiseprint:array(2));try value.validate();return value
    }
}
enum TruForJobResult:Sendable { case scores([Float]),evidence(TruForSpatial) }
