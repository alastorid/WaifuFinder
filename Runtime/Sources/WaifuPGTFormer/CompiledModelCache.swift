import Foundation
import CryptoKit
import CoreML

func compiledModel(_ root: URL, _ stage: String) throws -> URL {
    guard let metadata = try JSONSerialization.jsonObject(with:Data(contentsOf:root.appendingPathComponent("manifest.json"))) as? [String:Any],
          let hashes = metadata["splitPackageSHA256"] as? [String:String] else {
        throw NSError(domain:"PGTFormer",code:10,userInfo:[NSLocalizedDescriptionKey:"Missing validated model fingerprints"])
    }
    let identity = "pgt-compiled-v1\n" + ProcessInfo.processInfo.operatingSystemVersionString + "\n" + hashes.keys.filter { $0.hasPrefix(stage+".mlpackage/") }.sorted().map { $0+":"+hashes[$0]! }.joined(separator:"\n")
    let digest = SHA256.hash(data:Data(identity.utf8)).map { String(format:"%02x",$0) }.joined()
    let directory = FileManager.default.urls(for:.cachesDirectory,in:.userDomainMask)[0]
        .appendingPathComponent("Waifu/PGTFormerCompiled").appendingPathComponent(stage)
    let destination = directory.appendingPathComponent(digest+".mlmodelc")
    let fm = FileManager.default
    if fm.fileExists(atPath:destination.appendingPathComponent("coremldata.bin").path) { return destination }
    let temporary = try MLModel.compileModel(at:root.appendingPathComponent(stage+".mlpackage"))
    defer { try? fm.removeItem(at:temporary) }
    try fm.createDirectory(at:directory,withIntermediateDirectories:true)
    if fm.fileExists(atPath:destination.path) { try fm.removeItem(at:destination) }
    try fm.moveItem(at:temporary,to:destination)
    // Only retire this stage's compiled representations; never source weights.
    for old in try fm.contentsOfDirectory(at:directory,includingPropertiesForKeys:nil) where old.lastPathComponent != destination.lastPathComponent && old.pathExtension == "mlmodelc" {
        try fm.removeItem(at:old)
    }
    return destination
}
