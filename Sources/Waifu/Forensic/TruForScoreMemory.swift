import Foundation

/// Reuses the existing indexed append-only analysis cache engine, in a separate
/// scalar namespace/file. No anomaly, reliability or Noiseprint raster is persisted.
actor TruForScoreMemory {
    static let shared=TruForScoreMemory()
    private let disk:TagResultDiskCache
    init(fileURL:URL?=nil) {
        disk=TagResultDiskCache(fileURL:fileURL ?? FileManager.default.urls(for:.applicationSupportDirectory,in:.userDomainMask)[0].appendingPathComponent("Waifu/TruFor/scores-v1.bin"))
    }
    static func identity(_ photo:TruForPhoto,fingerprint:String)->TagResultCacheIdentity {
        .init(sourcePath:"trufor-score-v1/\(fingerprint)/\(TruForConfiguration.preprocessing)/\(photo.assetKey.identifier)",
            sourceSize:(UInt64(photo.assetKey.pixelWidth)<<32)|UInt64(photo.assetKey.pixelHeight),
            sourceModifiedMilliseconds:photo.assetKey.modifiedMilliseconds,analysisWidth:0,vocabularyFingerprint:0)
    }
    func read(_ photos:[TruForPhoto],fingerprint:String)async throws->[String:Float] {
        let identities=photos.map{Self.identity($0,fingerprint:fingerprint)}
        let values=try await disk.results(for:identities,retentionDays:3650).probabilitiesByIdentity
        var result:[String:Float]=[:]
        for (i,photo) in photos.enumerated() {
            guard let raw=values[identities[i]]?[0],raw.count==8,raw.prefix(4)==Data("TFS1".utf8) else { continue }
            let score=raw.withUnsafeBytes{$0.loadUnaligned(fromByteOffset:4,as:Float.self)}
            if score.isFinite,(0...1).contains(score) { result[photo.contentIdentity]=score }
        }
        return result
    }
    func remember(_ photo:TruForPhoto,score:Float,fingerprint:String)async throws {
        guard score.isFinite,(0...1).contains(score) else { throw ModelInstaller.fail("Invalid TruFor scalar") }
        var data=Data("TFS1".utf8),value=score;withUnsafeBytes(of:&value){data.append(contentsOf:$0)}
        _=try await disk.remember([.init(sampleIndex:0,compressedSparseScores:data)],for:Self.identity(photo,fingerprint:fingerprint),retentionDays:3650)
    }
}
