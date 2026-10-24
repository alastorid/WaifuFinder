import Foundation

/// SFM1: four ASCII bytes, UInt32 LE width/height/partition flag, followed by
/// contiguous IEEE Float32 LE probabilities and optional UInt16 LE labels.
/// Supported macOS architectures are little-endian. Bulk copies avoid per-pixel
/// JSON number parsing/formatting in workers, remote responses and disk caches.
enum SafireRasterCodec {
    static func encode(width:Int,height:Int,probabilities:[Float],partitions:[UInt16]?) -> Data {
        var data=Data("SFM1".utf8)
        for value in [UInt32(width),UInt32(height),partitions == nil ? UInt32(0) : UInt32(1)] {
            var value=value.littleEndian;withUnsafeBytes(of:&value) { data.append(contentsOf:$0) }
        }
        probabilities.withUnsafeBytes { data.append(contentsOf:$0) }
        partitions?.withUnsafeBytes { data.append(contentsOf:$0) }
        return data
    }
    static func decode(_ data:Data) throws -> (width:Int,height:Int,probabilities:[Float],partitions:[UInt16]?) {
        guard data.count>=16,data.prefix(4)==Data("SFM1".utf8) else { throw PhotosMedia.failure("Invalid SAFIRE raster header.") }
        func word(_ offset:Int)->UInt32 { data.withUnsafeBytes { UInt32(littleEndian:$0.loadUnaligned(fromByteOffset:offset,as:UInt32.self)) } }
        let w=Int(word(4)),h=Int(word(8)),flag=word(12)
        guard (1...1024).contains(w),(1...1024).contains(h),flag<=1 else { throw PhotosMedia.failure("Invalid SAFIRE raster dimensions.") }
        let count=w*h,probabilityBytes=count*4
        guard data.count==16+probabilityBytes+(flag==1 ? count*2 : 0) else { throw PhotosMedia.failure("Truncated SAFIRE raster.") }
        var probabilities=[Float](repeating:0,count:count)
        probabilities.withUnsafeMutableBytes { destination in data.withUnsafeBytes { source in destination.copyMemory(from:UnsafeRawBufferPointer(rebasing:source[16..<(16+probabilityBytes)])) } }
        var partitions:[UInt16]?
        if flag==1 {
            var values=[UInt16](repeating:0,count:count)
            values.withUnsafeMutableBytes { destination in data.withUnsafeBytes { source in destination.copyMemory(from:UnsafeRawBufferPointer(rebasing:source[(16+probabilityBytes)...])) } }
            partitions=values
        }
        return (w,h,probabilities,partitions)
    }
}

extension SafireSpatialResult {
    private struct Metadata:Codable {
        let assetID:String,contentIdentity:String,modelFingerprint:String,runtimeRevision:String,preprocessingVersion:String
    }
    func encodedSpatialData() throws -> Data {
        try validate()
        let header=try JSONEncoder().encode(Metadata(assetID:assetID,contentIdentity:contentIdentity,modelFingerprint:modelFingerprint,runtimeRevision:runtimeRevision,preprocessingVersion:preprocessingVersion))
        var size=UInt32(header.count).littleEndian
        var data=Data("SFE1".utf8)
        withUnsafeBytes(of:&size) { data.append(contentsOf:$0) };data.append(header)
        data.append(SafireRasterCodec.encode(width:width,height:height,probabilities:probabilities,partitions:partitions))
        return data
    }
    static func decodeSpatialData(_ data:Data) throws -> Self {
        // Read older JSON cache/remote results during upgrades as well.
        if data.first==123 { let result=try JSONDecoder().decode(Self.self,from:data);try result.validate();return result }
        guard data.count>=8,data.prefix(4)==Data("SFE1".utf8) else { throw PhotosMedia.failure("Invalid SAFIRE evidence header.") }
        let size=Int(data.withUnsafeBytes { UInt32(littleEndian:$0.loadUnaligned(fromByteOffset:4,as:UInt32.self)) })
        guard size>0,size<=8192,data.count>=8+size+16 else { throw PhotosMedia.failure("Invalid SAFIRE evidence metadata.") }
        let meta=try JSONDecoder().decode(Metadata.self,from:data.subdata(in:8..<(8+size)))
        let raster=try SafireRasterCodec.decode(data.subdata(in:(8+size)..<data.count))
        let result=Self(assetID:meta.assetID,contentIdentity:meta.contentIdentity,modelFingerprint:meta.modelFingerprint,runtimeRevision:meta.runtimeRevision,preprocessingVersion:meta.preprocessingVersion,width:raster.width,height:raster.height,probabilities:raster.probabilities,partitions:raster.partitions)
        try result.validate();return result
    }
}
