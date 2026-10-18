import Compression
import Foundation

/// Lossless, uncompressed Float32 storage for live results. Compression belongs
/// to the disk cache, so filtering and publishing never need a zlib round trip.
/// WDT1 remains readable for previously serialized frame metadata.
enum TagProbabilityCodec {
    private static let magic = Data([0x57, 0x44, 0x54, 0x32]) // WDT2
    private static let legacyMagic = Data([0x57, 0x44, 0x54, 0x31]) // WDT1
    private static let byteCount = TagVocabulary.headCount * MemoryLayout<UInt32>.size

    static func encode(_ probabilities: [Float]) -> Data? {
        guard probabilities.count == TagVocabulary.headCount,
              probabilities.allSatisfy({ $0.isFinite && (0...1).contains($0) }) else { return nil }
        let words = probabilities.map { $0.bitPattern.littleEndian }
        var data = magic
        words.withUnsafeBytes { data.append(contentsOf: $0) }
        return data
    }

    /// Reads only the requested heads from live Float32 storage. Layout and the
    /// requested values are validated; `decode` remains the whole-vector validator.
    /// Legacy compressed metadata uses the existing full decoder for compatibility.
    static func values(in data: Data, at indices: [Int]) -> [Float]? {
        guard indices.allSatisfy({ $0 >= 0 && $0 < TagVocabulary.headCount }) else { return nil }
        guard data.prefix(magic.count) == magic else {
            guard let probabilities = decode(data) else { return nil }
            return indices.map { probabilities[$0] }
        }
        guard data.count == magic.count + byteCount else { return nil }
        return data.withUnsafeBytes { buffer in
            var result: [Float] = []
            result.reserveCapacity(indices.count)
            for index in indices {
                let bits = buffer.loadUnaligned(fromByteOffset: magic.count + index * 4, as: UInt32.self)
                let value = Float(bitPattern: bits.littleEndian)
                guard value.isFinite && value >= 0 && value <= 1 else { return nil }
                result.append(value)
            }
            return result
        }
    }

    static func value(in data: Data, at index: Int) -> Float? {
        values(in: data, at: [index])?.first
    }

    static func decode(_ data: Data) -> [Float]? {
        let bytes: Data
        if data.prefix(magic.count) == magic {
            guard data.count == magic.count + byteCount else { return nil }
            bytes = data.dropFirst(magic.count)
        } else if data.count > legacyMagic.count, data.prefix(legacyMagic.count) == legacyMagic {
            var decoded = Data(count: byteCount + 1)
            let payload = data.dropFirst(legacyMagic.count)
            let count = payload.withUnsafeBytes { source in
                decoded.withUnsafeMutableBytes { destination in
                    compression_decode_buffer(destination.bindMemory(to: UInt8.self).baseAddress!, destination.count,
                                              source.bindMemory(to: UInt8.self).baseAddress!, source.count, nil, COMPRESSION_ZLIB)
                }
            }
            guard count == byteCount else { return nil }
            decoded.removeLast()
            bytes = decoded
        } else {
            return nil
        }
        let probabilities: [Float] = bytes.withUnsafeBytes { buffer in
            stride(from: 0, to: byteCount, by: 4).map { offset in
                Float(bitPattern: buffer.loadUnaligned(fromByteOffset: offset, as: UInt32.self).littleEndian)
            }
        }
        guard probabilities.allSatisfy({ $0.isFinite && (0...1).contains($0) }) else { return nil }
        return probabilities
    }
}
