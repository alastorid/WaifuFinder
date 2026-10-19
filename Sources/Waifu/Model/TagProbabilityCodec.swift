import Compression
import Foundation

/// Sparse E4M3 FP8 scores, compressed per frame with Apple's LZ4 implementation.
/// Scores below the fixed storage cutoff are zero. A compact row stays compressed
/// in frame records; selective consumers expand only its sparse pairs, never a
/// full Float32 vector. WDT1/2 remain readable for older serialized metadata.
enum TagProbabilityCodec {
    private static let compactMagic = Data("WDT3".utf8)
    private static let magic = Data([0x57, 0x44, 0x54, 0x32]) // WDT2
    // Fixed independently of the current display filter: raising a slider must
    // not permanently throw away additional scores on the next cache restore.
    static let storageThreshold: Float = 0.35
    private static let compactHeaderSize = 12
    private static let maximumSparseSize = 2 + TagVocabulary.headCount * 3
    private static let legacyMagic = Data([0x57, 0x44, 0x54, 0x31]) // WDT1
    private static let byteCount = TagVocabulary.headCount * MemoryLayout<UInt32>.size

    static func encode(_ probabilities: [Float]) -> Data? {
        guard probabilities.count == TagVocabulary.headCount,
              probabilities.allSatisfy({ $0.isFinite && (0...1).contains($0) }) else { return nil }
        var raw = Data([0, 0]) // little-endian pair count
        var count: UInt16 = 0
        for (index, value) in probabilities.enumerated() where value >= storageThreshold {
            count += 1
            append(UInt16(index), to: &raw)
            // A dedicated code preserves equality at the storage cutoff. For
            // included values rounded below it, decoding preserves strict >.
            raw.append(value == storageThreshold ? 1 : quantize(value))
        }
        raw[0] = UInt8(truncatingIfNeeded: count)
        raw[1] = UInt8(truncatingIfNeeded: count >> 8)
        var compressed = Data(count: raw.count + 1_024)
        let size = raw.withUnsafeBytes { source in
            compressed.withUnsafeMutableBytes { destination in
                compression_encode_buffer(destination.bindMemory(to: UInt8.self).baseAddress!, destination.count,
                                          source.bindMemory(to: UInt8.self).baseAddress!, source.count,
                                          nil, COMPRESSION_LZ4)
            }
        }
        guard size > 0 else { return nil }
        var result = compactMagic
        append(UInt32(raw.count), to: &result)
        append(UInt32(size), to: &result)
        result.append(compressed.prefix(size))
        return result
    }

    /// Selective reads preserve order and duplicate indices without retaining
    /// a decoded Float32 row alongside the compressed payload.
    static func values(in data: Data, at indices: [Int]) -> [Float]? {
        guard indices.allSatisfy({ $0 >= 0 && $0 < TagVocabulary.headCount }) else { return nil }
        if data.prefix(4) == compactMagic {
            guard let pairs = sparsePairs(in: data) else { return nil }
            return indices.map { index in
                // Pairs are sorted by vocabulary index; avoid a full-size map.
                var low = 0, high = pairs.count
                while low < high {
                    let middle = (low + high) / 2
                    if pairs[middle].index < index { low = middle + 1 } else { high = middle }
                }
                return low < pairs.count && pairs[low].index == index ? pairs[low].value : 0
            }
        }
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

    /// Sparse iteration for per-tag ranking. Compact FP8 rows never expand to
    /// dense Float32 vectors; older formats remain readable.
    static func sparseValues(in data: Data) -> [(index: Int, value: Float)]? {
        if data.prefix(4) == compactMagic { return sparsePairs(in: data) }
        guard let values = decode(data) else { return nil }
        return values.enumerated().compactMap { index, value in
            value > 0 ? (index, value) : nil
        }
    }

    static func value(in data: Data, at index: Int) -> Float? {
        values(in: data, at: [index])?.first
    }

    static func decode(_ data: Data) -> [Float]? {
        if data.prefix(4) == compactMagic {
            guard let pairs = sparsePairs(in: data) else { return nil }
            var result = [Float](repeating: 0, count: TagVocabulary.headCount)
            for pair in pairs { result[pair.index] = pair.value }
            return result
        }
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

    private static func append<T: FixedWidthInteger>(_ value: T, to data: inout Data) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
    }

    private static func quantize(_ value: Float) -> UInt8 {
        // Positive normalized E4M3; the admitted range is (0.35, 1].
        var exponent = Int(floor(log2(value)))
        var mantissa = Int(((value / pow(2, Float(exponent)) - 1) * 8).rounded(.toNearestOrEven))
        if mantissa == 8 { exponent += 1; mantissa = 0 }
        return UInt8(((exponent + 7) << 3) | mantissa)
    }

    private static func sparsePairs(in data: Data) -> [(index: Int, value: Float)]? {
        guard data.count >= compactHeaderSize else { return nil }
        return data.withUnsafeBytes { buffer in
            let rawSize = Int(UInt32(littleEndian: buffer.loadUnaligned(fromByteOffset: 4, as: UInt32.self)))
            let compressedSize = Int(UInt32(littleEndian: buffer.loadUnaligned(fromByteOffset: 8, as: UInt32.self)))
            guard (2...maximumSparseSize).contains(rawSize), (rawSize - 2) % 3 == 0,
                  compressedSize > 0, compressedSize <= maximumSparseSize + 1_024,
                  data.count == compactHeaderSize + compressedSize else { return nil }
            // An extra byte detects compressed streams expanding beyond the
            // advertised size rather than accepting a full destination prefix.
            var raw = Data(count: rawSize + 1)
            let decodedSize = raw.withUnsafeMutableBytes { destination in
                compression_decode_buffer(destination.bindMemory(to: UInt8.self).baseAddress!, destination.count,
                                          buffer.bindMemory(to: UInt8.self).baseAddress!.advanced(by: compactHeaderSize),
                                          compressedSize, nil, COMPRESSION_LZ4)
            }
            guard decodedSize == rawSize else { return nil }
            return raw.withUnsafeBytes { bytes in
                let count = Int(UInt16(littleEndian: bytes.loadUnaligned(as: UInt16.self)))
                guard rawSize == 2 + count * 3 else { return nil }
                var pairs: [(index: Int, value: Float)] = []
                pairs.reserveCapacity(count)
                var previous = -1
                for offset in stride(from: 2, to: rawSize, by: 3) {
                    let index = Int(UInt16(littleEndian: bytes.loadUnaligned(fromByteOffset: offset, as: UInt16.self)))
                    let code = bytes[offset + 2]
                    guard index > previous, index < TagVocabulary.headCount,
                          code == 1 || (0x2b...0x38).contains(code) else { return nil }
                    let value: Float = code == 1 ? storageThreshold
                        : max((1 + Float(code & 7) / 8) * pow(2, Float(Int(code >> 3) - 7)), storageThreshold.nextUp)
                    pairs.append((index, value))
                    previous = index
                }
                return pairs
            }
        }
    }
}
