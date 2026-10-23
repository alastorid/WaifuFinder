import Compression
import Foundation

/// Sparse normalized UInt8 scores, compressed per frame with Apple's LZ4 implementation.
/// Scores below the fixed storage cutoff are zero. A compact row stays compressed
/// in frame records; selective consumers expand only its sparse pairs, never a
/// full Float32 vector. Only the current UInt8 format is accepted.
enum TagProbabilityCodec {
    private static let compactMagic = Data("WDT4".utf8)
    // Fixed independently of the current display filter: raising a slider must
    // not permanently throw away additional scores on the next cache restore.
    static let storageThreshold: Float = 0.35
    private static let compactHeaderSize = 12
    private static let maximumSparseSize = 6 + (TagVocabulary.headCount - 4) * 3

    static func encode(_ probabilities: [Float]) -> Data? {
        guard probabilities.count == TagVocabulary.headCount,
              probabilities.allSatisfy({ $0.isFinite && (0...1).contains($0) }) else { return nil }
        let codes = probabilities.enumerated().map { index, value in
            index < 4 ? UInt8(value * 255) : quantize(value)
        }
        return encode(quantized: codes)
    }

    /// Accept already-quantized worker output without converting it through Float32.
    static func encode(quantized codes: [UInt8]) -> Data? {
        guard codes.count == TagVocabulary.headCount else { return nil }
        // Exactly four rating bytes, then a UInt16 trait-pair count.
        var raw = Data(codes.prefix(4))
        raw.append(contentsOf: [0, 0])
        var count: UInt16 = 0
        for index in 4..<codes.count {
            let code = codes[index]
            guard code == 0 || code == 1 || code >= 89 else { return nil }
            guard code != 0 else { continue }
            count += 1
            append(UInt16(index), to: &raw)
            raw.append(code)
        }
        raw[4] = UInt8(truncatingIfNeeded: count)
        raw[5] = UInt8(truncatingIfNeeded: count >> 8)
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
        guard let pairs = sparseBytes(in: data) else { return nil }
        return indices.map { index in
            var low = 0, high = pairs.count
            while low < high {
                let middle = (low + high) / 2
                if pairs[middle].index < index { low = middle + 1 } else { high = middle }
            }
            return low < pairs.count && pairs[low].index == index ? probability(pairs[low].value, at: index) : 0
        }
    }

    /// Sparse iteration for per-tag ranking. Compact UInt8 rows never expand to
    /// dense Float32 vectors.
    static func sparseValues(in data: Data) -> [(index: Int, value: Float)]? {
        sparseBytes(in: data)?.map { ($0.index, probability($0.value, at: $0.index)) }
    }

    static func value(in data: Data, at index: Int) -> Float? {
        values(in: data, at: [index])?.first
    }

    static func decode(_ data: Data) -> [Float]? {
        guard let pairs = sparseBytes(in: data) else { return nil }
        var result = [Float](repeating: 0, count: TagVocabulary.headCount)
        for pair in pairs { result[pair.index] = probability(pair.value, at: pair.index) }
        return result
    }

    /// Scheduling hint only. Actual consumers fully validate the payload.
    static func retainedCount(in data: Data) -> Int? {
        guard data.count >= compactHeaderSize, data.prefix(4) == compactMagic else { return nil }
        return data.withUnsafeBytes { bytes in
            let size = Int(UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: 4, as: UInt32.self)))
            guard (6...maximumSparseSize).contains(size), (size - 6) % 3 == 0 else { return nil }
            return 4 + (size - 6) / 3
        }
    }

    private static func append<T: FixedWidthInteger>(_ value: T, to data: inout Data) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
    }

    /// Uniform 0...255 confidence, truncated so quantization never raises a
    /// confidence across a display threshold. Error is less than 1/255.
    /// Code 1 preserves exact cutoff equality;
    /// code 89 is clamped just above the cutoff so admitted tags cannot vanish.
    static func quantize(_ value: Float) -> UInt8 {
        if value < storageThreshold { return 0 }
        if value == storageThreshold { return 1 }
        return UInt8(value * 255)
    }

    static func probability(_ code: UInt8) -> Float {
        if code == 0 { return 0 }
        if code == 1 { return storageThreshold }
        return max(Float(code) / 255, storageThreshold.nextUp)
    }

    static func probability(_ code: UInt8, at index: Int) -> Float {
        index < 4 ? Float(code) / 255 : probability(code)
    }

    /// Decode sparse bytes without allocating a dense Float32 head vector.
    static func sparseBytes(in data: Data) -> [(index: Int, value: UInt8)]? {
        guard data.count >= compactHeaderSize, data.prefix(4) == compactMagic else { return nil }
        return data.withUnsafeBytes { buffer in
            let rawSize = Int(UInt32(littleEndian: buffer.loadUnaligned(fromByteOffset: 4, as: UInt32.self)))
            let compressedSize = Int(UInt32(littleEndian: buffer.loadUnaligned(fromByteOffset: 8, as: UInt32.self)))
            guard (6...maximumSparseSize).contains(rawSize), (rawSize - 6) % 3 == 0,
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
                let count = Int(UInt16(littleEndian: bytes.loadUnaligned(fromByteOffset: 4, as: UInt16.self)))
                guard rawSize == 6 + count * 3 else { return nil }
                var pairs: [(index: Int, value: UInt8)] = (0..<4).map { ($0, bytes[$0]) }
                pairs.reserveCapacity(count + 4)
                var previous = 3
                for offset in stride(from: 6, to: rawSize, by: 3) {
                    let index = Int(UInt16(littleEndian: bytes.loadUnaligned(fromByteOffset: offset, as: UInt16.self)))
                    let code = bytes[offset + 2]
                    guard index > previous, index < TagVocabulary.headCount,
                          index < 4 || code == 1 || code >= 89 else { return nil }
                    pairs.append((index, code))
                    previous = index
                }
                return pairs
            }
        }
    }
}
