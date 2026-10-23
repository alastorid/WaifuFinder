import Foundation

/// Four UInt8 rating scores and sparse UInt8 traits, LZ4-compressed before
/// encryption. No winning-class field travels on the wire.
enum RemoteTagScorePayload {
    static func encode(_ results: [TagResult], fingerprint: UInt64) throws -> [UInt8] {
        guard (1...8).contains(results.count) else { throw TaggerError.invalidBatch }
        var writer = WireWriter()
        writer.u8(UInt8(results.count))
        writer.u64(fingerprint)
        for result in results {
            guard let scores = result.encodedProbabilities ?? TagProbabilityCodec.encode(result.probabilities) else {
                throw TaggerError.invalidProbabilities
            }
            writer.data(scores)
        }
        return writer.bytes
    }

    static func decode(_ payload: [UInt8], count expectedCount: Int,
                       fingerprint expected: UInt64?) throws -> [QuantizedTagRow] {
        var reader = WireReader(payload)
        let count = Int(try reader.u8())
        let fingerprint = try reader.u64()
        guard (1...8).contains(count), count == expectedCount else { throw TaggerError.invalidProbabilities }
        if let expected, expected != fingerprint {
            throw RemoteError.modelUnavailable("The server changed its tag vocabulary. Reconnect to resync.")
        }
        var rows: [QuantizedTagRow] = []
        for _ in 0..<count {
            let scores = Data(try reader.raw())
            guard let pairs = TagProbabilityCodec.sparseBytes(in: scores),
                  pairs.prefix(4).map(\.index) == [0, 1, 2, 3] else { throw TaggerError.invalidProbabilities }
            rows.append(QuantizedTagRow(scores: scores))
        }
        guard reader.isAtEnd else { throw RemoteError.framing("Trailing probability bytes") }
        return rows
    }
}
