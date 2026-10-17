import Compression
import Foundation
import Vision

/// `FeaturePrintObservation` ↔ `Data`, the only way in and out.
///
/// ## Why JSON and not the raw bytes
///
/// `FeaturePrintObservation` cannot be rebuilt from its raw payload through
/// any public API — the only initialiser takes another observation. Its
/// `Codable` conformance is the only route in, and it was verified bit-exact
/// in the reference implementation: worst distance error `0.0` across 400
/// measured pairs after a JSON round trip. So the bytes are JSON, and this is
/// the only place that knows it.
///
/// ## Why compress at all
///
/// A FeaturePrint is 768 floats — 3,072 B raw, ~4.3 KB as JSON. Waifu stores
/// one per analysed frame; at the default plan over the 47-minute sample video
/// that is ~2,800 vectors, ~12 MB uncompressed. zlib brings each to ~2.5 KB,
/// about a third of the size, for a decode cost of microseconds.
///
/// ## The one-byte marker
///
/// A payload written by this codec begins with the marker byte `1`. A payload
/// whose first byte is not the marker is read as plain JSON — the same
/// tolerance the reference implements for payloads written before compression
/// existed, kept so a future codec change can bump the marker and old payloads
/// still decode.
enum FeaturePrintCodec: Sendable {
    /// `zlib` streaming format, not raw deflate, so the stored bytes carry
    /// their own checksum: a truncated payload fails to decompress instead of
    /// decoding into plausible-looking nonsense.
    private static let algorithm = COMPRESSION_ZLIB

    /// The codec marker. A payload whose first byte is not this is treated as
    /// uncompressed JSON.
    private static let marker: UInt8 = 1

    /// The JSON of a FeaturePrint is ~4.3 KB. This only bounds a corrupt length
    /// prefix; normal payloads are far below it.
    private static let maximumDecompressedSize = 1 << 20

    /// Encodes an observation to a storable payload: JSON, then zlib.
    ///
    /// `nil` only when the observation cannot be encoded or the buffer could
    /// not hold the result — a degradation the scorer turns into a missing
    /// vector, never a failed frame.
    static func encode(_ observation: FeaturePrintObservation) -> Data? {
        guard let json = try? JSONEncoder().encode(observation) else { return nil }
        return compress(json)
    }

    /// Decodes a payload back into an observation.
    ///
    /// `nil` for anything that does not inflate or decode — which is the point
    /// of the zlib container: a truncated or corrupt payload fails here rather
    /// than decoding into plausible-looking nonsense.
    static func decode(_ data: Data) -> FeaturePrintObservation? {
        guard let json = decompress(data) else { return nil }
        return try? JSONDecoder().decode(FeaturePrintObservation.self, from: json)
    }

    /// Compresses `data` with a marker byte prefix. `nil` for empty input or a
    /// buffer that could not hold the result.
    ///
    /// Empty in, `nil` out rather than the empty `Data` back: a zero-length
    /// blob reads back as *no* value at all, which would leave a stored row
    /// that looks written and never is.
    static func compress(_ data: Data) -> Data? {
        guard !data.isEmpty else { return nil }
        let capacity = data.count + 1024
        var out = Data(count: capacity + 1)
        let written = out.withUnsafeMutableBytes { destination -> Int in
            data.withUnsafeBytes { source in
                guard let base = destination.bindMemory(to: UInt8.self).baseAddress,
                      let sourceBase = source.bindMemory(to: UInt8.self).baseAddress else { return 0 }
                return compression_encode_buffer(base + 1, capacity,
                                                 sourceBase, data.count,
                                                 nil, algorithm)
            }
        }
        // 0 means the buffer was too small; a negative value is an encoding error.
        guard written > 0 else { return nil }
        out[0] = marker
        out.removeLast(capacity - written)
        return out
    }

    /// Decompresses a payload, tolerating an unmarked one written before
    /// compression existed. `nil` for anything that does not inflate.
    static func decompress(_ data: Data) -> Data? {
        guard let first = data.first else { return nil }
        guard first == marker else { return data }
        let body = data.dropFirst()
        var out = Data(count: maximumDecompressedSize)
        let written = out.withUnsafeMutableBytes { destination -> Int in
            body.withUnsafeBytes { source in
                guard let base = destination.bindMemory(to: UInt8.self).baseAddress,
                      let sourceBase = source.bindMemory(to: UInt8.self).baseAddress else { return 0 }
                return compression_decode_buffer(base, maximumDecompressedSize,
                                                 sourceBase, body.count,
                                                 nil, algorithm)
            }
        }
        guard written > 0 else { return nil }
        out.removeLast(maximumDecompressedSize - written)
        return out
    }

    /// Verifies that a round trip through this codec preserves distances
    /// **bit-exactly**: the distance from a decoded copy of `observation` to
    /// `other` must equal the distance from the original to `other`, to the
    /// last bit.
    ///
    /// Returns the absolute distance error — `0.0` is the only acceptable
    /// value — or `nil` when the round trip itself failed (an observation that
    /// would not encode, or a distance that would not compute).
    ///
    /// This is the check the reference ran over 400 measured pairs (worst error
    /// `0.0`); it exists so a future change to the codec cannot quietly break
    /// the property that makes caching one vector per frame worthwhile.
    static func roundTripDistanceError(_ observation: FeaturePrintObservation,
                                       comparedWith other: FeaturePrintObservation) -> Double? {
        guard let encoded = encode(observation),
              let decoded = decode(encoded) else { return nil }
        guard let original = try? observation.distance(to: other),
              let roundTripped = try? decoded.distance(to: other) else { return nil }
        return abs(original - roundTripped)
    }
}
