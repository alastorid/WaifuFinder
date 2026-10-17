import Foundation
import Vision

/// Per-pass cache, owned by the grouping worker. Completed rows are discarded;
/// remaining entries belong only to the active forward comparison window.
struct DecodedFeaturePrintWindow {
    private let payloads: [Data?]
    private let decode: (Data) -> FeaturePrintObservation?
    // An existing key with a nil value records corrupt data, avoiding repeated
    // failed decompression/JSON decoding for every neighbour of that frame.
    private var observations: [Int: FeaturePrintObservation?] = [:]

    init(payloads: [Data?], decode: @escaping (Data) -> FeaturePrintObservation? = FeaturePrintCodec.decode) {
        self.payloads = payloads
        self.decode = decode
    }

    mutating func observation(at index: Int) -> FeaturePrintObservation? {
        if let cached = observations[index] { return cached }
        let observation = payloads[index].flatMap(decode)
        observations.updateValue(observation, forKey: index)
        return observation
    }

    mutating func discard(_ index: Int) {
        observations.removeValue(forKey: index)
    }

    var retainedCount: Int { observations.count }
}
