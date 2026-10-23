import Foundation
import Vision

/// The public Vision payload is native-endian Float32 only when its type, byte
/// length and request descriptor agree. No serialized JSON bytes are interpreted
/// as vectors. Unknown representations remain eligible for native Vision search.
struct WaifuVisionVectorIndex: Sendable {
    let source: UUID
    private(set) var packed: WaifuVectorIndex?
    private var template: FeaturePrintObservation?
    private(set) var observations: [String: FeaturePrintObservation] = [:]
    private(set) var unrepresentable: Set<String> = []
    init(source: UUID) { self.source = source }
    static func vector(_ observation: FeaturePrintObservation) -> [Float]? {
        guard observation.elementType == .float, observation.elementCount > 0, observation.elementCount <= 4096,
              observation.data.count == observation.elementCount * 4 else { return nil }
        let values = observation.data.withUnsafeBytes { bytes in
            (0..<observation.elementCount).map { bytes.loadUnaligned(fromByteOffset: $0*4, as: Float.self) }
        }
        return values.allSatisfy(\.isFinite) ? values : nil
    }
    mutating func append(id: String, observation: FeaturePrintObservation) {
        guard observations[id] == nil else { return }
        observations[id] = observation
        guard let vector = Self.vector(observation) else { unrepresentable.insert(id); return }
        if template == nil {
            template = observation
            // The metric is validated with non-identical observations/query before use.
            packed = WaifuVectorIndex(source: source, dimensions: vector.count, metric: .squaredEuclidean)
        }
        guard let template, observation.originatingRequestDescriptor == template.originatingRequestDescriptor,
              packed?.append(id: id, vector: vector) == true else { unrepresentable.insert(id); return }
    }
    /// Validate query-to-index distances too, rather than relying only on pairs
    /// within the database. Every source generation/request format gets checked.
    func validatedQuery(_ query: FeaturePrintObservation) -> [Float]? {
        guard let template, let packed, let vector = Self.vector(query), vector.count == packed.dimensions,
              query.originatingRequestDescriptor == template.originatingRequestDescriptor else { return nil }
        let probes = VisualFeatureVectors.validationPairs(count: packed.count).flatMap { [$0.0,$0.1] }
        for row in Set([0] + probes) {
            guard row < packed.count, let observation = observations[packed.ids[row]],
                  let native = try? query.distance(to: observation) else { return nil }
            var squared: Float = 0
            for d in 0..<packed.dimensions { let delta = vector[d] - packed.values[row*packed.rowStride+d]; squared += delta*delta }
            guard VisualFeatureVectors.matches(native, Double(squared)) else { return nil }
        }
        return vector
    }
}
