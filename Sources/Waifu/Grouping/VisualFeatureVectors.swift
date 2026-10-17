import Accelerate
import Foundation
import Vision

/// Float embeddings are used only after checking their distance convention against
/// Vision. Unknown formats or a different metric retain the native Vision path.
struct VisualFeatureVectors: Sendable {
    enum Metric: UInt32, Sendable {
        case euclidean = 0
        case squaredEuclidean = 1
    }

    let count: Int
    let dimensions: Int
    let values: [Float]
    let metric: Metric

    init?(observations: [FeaturePrintObservation]) {
        guard let first = observations.first, first.elementType == .float,
              first.elementCount > 0, first.elementCount <= 4_096 else { return nil }
        let dimensions = first.elementCount
        var values: [Float] = []
        values.reserveCapacity(observations.count * dimensions)
        for observation in observations {
            if Task.isCancelled { return nil }
            guard observation.elementType == .float, observation.elementCount == dimensions,
                  observation.originatingRequestDescriptor == first.originatingRequestDescriptor,
                  observation.data.count == dimensions * MemoryLayout<Float>.stride else { return nil }
            observation.data.withUnsafeBytes { bytes in
                for index in 0..<dimensions {
                    values.append(bytes.loadUnaligned(fromByteOffset: index * 4, as: Float.self))
                }
            }
        }
        guard values.allSatisfy(\.isFinite) else { return nil }
        // Current Vision revision 2 returns squared Euclidean distance. Infer
        // and verify the convention rather than assuming a norm from raw bytes.
        for metric in [Metric.squaredEuclidean, .euclidean] {
            guard let candidate = Self(count: observations.count, dimensions: dimensions, values: values, metric: metric) else { return nil }
            let matches = Self.validationPairs(count: observations.count).allSatisfy { left, right in
                guard let native = try? observations[left].distance(to: observations[right]) else { return false }
                return Self.matches(native, Double(candidate.distance(left, right)))
            }
            if matches { self = candidate; return }
        }
        return nil
    }

    /// Explicit vectors are also used by the solver's numerical tests/benchmarks.
    init?(count: Int, dimensions: Int, values: [Float], metric: Metric = .euclidean) {
        guard count > 0, dimensions > 0, dimensions <= 4_096,
              count <= Int.max / dimensions, values.count == count * dimensions,
              values.allSatisfy(\.isFinite) else { return nil }
        self.count = count
        self.dimensions = dimensions
        self.values = values
        self.metric = metric
    }

    func distance(_ left: Int, _ right: Int) -> Float {
        values.withUnsafeBufferPointer { buffer in
            var squared: Float = 0
            vDSP_distancesq(buffer.baseAddress! + left * dimensions, 1,
                            buffer.baseAddress! + right * dimensions, 1,
                            &squared, vDSP_Length(dimensions))
            return metric == .squaredEuclidean ? max(0, squared) : sqrt(max(0, squared))
        }
    }

    func matrix() throws -> TSPDistanceMatrix {
        guard count <= CuOptMetalTSP.denseLimit else { throw TSPError.allocationFailed }
        var values = [Float](repeating: 0, count: count * count)
        for left in 0..<count {
            try Task.checkCancellation()
            for right in (left + 1)..<count {
                let value = distance(left, right)
                values[left * count + right] = value
                values[right * count + left] = value
            }
        }
        return try TSPDistanceMatrix(count: count, values: values)
    }

    static func validationPairs(count: Int) -> [(Int, Int)] {
        guard count > 1 else { return [] }
        var pairs = [(0, count - 1)]
        for index in 0..<min(count, 32) {
            pairs.append((index * count / min(count, 32), (index * 7_919 + count / 2) % count))
        }
        return pairs
    }

    static func matches(_ native: Double, _ proposed: Double) -> Bool {
        native.isFinite && proposed.isFinite && abs(native - proposed) <= 2e-5 * max(1, abs(native))
    }
}
