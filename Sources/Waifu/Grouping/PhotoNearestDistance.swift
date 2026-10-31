import Accelerate
import Foundation
import Vision

/// Distances use the same validated embedding and metric as Minimal Visual Distance.
enum PhotoNearestDistance {
    static func scores(_ frames: [AnalyzedFrame]) throws -> [FrameID: Float] {
        var ids: [FrameID] = []
        var observations: [FeaturePrintObservation] = []
        for frame in frames {
            try Task.checkCancellation()
            guard let data = frame.featurePrint, let observation = FeaturePrintCodec.decode(data) else { continue }
            ids.append(frame.id)
            observations.append(observation)
        }
        guard observations.count > 1, let vectors = VisualFeatureVectors(observations: observations) else { return [:] }
        let distances = try normalizedDistances(vectors)
        return Dictionary(uniqueKeysWithValues: zip(ids, distances))
    }

    static func normalizedDistances(_ vectors: VisualFeatureVectors) throws -> [Float] {
        guard vectors.count > 1 else { return [0] }
        // Visit each distinct pair once and retain only its endpoints' minima.
        // Accelerate evaluates the vectors without allocating a quadratic matrix.
        var distances = [Float](repeating: .infinity, count: vectors.count)
        try vectors.values.withUnsafeBufferPointer { values in
            for left in 0..<vectors.count {
                try Task.checkCancellation()
                for right in (left + 1)..<vectors.count {
                    if right % 256 == 0 { try Task.checkCancellation() }
                    var squared: Float = 0
                    vDSP_distancesq(values.baseAddress! + left * vectors.dimensions, 1,
                                    values.baseAddress! + right * vectors.dimensions, 1,
                                    &squared, vDSP_Length(vectors.dimensions))
                    distances[left] = min(distances[left], max(0, squared))
                    distances[right] = min(distances[right], max(0, squared))
                }
            }
        }
        if vectors.metric == .euclidean {
            distances = distances.map { sqrt($0) }
        }
        try Task.checkCancellation()
        guard let low = distances.min(), let high = distances.max(), high > low else {
            return [Float](repeating: 0, count: vectors.count)
        }
        return distances.map { min(1, max(0, ($0 - low) / (high - low))) }
    }
}
