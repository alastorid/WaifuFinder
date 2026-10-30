import Accelerate
import Foundation
import Vision

/// Distances use the same validated embedding and metric as Minimal Visual Distance.
enum PhotoCenterDistance {
    static func scores(_ frames: [AnalyzedFrame]) throws -> [FrameID: Float] {
        var ids: [FrameID] = []
        var observations: [FeaturePrintObservation] = []
        for frame in frames {
            try Task.checkCancellation()
            guard let data = frame.featurePrint, let observation = FeaturePrintCodec.decode(data) else { continue }
            ids.append(frame.id)
            observations.append(observation)
        }
        guard let vectors = VisualFeatureVectors(observations: observations) else { return [:] }
        let distances = try normalizedDistances(vectors)
        return Dictionary(uniqueKeysWithValues: zip(ids, distances))
    }

    static func normalizedDistances(_ vectors: VisualFeatureVectors) throws -> [Float] {
        // Double accumulation keeps large collections' centers numerically stable.
        var sums = [Double](repeating: 0, count: vectors.dimensions)
        for row in 0..<vectors.count {
            try Task.checkCancellation()
            for column in 0..<vectors.dimensions {
                sums[column] += Double(vectors.values[row * vectors.dimensions + column])
            }
        }
        let center = sums.map { Float($0 / Double(vectors.count)) }
        var distances = [Float](repeating: 0, count: vectors.count)
        try vectors.values.withUnsafeBufferPointer { values in
            for row in 0..<vectors.count {
                try Task.checkCancellation()
                var squared: Float = 0
                vDSP_distancesq(values.baseAddress! + row * vectors.dimensions, 1,
                                center, 1, &squared, vDSP_Length(vectors.dimensions))
                distances[row] = vectors.metric == .squaredEuclidean ? max(0, squared) : sqrt(max(0, squared))
            }
        }
        try Task.checkCancellation()
        guard let low = distances.min(), let high = distances.max(), high > low else {
            return [Float](repeating: 0, count: vectors.count)
        }
        return distances.map { min(1, max(0, ($0 - low) / (high - low))) }
    }
}
