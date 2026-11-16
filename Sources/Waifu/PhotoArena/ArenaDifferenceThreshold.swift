import Foundation

/// Otsu thresholding of native-pixel RGB residuals, with a robust estimate of
/// background variation. Thresholds use the same strict `residual > tolerance`
/// convention as the masks. One-level edits against exact agreement are visible.
enum ArenaDifferenceThreshold {
    static func select(histogram: [Int]) -> Int {
        guard histogram.count == 256, histogram.allSatisfy({ $0 >= 0 }) else { return 0 }
        let count = Double(histogram.reduce(0, +))
        guard count > 0 else { return 0 }
        let sum = histogram.indices.reduce(0.0) { $0 + Double($1) * Double(histogram[$1]) }
        let squares = histogram.indices.reduce(0.0) { $0 + Double($1 * $1) * Double(histogram[$1]) }
        let variance = max(0, squares / count - pow(sum / count, 2))
        // Identical images produce an empty mask. A constant nonzero residual is
        // a genuine whole-image difference and must remain visible to admission.
        guard variance > 0 else { return max(0, Int((sum / count).rounded()) - 1) }
        var lowerCount = 0.0
        var lowerSum = 0.0
        var bestScore = 0.0
        var threshold = 0
        for value in 0..<255 {
            let frequency = Double(histogram[value])
            lowerCount += frequency
            lowerSum += Double(value) * frequency
            let upperCount = count - lowerCount
            guard lowerCount > 0, upperCount > 0 else { continue }
            let lowerMean = lowerSum / lowerCount
            let upperMean = (sum - lowerSum) / upperCount
            let score = (lowerCount / count) * (upperCount / count) * pow(upperMean - lowerMean, 2)
            if score > bestScore {
                bestScore = score
                threshold = value
            }
        }
        // A weak split gets a robust noise estimate rather than a fixed fallback.
        // MAD avoids letting a few large edits inflate the ordinary noise level.
        let background = bestScore / variance >= 0.65
            ? Array(histogram.prefix(threshold + 1)) : histogram
        let median = percentileHalf(background)
        var deviations = [Int](repeating: 0, count: 256)
        for value in background.indices { deviations[abs(value - median)] += background[value] }
        let noiseFloor = min(255, Int(ceil(Double(median) + 3 * 1.4826 * Double(percentileHalf(deviations)))))
        guard bestScore / variance >= 0.65 else { return noiseFloor }
        // Retain the upper population even when the lower population is broad.
        let upperStart = ((threshold + 1)..<256).first { histogram[$0] > 0 } ?? 256
        return min(upperStart - 1, max(threshold, noiseFloor))
    }

    private static func percentileHalf(_ histogram: [Int]) -> Int {
        let target = max(1, (histogram.reduce(0, +) + 1) / 2)
        var cumulative = 0
        for value in histogram.indices {
            cumulative += histogram[value]
            if cumulative >= target { return value }
        }
        return 0
    }
}
