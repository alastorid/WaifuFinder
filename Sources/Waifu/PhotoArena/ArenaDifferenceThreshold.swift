import Foundation

/// Otsu thresholding of native-pixel RGB residuals. A histogram split is used
/// only when it separates two populations; a uniform global shift stays visible.
enum ArenaDifferenceThreshold {
    static func select(histogram: [Int]) -> Int {
        guard histogram.count == 256 else { return 12 }
        let count = Double(histogram.reduce(0, +))
        guard count > 0 else { return 12 }
        let sum = histogram.indices.reduce(0.0) { $0 + Double($1 * histogram[$1]) }
        let squares = histogram.indices.reduce(0.0) { $0 + Double($1 * $1 * histogram[$1]) }
        let variance = squares / count - pow(sum / count, 2)
        guard variance > 0 else { return 12 }
        var lowerCount = 0.0
        var lowerSum = 0.0
        var lowerSquares = 0.0
        var bestScore = 0.0
        var threshold = 12
        var backgroundMean = 0.0
        var backgroundDeviation = 0.0
        var separation = 0.0
        for value in 0..<255 {
            let frequency = Double(histogram[value])
            lowerCount += frequency
            lowerSum += Double(value) * frequency
            lowerSquares += Double(value * value) * frequency
            let upperCount = count - lowerCount
            guard lowerCount > 0, upperCount > 0 else { continue }
            let lowerMean = lowerSum / lowerCount
            let upperMean = (sum - lowerSum) / upperCount
            let score = (lowerCount / count) * (upperCount / count) * pow(upperMean - lowerMean, 2)
            if score > bestScore {
                bestScore = score
                threshold = value
                backgroundMean = lowerMean
                backgroundDeviation = sqrt(max(0, lowerSquares / lowerCount - lowerMean * lowerMean))
                separation = upperMean - lowerMean
            }
        }
        // Avoid inventing edit classes within a narrow, unimodal noise distribution.
        guard bestScore / variance >= 0.65, separation >= 12 else { return 12 }
        // Keep the historical noise floor. The lower population estimates the
        // ordinary residual variation, independently of the admission area limit.
        let noiseFloor = Int(ceil(backgroundMean + 3 * backgroundDeviation))
        guard Double(noiseFloor) < backgroundMean + separation else { return 12 }
        return min(254, max(12, threshold, noiseFloor))
    }
}
