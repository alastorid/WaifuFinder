import CoreGraphics
import CoreImage
import Foundation

/// Bounded native-tile calibration. Pixel buffers are released after each tile.
enum MergerPreprocessor {
    static func prepare(_ inputs: [ArenaSource]) throws -> MergerAdmission.Assessment {
        var measured = try ArenaForensics.analyzeFiles(inputs, tileSize: 256)
        // A large unrelated input must not monopolize the reference canvas.
        if measured.sources.filter(\.accepted).count < 2, inputs.count > 2 {
            let remaining = inputs.filter { $0.id != measured.referenceID }
            let alternative = try ArenaForensics.analyzeFiles(remaining, tileSize: 256)
            if alternative.sources.filter(\.accepted).count >= 2 {
                var excluded = measured.sources.first { $0.id == measured.referenceID }!
                excluded.accepted = false
                excluded.issues = ["Excluded reference outlier; the remaining sources have mutually supported registration."]
                measured = alternative
                measured.sources.append(excluded)
            }
        }
        let reference = measured.sources.first { $0.id == measured.referenceID }!
        let referenceImage = try ArenaForensics.alignedImage(reference)
        let tiles = measured.regions
        var calibrations: [String: Calibration] = [:]
        for index in measured.sources.indices where measured.sources[index].accepted {
            try Task.checkCancellation()
            let source = measured.sources[index]
            if source.fileSHA256 == reference.fileSHA256 {
                calibrations[source.id] = Calibration(offset: [0, 0, 0], threshold: 12, support: 1)
                continue
            }
            let calibration = try calibrate(ArenaForensics.alignedImage(source), reference: referenceImage, tiles: tiles)
            if calibration.support < 0.70 {
                measured.sources[index].accepted = false
                measured.sources[index].issues.append("Local structure agrees in only \(Int(calibration.support * 100))% of sampled areas.")
            } else {
                calibrations[source.id] = calibration
                measured.sources[index].issues.append("Local structure support \(Int(calibration.support * 100))%; calibrated residual threshold \(Int(calibration.threshold))/255.")
            }
        }
        let accepted = measured.sources.filter(\.accepted)
        guard accepted.count >= 2 else { return try MergerAdmission.assess(measured) }
        let images = try accepted.map(ArenaForensics.alignedImage)
        let orderedCalibrations = accepted.map { calibrations[$0.id]! }
        var retainedRuns = 0
        for index in measured.regions.indices {
            try Task.checkCancellation()
            let rect = measured.regions[index].rect.cgRect
            let buffers = images.map { ArenaForensics.bytes($0, rect: rect) }
            let width = Int(rect.width), height = Int(rect.height)
            var runs: [ArenaMaskRun] = []
            for y in 0..<height {
                var start: Int?
                for x in 0...width {
                    var changed = false
                    if x < width {
                        let pixel = (y * width + x) * 4
                        for channel in 0..<3 {
                            var minimum = Double.infinity, maximum = -Double.infinity
                            var threshold = 12.0, covered = 0
                            for sourceIndex in accepted.indices where buffers[sourceIndex][pixel + 3] > 250 {
                                let calibration = orderedCalibrations[sourceIndex]
                                let value = Double(buffers[sourceIndex][pixel + channel]) - calibration.offset[channel]
                                minimum = min(minimum, value); maximum = max(maximum, value)
                                threshold = max(threshold, calibration.threshold)
                                covered += 1
                            }
                            if covered >= 2 && maximum - minimum > threshold { changed = true }
                        }
                    }
                    if changed && start == nil { start = x }
                    if !changed, let first = start {
                        runs.append(ArenaMaskRun(y: Int(rect.minY) + y, x: Int(rect.minX) + first, length: x - first))
                        start = nil
                    }
                }
            }
            retainedRuns += runs.count
            measured.regions[index].changeMask = retainedRuns <= 500_000 ? ArenaPixelMask(runs: runs) : nil
            measured.regions[index].maskDeferred = retainedRuns > 500_000 ? true : nil
        }
        measured.warnings.append("Difference detection compensates robust global RGB offsets and measured low-amplitude residual noise. Reconstruction always uses the original source pixels.")
        return try MergerAdmission.assess(measured)
    }

    private struct Calibration {
        let offset: [Double]
        let threshold: Double
        let support: Double
    }

    private static func calibrate(_ source: CIImage, reference: CIImage,
                                  tiles: [ArenaRegion]) throws -> Calibration {
        // Stratified samples bound calibration work independently of image size.
        let step = max(1, (tiles.count + 63) / 64)
        var differences = [[Double]](repeating: [], count: 3)
        var sampled: [([UInt8], [UInt8])] = []
        for index in stride(from: 0, to: tiles.count, by: step) {
            try Task.checkCancellation()
            let bounds = tiles[index].rect.cgRect
            let rect = CGRect(x: bounds.midX - min(32, bounds.width / 2),
                              y: bounds.midY - min(32, bounds.height / 2),
                              width: min(64, bounds.width), height: min(64, bounds.height)).integral
            let a = ArenaForensics.bytes(reference, rect: rect)
            let b = ArenaForensics.bytes(source, rect: rect)
            guard a.count == b.count else { continue }
            var pairs = 0
            for pixel in stride(from: 0, to: a.count, by: 16) where a[pixel + 3] > 250 && b[pixel + 3] > 250 {
                for channel in 0..<3 { differences[channel].append(Double(b[pixel + channel]) - Double(a[pixel + channel])) }
                pairs += 1
            }
            if pairs >= 64 { sampled.append((a, b)) }
        }
        guard !sampled.isEmpty else { return Calibration(offset: [0, 0, 0], threshold: 12, support: 0) }
        let offset = differences.map { median($0) }
        var residuals: [Double] = []
        var supported = 0
        for (a, b) in sampled {
            var errors: [Double] = []
            for pixel in stride(from: 0, to: a.count, by: 16) where a[pixel + 3] > 250 && b[pixel + 3] > 250 {
                let error = (0..<3).map { abs(Double(b[pixel + $0]) - Double(a[pixel + $0]) - offset[$0]) }.max()!
                errors.append(error)
            }
            let error = median(errors)
            if error <= 12 { supported += 1; residuals.append(contentsOf: errors) }
        }
        let noise = median(residuals)
        let deviation = median(residuals.map { abs($0 - noise) })
        return Calibration(offset: offset, threshold: min(32, max(12, noise + 6 * deviation)),
                           support: Double(supported) / Double(sampled.count))
    }

    private static func median(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        return sorted[sorted.count / 2]
    }
}
