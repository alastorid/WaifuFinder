import Foundation
import Metal

@main
struct TagSummaryBenchmark {
    static func main() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("tag-summary-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        var csv = "name,category\n"
        for rating in TagRating.allCases { csv += "\(rating.rawValue),9\n" }
        for index in 0..<TagVocabulary.generalCount { csv += "tag_\(index),0\n" }
        for index in 0..<2_751 { csv += "character_\(index),4\n" }
        let labels = directory.appendingPathComponent("labels.csv")
        try csv.write(to: labels, atomically: true, encoding: .utf8)
        let vocabulary = try TagVocabulary.load(from: labels)
        let metal = try MetalTagSummary()
        let payloads = (0..<8).map { variant in
            var probabilities = (0..<TagVocabulary.headCount).map { Float(($0 * 7_919 + variant) % 101) / 10_100 }
            for index in 4..<24 { probabilities[index] = Float((index + variant) % 10 + 1) / 10 }
            return TagProbabilityCodec.encode(probabilities)!
        }
        let clock = ContinuousClock()
        func milliseconds(_ duration: Duration) -> Double {
            let parts = duration.components
            return Double(parts.seconds) * 1_000 + Double(parts.attoseconds) / 1e15
        }
        print("Optimized tag summary: 8,106 general heads; threshold 0.35; sparse synthetic probabilities")
        print("Device: \(MTLCreateSystemDefaultDevice()?.name ?? "unavailable")")
        print("Each warm measurement includes cache lookup, view selection, reduction and result construction.")
        print("Photos\tCold decode + CPU ms\tWarm CPU ms\tWarm Metal ms\tMax mean error")
        for count in [64, 512, 1_024, 1_536, 1_800] {
            let frames = (0..<count).map { index in
                var frame = AnalyzedFrame(sampleIndex: index, timeSeconds: Double(index), artScore: 1,
                                          faceScore: nil, faceCount: 0, faceAreaFraction: nil, featurePrint: nil)
                frame.tagProbabilities = payloads[index % payloads.count]
                return frame
            }
            let computer = TagSummaryComputer(metalProvider: { metal })
            let coldStart = clock.now
            let reference = try await computer.summarize(frames, vocabulary: vocabulary, threshold: 0.35, backend: .cpu)
            let cold = milliseconds(coldStart.duration(to: clock.now))
            var cpuTimes: [Double] = [], gpuTimes: [Double] = []
            var maximumError = 0.0
            for _ in 0..<7 {
                let cpuStart = clock.now
                _ = try await computer.summarize(frames, vocabulary: vocabulary, threshold: 0.35, backend: .cpu)
                cpuTimes.append(milliseconds(cpuStart.duration(to: clock.now)))
                let gpuStart = clock.now
                let gpu = try await computer.summarize(frames, vocabulary: vocabulary, threshold: 0.35, backend: .metal)
                gpuTimes.append(milliseconds(gpuStart.duration(to: clock.now)))
                for (left, right) in zip(reference.tags, gpu.tags) {
                    guard left.index == right.index, left.matchingFrames == right.matchingFrames,
                          left.scoredFrames == right.scoredFrames else { throw TagSummaryError.invalidInput }
                    maximumError = max(maximumError, abs((left.mean ?? 0) - (right.mean ?? 0)))
                }
            }
            print(String(format: "%d\t%.3f\t%.3f\t%.3f\t%.9f", count, cold,
                         cpuTimes.sorted()[3], gpuTimes.sorted()[3], maximumError))
        }
    }
}
