import Foundation

/// Background session benchmark: no windows, UI automation or photo-library access.
@main
struct InverseTagBenchmark {
    @MainActor
    static func main() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("inverse-tags-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        var csv = "name,category\n"
        for rating in TagRating.allCases { csv += "\(rating.rawValue),9\n" }
        for index in 0..<TagVocabulary.generalCount { csv += "tag_\(index),0\n" }
        for index in 0..<2_751 { csv += "character_\(index),4\n" }
        let labels = directory.appendingPathComponent("labels.csv")
        try csv.write(to: labels, atomically: true, encoding: .utf8)
        let vocabulary = try TagVocabulary.load(from: labels)
        let payloads = (0..<128).map { variant in
            var scores = [Float](repeating: 0, count: TagVocabulary.headCount)
            scores[0] = 0.9
            for head in 0..<32 { scores[4 + (head + variant) % 256] = 0.8 }
            return TagProbabilityCodec.encode(scores)!
        }
        let counts = ProcessInfo.processInfo.environment["WAIFU_INVERSE_BENCH_COUNTS"]?
            .split(separator: ",").compactMap { Int($0) } ?? [1_000, 10_000, 100_000]
        let clock = ContinuousClock()
        func ms(_ start: ContinuousClock.Instant) -> Double {
            let c = start.duration(to: clock.now).components
            return Double(c.seconds) * 1_000 + Double(c.attoseconds) / 1e15
        }
        print("Inverse tag generation, optimized production code; 8,106 general tags; 32 retained general scores/photo")
        print("Timings include synchronous generateUnwantedTagGroup() and its triggered work. Synthetic fixtures; no UI interaction.")
        for count in counts {
            let frames = (0..<count).map { index in
                var frame = AnalyzedFrame(sampleIndex: index, timeSeconds: Double(index), artScore: 1,
                                          faceScore: nil, faceCount: 0, faceAreaFraction: nil, featurePrint: nil)
                frame.tagProbabilities = payloads[index % payloads.count]
                return frame
            }
            let session = WaifuSession(settingsURL: directory.appendingPathComponent("settings-\(count).json"), vocabulary: vocabulary)
            session.frameOrder = .chronological
            session.groupingEnabled = false
            var settings = session.settings
            settings.tagTopCount = 99
            session.settings = settings
            session.updateAnalyzedFrames(frames)
            let original = session.displayedFrames.map(\.id)
            while !session.canGenerateUnwantedTagGroup {
                try await Task.sleep(for: .milliseconds(10))
            }
            let revision = session.gridRevision
            let start = clock.now
            session.generateUnwantedTagGroup()
            let elapsed = ms(start)
            precondition(session.displayedFrames.map(\.id) == original)
            precondition(session.gridRevision == revision)
            precondition(session.tagFilter.unwantedGroups.count == 1)
            print(String(format: "%d photos: generation %.3f ms; inverse %d tags; original %d photos unchanged",
                         count, elapsed, session.tagFilter.unwantedGroupTags.count, original.count))
        }
    }
}
