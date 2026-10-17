import CoreGraphics
import Foundation

private func streamingPosition(_ data: Data) -> Double? {
    data.first.map(Double.init)
}

private let streamingDistance: @Sendable (Data, Data) -> Double? = { lhs, rhs in
    guard let a = streamingPosition(lhs), let b = streamingPosition(rhs) else { return nil }
    return abs(a - b) * 0.1
}

private func streamingReferenceFrames() -> [AnalyzedFrame] {
    var frames: [AnalyzedFrame] = []
    for index in 0..<180 {
        let time = Double(index / 60) * 180 + Double(index % 60) * 2
        let printData: Data? = index % 13 == 0 ? nil : Data([UInt8((index * 17) % 31)])
        frames.append(AnalyzedFrame(sampleIndex: index * 30, timeSeconds: time,
                                    artScore: 0, faceScore: nil, faceCount: 0,
                                    faceAreaFraction: nil, featurePrint: printData))
    }
    return frames
}

private func testStreamingChainsMatchBatchGrouping() throws {
    let frames = streamingReferenceFrames()
    let thresholds: [Float] = [0.01, 0.1, 0.3, 0.8]
    for threshold in thresholds {
        var settings = GroupingSettings.default
        settings.similarityThreshold = threshold
        settings.maximumGroupSize = 7
        let incremental = IncrementalFrameGrouper(settings: settings, distance: streamingDistance)
        for frame in frames { try incremental.append(frame) }
        let groups = try incremental.finish()
        let grouper = FrameGrouper(distance: streamingDistance)
        let reference = try grouper.groupCancellable(frames.reversed(), settings: settings)
        checkEqual(groups, reference, "same graph components, stable IDs and contiguous splitting at \(threshold)")
        checkEqual(incremental.processed, frames.count, "missing prints still count as processed")
        checkEqual(incremental.retainedPrintCount, 0, "EOF releases active prints")
    }
}

func registerIncrementalGroupingTests() {
    let suite = "incremental grouping"
    Registry.shared.add(suite: suite, TestCase(name: "live group snapshots preserve comparisons with subsequent frames", knownBug: nil) {
        let settings = GroupingSettings.default
        let frames = streamingReferenceFrames()
        let incremental = IncrementalFrameGrouper(settings: settings, distance: streamingDistance)
        for (index, frame) in frames.enumerated() {
            try incremental.append(frame)
            if index % 7 == 0 {
                let retained = incremental.retainedPrintCount
                _ = try incremental.snapshot()
                checkEqual(incremental.retainedPrintCount, retained, "live publication keeps the active window")
            }
        }
        let reference = try FrameGrouper(distance: streamingDistance).groupCancellable(frames, settings: settings)
        checkEqual(try incremental.finish(), reference, "live snapshots do not alter final grouping")
    })

    Registry.shared.add(suite: suite, TestCase(name: "streamed chains, windows, missing prints and splits match batch grouping",
                                               knownBug: nil, body: { try testStreamingChainsMatchBatchGrouping() }))
    Registry.shared.add(suite: suite, TestCase(name: "groups advance before input ends and old prints expire at the window boundary", knownBug: nil) {
        let grouping = IncrementalFrameGrouper(settings: .default, distance: { _, _ in 0 })
        for index in 0..<125 {
            let frame = AnalyzedFrame(sampleIndex: index, timeSeconds: Double(index), artScore: 0,
                                      faceScore: nil, faceCount: 0, faceAreaFraction: nil, featurePrint: Data([1]))
            try grouping.append(frame)
            if index == 1 { checkEqual(grouping.linkedGroups, 1, "first group exists while later frames have yet to arrive") }
            check(grouping.retainedPrintCount <= 121, "decoded storage never retains expired frames")
        }
        try grouping.append(AnalyzedFrame(sampleIndex: 125, timeSeconds: 400, artScore: 0,
                                           faceScore: nil, faceCount: 0, faceAreaFraction: nil, featurePrint: nil))
        checkEqual(grouping.retainedPrintCount, 0, "even a missing-print frame expires prior observations")
        let result = try grouping.finish()
        checkEqual(Set(result.flatMap(\.memberIDs)).count, 125, "expiry preserves historical component membership")
    })
    Registry.shared.add(suite: suite, TestCase(name: "incremental grouping rejects out-of-order input and observes cancellation", knownBug: nil) {
        let grouping = IncrementalFrameGrouper(settings: .default, distance: streamingDistance)
        let first = AnalyzedFrame(sampleIndex: 0, timeSeconds: 10, artScore: 0, faceScore: nil,
                                  faceCount: 0, faceAreaFraction: nil, featurePrint: Data([1]))
        try grouping.append(first)
        let earlier = AnalyzedFrame(sampleIndex: 1, timeSeconds: 5, artScore: 0, faceScore: nil,
                                    faceCount: 0, faceAreaFraction: nil, featurePrint: Data([1]))
        do { try grouping.append(earlier); check(false, "a broken stream must not silently drop comparison edges") }
        catch IncrementalFrameGrouper.InputError.outOfOrder { }
        let task = Task.detached {
            let cancelled = IncrementalFrameGrouper(settings: .default)
            withUnsafeCurrentTask { $0?.cancel() }
            try cancelled.append(first)
        }
        do { try await task.value; check(false, "cancelled append must throw") }
        catch is CancellationError { }
    })
    Registry.shared.add(suite: suite, TestCase(name: "grouping runs while later scoring workers are still waiting", knownBug: nil) {
        let gate = GroupingScoreGate()
        let pipeline = FrameScorePipeline(scorer: GatedGroupingScorer(gate: gate),
                                          thumbnailStore: GroupingDiscardWriter(), analysisWidth: 16, workerCount: 2)
        let image = try ImageFixture.cgImage(width: 16, height: 16, gray: 0.5)
        let source = AsyncThrowingStream<SampledFrame, Error> { continuation in
            for index in 0..<6 {
                continuation.yield(SampledFrame(sampleIndex: index, timeSeconds: Double(index), image: image))
            }
            continuation.finish()
        }
        let grouping = IncrementalFrameGrouper(settings: .default, distance: { _, _ in 0 })
        for try await frame in pipeline.run(frames: source, minimumFaceAreaFraction: 0.01, onProgress: { _ in }) {
            try grouping.append(frame)
            if grouping.processed == 2 {
                checkEqual(grouping.linkedGroups, 1, "group exists before later frames can finish scoring")
                await gate.open()
            }
        }
        checkEqual(grouping.processed, 6, "all ordered scores consumed")
        checkEqual(try grouping.finish().first?.memberIDs.count, 6, "one complete final group")
    })
}

private actor GroupingScoreGate {
    private var opened = false
    private var waiting: [CheckedContinuation<Void, Never>] = []
    func wait() async {
        if opened { return }
        await withCheckedContinuation { waiting.append($0) }
    }
    func open() {
        opened = true
        for continuation in waiting { continuation.resume() }
        waiting.removeAll()
    }
}

private struct GatedGroupingScorer: FrameScoring {
    let gate: GroupingScoreGate
    func score(_ frame: SampledFrame) async throws -> FrameScoreResult {
        if frame.sampleIndex >= 2 { await gate.wait() }
        return FrameScoreResult(artScore: 0, faceScore: nil, faceCount: 0, faceAreaFraction: nil, featurePrint: Data([1]))
    }
}

private struct GroupingDiscardWriter: ThumbnailWriter {
    func write(_ image: CGImage, key: String) async throws { }
}
