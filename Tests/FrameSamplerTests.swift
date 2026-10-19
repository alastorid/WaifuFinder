import AVFoundation
import Foundation

// Sampling a clip, one frame at a time and only when asked.
//
// `AVFoundationFrameSampler.sample` returns a stream built with
// `AsyncThrowingStream(unfolding:)`, so a frame is decoded when the consumer
// pulls one and not before. That is a memory property first — a push-based
// stream buffers without bound, which is what made a long video cost
// gigabytes — and these cases are the observable half of it: the stream
// starts at the clip's first frame, it is contiguous and ascending, and a
// consumer that takes two frames gets two.
//
// The pass is frame by frame: every frame the file holds is processed, and
// the count case pins that against the probe's own estimate (`duration ×
// frameRate`), so a sampler that silently skipped frames — the grid the old
// per-second plan sampled on, for instance — fails here even though every
// frame it *did* deliver was well-formed.
//
// These cases use real containers generated and decoded with Apple frameworks.

func registerFrameSamplerTests() {
    let suite = "frame sampling"

    Registry.shared.add(suite: suite, TestCase(
        name: "native-frame concurrency scales down for large sources and small memory budgets", knownBug: nil) {
        let normal = FrameProcessingConcurrency(width: 1920, height: 1080,
                                               processorCount: 8, physicalMemory: 16 * 1024 * 1024 * 1024,
                                               performanceCoreCount: 4)
        checkEqual(normal.preparationWorkers, 4, "ordinary video can prepare four frames concurrently")
        checkEqual(normal.scoringWorkers, 4, "ordinary video retains four scoring workers alongside preparation")
        let large = FrameProcessingConcurrency(width: 7680, height: 4320,
                                              processorCount: 8, physicalMemory: 16 * 1024 * 1024 * 1024,
                                              performanceCoreCount: 4)
        check(large.preparationWorkers + large.scoringWorkers < normal.preparationWorkers + normal.scoringWorkers,
              "8K inputs reduce native image residency")
        let small = FrameProcessingConcurrency(width: 3840, height: 2160,
                                              processorCount: 8, physicalMemory: 4 * 1024 * 1024 * 1024,
                                              performanceCoreCount: 4)
        checkEqual(small.preparationWorkers, 1, "a tight memory budget avoids parallel native rasters")
        checkEqual(small.scoringWorkers, 1, "one scoring slot still permits progress under a tight budget")
        let single = FrameProcessingConcurrency(width: 1920, height: 1080,
                                               processorCount: 1, physicalMemory: 16 * 1024 * 1024 * 1024,
                                               performanceCoreCount: 4)
        checkEqual(single.preparationWorkers, 1, "one processor avoids parallel preparation")
        checkEqual(single.scoringWorkers, 1, "one processor avoids oversubscribing CPU scoring")
        let enlarged = FrameProcessingConcurrency(width: 1920, height: 1080, analysisWidth: 4096,
                                                 processorCount: 8, physicalMemory: 16 * 1024 * 1024 * 1024,
                                                 performanceCoreCount: 4)
        check(enlarged.preparationWorkers + enlarged.scoringWorkers < normal.preparationWorkers + normal.scoringWorkers,
              "large analysis rasters count toward the same working budget")
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "cancelling a consumer stops an active preparation window without emitting extra frames", knownBug: nil) {
        let directory = try Temporary.directory("sampler-cancellation")
        let clip = try VideoFixture.makeMP4(in: directory, durationSeconds: 3,
                                          width: 320, height: 240, frameRate: 30)
        let ticks = TickLog()
        let consumer = Task {
            var count = 0
            for try await _ in AVFoundationFrameSampler().sample(
                clip, analysisWidth: 64, mode: .full, onProgress: { ticks.record($0) }
            ) {
                count += 1
                if count == 2 { withUnsafeCurrentTask { $0?.cancel() } }
            }
            return count
        }
        switch await consumer.result {
        case .success(let count): checkEqual(count, 2, "a cancelled stream may finish normally but must stop")
        case .failure(let error): check(error is CancellationError, "cancellation retains its error type")
        }
        checkEqual(ticks.all, [1, 2], "prepared frames do not escape after cancellation")
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "a clip is processed frame by frame, from its first frame, ascending and contiguous", knownBug: nil) {
        let directory = try Temporary.directory("sampler")
        let clip = try VideoFixture.makeMP4(in: directory, durationSeconds: 3,
                                            width: 320, height: 240, frameRate: 30)
        let info = try await VideoProbe.probe(clip)

        var frames: [SampledFrame] = []
        for try await frame in AVFoundationFrameSampler().sample(
            clip, analysisWidth: 64, mode: .full, onProgress: { _ in }
        ) {
            frames.append(frame)
        }

        checkNotNil(frames.first, "a three-second clip yields at least one frame")
        guard let first = frames.first else { return }

        checkEqual(frames.map(\.sampleIndex), Array(0..<frames.count),
                   "sample indices are contiguous from zero — no frame is skipped")
        let times = frames.map(\.timeSeconds)
        checkEqual(times, times.sorted(), "timestamps ascend, because the stream is one pass through the file")
        check(first.timeSeconds < 0.1,
              "the first frame is the clip's first frame, not one step in — got \(first.timeSeconds)s")
        checkEqual(first.image.width, 64, "and it is decoded at the analysis width, not the source width")
        check(abs(frames.count - info.estimatedFrameCount) <= 2,
              "every frame of the clip is processed: \(frames.count) frames against the probe's estimate of \(info.estimatedFrameCount)")
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "frames are produced on demand, so a partial read is a partial decode", knownBug: nil) {
        // The point is that the stream does not demand to be drained: two frames
        // taken is two frames produced. Under the old push-based producer the whole
        // clip was decoded and buffered regardless of how much of it was used, and
        // that buffering is what the memory budget could not afford.
        let directory = try Temporary.directory("sampler-pull")
        let clip = try VideoFixture.makeMP4(in: directory, durationSeconds: 4,
                                            width: 320, height: 240, frameRate: 30)

        var taken: [SampledFrame] = []
        for try await frame in AVFoundationFrameSampler().sample(
            clip, analysisWidth: 64, mode: .full, onProgress: { _ in }
        ) {
            taken.append(frame)
            if taken.count == 2 { break }
        }

        checkEqual(taken.count, 2, "exactly what was asked for")
        checkEqual(taken.map(\.sampleIndex), [0, 1], "and it is the head of the stream")
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "progress counts the frames emitted, in order", knownBug: nil) {
        let directory = try Temporary.directory("sampler-progress")
        let clip = try VideoFixture.makeMP4(in: directory, durationSeconds: 3,
                                            width: 320, height: 240, frameRate: 30)

        let ticks = TickLog()
        var frames: [SampledFrame] = []
        for try await frame in AVFoundationFrameSampler().sample(
            clip, analysisWidth: 64, mode: .full, onProgress: { ticks.record($0) }
        ) {
            frames.append(frame)
        }

        checkEqual(ticks.all, Array(1...frames.count),
                   "one tick per emitted frame, counting up from one")
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "a quick pass analyses one frame a second, keeping decoded ordinals", knownBug: nil) {
        let directory = try Temporary.directory("sampler-quick")
        let clip = try VideoFixture.makeMP4(in: directory, durationSeconds: 3,
                                            width: 320, height: 240, frameRate: 30)
        let info = try await VideoProbe.probe(clip)

        var frames: [SampledFrame] = []
        for try await frame in AVFoundationFrameSampler().sample(
            clip, analysisWidth: 64, mode: .quick, onProgress: { _ in }
        ) {
            frames.append(frame)
        }

        checkNotNil(frames.first, "a three-second clip yields at least one frame")
        guard let first = frames.first else { return }

        check(first.timeSeconds < 0.1,
              "the first frame is the clip's first frame, not one step in — got \(first.timeSeconds)s")
        for frame in frames {
            check(frame.sampleIndex % 30 == 0,
                  "a quick pass keeps one frame every 30 decoded frames — index \(frame.sampleIndex)")
        }
        let times = frames.map(\.timeSeconds)
        checkEqual(times, times.sorted(), "timestamps ascend, because the stream is one pass through the file")
        let expected = max(1, info.estimatedFrameCount / 30)
        check(abs(frames.count - expected) <= 1,
              "a quick pass analyses roughly one frame a second: \(frames.count) frames against \(expected)")
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "a keyframe pass keeps the encoder's sync samples and nothing else", knownBug: nil) {
        let gop = 15
        let directory = try Temporary.directory("sampler-keyframe")
        let clip = try VideoFixture.makeFixedGOPMP4(in: directory, durationSeconds: 3,
                                                    width: 320, height: 240, frameRate: 30, gop: gop)

        var frames: [SampledFrame] = []
        for try await frame in AVFoundationFrameSampler().sample(
            clip, analysisWidth: 64, mode: .keyframe, onProgress: { _ in }
        ) {
            frames.append(frame)
        }

        checkEqual(frames.count, 6, "a 3-second 30fps clip with a 15-frame GOP holds 6 keyframes")
        for frame in frames {
            check(frame.sampleIndex % gop == 0,
                  "a keyframe pass keeps only sync samples — index \(frame.sampleIndex) is not on the \(gop)-frame GOP")
        }
        let times = frames.map(\.timeSeconds)
        checkEqual(times, times.sorted(), "keyframes ascend, because the stream is one pass through the file")

        // The same clip in full mode is the regression guard: the sync-sample
        // filter must be invisible to the two modes that never asked for it.
        var all: [SampledFrame] = []
        for try await frame in AVFoundationFrameSampler().sample(
            clip, analysisWidth: 64, mode: .full, onProgress: { _ in }
        ) {
            all.append(frame)
        }
        check(all.count > frames.count * 5,
              "full still takes every frame — \(all.count) against \(frames.count) keyframes")
        checkEqual(all.count, 90, "full on a 3-second 30fps clip is all 90 frames")
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "the mode's frame step and analysed count", knownBug: nil) {
        checkEqual(AnalysisMode.full.frameStep(frameRate: 29.97), 1,
                   "the full pass keeps every frame")
        checkEqual(AnalysisMode.quick.frameStep(frameRate: 29.97), 30,
                   "a quick pass at 29.97 fps keeps every 30th frame")
        checkEqual(AnalysisMode.quick.frameStep(frameRate: 0), 1,
                   "an unknown rate cannot become a step of zero")
        checkEqual(AnalysisMode.full.analysedFrameCount(frameCount: 84_067, frameRate: 29.97), 84_067,
                   "the full pass analyses every frame")
        checkEqual(AnalysisMode.quick.analysedFrameCount(frameCount: 84_067, frameRate: 29.97), 2_803,
                   "the 47-minute reference analyses 2,803 frames, one a second")
        checkEqual(AnalysisMode.quick.analysedFrameCount(frameCount: 0, frameRate: 30), 1,
                   "the analysed count floors at one frame")
        checkEqual(AnalysisMode.keyframe.frameStep(frameRate: 29.97), 1,
                   "a keyframe pass has no ordinal grid — its step is one and the sync flag decides")
        checkEqual(AnalysisMode.keyframe.analysedFrameCount(frameCount: 84_067, frameRate: 29.97), 1,
                   "with no duration the keyframe estimate floors at one frame")
        checkEqual(AnalysisMode.keyframe.analysedFrameCount(frameCount: 84_067, frameRate: 29.97,
                                                            durationSeconds: 2_820), 1_410,
                   "the 47-minute reference seeds 1,410 frames at a two-second GOP")
    })
}
