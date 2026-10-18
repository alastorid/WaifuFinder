import Foundation
import Metal

private func summaryVocabulary(firstTag: String = "tag_0") throws -> TagVocabulary {
    let directory = try Temporary.directory("tag-summary-labels")
    let file = directory.appendingPathComponent("labels.csv")
    var csv = "name,category\n"
    for rating in TagRating.allCases { csv += "\(rating.rawValue),9\n" }
    for index in 0..<TagVocabulary.generalCount { csv += "\(index == 0 ? firstTag : "tag_\(index)"),0\n" }
    for index in 0..<2_751 { csv += "character_\(index),4\n" }
    try csv.write(to: file, atomically: true, encoding: .utf8)
    return try TagVocabulary.load(from: file)
}

private func summaryFrame(_ id: Int, _ scores: [Int: Float]) -> AnalyzedFrame {
    var probabilities = [Float](repeating: 0, count: TagVocabulary.headCount)
    probabilities[0] = 0.95
    for (index, value) in scores { probabilities[index] = value }
    var frame = FrameFixture.make(id: String(format: "f%06d", id), time: Double(id), artScore: 1)
    frame.tagProbabilities = TagProbabilityCodec.encode(probabilities)
    frame.tagRating = .general
    return frame
}

func registerTagSummaryTests() {
    let suite = "frame arrangement: tag summary"

    Registry.shared.add(suite: suite, TestCase(name: "means include only values strictly above threshold and preserve legacy membership", knownBug: nil) {
        let vocabulary = try summaryVocabulary()
        let frames = [summaryFrame(0, [4: 0.9, 5: 0.8, 6: 0.5, 7: 0.7]),
                      summaryFrame(1, [4: 0.5, 5: 0.8, 7: 0.7]),
                      summaryFrame(2, [4: 0.1, 7: 0.7])]
        let computer = TagSummaryComputer()
        let result = try await computer.summarize(frames, vocabulary: vocabulary, threshold: 0.5, backend: .cpu)
        let byIndex = Dictionary(uniqueKeysWithValues: result.tags.map { ($0.index, $0) })
        checkEqual(byIndex[4]?.matchingFrames, 1, "only 0.9 crosses 0.5")
        checkEqual(byIndex[4]?.scoredFrames, 1, "a low probability never dilutes the mean")
        checkClose(byIndex[4]?.mean ?? -1, 0.9, tolerance: 1e-6, "conditional mean")
        checkNil(byIndex[6], "a probability equal to threshold is excluded")
        checkEqual(result.labels(using: vocabulary).common, ["tag 3"], "common requires every displayed frame")
        checkEqual(result.labels(using: vocabulary).ranked, ["tag 0", "tag 1", "tag 3"], "rank by conditional confidence")

        var legacy = FrameFixture.make(id: "f000003", time: 3, artScore: 1)
        legacy.tagProbabilities = Data("corrupt".utf8)
        legacy.topTags = ["tag_0", "tag_3", "tag_9"]
        let mixed = try await computer.summarize(frames + [legacy], vocabulary: vocabulary, threshold: 0.5, backend: .cpu)
        let mixedTags = Dictionary(uniqueKeysWithValues: mixed.tags.map { ($0.index, $0) })
        checkEqual(mixedTags[4]?.matchingFrames, 2, "legacy tags contribute presence")
        checkEqual(mixedTags[4]?.scoredFrames, 1, "legacy tags do not invent confidence")
        checkNil(mixedTags[13]?.mean, "a legacy-only tag has no numeric mean")
        checkEqual(mixed.labels(using: vocabulary).common, ["tag 3"], "legacy common tags remain available")
    })

    Registry.shared.add(suite: suite, TestCase(name: "Metal and CPU agree across chunks, missing rows and threshold boundaries", knownBug: nil) {
        guard MTLCreateSystemDefaultDevice()?.hasUnifiedMemory == true else { throw SkipTest(reason: "Apple Silicon Metal unavailable") }
        let metal = try MetalTagSummary()
        let vocabulary = try summaryVocabulary()
        var frames = (0..<133).map { row in
            summaryFrame(row, [4: Float(row % 101) / 101, 5: 0.35, 6: Float(0.35).nextUp,
                               7: Float(0.35).nextDown, 8: 0.95])
        }
        var legacy = FrameFixture.make(id: "f000133", time: 133, artScore: 1)
        legacy.topTags = ["tag_0", "tag_4"]
        frames.append(legacy)
        frames.append(FrameFixture.make(id: "f000134", time: 134, artScore: 1))
        let computer = TagSummaryComputer(chunkByteLimit: vocabulary.generalIndices.count * 4 * 17,
                                          metalProvider: { metal })
        let cpu = try await computer.summarize(frames, vocabulary: vocabulary, threshold: 0.35, backend: .cpu)
        let gpu = try await computer.summarize(frames, vocabulary: vocabulary, threshold: 0.35, backend: .metal)
        check(gpu.gpuChunks > 1, "the real Metal reducer processes multiple chunks")
        checkEqual(gpu.tags.map(\.index), cpu.tags.map(\.index), "threshold membership is exact")
        for (left, right) in zip(cpu.tags, gpu.tags) {
            checkEqual(left.matchingFrames, right.matchingFrames, "presence count for \(left.index)")
            checkEqual(left.scoredFrames, right.scoredFrames, "scored count for \(left.index)")
            checkClose(left.mean ?? -1, right.mean ?? -1, tolerance: 1e-6, "mean for \(left.index)")
        }
        checkEqual(gpu.labels(using: vocabulary).ranked, cpu.labels(using: vocabulary).ranked, "GPU preserves confidence ordering")
        checkEqual(await computer.decodedVectorCount, 133, "GPU reuses decoded vectors from the CPU pass")
        _ = try await computer.summarize(Array(frames.reversed()), vocabulary: vocabulary, threshold: 0.8, backend: .metal)
        checkEqual(await computer.uploadedVectorCount, 134, "reordering and threshold changes reuse Metal vectors, including the legacy row")
    })

    Registry.shared.add(suite: suite, TestCase(name: "Metal cache masks hidden rows and refreshes reused slots with bounded overflow", knownBug: nil) {
        guard MTLCreateSystemDefaultDevice()?.hasUnifiedMemory == true else { throw SkipTest(reason: "Apple Silicon Metal unavailable") }
        let vocabulary = try summaryVocabulary()
        let metal = try MetalTagSummary()
        let rowBytes = vocabulary.generalIndices.count * 4
        let computer = TagSummaryComputer(cacheByteLimit: rowBytes * 3, chunkByteLimit: rowBytes * 2, metalProvider: { metal })
        let reference = TagSummaryComputer(cacheByteLimit: 0)
        var frames = (0..<7).map { summaryFrame($0, [4: Float($0 + 1) / 10, 5: 0.9]) }
        _ = try await computer.summarize(frames, vocabulary: vocabulary, threshold: 0.35, backend: .metal)
        let subset = [frames[1], frames[4], frames[6]]
        let cpu = try await reference.summarize(subset, vocabulary: vocabulary, threshold: 0.35, backend: .cpu)
        let gpu = try await computer.summarize(subset, vocabulary: vocabulary, threshold: 0.35, backend: .metal)
        checkEqual(gpu.tags.map(\.matchingFrames), cpu.tags.map(\.matchingFrames), "hidden cached rows contribute nothing")
        for (left, right) in zip(cpu.tags, gpu.tags) {
            checkClose(left.mean ?? -1, right.mean ?? -1, tolerance: 1e-6, "filtered mean")
        }
        checkEqual(await computer.uploadedVectorCount, 3, "only the bounded cached rows are kept on Metal")
        frames[4] = summaryFrame(4, [4: 0.99, 5: 0.1])
        let remaining = Array(frames.suffix(4))
        let expected = try await reference.summarize(remaining, vocabulary: vocabulary, threshold: 0.35, backend: .cpu)
        let replaced = try await computer.summarize(remaining, vocabulary: vocabulary, threshold: 0.35,
                                                    retaining: Set(remaining.map(\.id)), backend: .metal)
        checkEqual(replaced.tags.map(\.matchingFrames), expected.tags.map(\.matchingFrames), "pruned slots contain the new source photos")
        for (left, right) in zip(expected.tags, replaced.tags) {
            checkClose(left.mean ?? -1, right.mean ?? -1, tolerance: 1e-6, "reused slots replace old probabilities")
        }
        var edited = remaining
        edited[1] = summaryFrame(4, [4: 0.4, 5: 0.99])
        let updated = try await computer.summarize(edited, vocabulary: vocabulary, threshold: 0.35, backend: .metal)
        let expectedUpdate = try await reference.summarize(edited, vocabulary: vocabulary, threshold: 0.35, backend: .cpu)
        checkEqual(updated.tags.map(\.matchingFrames), expectedUpdate.tags.map(\.matchingFrames), "changed probabilities refresh an existing GPU slot")
        for (left, right) in zip(expectedUpdate.tags, updated.tags) {
            checkClose(left.mean ?? -1, right.mean ?? -1, tolerance: 1e-6, "updated GPU mean")
        }
    })

    Registry.shared.add(suite: suite, TestCase(name: "automatic reduction falls back when Metal is unavailable", knownBug: nil) {
        let vocabulary = try summaryVocabulary()
        let frames = (0..<1_536).map { summaryFrame($0, [4: 0.8, 5: $0 % 2 == 0 ? 0.9 : 0.1]) }
        let attempts = TickLog()
        let computer = TagSummaryComputer(metalProvider: { attempts.record(1); return nil })
        let automatic = try await computer.summarize(frames, vocabulary: vocabulary, threshold: 0.35)
        let cpu = try await computer.summarize(frames, vocabulary: vocabulary, threshold: 0.35, backend: .cpu)
        checkEqual(automatic.gpuChunks, 0, "no unavailable GPU is used")
        checkEqual(attempts.all, [1], "a GPU-eligible selection actually attempts Metal before falling back")
        checkEqual(automatic.labels(using: vocabulary).ranked, cpu.labels(using: vocabulary).ranked, "fallback uses the same conditional average")
        checkEqual(automatic.labels(using: vocabulary).common, cpu.labels(using: vocabulary).common, "fallback preserves common tags")
    })

    Registry.shared.add(suite: suite, TestCase(name: "vector cache survives ordering, thresholds and derived tag changes but refreshes edited probabilities", knownBug: nil) {
        let vocabulary = try summaryVocabulary()
        let computer = TagSummaryComputer()
        var frames = [summaryFrame(0, [4: 0.6]), summaryFrame(1, [4: 0.9])]
        _ = try await computer.summarize(frames, vocabulary: vocabulary, threshold: 0.35, backend: .cpu)
        frames[0].topTags = ["derived tag changed"]
        let high = try await computer.summarize(Array(frames.reversed()), vocabulary: vocabulary, threshold: 0.8, backend: .cpu)
        checkEqual(await computer.decodedVectorCount, 2, "no repeated decoding after reorder or threshold change")
        checkEqual(high.tags.first?.scoredFrames, 1, "a new threshold still updates the result")
        frames[0] = summaryFrame(0, [4: 0.99])
        let edited = try await computer.summarize(frames, vocabulary: vocabulary, threshold: 0.8, backend: .cpu)
        checkEqual(await computer.decodedVectorCount, 3, "only changed probabilities are decoded")
        checkEqual(edited.tags.first?.scoredFrames, 2, "the changed row replaces the old scores")
        _ = try await computer.summarize([], vocabulary: vocabulary, threshold: 0.8, retaining: [], backend: .cpu)
        _ = try await computer.summarize(frames, vocabulary: vocabulary, threshold: 0.8, backend: .cpu)
        checkEqual(await computer.decodedVectorCount, 5, "released source photos do not remain cached")
        let renamed = try summaryVocabulary(firstTag: "changed_tag")
        let newVocabulary = try await computer.summarize(frames, vocabulary: renamed, threshold: 0.8, backend: .cpu)
        checkEqual(await computer.decodedVectorCount, 7, "another vocabulary invalidates decoded head indices")
        checkEqual(newVocabulary.labels(using: renamed).common, ["changed tag"], "new labels describe the refreshed vectors")
        let uncached = TagSummaryComputer(cacheByteLimit: 0, chunkByteLimit: 4)
        let bounded = try await uncached.summarize(frames, vocabulary: vocabulary, threshold: 0.8, backend: .cpu)
        checkEqual(bounded.tags.first?.mean, edited.tags.first?.mean, "bounded storage retains the same result")
        _ = try await uncached.summarize(frames, vocabulary: vocabulary, threshold: 0.8, backend: .cpu)
        checkEqual(await uncached.decodedVectorCount, 4, "zero cache budget retains no vectors")
    })

    Registry.shared.add(suite: suite, TestCase(name: "canceled summaries stop before decoding", knownBug: nil) {
        let vocabulary = try summaryVocabulary()
        let computer = TagSummaryComputer()
        let frames = [summaryFrame(0, [4: 0.9])]
        let task = Task.detached { () -> Bool in
            withUnsafeCurrentTask { $0?.cancel() }
            do {
                _ = try await computer.summarize(frames, vocabulary: vocabulary, threshold: 0.35)
                return false
            } catch is CancellationError { return true }
            catch { return false }
        }
        check(await task.value, "obsolete work exits through cancellation")
        checkEqual(await computer.decodedVectorCount, 0, "cancellation avoids decoding")
    })

    Registry.shared.add(suite: suite, TestCase(name: "invalid thresholds and duplicate photo IDs are rejected", knownBug: nil) {
        let vocabulary = try summaryVocabulary()
        let computer = TagSummaryComputer()
        let frame = summaryFrame(0, [4: 0.9])
        for threshold: Float in [.nan, -.infinity, -0.1, 1.1] {
            do {
                _ = try await computer.summarize([frame], vocabulary: vocabulary, threshold: threshold)
                check(false, "invalid threshold must fail")
            } catch TagSummaryError.invalidInput { }
        }
        do {
            _ = try await computer.summarize([frame, frame], vocabulary: vocabulary, threshold: 0.35)
            check(false, "duplicates must not produce different CPU/GPU counts")
        } catch TagSummaryError.invalidInput { }
        checkEqual(await computer.decodedVectorCount, 0, "invalid input does not populate the cache")
    })

    Registry.shared.add(suite: suite, TestCase(name: "session publishes only current tag summaries without rebuilding the grid", knownBug: nil) {
        let vocabulary = try summaryVocabulary()
        let directory = try Temporary.directory("tag-summary-session")
        let frames = (0..<128).map { summaryFrame($0, [$0 < 64 ? 4 : 5: 0.9]) }
        try await Task { @MainActor in
            let session = WaifuSession(settingsURL: directory.appendingPathComponent("settings.json"), vocabulary: vocabulary)
            session.frameOrder = .only([.time])
            session.updateAnalyzedFrames(frames)
            session.frameLimits.time.high = 1
            session.frameLimits.time.high = nil
            session.frameLimits.time.low = 126
            let revision = session.gridRevision
            for _ in 0..<500 {
                if !session.isSummarizingTags { break }
                try await Task.sleep(for: .milliseconds(20))
            }
            check(!session.isSummarizingTags, "current summary completes")
            checkEqual(session.tagCommonTags, ["tag 1"], "obsolete jobs cannot restore a filtered-out tag")
            checkEqual(session.gridRevision, revision, "a summary publication does not rebuild the photo grid")
            session.settings.taggerThreshold = 1
            session.settings.taggerThreshold = 0.35
            check(!session.isSummarizingTags, "undoing a pending threshold reuses the completed result")
            try await Task.sleep(for: .milliseconds(100))
            checkEqual(session.tagCommonTags, ["tag 1"], "a canceled threshold cannot overwrite the restored summary")
            session.frameOrder.toggleDirection(of: .time)
            check(!session.isSummarizingTags, "another display order reuses the completed summary")
            checkEqual(session.tagCommonTags, ["tag 1"], "ordering does not invalidate tag vectors or summaries")
            session.settings.taggerThreshold = 1
            for _ in 0..<500 {
                if !session.isSummarizingTags { break }
                try await Task.sleep(for: .milliseconds(20))
            }
            checkEqual(session.tagCommonTags, [], "threshold changes update cached vectors' summary")
            checkEqual(session.tagFacetTags, [], "no confidence can cross a threshold of one")
            session.updateAnalyzedFrames(frames)
            session.updateAnalyzedFrames([])
            try await Task.sleep(for: .milliseconds(50))
            check(!session.isSummarizingTags, "clearing the album cancels pending work")
            checkEqual(session.tagCommonTags, [], "obsolete work cannot restore cleared tags")
        }.value
    })
}
