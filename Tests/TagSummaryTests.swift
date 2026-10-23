import Foundation
import CoreServices
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

    Registry.shared.add(suite: suite, TestCase(name: "Photos keywords use live threshold, all qualifying tags and rating first", knownBug: nil) {
        let vocabulary = try summaryVocabulary()
        var scores = Dictionary(uniqueKeysWithValues: (4..<24).map { ($0, Float(0.75)) })
        scores[24] = 0.5
        scores[25] = 0.25
        scores[26] = 0.875
        scores[4 + TagVocabulary.generalCount] = 0.75
        var frame = summaryFrame(0, scores)
        frame.tagRating = .explicit
        frame.topTags = ["stale top tag"]
        let keywords = try PhotosKeywords.keywords(for: frame, vocabulary: vocabulary)
        checkEqual(keywords.first, "explicit", "rating is a plain keyword at the start")
        checkEqual(keywords.count, 24, "includes general and character tags beyond the top 15")
        checkEqual(keywords[1], "tag 22", "other keywords follow confidence order")
        check(!keywords.contains("stale top tag"), "never uses cached top tags")
        let raised = try PhotosKeywords.keywords(for: frame, vocabulary: vocabulary, threshold: 0.75)
        checkEqual(raised, ["explicit", "tag 22"], "strictly above the current threshold")
        checkEqual(try PhotosKeywords.keywords(for: frame, vocabulary: vocabulary, threshold: 1),
                   ["explicit"], "rating remains when no tags qualify")
    })

    Registry.shared.add(suite: suite, TestCase(name: "Photos keyword replacement rejects unavailable or mismatched scores", knownBug: nil) {
        let vocabulary = try summaryVocabulary()
        var frame = summaryFrame(0, [4: 0.75])
        frame.tagVocabularyFingerprint = vocabulary.cacheFingerprint &+ 1
        do {
            _ = try PhotosKeywords.keywords(for: frame, vocabulary: vocabulary)
            check(false, "mismatched vocabulary must not write keywords")
        } catch {}
        frame.tagVocabularyFingerprint = nil
        frame.tagProbabilities = nil
        do {
            _ = try PhotosKeywords.keywords(for: frame, vocabulary: vocabulary)
            check(false, "missing scores must not clear existing keywords")
        } catch {}
    })

    Registry.shared.add(suite: suite, TestCase(name: "Photos keyword event targets the exact photo and preserves plain text keywords", knownBug: nil) {
        let tags = ["explicit", "a,b", "quoted\"tag", "line\nbreak", "\\backslash", "猫"]
        let event = try PhotosKeywordWriter.event(for: .init(assetIdentifier: "asset\"\\id", keywords: tags))
        checkEqual(event.eventClass, AEEventClass(kAECoreSuite), "standard Apple event suite")
        checkEqual(event.eventID, AEEventID(kAESetData), "replaces the keyword property")
        let property = event.paramDescriptor(forKeyword: keyDirectObject)
        checkEqual(property?.descriptorType, DescType(typeObjectSpecifier), "property is an object specifier")
        checkEqual(property?.forKeyword(OSType(keyAEKeyData))?.typeCodeValue, PhotosKeywordWriter.keywordsProperty, "keywords property")
        let photo = property?.forKeyword(OSType(keyAEContainer))
        checkEqual(photo?.forKeyword(OSType(keyAEDesiredClass))?.typeCodeValue, PhotosKeywordWriter.mediaItemClass, "media item")
        checkEqual(photo?.forKeyword(OSType(keyAEKeyForm))?.enumCodeValue, OSType(formUniqueID), "uses stable ID, not an index")
        checkEqual(photo?.forKeyword(OSType(keyAEKeyData))?.stringValue, "asset\"\\id", "exact asset ID")
        let data = event.paramDescriptor(forKeyword: keyAEData)
        let values = (1...tags.count).compactMap { data?.atIndex($0)?.stringValue }
        checkEqual(values, tags, "text is passed directly, with no script escaping or delimiters")
    })

    Registry.shared.add(suite: suite, TestCase(name: "Photos keyword cancellation finishes the in-flight photo then stops", knownBug: nil) {
        let probe = KeywordWriteProbe(pauseFirst: true)
        let updates = (0..<3).map { PhotosKeywords.Update(assetIdentifier: "photo \($0)", keywords: ["general"]) }
        let task = Task {
            try await PhotosKeywordWriter.run(updates, write: { try await probe.write($0) },
                                              progress: { await probe.report($0) })
        }
        await probe.waitForFirstWrite()
        task.cancel()
        await probe.release()
        do {
            try await task.value
            check(false, "cancellation must stop the job")
        } catch is CancellationError {} catch { throw error }
        checkEqual(await probe.identifiers, ["photo 0"], "never starts the next photo")
        checkEqual(await probe.counts, [1], "confirms the photo that finished before stopping")
    })

    Registry.shared.add(suite: suite, TestCase(name: "Photos keyword progress counts only confirmed writes and stops on failure", knownBug: nil) {
        let probe = KeywordWriteProbe(failSecond: true)
        let updates = (0..<3).map { PhotosKeywords.Update(assetIdentifier: "photo \($0)", keywords: ["general"]) }
        do {
            try await PhotosKeywordWriter.run(updates, write: { try await probe.write($0) },
                                              progress: { await probe.report($0) })
            check(false, "a failed write must surface an error")
        } catch {}
        checkEqual(await probe.identifiers, ["photo 0", "photo 1"], "stops before later photos")
        checkEqual(await probe.counts, [1], "failed photo is not counted")
        let completed = KeywordWriteProbe()
        try await PhotosKeywordWriter.run(updates, write: { try await completed.write($0) },
                                          progress: { await completed.report($0) })
        checkEqual(await completed.counts, [1, 2, 3], "each confirmed photo advances progress")
    })

    Registry.shared.add(suite: suite, TestCase(name: "thumbnail Top reasons follow rankings, group caps and partial reveals", knownBug: nil) {
        let vocabulary = try summaryVocabulary()
        let group = FrameGroupID(raw: "top-badges")
        var frames = [summaryFrame(0, [4: 0.9, 5: 0.5]),
                      summaryFrame(1, [4: 0.8, 5: 0.8]),
                      summaryFrame(2, [4: 0.5, 5: 0.9])]
        for index in frames.indices { frames[index].groupID = group }
        let capped = FrameArrangement.arrange(frames, order: .chronological,
            perGroupTopCount: 1, grouping: .default, vocabulary: vocabulary, tagTopCount: 1)
        checkEqual(capped.topTagsByFrame, [frames[0].id: ["tag 0"]], "only visible winners receive badges")
        let revealed = FrameArrangement.arrange(frames, order: .chronological,
            perGroupTopCount: 1, revealedGroups: [group], grouping: .default,
            vocabulary: vocabulary, tagTopCount: 1)
        checkEqual(revealed.topTagsByFrame, [frames[0].id: ["tag 0"], frames[2].id: ["tag 1"]],
                   "partial reveal retains the exact ranking reason for each restored winner")
        check(revealed.topTagsByFrame[frames[1].id] == nil, "a high-scoring nonwinner never receives a badge")
        let high = FrameArrangement.arrange(frames, order: .chronological,
            perGroupTopCount: 33, grouping: .default, tagFilter: TagFilter(threshold: 0.95),
            vocabulary: vocabulary, tagTopCount: 1)
        check(high.topTagsByFrame.isEmpty, "threshold changes remove obsolete reasons")
        let shared = TagFilter().topFrameSelection([summaryFrame(3, [4: 0.9, 5: 0.9])],
                                                  count: 1, vocabulary: vocabulary)
        checkEqual(shared.tagsByFrame[FrameID(sampleIndex: 3)], ["tag 0", "tag 1"],
                   "a frame can explain multiple independent winning tags")
    })

    Registry.shared.add(suite: suite, TestCase(name: "Advanced mode applies Top without poisoning cached candidates", knownBug: nil) {
        let vocabulary = try summaryVocabulary()
        let directory = try Temporary.directory("tag-advanced-session")
        let frames = [summaryFrame(0, [4: 0.9, 5: 0.5]),
                      summaryFrame(1, [4: 0.8, 5: 0.8]),
                      summaryFrame(2, [4: 0.5, 5: 0.9])]
        await MainActor.run {
            let session = WaifuSession(settingsURL: directory.appendingPathComponent("settings.json"), vocabulary: vocabulary)
            session.frameOrder = .chronological
            session.updateAnalyzedFrames(frames)
            session.tagFilter = TagFilter(requiredTags: ["tag_0", "tag_1"])
            session.settings.tagTopCount = 1
            session.settings.tagAdvancedMode = true
            session.tagFilter = TagFilter()
            checkEqual(Set(session.displayedFrames.map(\.id)), Set([frames[0].id, frames[2].id]),
                       "Advanced ranks every tag without a selection")
            checkEqual(session.topTagsByFrame[frames[0].id], ["tag 0"], "first thumbnail explains its winning tag")
            checkEqual(session.topTagsByFrame[frames[2].id], ["tag 1"], "last thumbnail explains a different winning tag")
            session.settings.tagAdvancedMode = false
            session.tagFilter = TagFilter(requiredTags: ["tag_0", "tag_1"])
            session.settings.tagTopCount = 2
            checkEqual(session.displayedFrames.count, 3, "Basic leaves Top off")
            check(session.topTagsByFrame.isEmpty, "Basic removes thumbnail Top badges")
            session.settings.tagAdvancedMode = true
            checkEqual(session.displayedFrames.map(\.id), [frames[1].id], "Advanced applies per-tag Top")
            session.settings.tagTopCount = 99
            checkEqual(session.displayedFrames.count, 3, "all restores cached candidates")
            check(session.topTagsByFrame.isEmpty, "all removes thumbnail Top badges")
            session.settings.tagTopCount = 1
            checkEqual(session.displayedFrames.count, 0, "separate top-one winners do not satisfy AND")
            session.settings.tagAdvancedMode = false
            checkEqual(session.displayedFrames.count, 3, "returning to Basic restores the candidate set")
            session.settings.taggerThreshold = 0.85
            checkEqual(session.displayedFrames.count, 0, "dynamic threshold applies to each required tag")
            session.settings.taggerThreshold = 0.35
            checkEqual(session.displayedFrames.count, 3, "lowering threshold restores UInt8-backed matches")
        }
    })

    Registry.shared.add(suite: suite, TestCase(name: "Top intersects each selected tag before group caps and all removes the limit", knownBug: nil) {
        let vocabulary = try summaryVocabulary()
        let frames = [summaryFrame(0, [4: 0.9, 5: 0.5]),
                      summaryFrame(1, [4: 0.8, 5: 0.8]),
                      summaryFrame(2, [4: 0.5, 5: 0.9])]
        let filter = TagFilter(requiredTags: ["tag_0", "tag_1"])
        let top = FrameArrangement.arrange(frames, order: .chronological,
            perGroupTopCount: 33, grouping: .default, tagFilter: filter,
            vocabulary: vocabulary, tagTopCount: 2)
        checkEqual(top.visible.map(\.id), [frames[1].id], "only the middle frame is in both top-two sets")
        checkEqual(top.topTagsByFrame, [frames[1].id: ["tag 0", "tag 1"]], "selected-tag explanation includes both rankings")
        let all = FrameArrangement.arrange(frames, order: .chronological,
            perGroupTopCount: 33, grouping: .default, tagFilter: filter,
            vocabulary: vocabulary, tagTopCount: 99)
        checkEqual(all.visible.map(\.id), frames.map(\.id), "99 removes the top limit")
        let reversed = filter.keepingTopFrames(Array(frames.reversed()), count: 2, vocabulary: vocabulary)
        checkEqual(reversed.map(\.id), [frames[1].id], "membership does not depend on display order")
        checkEqual(Set(TagFilter().keepingTopFrames(frames, count: 1, vocabulary: vocabulary).map(\.id)),
                   Set([frames[0].id, frames[2].id]), "without a selection combine each tag's top frame")
        checkEqual(TagFilter().keepingTopFrames(frames, count: 99, vocabulary: vocabulary).count, 3,
                   "all removes the limit without selected tags too")
    })

    Registry.shared.add(suite: suite, TestCase(name: "raising and lowering dynamic threshold preserves sparse UInt8 scores", knownBug: nil) {
        let vocabulary = try summaryVocabulary()
        let frames = [summaryFrame(0, [4: 0.5]), summaryFrame(1, [4: 0.9])]
        let originalPayloads = frames.map(\.tagProbabilities)
        let low = TagFilter(requiredTags: ["tag_0"], threshold: 0.35)
        let high = TagFilter(requiredTags: ["tag_0"], threshold: 0.8)
        checkEqual(low.keepingTopFrames(frames, count: 7, vocabulary: vocabulary).count, 2, "low threshold admits both")
        checkEqual(high.keepingTopFrames(frames, count: 7, vocabulary: vocabulary).count, 1, "raised threshold narrows matches")
        checkEqual(low.keepingTopFrames(frames, count: 7, vocabulary: vocabulary).count, 2, "lowering threshold restores matches")
        checkEqual(frames.map(\.tagProbabilities), originalPayloads, "compressed scores remain intact")
        checkEqual(frames[0].tagProbabilities?.prefix(4), Data("WDT4".utf8), "UInt8 storage is retained")
    })

    Registry.shared.add(suite: suite, TestCase(name: "selected tag scores compose with ratings and strict trait filters", knownBug: nil) {
        let vocabulary = try summaryVocabulary()
        let frames = [summaryFrame(0, [0: 0.9, 4: 0.8, 5: 0.5]),
                      summaryFrame(1, [0: 0.1, 4: 0.9, 5: 0.1]),
                      summaryFrame(2, [0: 0.8, 4: 0.8, 5: 0.2])]
        var filter = TagFilter(requiredTags: ["tag_0"], threshold: 0.5, scoreSortTag: "tag_0")
        let forward = FrameArrangement.arrange(frames, order: .only([.ratingGeneral]),
            perGroupTopCount: 10, grouping: .default, tagFilter: filter, vocabulary: vocabulary)
        checkEqual(forward.visible.map(\.id), [frames[1].id, frames[0].id, frames[2].id],
                   "tag score is primary; equal trait scores fall through to the distinct rating heads")
        filter.scoreSortDescending = false
        let reverse = FrameArrangement.arrange(frames, order: .only([.ratingGeneral]),
            perGroupTopCount: 10, grouping: .default, tagFilter: filter, vocabulary: vocabulary)
        checkEqual(reverse.visible.map(\.id), [frames[0].id, frames[2].id, frames[1].id],
                   "tag direction does not reverse the rating tie breaker")
        let required = TagFilter(requiredTags: ["tag_0", "tag_1"], threshold: 0.5)
        check(!required.admits(frames[0], vocabulary: vocabulary), "equal threshold is excluded from all-tags match")
        let excluded = TagFilter(requiredTags: ["tag_0"], excludedTags: ["tag_1"], threshold: 0.2)
        check(!excluded.admits(frames[0], vocabulary: vocabulary), "an above-threshold excluded trait rejects")
        check(excluded.admits(frames[2], vocabulary: vocabulary), "an equal-threshold excluded trait admits")
        var corrupt = frames[0]
        corrupt.tagProbabilities = Data("corrupt".utf8)
        corrupt.topTags = ["tag_0", "tag_1"]
        check(required.admits(corrupt, vocabulary: vocabulary), "malformed data retains saved-tag fallback")
        check(!excluded.admits(corrupt, vocabulary: vocabulary), "saved excluded tags still reject")
        check(!TagFilter(requiredTags: ["unknown"]).admits(frames[0], vocabulary: vocabulary),
              "unknown required heads do not match")
    })

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
        checkClose(byIndex[4]?.mean ?? -1, Double(229) / 255, tolerance: 1e-6, "conditional mean uses the quantized confidence")
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
        let computer = TagSummaryComputer(chunkByteLimit: vocabulary.generalIndices.count * 17,
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
        let rowBytes = vocabulary.generalIndices.count + 2 * MemoryLayout<Int>.stride
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

    Registry.shared.add(suite: suite, TestCase(name: "integer reductions match decoded scores at every byte and cutoff boundary", knownBug: nil) {
        let vocabulary = try summaryVocabulary()
        let codes = Array(89...255)
        let frames = codes.enumerated().map { row, code in
            summaryFrame(row, [4: code == 89 ? Float(0.35).nextUp : Float(code) / 255, 5: 0.35])
        }
        let values = frames.map { TagProbabilityCodec.value(in: $0.tagProbabilities!, at: 4)! }
        let computer = TagSummaryComputer()
        let thresholds: [Float] = [0, Float(0.35).nextDown, 0.35, Float(0.35).nextUp, 0.5, Float(127) / 255, Float(254) / 255, 1]
        for threshold in thresholds {
            let admitted = values.filter { $0 > threshold }
            let cpu = try await computer.summarize(frames, vocabulary: vocabulary, threshold: threshold, backend: .cpu)
            let tag = cpu.tags.first { $0.index == 4 }
            checkEqual(tag?.scoredFrames ?? 0, admitted.count, "integer threshold matches decoded probabilities")
            if !admitted.isEmpty {
                let expected = admitted.reduce(0.0) { $0 + Double($1) } / Double(admitted.count)
                checkClose(tag?.mean ?? -1, expected, tolerance: 1e-7, "integer sum converts to the correct mean")
            }
            if MTLCreateSystemDefaultDevice()?.hasUnifiedMemory == true {
                let gpu = try await computer.summarize(frames, vocabulary: vocabulary, threshold: threshold, backend: .metal)
                checkEqual(gpu.tags.map(\.matchingFrames), cpu.tags.map(\.matchingFrames), "byte GPU membership matches CPU")
                checkEqual(gpu.tags.map(\.mean), cpu.tags.map(\.mean), "integer CPU/GPU means agree exactly")
            }
        }
    })

    Registry.shared.add(suite: suite, TestCase(name: "automatic sparse summaries avoid GPU dispatch and keep byte-sized matrices", knownBug: nil) {
        let vocabulary = try summaryVocabulary()
        let frames = (0..<1_800).map { summaryFrame($0, [4: 0.8, 5: 0.9]) }
        let attempts = TickLog()
        let computer = TagSummaryComputer(metalProvider: { attempts.record(1); return nil })
        let automatic = try await computer.summarize(frames, vocabulary: vocabulary, threshold: 0.35)
        checkEqual(attempts.all, [], "sparse selections use the faster populated-column CPU path")
        checkEqual(automatic.tags.map(\.matchingFrames), [1_800, 1_800], "all sparse tag memberships are counted")
        if MTLCreateSystemDefaultDevice()?.hasUnifiedMemory == true {
            let metal = try MetalTagSummary()
            let buffer = try metal.matrixBuffer(rows: 17, columns: vocabulary.generalIndices.count)
            checkEqual(buffer.length, 17 * vocabulary.generalIndices.count, "Metal retains one byte per score")
        }
    })

    Registry.shared.add(suite: suite, TestCase(name: "automatic reduction falls back when Metal is unavailable", knownBug: nil) {
        let vocabulary = try summaryVocabulary()
        let dense = summaryFrame(0, Dictionary(uniqueKeysWithValues: (4..<1_200).map { ($0, Float(0.8)) })).tagProbabilities
        let attempts = TickLog()
        let computer = TagSummaryComputer(metalProvider: { attempts.record(1); return nil })
        for count in [128, 512, 1_536] {
            let frames = (0..<count).map { index in
                var frame = FrameFixture.make(id: String(format: "f%06d", index), time: Double(index), artScore: 1)
                frame.tagProbabilities = dense
                return frame
            }
            let automatic = try await computer.summarize(frames, vocabulary: vocabulary, threshold: 0.35)
            let cpu = try await computer.summarize(frames, vocabulary: vocabulary, threshold: 0.35, backend: .cpu)
            checkEqual(automatic.gpuChunks, 0, "no unavailable GPU is used")
            checkEqual(automatic.labels(using: vocabulary).ranked, cpu.labels(using: vocabulary).ranked, "fallback uses the same conditional average")
            checkEqual(automatic.labels(using: vocabulary).common, cpu.labels(using: vocabulary).common, "fallback preserves common tags")
        }
        checkEqual(attempts.all, [1, 1], "dense selections attempt Metal at 512 frames; small ones use CPU")
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

private actor KeywordWriteProbe {
    var identifiers: [String] = []
    var counts: [Int] = []
    let pauseFirst: Bool
    let failSecond: Bool
    var firstStarted = false
    var startWaiter: CheckedContinuation<Void, Never>?
    var writeWaiter: CheckedContinuation<Void, Never>?

    init(pauseFirst: Bool = false, failSecond: Bool = false) {
        self.pauseFirst = pauseFirst
        self.failSecond = failSecond
    }

    func write(_ update: PhotosKeywords.Update) async throws {
        identifiers.append(update.assetIdentifier)
        if pauseFirst && identifiers.count == 1 {
            await withCheckedContinuation { continuation in
                writeWaiter = continuation
                firstStarted = true
                startWaiter?.resume()
                startWaiter = nil
            }
        }
        if failSecond && identifiers.count == 2 { throw PhotosMedia.failure("Simulated failure") }
    }

    func waitForFirstWrite() async {
        if firstStarted { return }
        await withCheckedContinuation { startWaiter = $0 }
    }

    func release() { writeWaiter?.resume(); writeWaiter = nil }
    func report(_ count: Int) { counts.append(count) }
}
