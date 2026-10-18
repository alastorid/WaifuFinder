import Foundation

/// The tagging cache is keyed per photo for Photos albums. Album position is not
/// part of that key on purpose: PhotoKit does not guarantee the member order of a
/// fetch without a sort descriptor, so an album-wide key changed between sessions
/// even when no photo had been edited, and every cached tag was discarded.
func registerTagResultMemoryTests() {
    let suite = "Tag result memory"

    Registry.shared.add(suite: suite, TestCase(name: "scores rounded across the sparse threshold survive a restart", knownBug: nil) {
        let vocabulary = try tagVocabularyFixture()
        let directory = try Temporary.directory("tag-result-threshold")
        let fileURL = directory.appendingPathComponent("tag-results.db")
        let memory = await MainActor.run { TagResultMemory(disk: TagResultDiskCache(fileURL: fileURL)) }
        let identity = TagResultCacheIdentity(
            photosAssetKey: PhotosMedia.AssetCacheKey(identifier: "threshold-photo", modifiedMilliseconds: 1_000,
                                                      pixelWidth: 4_000, pixelHeight: 3_000),
            vocabulary: vocabulary)!
        let keys = TagResultCacheKeys(source: nil, bySampleIndex: [0: identity])
        let blank = AnalyzedFrame(sampleIndex: 0, timeSeconds: 0, artScore: 0, faceScore: nil, faceCount: 0,
                                  faceAreaFraction: nil, featurePrint: nil)
        let boundaryScores: [Float] = [Float(0.35).nextUp, 0.3501, 0.351, 0.359, 0.36, 0.375, 0.9, 1]
        let indices = Array(vocabulary.generalIndices.prefix(boundaryScores.count + 2))
        var probabilities = [Float](repeating: 0, count: TagVocabulary.headCount)
        probabilities[1] = 0.9
        for (index, score) in zip(indices, boundaryScores) { probabilities[index] = score }
        probabilities[indices[boundaryScores.count]] = 0.35
        probabilities[indices[boundaryScores.count + 1]] = 0.3499
        var tagged = blank
        tagged.tagProbabilities = TagProbabilityCodec.encode(probabilities)
        tagged.tagVocabularyFingerprint = vocabulary.cacheFingerprint
        await memory.remember(frames: [tagged], keys: keys)

        // A new cache actor must rebuild the index and read the on-disk payload.
        let reopened = await MainActor.run { TagResultMemory(disk: TagResultDiskCache(fileURL: fileURL)) }
        let restored = await reopened.restore([blank], keys: keys, vocabulary: vocabulary, threshold: 0.35)
        checkEqual(restored.restoredCount, 1, "one rounded score must not discard the photo's entire result")
        check(restored.frames[0].hasTaggingResult, "the restored photo does not need tagging again")
        let decoded = restored.frames[0].tagProbabilities.flatMap { TagProbabilityCodec.decode($0) }
        checkNotNil(decoded, "restored runtime probabilities are valid")
        guard let decoded else { return }
        for index in indices.prefix(boundaryScores.count) {
            check(decoded[index] > 0.35, "every retained score stays above the sparse threshold")
            check(restored.frames[0].topTags?.contains(vocabulary.labels[index].name) == true,
                  "a retained tag remains visible after restart")
        }
        for index in indices.suffix(2) {
            checkEqual(decoded[index], 0, "scores at or below the threshold remain omitted")
        }
        checkEqual(restored.frames[0].tagRating, TagRating.allCases[1], "the photo's rating is also restored")

        // Older payloads have the same compressed body without the size field.
        let cached = await reopened.cachedResults(for: keys)
        guard let payload = cached[identity]?[0] else {
            check(false, "the photo's saved payload exists")
            return
        }
        var legacyPayload = Data("WDF8".utf8)
        legacyPayload.append(payload.dropFirst(6))
        let legacyDisk = TagResultDiskCache(fileURL: directory.appendingPathComponent("legacy.db"))
        _ = try await legacyDisk.remember(
            [TagResultMemoryRecord(sampleIndex: 0, compressedSparseScores: legacyPayload)],
            for: identity, retentionDays: 30)
        let legacyMemory = await MainActor.run { TagResultMemory(disk: legacyDisk) }
        let legacy = await legacyMemory.restore([blank], keys: keys, vocabulary: vocabulary, threshold: 0.35)
        checkEqual(legacy.restoredCount, 1, "existing WDF8 results also recover without retagging")
        checkEqual(legacy.frames[0].topTags, restored.frames[0].topTags, "both formats recover the same tags")
    })

    Registry.shared.add(suite: suite, TestCase(name: "photo keys ignore album order and membership", knownBug: nil) {
        let vocabulary = try tagVocabularyFixture()
        let first = PhotosMedia.AssetCacheKey(identifier: "asset-A", modifiedMilliseconds: 1_000,
                                              pixelWidth: 4_000, pixelHeight: 3_000)
        let second = PhotosMedia.AssetCacheKey(identifier: "asset-B", modifiedMilliseconds: 2_000,
                                               pixelWidth: 3_000, pixelHeight: 4_000)
        let firstIdentity = TagResultCacheIdentity(photosAssetKey: first, vocabulary: vocabulary)
        let secondIdentity = TagResultCacheIdentity(photosAssetKey: second, vocabulary: vocabulary)
        checkNotNil(firstIdentity, "a still photo is cacheable")
        checkNotNil(secondIdentity, "a still photo is cacheable")

        // Reopening an album hands back the same assets at different ordinals.
        // Ordinal lives in the record key, never in the identity, so moving a
        // photo within its album changes nothing about its cache identity.
        let reordered = TagResultCacheKeys(source: nil, bySampleIndex: [
            PhotosMedia.frameIndex(seconds: 7): secondIdentity!,
            PhotosMedia.frameIndex(seconds: 3): firstIdentity!
        ])
        let original = TagResultCacheKeys(source: nil, bySampleIndex: [
            PhotosMedia.frameIndex(seconds: 3): firstIdentity!,
            PhotosMedia.frameIndex(seconds: 7): secondIdentity!
        ])
        checkEqual(reordered, original, "album order is not part of the key")

        // Editing a photo changes its revision, and only its own key.
        let edited = TagResultCacheIdentity(
            photosAssetKey: PhotosMedia.AssetCacheKey(identifier: "asset-A", modifiedMilliseconds: 3_000,
                                                      pixelWidth: 4_000, pixelHeight: 3_000),
            vocabulary: vocabulary)
        check(edited != firstIdentity, "an edit invalidates the photo it changed")
        checkEqual(edited!.sourcePath, firstIdentity!.sourcePath, "the asset identifier is stable across edits")

        // Transposing the photo needs both dimensions: 4000x3000 and 3000x4000
        // render different 448-pixel renditions.
        let rotated = TagResultCacheIdentity(
            photosAssetKey: PhotosMedia.AssetCacheKey(identifier: "asset-A", modifiedMilliseconds: 1_000,
                                                      pixelWidth: 3_000, pixelHeight: 4_000),
            vocabulary: vocabulary)
        check(rotated != firstIdentity, "a change of pixel geometry invalidates the photo")

        checkNil(TagResultCacheIdentity(photosAssetKey: PhotosMedia.AssetCacheKey(identifier: "",
                                                                                   modifiedMilliseconds: 0,
                                                                                   pixelWidth: 1, pixelHeight: 1),
                                        vocabulary: vocabulary),
                 "an unidentified asset is not cached")
        checkEqual(firstIdentity!.analysisWidth, UInt32(WDTaggerBackend.imageSize),
                   "photos are always scored from the tagger's own input size")
        check(firstIdentity!.sourcePath.hasPrefix("photos-asset-v1/"),
              "photo keys are namespaced away from file and album paths")
    })

    Registry.shared.add(suite: suite, TestCase(name: "a frame with no photo is never cached under another photo's key", knownBug: nil) {
        let vocabulary = try tagVocabularyFixture()
        let identity = TagResultCacheIdentity(
            photosAssetKey: PhotosMedia.AssetCacheKey(identifier: "asset-A", modifiedMilliseconds: 1_000,
                                                      pixelWidth: 4_000, pixelHeight: 3_000),
            vocabulary: vocabulary)!
        // One frame could not be matched to a photo — an album video item, or a
        // timestamp that fell outside every member. Writing it under a neighbour's
        // key would hand the wrong tags to the wrong photo later.
        let partial = TagResultCacheKeys(source: nil, bySampleIndex: [
            PhotosMedia.frameIndex(seconds: 1): identity
        ])
        let known = AnalyzedFrame(sampleIndex: PhotosMedia.frameIndex(seconds: 1), timeSeconds: 1,
                                  artScore: 0, faceScore: nil, faceCount: 0,
                                  faceAreaFraction: nil, featurePrint: nil)
        let unknown = AnalyzedFrame(sampleIndex: PhotosMedia.frameIndex(seconds: 2), timeSeconds: 2,
                                    artScore: 0, faceScore: nil, faceCount: 0,
                                    faceAreaFraction: nil, featurePrint: nil)
        checkEqual(partial.key(for: known), identity, "the matched frame keeps its own key")
        checkNil(partial.key(for: unknown), "an unmatched frame is left uncached")

        // A video file has one identity for every frame, which is the other case.
        let source = TagResultCacheKeys(source: identity)
        checkEqual(source.key(for: unknown), identity, "a file source shares one identity")
        checkNoDuplicates(source.identities, "a file source reads one identity")
        check(TagResultCacheKeys.none.isEmpty, "no source means no keys")
    })

    Registry.shared.add(suite: suite, TestCase(name: "photo tagging results survive an album reorder and a photo being edited", knownBug: nil) {
        let vocabulary = try tagVocabularyFixture()
        let directory = try Temporary.directory("tag-result-photos")
        let disk = TagResultDiskCache(fileURL: directory.appendingPathComponent("tag-results.db"))
        let memory = await MainActor.run { TagResultMemory(disk: disk) }
        let fingerprint = vocabulary.cacheFingerprint

        func assetKey(_ identifier: String, milliseconds: UInt64) -> PhotosMedia.AssetCacheKey {
            PhotosMedia.AssetCacheKey(identifier: identifier, modifiedMilliseconds: milliseconds,
                                      pixelWidth: 4_000, pixelHeight: 3_000)
        }
        func identity(_ identifier: String, milliseconds: UInt64) -> TagResultCacheIdentity {
            TagResultCacheIdentity(photosAssetKey: assetKey(identifier, milliseconds: milliseconds),
                                   vocabulary: vocabulary)!
        }
        /// A photo scores its own recognisable tag vector, so a restored frame can
        /// be traced back to the photo that produced it.
        func frame(_ identifier: String, seconds: Double, milliseconds: UInt64 = 1_000) -> AnalyzedFrame {
            var probabilities = [Float](repeating: 0, count: TagVocabulary.headCount)
            probabilities[vocabulary.index(for: identifier) ?? 4] = 0.9
            var frame = AnalyzedFrame(sampleIndex: PhotosMedia.frameIndex(seconds: seconds),
                                      timeSeconds: seconds, artScore: 0, faceScore: nil, faceCount: 0,
                                      faceAreaFraction: nil, featurePrint: nil)
            frame.tagProbabilities = TagProbabilityCodec.encode(probabilities)
            frame.tagVocabularyFingerprint = fingerprint
            return frame
        }
        /// Album seconds become frame ordinals exactly as the Photos sampler does
        /// it, so a test key always addresses the frame it was built beside.
        func keys(_ members: [(seconds: Double, identifier: String, milliseconds: UInt64)]) -> TagResultCacheKeys {
            TagResultCacheKeys(source: nil, bySampleIndex: Dictionary(uniqueKeysWithValues: members.map {
                (PhotosMedia.frameIndex(seconds: $0.seconds),
                 identity($0.identifier, milliseconds: $0.milliseconds))
            }))
        }

        // Tag three photos, one second apart, in one album order.
        let untagged = [frame("cat ears", seconds: 0), frame("long hair", seconds: 1), frame("blue hair", seconds: 2)]
        let firstRun = keys([(0, "cat ears", 1_000), (1, "long hair", 1_000), (2, "blue hair", 1_000)])
        await memory.remember(frames: untagged, keys: firstRun)
        checkEqual(await MainActor.run { memory.entryCount }, 3, "one entry per photo, not one per album")

        // Reopen the same album. The same three photos come back, moved around.
        let reordered = [frame("blue hair", seconds: 0), frame("cat ears", seconds: 1), frame("long hair", seconds: 2)]
        let secondRun = keys([(0, "blue hair", 1_000), (1, "cat ears", 1_000), (2, "long hair", 1_000)])
        let restored = await memory.restore(reordered, keys: secondRun, vocabulary: vocabulary, threshold: 0.35)
        checkEqual(restored.restoredCount, 3, "every photo is restored after the album reorders")
        checkEqual(restored.frames.compactMap(\.topTags?.first),
                   ["blue hair", "cat ears", "long hair"],
                   "each photo gets its own tags back, in its new position")

        // Editing one photo in Photos invalidates that photo and nothing else.
        let edited = [frame("blue hair", seconds: 0, milliseconds: 5_000),
                      frame("cat ears", seconds: 1), frame("long hair", seconds: 2)]
        let thirdRun = keys([(0, "blue hair", 5_000), (1, "cat ears", 1_000), (2, "long hair", 1_000)])
        let afterEdit = await memory.restore(edited, keys: thirdRun, vocabulary: vocabulary, threshold: 0.35)
        checkEqual(afterEdit.restoredCount, 2, "only the edited photo is scored again")
        checkEqual(afterEdit.frames[0].topTags, nil, "the edited photo has no cached tags")
        checkEqual(afterEdit.frames[1].topTags?.first, "cat ears", "the untouched photo beside it is reused")

        // The same photo in a different album, at a different ordinal, still hits.
        let elsewhere = [frame("cat ears", seconds: 41)]
        let otherAlbum = keys([(41, "cat ears", 1_000)])
        let shared = await memory.restore(elsewhere, keys: otherAlbum, vocabulary: vocabulary, threshold: 0.35)
        checkEqual(shared.restoredCount, 1, "one photo tagged in one album is reused in another")
        checkEqual(shared.frames[0].topTags?.first, "cat ears", "the reused tags describe that photo")
    })

    Registry.shared.add(suite: suite, TestCase(name: "a changed vocabulary never reuses stored tag results", knownBug: nil) {
        let vocabulary = try tagVocabularyFixture()
        let directory = try Temporary.directory("tag-result-vocabulary")
        let disk = TagResultDiskCache(fileURL: directory.appendingPathComponent("tag-results.db"))
        let memory = await MainActor.run { TagResultMemory(disk: disk) }

        let assetKey = PhotosMedia.AssetCacheKey(identifier: "asset-A", modifiedMilliseconds: 1_000,
                                                  pixelWidth: 4_000, pixelHeight: 3_000)
        let identity = TagResultCacheIdentity(photosAssetKey: assetKey, vocabulary: vocabulary)!
        var probabilities = [Float](repeating: 0, count: TagVocabulary.headCount)
        probabilities[vocabulary.index(for: "cat ears") ?? 4] = 0.9
        var frame = AnalyzedFrame(sampleIndex: 0, timeSeconds: 0, artScore: 0, faceScore: nil, faceCount: 0,
                                  faceAreaFraction: nil, featurePrint: nil)
        frame.tagProbabilities = TagProbabilityCodec.encode(probabilities)
        frame.tagVocabularyFingerprint = vocabulary.cacheFingerprint

        let keys = TagResultCacheKeys(source: nil, bySampleIndex: [0: identity])
        await memory.remember(frames: [frame], keys: keys)
        checkEqual(await MainActor.run { memory.entryCount }, 1, "the result is stored")

        // Results carrying another vocabulary's index order are not storable, and
        // the frames themselves carry the fingerprint the restore checks.
        var foreign = frame
        foreign.tagVocabularyFingerprint = vocabulary.cacheFingerprint &+ 1
        await memory.remember(frames: [foreign], keys: keys)
        checkEqual(await MainActor.run { memory.entryCount }, 1, "a mismatched fingerprint adds nothing")
    })
}

/// The real label list is the only honest vocabulary: the cache fingerprint and
/// the restored rating both come out of it.
private func tagVocabularyFixture() throws -> TagVocabulary {
    let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
    let directory = ProcessInfo.processInfo.environment["WAIFU_TEST_TAGGER_DIR"].map { URL(fileURLWithPath: $0) }
        ?? WDTaggerConfiguration.availableDirectory
        ?? root.appendingPathComponent("models/published/WD-SwinV2-Tagger-v3-CoreML")
    guard FileManager.default.fileExists(atPath: directory.appendingPathComponent("wd-tagger-labels.csv").path) else {
        throw SkipTest(reason: "offline WDTagger fixture is absent")
    }
    return try TagVocabulary.load(from: directory.appendingPathComponent("wd-tagger-labels.csv"))
}
