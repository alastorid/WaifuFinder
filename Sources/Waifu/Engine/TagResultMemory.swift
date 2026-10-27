import Combine
import Compression
import Foundation

/// Identifies tagging results for one source file and one vocabulary ordering.
/// The source path and file metadata prevent a replaced video at the same path
/// from reusing old results; the vocabulary fingerprint protects numeric indices.
struct TagResultCacheIdentity: Hashable, Sendable {
    let sourcePath: String
    let sourceSize: UInt64
    let sourceModifiedMilliseconds: UInt64
    let analysisWidth: UInt32
    let vocabularyFingerprint: UInt64

    init?(videoURL: URL, vocabulary: TagVocabulary, analysisWidth: Int, analysisMode: AnalysisMode = .full) {
        let source = videoURL.standardizedFileURL.resolvingSymlinksInPath()
        guard source.isFileURL,
              analysisWidth > 0, analysisWidth <= Int(UInt32.max),
              let attributes = try? FileManager.default.attributesOfItem(atPath: source.path),
              let size = (attributes[.size] as? NSNumber)?.uint64Value,
              let modified = attributes[.modificationDate] as? Date else { return nil }

        sourcePath = source.path
        sourceSize = size
        sourceModifiedMilliseconds = UInt64(max(0, modified.timeIntervalSince1970 * 1_000))
        self.analysisWidth = UInt32(analysisWidth)
        vocabularyFingerprint = vocabulary.cacheFingerprint
    }

    /// One Photos photo, keyed by the asset rather than by its position in an
    /// album. The rendition the tagger scored is a pure function of the asset's
    /// own revision and pixel geometry, so those three values are the whole key:
    /// reopening the album in a different order, adding a photo to it, or taking
    /// one away cannot invalidate another photo's tags, and the same photo tagged
    /// in another album is reused instead of scored again.
    init?(photosAssetKey: PhotosMedia.AssetCacheKey, vocabulary: TagVocabulary) {
        let identifier = photosAssetKey.identifier
        guard !identifier.isEmpty, identifier.utf8.count <= 1_048_576 else { return nil }
        sourcePath = "photos-asset-v1/\(identifier)"
        // Both dimensions are needed: 4000x3000 and 3000x4000 render different
        // 448-pixel renditions even though they hold the same pixel count.
        sourceSize = (UInt64(photosAssetKey.pixelWidth) << 32) | UInt64(photosAssetKey.pixelHeight)
        sourceModifiedMilliseconds = photosAssetKey.modifiedMilliseconds
        self.analysisWidth = UInt32(WDTaggerBackend.imageSize)
        vocabularyFingerprint = vocabulary.cacheFingerprint
    }

    init(sourcePath: String, sourceSize: UInt64, sourceModifiedMilliseconds: UInt64,
                     analysisWidth: UInt32, vocabularyFingerprint: UInt64) {
        self.sourcePath = sourcePath
        self.sourceSize = sourceSize
        self.sourceModifiedMilliseconds = sourceModifiedMilliseconds
        self.analysisWidth = analysisWidth
        self.vocabularyFingerprint = vocabularyFingerprint
    }
}

/// The cache keys for the frames of one loaded source.
///
/// A video file has exactly one identity, shared by every frame: its path, size
/// and modification time describe the whole file. A Photos album has none worth
/// trusting — PhotoKit does not guarantee the member order of a fetch without a
/// sort descriptor, so an album-wide key changes whenever that order or the
/// membership shifts and every cached tag is thrown away. Each photo therefore
/// carries its own key and nothing is written for a frame that has no photo.
struct TagResultCacheKeys: Hashable, Sendable {
    let source: TagResultCacheIdentity?
    private let bySampleIndex: [Int: TagResultCacheIdentity]
    private let recordIndices: [Int: Int]

    init(source: TagResultCacheIdentity?, bySampleIndex: [Int: TagResultCacheIdentity] = [:], recordIndices: [Int: Int] = [:]) {
        self.source = source
        self.bySampleIndex = bySampleIndex
        self.recordIndices = recordIndices
    }

    static let none = TagResultCacheKeys(source: nil)

    var isEmpty: Bool { source == nil && bySampleIndex.isEmpty }

    /// The identity for one frame, or nil when that frame is not cacheable.
    /// Falling back to the source key is only correct when every frame shares it.
    func key(for frame: AnalyzedFrame) -> TagResultCacheIdentity? {
        if let identity = bySampleIndex[frame.id.sampleIndex] { return identity }
        return bySampleIndex.isEmpty ? source : nil
    }

    /// The ordinal a frame's record occupies within its identity. A file shares
    /// one identity across many frames, so each keeps its own ordinal. A photo
    /// identity already names exactly one photo, so its single record always sits
    /// at the same ordinal: storing the album position as well would make the
    /// entry unreachable the next time the album is fetched in another order.
    func recordIndex(for frame: AnalyzedFrame) -> Int {
        recordIndices[frame.id.sampleIndex] ?? (bySampleIndex.isEmpty ? frame.id.sampleIndex : 0)
    }

    var identities: [TagResultCacheIdentity] {
        if !bySampleIndex.isEmpty { return Array(Set(bySampleIndex.values)) }
        return source.map { [$0] } ?? []
    }

    /// Our own keyword writes change PhotoKit's modification date without
    /// changing the pixels. Save the same scores under the confirmed post-write
    /// revisions too, so they remain reachable after the app restarts. Never
    /// extend this exception to an unknown asset or changed pixel geometry.
    func afterPhotoMetadataWrite(_ assets: [PhotosMedia.AssetCacheKey]) -> TagResultCacheKeys {
        let updated = Dictionary(assets.map { ("photos-asset-v1/\($0.identifier)", $0) },
                                 uniquingKeysWith: { _, latest in latest })
        var revised: [Int: TagResultCacheIdentity] = [:]
        for (index, identity) in bySampleIndex {
            guard let asset = updated[identity.sourcePath],
                  identity.sourceSize == (UInt64(asset.pixelWidth) << 32) | UInt64(asset.pixelHeight) else { continue }
            revised[index] = TagResultCacheIdentity(sourcePath: identity.sourcePath,
                sourceSize: identity.sourceSize, sourceModifiedMilliseconds: asset.modifiedMilliseconds,
                analysisWidth: identity.analysisWidth, vocabularyFingerprint: identity.vocabularyFingerprint)
        }
        return TagResultCacheKeys(source: nil, bySampleIndex: revised)
    }

    /// Photos frames are keyed individually; every other source is a file.
    static func make(videoURL: URL?, frames: [AnalyzedFrame], vocabulary: TagVocabulary?,
                     analysisWidth: Int, analysisMode: AnalysisMode) -> TagResultCacheKeys {
        guard let videoURL, let vocabulary else { return .none }
        if let collection = VideoCollection.get(videoURL) {
            let identities = collection.members.map { member in
                TagResultCacheIdentity(videoURL: member.video.url, vocabulary: vocabulary,
                    analysisWidth: analysisWidth, analysisMode: analysisMode)
            }
            var bySampleIndex: [Int: TagResultCacheIdentity] = [:]
            var indices: [Int: Int] = [:]
            for frame in frames {
                let source = VideoCollection.sourceIndex(frame.id.sampleIndex)
                guard identities.indices.contains(source), let identity = identities[source] else { continue }
                bySampleIndex[frame.id.sampleIndex] = identity
                indices[frame.id.sampleIndex] = VideoCollection.localIndex(frame.id.sampleIndex)
            }
            return TagResultCacheKeys(source: nil, bySampleIndex: bySampleIndex, recordIndices: indices)
        }
        if PhotosMedia.shared.album(videoURL) != nil {
            var bySampleIndex: [Int: TagResultCacheIdentity] = [:]
            bySampleIndex.reserveCapacity(frames.count)
            for frame in frames {
                guard let assetKey = PhotosMedia.shared.photoAssetCacheKey(for: videoURL,
                                                                          seconds: frame.timeSeconds),
                      let identity = TagResultCacheIdentity(photosAssetKey: assetKey,
                                                            vocabulary: vocabulary) else { continue }
                bySampleIndex[frame.id.sampleIndex] = identity
            }
            return TagResultCacheKeys(source: nil, bySampleIndex: bySampleIndex)
        }
        return TagResultCacheKeys(source: TagResultCacheIdentity(videoURL: videoURL, vocabulary: vocabulary,
                                                                 analysisWidth: analysisWidth,
                                                                 analysisMode: analysisMode))
    }
}

struct TagResultMemoryRecord: Sendable {
    let sampleIndex: Int
    /// WDU1: zlib-compressed sparse pairs of vocabulary index and UInt8 score.
    let compressedSparseScores: Data
}

struct TagResultCacheStats: Sendable {
    let entryCount: Int
    let fileSize: Int64
}

struct TagResultCacheRead: Sendable {
    /// Grouped by cache identity first and frame ordinal second. The identity
    /// comes first because a Photos photo carries its own: the ordinal alone
    /// would hand one photo's stored tags to whichever photo now sits at that
    /// position in the album.
    let probabilitiesByIdentity: [TagResultCacheIdentity: [Int: Data]]
    let stats: TagResultCacheStats
}

struct RestoredTagResults: Sendable {
    let frames: [AnalyzedFrame]
    let restoredCount: Int
}

/// App-facing settings and statistics for the compressed tag cache.
@MainActor
final class TagResultMemory: ObservableObject {
    static let shared = TagResultMemory()

    @Published var rememberTaggingResults: Bool = UserDefaults.standard.object(forKey: "rememberTaggingResults") as? Bool ?? true {
        didSet {
            UserDefaults.standard.set(rememberTaggingResults, forKey: "rememberTaggingResults")
            Task { await refresh() }
        }
    }

    @Published var retentionDays: Int = min(max(UserDefaults.standard.object(forKey: "tagResultRetentionDays") as? Int ?? 30, 1), 36_500) {
        didSet {
            let valid = min(max(retentionDays, 1), 36_500)
            guard valid == retentionDays else {
                retentionDays = valid
                return
            }
            UserDefaults.standard.set(valid, forKey: "tagResultRetentionDays")
            Task { await refresh() }
        }
    }

    @Published private(set) var entryCount = 0
    @Published private(set) var fileSize: Int64 = 0
    @Published private(set) var errorMessage: String?

    let fileURL = TagResultDiskCache.defaultFileURL
    private let disk: TagResultDiskCache

    init(disk: TagResultDiskCache = .shared) {
        self.disk = disk
        Task { await refresh() }
    }

    func refresh() async {
        do {
            let stats = try await disk.stats(retentionDays: retentionDays)
            apply(stats)
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func deleteAll() async {
        do {
            apply(try await disk.deleteAll())
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func cachedResults(for keys: TagResultCacheKeys) async -> [TagResultCacheIdentity: [Int: Data]] {
        guard rememberTaggingResults else { return [:] }
        let identities = keys.identities
        guard !identities.isEmpty else { return [:] }
        do {
            let result = try await disk.results(for: identities, retentionDays: retentionDays)
            apply(result.stats)
            return result.probabilitiesByIdentity
        } catch {
            errorMessage = error.localizedDescription
            return [:]
        }
    }

    /// Adds missing or changed runtime results. Names are regenerated from the
    /// vocabulary when read and never written into the cache.
    func remember(frames: [AnalyzedFrame], keys: TagResultCacheKeys) async {
        guard rememberTaggingResults else { return }
        // Decoding every full score vector and compressing its sparse form can
        // take seconds for a long video. Never do that on the UI executor.
        let records = await Task.detached(priority: .utility) {
            Self.records(from: frames, keys: keys)
        }.value
        guard rememberTaggingResults else { return }
        guard !records.isEmpty else { return }
        for (identity, group) in records {
            do {
                apply(try await disk.remember(group, for: identity, retentionDays: retentionDays))
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    /// Shutdown hook: all normal writes are asynchronous, but wait for this final
    /// delta so process exit cannot discard a newly completed frame result.
    func flushAtTermination(frames: [AnalyzedFrame], keys: TagResultCacheKeys) {
        guard rememberTaggingResults else { return }
        let disk = self.disk
        let retentionDays = self.retentionDays
        let finished = DispatchSemaphore(value: 0)
        Task.detached(priority: .utility) {
            for (identity, group) in Self.records(from: frames, keys: keys) {
                _ = try? await disk.remember(group, for: identity, retentionDays: retentionDays)
            }
            finished.signal()
        }
        finished.wait()
    }

    func restore(_ frames: [AnalyzedFrame], keys: TagResultCacheKeys,
                 vocabulary: TagVocabulary, threshold: Float) async -> RestoredTagResults {
        guard rememberTaggingResults else { return RestoredTagResults(frames: frames, restoredCount: 0) }
        let cached = await cachedResults(for: keys)
        guard !cached.isEmpty else { return RestoredTagResults(frames: frames, restoredCount: 0) }

        return await Task.detached(priority: .utility) {
            var restoredCount = 0
            let restoredFrames = frames.map { original -> AnalyzedFrame in
                // Restore exists to turn cached probabilities into a rating and
                // its tags, so it fills any frame still missing that rating. A
                // frame carrying probabilities alone is unfinished here even
                // though `hasTaggingResult` counts it for re-tagging purposes.
                guard let identity = keys.key(for: original),
                      original.tagRating == nil,
                      let encoded = cached[identity]?[keys.recordIndex(for: original)],
                      let probabilities = TagResultCacheProbabilityCodec.decode(encoded),
                      let runtimeProbabilities = TagProbabilityCodec.encode(probabilities),
                      let result = try? vocabulary.result(probabilities: probabilities, threshold: threshold) else {
                    return original
                }
                var frame = original
                frame.tagProbabilities = runtimeProbabilities
                frame.tagVocabularyFingerprint = identity.vocabularyFingerprint
                frame.tagRating = result.rating
                frame.topTags = result.topTags
                frame.tagGeneralScore = result.generalScore
                restoredCount += 1
                return frame
            }
            return RestoredTagResults(frames: restoredFrames, restoredCount: restoredCount)
        }.value
    }

    private func apply(_ stats: TagResultCacheStats) {
        entryCount = stats.entryCount
        fileSize = stats.fileSize
        errorMessage = nil
    }

    /// One group per cache identity, each sorted by frame ordinal so a run
    /// appends in a stable order regardless of how the frames were produced.
    private nonisolated static func records(from frames: [AnalyzedFrame],
                                            keys: TagResultCacheKeys) -> [(TagResultCacheIdentity, [TagResultMemoryRecord])] {
        var grouped: [TagResultCacheIdentity: [Int: Data]] = [:]
        for frame in frames {
            guard let identity = keys.key(for: frame),
                  frame.tagVocabularyFingerprint == identity.vocabularyFingerprint,
                  let probabilities = frame.tagProbabilities,
                  let sparseScores = TagResultCacheProbabilityCodec.encode(probabilities) else { continue }
            grouped[identity, default: [:]][keys.recordIndex(for: frame)] = sparseScores
        }
        return grouped.map { identity, entries in
            (identity, entries.map { TagResultMemoryRecord(sampleIndex: $0.key,
                                                           compressedSparseScores: $0.value) }
                .sorted { $0.sampleIndex < $1.sampleIndex })
        }.sorted { $0.0.sourcePath < $1.0.sourcePath }
    }
}

/// Persist only scores above the fixed default display threshold. Each saved
/// item is a two-byte vocabulary index followed by its normalized UInt8 value;
/// all omitted indices reconstruct as zero. Tag names are never stored.
private enum TagResultCacheProbabilityCodec {
    private static let magic = Data("WDU1".utf8)
    private static let threshold = WaifuSettings.default.taggerThreshold
    private static let headerByteCount = 4
    private static let pairByteCount = 3
    private static let maximumUncompressedByteCount = headerByteCount + 4 + TagVocabulary.headCount * pairByteCount

    static func encode(_ runtimeData: Data) -> Data? {
        guard let pairs = TagProbabilityCodec.sparseBytes(in: runtimeData) else { return nil }
        let sparsePairs = pairs.filter { $0.index < 4 || TagProbabilityCodec.probability($0.value) > threshold }
        guard sparsePairs.count <= UInt16.max else { return nil }

        var raw = Data()
        raw.appendLittleEndian(UInt16(TagVocabulary.headCount))
        raw.appendLittleEndian(UInt16(sparsePairs.count))
        for (index, score) in sparsePairs {
            raw.appendLittleEndian(UInt16(index))
            raw.append(score)
        }

        var compressed = [UInt8](repeating: 0, count: raw.count + 1_024)
        let compressedCount = raw.withUnsafeBytes { source in
            compressed.withUnsafeMutableBytes { destination in
                compression_encode_buffer(destination.bindMemory(to: UInt8.self).baseAddress!, destination.count,
                                          source.bindMemory(to: UInt8.self).baseAddress!, source.count,
                                          nil, COMPRESSION_ZLIB)
            }
        }
        guard compressedCount > 0 else { return nil }
        var result = magic
        result.appendLittleEndian(UInt16(raw.count))
        result.append(contentsOf: compressed.prefix(compressedCount))
        return result
    }

    static func decode(_ encoded: Data) -> [Float]? {
        guard encoded.count > magic.count, encoded.prefix(magic.count) == magic else { return nil }
        var headerOffset = magic.count
        guard let rawLength = encoded.readUInt16(at: &headerOffset),
              (headerByteCount...maximumUncompressedByteCount).contains(Int(rawLength)) else { return nil }
        let compressed = encoded.dropFirst(headerOffset)
        var raw = Data(count: Int(rawLength) + 1)
        let decodedSize = compressed.withUnsafeBytes { source in
            raw.withUnsafeMutableBytes { destination in
                compression_decode_buffer(destination.bindMemory(to: UInt8.self).baseAddress!, destination.count,
                                          source.bindMemory(to: UInt8.self).baseAddress!, source.count, nil, COMPRESSION_ZLIB)
            }
        }
        guard decodedSize == Int(rawLength) else { return nil }
        raw.removeLast()

        var offset = 0
        guard let vocabularyCount = raw.readUInt16(at: &offset), Int(vocabularyCount) == TagVocabulary.headCount,
              let pairCount = raw.readUInt16(at: &offset),
              raw.count == headerByteCount + Int(pairCount) * pairByteCount else { return nil }

        var probabilities = [Float](repeating: 0, count: TagVocabulary.headCount)
        var previousIndex = -1
        for _ in 0..<Int(pairCount) {
            guard let indexValue = raw.readUInt16(at: &offset) else { return nil }
            let index = Int(indexValue)
            guard index > previousIndex, index < probabilities.count, offset < raw.count else { return nil }
            let code = raw[offset]
            offset += 1
            guard index < 4 || code >= 89 else { return nil }
            probabilities[index] = TagProbabilityCodec.probability(code, at: index)
            previousIndex = index
        }
        return probabilities
    }
}

/// Append-only key/value file. Each record keeps one source path and one sample
/// index beside the already-compressed model output. Startup and expiry scans read
/// record headers only; score vectors are decompressed only for the loaded video.
actor TagResultDiskCache {
    static let shared = TagResultDiskCache()
    nonisolated static var defaultFileURL: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Waifu", isDirectory: true)
            .appendingPathComponent("tag-results.db")
    }

    /// The production cache lives in Application Support; tests point this at a
    /// temporary directory so a regression run never touches real results.
    init(fileURL: URL = TagResultDiskCache.defaultFileURL) {
        self.fileURL = fileURL
    }

    private static let magic = Data([0x57, 0x54, 0x52, 0x31]) // WTR1
    private let fileURL: URL
    private var isLoaded = false
    private var mappedData: Data?
    private var latestRecordsByIdentity: [TagResultCacheIdentity: [Int: Location]] = [:]
    private var recordCount = 0
    private var liveRecordCount = 0
    private var knownFileSize: UInt64 = 0

    private struct Location {
        let recordOffset: Int
        let recordLength: Int
        let payloadOffset: Int
        let payloadLength: Int
        let savedAtMilliseconds: UInt64
    }

    func stats(retentionDays: Int) throws -> TagResultCacheStats {
        try loadIndexIfNeeded()
        try pruneExpired(retentionDays: retentionDays)
        return makeStats()
    }

    func deleteAll() throws -> TagResultCacheStats {
        if FileManager.default.fileExists(atPath: fileURL.path) {
            try FileManager.default.removeItem(at: fileURL)
        }
        mappedData = nil
        latestRecordsByIdentity.removeAll(keepingCapacity: true)
        recordCount = 0
        liveRecordCount = 0
        knownFileSize = 0
        isLoaded = true
        return makeStats()
    }

    /// Reads every identity a source addresses at once. A Photos album
    /// contributes one identity per photo, so this is many keys per lookup
    /// rather than one key per lookup.
    func results(for identities: [TagResultCacheIdentity], retentionDays: Int) throws -> TagResultCacheRead {
        try loadIndexIfNeeded()
        try pruneExpired(retentionDays: retentionDays)
        guard let data = try readMappedDataIfPresent() else {
            return TagResultCacheRead(probabilitiesByIdentity: [:], stats: makeStats())
        }

        var results: [TagResultCacheIdentity: [Int: Data]] = [:]
        for identity in identities {
            guard let locations = latestRecordsByIdentity[identity], !locations.isEmpty else { continue }
            var payloads: [Int: Data] = [:]
            payloads.reserveCapacity(locations.count)
            for (sampleIndex, location) in locations {
                let range = location.payloadOffset..<(location.payloadOffset + location.payloadLength)
                payloads[sampleIndex] = data[range]
            }
            results[identity] = payloads
        }
        return TagResultCacheRead(probabilitiesByIdentity: results, stats: makeStats())
    }

    func remember(_ records: [TagResultMemoryRecord], for identity: TagResultCacheIdentity,
                  retentionDays: Int) throws -> TagResultCacheStats {
        try loadIndexIfNeeded()
        try pruneExpired(retentionDays: retentionDays)
        guard !records.isEmpty else { return makeStats() }
        guard let pathData = identity.sourcePath.data(using: .utf8),
              !pathData.isEmpty, pathData.count <= 1_048_576 else { return makeStats() }

        try ensureFileExists()
        let existingData = try readMappedDataIfPresent()
        let now = Self.nowMilliseconds
        var appendData = Data()
        let startOffset = Int(knownFileSize)

        for record in records where record.sampleIndex >= 0 && !record.compressedSparseScores.isEmpty {
            let isNewEntry = latestRecordsByIdentity[identity]?[record.sampleIndex] == nil
            if let existing = latestRecordsByIdentity[identity]?[record.sampleIndex],
               existing.payloadLength == record.compressedSparseScores.count,
               let existingData {
                let range = existing.payloadOffset..<(existing.payloadOffset + existing.payloadLength)
                if existingData[range] == record.compressedSparseScores { continue }
            }

            guard record.compressedSparseScores.count <= 1_000_000 else { continue }

            var encoded = Data()
            encoded.appendLittleEndian(UInt32(pathData.count))
            encoded.appendLittleEndian(UInt64(record.sampleIndex))
            encoded.appendLittleEndian(identity.sourceSize)
            encoded.appendLittleEndian(identity.sourceModifiedMilliseconds)
            encoded.appendLittleEndian(identity.analysisWidth)
            encoded.appendLittleEndian(identity.vocabularyFingerprint)
            encoded.appendLittleEndian(now)
            encoded.appendLittleEndian(UInt32(record.compressedSparseScores.count))
            encoded.append(pathData)
            encoded.append(record.compressedSparseScores)

            let recordOffset = startOffset + appendData.count
            let payloadOffset = recordOffset + encoded.count - record.compressedSparseScores.count
            latestRecordsByIdentity[identity, default: [:]][record.sampleIndex] =
                Location(recordOffset: recordOffset,
                         recordLength: encoded.count,
                         payloadOffset: payloadOffset,
                         payloadLength: record.compressedSparseScores.count,
                         savedAtMilliseconds: now)
            if isNewEntry { liveRecordCount += 1 }
            appendData.append(encoded)
            recordCount += 1
        }

        if !appendData.isEmpty {
            let handle = try FileHandle(forWritingTo: fileURL)
            try handle.seekToEnd()
            try handle.write(contentsOf: appendData)
            try handle.close()
            knownFileSize += UInt64(appendData.count)
            mappedData = nil
        }

        try compactIfNeeded()
        return makeStats()
    }

    private static var nowMilliseconds: UInt64 {
        UInt64(max(0, Date().timeIntervalSince1970 * 1_000))
    }

    private func ensureFileExists() throws {
        guard !FileManager.default.fileExists(atPath: fileURL.path) else { return }
        try FileManager.default.createDirectory(at: fileURL.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        try Self.magic.write(to: fileURL, options: .atomic)
        knownFileSize = UInt64(Self.magic.count)
        mappedData = Self.magic
        isLoaded = true
    }

    private func loadIndexIfNeeded() throws {
        guard !isLoaded else { return }
        guard let data = try readMappedDataIfPresent() else {
            latestRecordsByIdentity = [:]
            recordCount = 0
            liveRecordCount = 0
            knownFileSize = 0
            isLoaded = true
            return
        }
        guard data.starts(with: Self.magic) else {
            throw CocoaError(.fileReadCorruptFile, userInfo: [NSFilePathErrorKey: fileURL.path])
        }

        var parsed: [TagResultCacheIdentity: [Int: Location]] = [:]
        var previousPathBytes: Data?
        var previousIdentity: TagResultCacheIdentity?
        var offset = Self.magic.count
        var parsedRecordCount = 0
        var malformedTail = false

        while offset < data.count {
            let recordOffset = offset
            guard let pathLength = data.readUInt32(at: &offset), pathLength > 0,
                  pathLength <= 1_048_576,
                  let sampleIndex = data.readUInt64(at: &offset), sampleIndex <= UInt64(Int.max),
                  let sourceSize = data.readUInt64(at: &offset),
                  let sourceModified = data.readUInt64(at: &offset),
                  let analysisWidth = data.readUInt32(at: &offset), analysisWidth > 0,
                  let vocabularyFingerprint = data.readUInt64(at: &offset),
                  let savedAt = data.readUInt64(at: &offset),
                  let payloadLength = data.readUInt32(at: &offset), payloadLength > 0,
                  payloadLength <= 1_000_000,
                  offset <= data.count - Int(pathLength) - Int(payloadLength) else {
                malformedTail = true
                break
            }

            let pathEnd = offset + Int(pathLength)
            let pathBytes = data[offset..<pathEnd]
            let identity: TagResultCacheIdentity
            if let previousPathBytes, previousPathBytes == pathBytes,
               let previousIdentity,
               previousIdentity.sourceSize == sourceSize,
               previousIdentity.sourceModifiedMilliseconds == sourceModified,
               previousIdentity.analysisWidth == analysisWidth,
               previousIdentity.vocabularyFingerprint == vocabularyFingerprint {
                identity = previousIdentity
            } else {
                guard let path = String(data: pathBytes, encoding: .utf8) else {
                    malformedTail = true
                    break
                }
                identity = TagResultCacheIdentity(sourcePath: path, sourceSize: sourceSize,
                                                  sourceModifiedMilliseconds: sourceModified,
                                                  analysisWidth: analysisWidth,
                                                  vocabularyFingerprint: vocabularyFingerprint)
                previousPathBytes = pathBytes
                previousIdentity = identity
            }
            offset = pathEnd
            let payloadOffset = offset
            offset += Int(payloadLength)
            parsedRecordCount += 1

            let frameIndex = Int(sampleIndex)
            let location = Location(recordOffset: recordOffset, recordLength: offset - recordOffset,
                                    payloadOffset: payloadOffset, payloadLength: Int(payloadLength),
                                    savedAtMilliseconds: savedAt)
            if (parsed[identity]?[frameIndex]?.savedAtMilliseconds ?? 0) <= savedAt {
                parsed[identity, default: [:]][frameIndex] = location
            }
        }

        latestRecordsByIdentity = parsed
        recordCount = parsedRecordCount
        liveRecordCount = parsed.values.reduce(0) { $0 + $1.count }
        knownFileSize = UInt64(data.count)
        mappedData = data
        isLoaded = true
        if malformedTail { try compact() }
    }

    private func readMappedDataIfPresent() throws -> Data? {
        if let mappedData { return mappedData }
        guard FileManager.default.fileExists(atPath: fileURL.path) else { return nil }
        let data = try Data(contentsOf: fileURL, options: .mappedIfSafe)
        mappedData = data
        return data
    }

    private func pruneExpired(retentionDays: Int) throws {
        guard !latestRecordsByIdentity.isEmpty else { return }
        let validDays = min(max(retentionDays, 1), 36_500)
        let retentionMilliseconds = UInt64(validDays) * 86_400_000
        let now = Self.nowMilliseconds
        let cutoff = now > retentionMilliseconds ? now - retentionMilliseconds : 0
        var expired: [(TagResultCacheIdentity, Int)] = []
        for (identity, records) in latestRecordsByIdentity {
            for (sampleIndex, location) in records where location.savedAtMilliseconds < cutoff {
                expired.append((identity, sampleIndex))
            }
        }
        guard !expired.isEmpty else {
            try compactIfNeeded()
            return
        }
        for (identity, sampleIndex) in expired {
            latestRecordsByIdentity[identity]?.removeValue(forKey: sampleIndex)
            liveRecordCount -= 1
            if latestRecordsByIdentity[identity]?.isEmpty == true {
                latestRecordsByIdentity.removeValue(forKey: identity)
            }
        }
        try compact()
    }

    private func compactIfNeeded() throws {
        guard recordCount > max(liveRecordCount * 2, liveRecordCount + 1_024) else { return }
        try compact()
    }

    private func compact() throws {
        guard let data = try readMappedDataIfPresent() else { return }
        var compacted = Self.magic
        let locations = latestRecordsByIdentity.values.flatMap(\.values).sorted {
            if $0.recordOffset == $1.recordOffset { return $0.recordLength < $1.recordLength }
            return $0.recordOffset < $1.recordOffset
        }
        compacted.reserveCapacity(Self.magic.count + locations.reduce(0) { $0 + $1.recordLength })
        for location in locations {
            let range = location.recordOffset..<(location.recordOffset + location.recordLength)
            compacted.append(contentsOf: data[range])
        }
        try FileManager.default.createDirectory(at: fileURL.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        try compacted.write(to: fileURL, options: .atomic)
        latestRecordsByIdentity = [:]
        recordCount = 0
        liveRecordCount = 0
        knownFileSize = UInt64(compacted.count)
        mappedData = compacted
        isLoaded = false
        try loadIndexIfNeeded()
    }

    private func makeStats() -> TagResultCacheStats {
        return TagResultCacheStats(entryCount: liveRecordCount, fileSize: Int64(knownFileSize))
    }
}

private extension Data {
    mutating func appendLittleEndian<T: FixedWidthInteger>(_ value: T) {
        var littleEndian = value.littleEndian
        Swift.withUnsafeBytes(of: &littleEndian) { append(contentsOf: $0) }
    }

    func readUInt16(at offset: inout Int) -> UInt16? {
        guard offset >= 0, offset <= count - 2 else { return nil }
        let value = UInt16(self[startIndex + offset]) | UInt16(self[startIndex + offset + 1]) << 8
        offset += 2
        return value
    }

    func readUInt32(at offset: inout Int) -> UInt32? {
        guard offset >= 0, offset <= count - 4 else { return nil }
        var value: UInt32 = 0
        for byte in 0..<4 { value |= UInt32(self[startIndex + offset + byte]) << (byte * 8) }
        offset += 4
        return value
    }

    func readUInt64(at offset: inout Int) -> UInt64? {
        guard offset >= 0, offset <= count - 8 else { return nil }
        var value: UInt64 = 0
        for byte in 0..<8 { value |= UInt64(self[startIndex + offset + byte]) << (byte * 8) }
        offset += 8
        return value
    }
}
