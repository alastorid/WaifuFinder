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
        if let photos = PhotosMedia.shared.cacheSource(videoURL, mode: analysisMode) {
            sourcePath = photos.path + "/tagger-direct-v1"
            sourceSize = photos.count
            sourceModifiedMilliseconds = 0
            self.analysisWidth = UInt32(WDTaggerBackend.imageSize)
            vocabularyFingerprint = vocabulary.cacheFingerprint
            return
        }
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

    fileprivate init(sourcePath: String, sourceSize: UInt64, sourceModifiedMilliseconds: UInt64,
                     analysisWidth: UInt32, vocabularyFingerprint: UInt64) {
        self.sourcePath = sourcePath
        self.sourceSize = sourceSize
        self.sourceModifiedMilliseconds = sourceModifiedMilliseconds
        self.analysisWidth = analysisWidth
        self.vocabularyFingerprint = vocabularyFingerprint
    }
}

struct TagResultMemoryRecord: Sendable {
    let sampleIndex: Int
    /// WDF8: zlib-compressed sparse pairs of vocabulary index and FP8 score.
    let compressedSparseScores: Data
}

struct TagResultCacheStats: Sendable {
    let entryCount: Int
    let fileSize: Int64
}

struct TagResultCacheRead: Sendable {
    let probabilitiesBySampleIndex: [Int: Data]
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
    private let disk = TagResultDiskCache.shared

    private init() {
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

    func cachedResults(for identity: TagResultCacheIdentity) async -> [Int: Data] {
        guard rememberTaggingResults else { return [:] }
        do {
            let result = try await disk.results(for: identity, retentionDays: retentionDays)
            apply(result.stats)
            return result.probabilitiesBySampleIndex
        } catch {
            errorMessage = error.localizedDescription
            return [:]
        }
    }

    /// Adds missing or changed runtime results. Names are regenerated from the
    /// vocabulary when read and never written into the cache.
    func remember(identity: TagResultCacheIdentity, frames: [AnalyzedFrame]) async {
        guard rememberTaggingResults else { return }
        // Decoding every full score vector and compressing its sparse form can
        // take seconds for a long video. Never do that on the UI executor.
        let records = await Task.detached(priority: .utility) {
            Self.records(from: frames, identity: identity)
        }.value
        guard rememberTaggingResults else { return }
        guard !records.isEmpty else { return }
        do {
            let stats = try await disk.remember(records, for: identity, retentionDays: retentionDays)
            apply(stats)
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    /// Shutdown hook: all normal writes are asynchronous, but wait for this final
    /// delta so process exit cannot discard a newly completed frame result.
    func flushAtTermination(identity: TagResultCacheIdentity, frames: [AnalyzedFrame]) {
        guard rememberTaggingResults else { return }
        let disk = self.disk
        let retentionDays = self.retentionDays
        let finished = DispatchSemaphore(value: 0)
        Task.detached(priority: .utility) {
            let records = Self.records(from: frames, identity: identity)
            _ = try? await disk.remember(records, for: identity, retentionDays: retentionDays)
            finished.signal()
        }
        finished.wait()
    }

    func restore(_ frames: [AnalyzedFrame], identity: TagResultCacheIdentity,
                 vocabulary: TagVocabulary, threshold: Float) async -> RestoredTagResults {
        guard rememberTaggingResults else { return RestoredTagResults(frames: frames, restoredCount: 0) }
        let cached = await cachedResults(for: identity)
        guard !cached.isEmpty else { return RestoredTagResults(frames: frames, restoredCount: 0) }

        return await Task.detached(priority: .utility) {
            var restoredCount = 0
            let restoredFrames = frames.map { original -> AnalyzedFrame in
                guard original.tagRating == nil,
                      let encoded = cached[original.id.sampleIndex],
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

    private nonisolated static func records(from frames: [AnalyzedFrame],
                                identity: TagResultCacheIdentity) -> [TagResultMemoryRecord] {
        var recordsBySampleIndex: [Int: Data] = [:]
        for frame in frames {
            if frame.tagVocabularyFingerprint == identity.vocabularyFingerprint,
               let probabilities = frame.tagProbabilities,
               let sparseScores = TagResultCacheProbabilityCodec.encode(probabilities) {
                recordsBySampleIndex[frame.id.sampleIndex] = sparseScores
            }
        }
        return recordsBySampleIndex.map {
            TagResultMemoryRecord(sampleIndex: $0.key, compressedSparseScores: $0.value)
        }.sorted { $0.sampleIndex < $1.sampleIndex }
    }
}

/// Persist only scores above the fixed default display threshold. Each saved
/// item is a two-byte vocabulary index followed by its positive E4M3 FP8 value;
/// all omitted indices reconstruct as zero. Tag names are never stored.
private enum TagResultCacheProbabilityCodec {
    private static let legacyMagic = Data([0x57, 0x44, 0x46, 0x38]) // WDF8
    private static let magic = Data([0x57, 0x44, 0x46, 0x39]) // WDF9
    private static let threshold = WaifuSettings.default.taggerThreshold
    private static let headerByteCount = 4
    private static let pairByteCount = 3
    private static let maximumUncompressedByteCount = headerByteCount + 4 + TagVocabulary.headCount * pairByteCount

    static func encode(_ runtimeData: Data) -> Data? {
        guard let probabilities = TagProbabilityCodec.decode(runtimeData),
              probabilities.count == TagVocabulary.headCount else { return nil }

        let sparsePairs = probabilities.enumerated().compactMap { index, ratio -> (Int, UInt8)? in
            guard ratio > threshold, let fp8 = encodeE4M3(ratio) else { return nil }
            return (index, fp8)
        }
        guard sparsePairs.count <= UInt16.max else { return nil }

        var raw = Data()
        raw.appendLittleEndian(UInt16(TagVocabulary.headCount))
        raw.appendLittleEndian(UInt16(sparsePairs.count))
        for (index, fp8) in sparsePairs {
            raw.appendLittleEndian(UInt16(index))
            raw.append(fp8)
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
        let compressed: Data.SubSequence
        let expectedRawCount: Int?
        if encoded.count > magic.count, encoded.prefix(magic.count) == magic {
            var offset = magic.count
            guard let rawLength = encoded.readUInt16(at: &offset),
                  let exactLength = Int(exactly: rawLength),
                  (headerByteCount...maximumUncompressedByteCount).contains(exactLength) else { return nil }
            expectedRawCount = exactLength
            compressed = encoded.dropFirst(offset)
        } else if encoded.count > legacyMagic.count, encoded.prefix(legacyMagic.count) == legacyMagic {
            expectedRawCount = nil
            compressed = encoded.dropFirst(legacyMagic.count)
        } else {
            return nil
        }

        guard let raw = decompress(compressed, expectedRawCount: expectedRawCount),
              raw.count >= headerByteCount else { return nil }

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
            let fp8 = raw[offset]
            offset += 1
            guard let ratio = decodeE4M3(fp8), ratio > threshold else { return nil }
            probabilities[index] = ratio
            previousIndex = index
        }
        return probabilities
    }

    private static func decompress(_ compressed: Data.SubSequence,
                                   expectedRawCount: Int?) -> Data? {
        var capacity = expectedRawCount ?? min(maximumUncompressedByteCount, max(16, compressed.count * 2))
        while capacity <= maximumUncompressedByteCount {
            var raw = Data(count: capacity)
            let rawCount = compressed.withUnsafeBytes { source in
                raw.withUnsafeMutableBytes { destination in
                    compression_decode_buffer(destination.bindMemory(to: UInt8.self).baseAddress!, destination.count,
                                              source.bindMemory(to: UInt8.self).baseAddress!, source.count,
                                              nil, COMPRESSION_ZLIB)
                }
            }
            if rawCount > 0 {
                if let expectedRawCount, rawCount != expectedRawCount { return nil }
                if expectedRawCount != nil || rawCount < capacity || capacity == maximumUncompressedByteCount {
                    raw.removeSubrange(rawCount..<raw.count)
                    return raw
                }
            } else if expectedRawCount != nil {
                return nil
            }
            guard capacity < maximumUncompressedByteCount else { return nil }
            capacity = min(maximumUncompressedByteCount, capacity * 2)
        }
        return nil
    }

    /// Positive, normalized E4M3. The stored range is greater than 0.35 through 1.
    private static func encodeE4M3(_ value: Float) -> UInt8? {
        guard value.isFinite, value > threshold, value <= 1 else { return nil }
        var exponent = Int(floor(log2(value)))
        let significand = value / pow(2, Float(exponent))
        var mantissa = Int(((significand - 1) * 8).rounded())
        if mantissa == 8 {
            exponent += 1
            mantissa = 0
        }
        let exponentField = exponent + 7
        guard (1...14).contains(exponentField), (0..<8).contains(mantissa) else { return nil }
        return UInt8((exponentField << 3) | mantissa)
    }

    private static func decodeE4M3(_ value: UInt8) -> Float? {
        let exponentField = Int((value >> 3) & 0x0f)
        let mantissa = Int(value & 0x07)
        guard value & 0x80 == 0, (1...14).contains(exponentField) else { return nil }
        let significand = 1 + Float(mantissa) / 8
        let decoded = significand * pow(2, Float(exponentField - 7))
        guard decoded.isFinite, decoded <= 1 else { return nil }
        return decoded
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

    private static let magic = Data([0x57, 0x54, 0x52, 0x31]) // WTR1
    private let fileURL = TagResultDiskCache.defaultFileURL
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

    func results(for identity: TagResultCacheIdentity, retentionDays: Int) throws -> TagResultCacheRead {
        try loadIndexIfNeeded()
        try pruneExpired(retentionDays: retentionDays)
        guard let data = try readMappedDataIfPresent() else {
            return TagResultCacheRead(probabilitiesBySampleIndex: [:], stats: makeStats())
        }

        var results: [Int: Data] = [:]
        for (sampleIndex, location) in latestRecordsByIdentity[identity] ?? [:] {
            let range = location.payloadOffset..<(location.payloadOffset + location.payloadLength)
            results[sampleIndex] = data[range]
        }
        return TagResultCacheRead(probabilitiesBySampleIndex: results, stats: makeStats())
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
