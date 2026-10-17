import Combine
import Compression
import Foundation

/// Source identity for reusable per-frame analysis outputs. The selected face
/// depends on the minimum-face-area setting, so that value participates in the
/// key while ranking and grouping preferences do not.
struct AnalysisResultCacheIdentity: Hashable, Sendable {
    let sourcePath: String
    let sourceSize: UInt64
    let sourceModifiedMilliseconds: UInt64
    let analysisWidth: UInt32
    let minimumFaceAreaBits: UInt32

    init?(videoURL: URL, analysisWidth: Int, minimumFaceAreaFraction: Float, analysisMode: AnalysisMode = .full) {
        if let photos = PhotosMedia.shared.cacheSource(videoURL, mode: analysisMode) {
            guard minimumFaceAreaFraction.isFinite else { return nil }
            sourcePath = photos.path
            sourceSize = photos.count
            sourceModifiedMilliseconds = 0
            self.analysisWidth = UInt32(PhotosMedia.analysisPixelSize)
            minimumFaceAreaBits = minimumFaceAreaFraction.bitPattern
            return
        }
        let source = videoURL.standardizedFileURL.resolvingSymlinksInPath()
        guard source.isFileURL,
              analysisWidth > 0, analysisWidth <= Int(UInt32.max),
              minimumFaceAreaFraction.isFinite,
              let attributes = try? FileManager.default.attributesOfItem(atPath: source.path),
              let size = (attributes[.size] as? NSNumber)?.uint64Value,
              let modified = attributes[.modificationDate] as? Date else { return nil }

        sourcePath = source.path
        sourceSize = size
        sourceModifiedMilliseconds = UInt64(max(0, modified.timeIntervalSince1970 * 1_000))
        self.analysisWidth = UInt32(analysisWidth)
        minimumFaceAreaBits = minimumFaceAreaFraction.bitPattern
    }

    fileprivate init(sourcePath: String, sourceSize: UInt64, sourceModifiedMilliseconds: UInt64,
                     analysisWidth: UInt32, minimumFaceAreaBits: UInt32) {
        self.sourcePath = sourcePath
        self.sourceSize = sourceSize
        self.sourceModifiedMilliseconds = sourceModifiedMilliseconds
        self.analysisWidth = analysisWidth
        self.minimumFaceAreaBits = minimumFaceAreaBits
    }
}

struct AnalysisResultMemoryRecord: Sendable {
    let sampleIndex: Int
    /// WAF1 envelope with a zlib-compressed frame-analysis payload.
    let compressedResult: Data
}

struct AnalysisResultCacheStats: Sendable {
    let entryCount: Int
    let fileSize: Int64
}

/// User controls and statistics for remembered frame analysis.
@MainActor
final class AnalysisResultMemory: ObservableObject {
    static let shared = AnalysisResultMemory()

    @Published var rememberAnalysisResults: Bool = UserDefaults.standard.object(forKey: "rememberAnalysisResults") as? Bool ?? true {
        didSet {
            UserDefaults.standard.set(rememberAnalysisResults, forKey: "rememberAnalysisResults")
            Task { await refresh() }
        }
    }

    @Published var retentionDays: Int = min(max(UserDefaults.standard.object(forKey: "analysisResultRetentionDays") as? Int ?? 30, 1), 36_500) {
        didSet {
            let valid = min(max(retentionDays, 1), 36_500)
            guard valid == retentionDays else {
                retentionDays = valid
                return
            }
            UserDefaults.standard.set(valid, forKey: "analysisResultRetentionDays")
            Task { await refresh() }
        }
    }

    @Published private(set) var entryCount = 0
    @Published private(set) var fileSize: Int64 = 0
    @Published private(set) var errorMessage: String?

    let fileURL = AnalysisResultDiskCache.defaultFileURL
    private let disk = AnalysisResultDiskCache.shared

    private init() {
        Task { await refresh() }
    }

    func refresh() async {
        do {
            apply(try await disk.stats(retentionDays: retentionDays))
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

    func cachedResults(for identity: AnalysisResultCacheIdentity) async -> [Int: Data] {
        guard rememberAnalysisResults else { return [:] }
        do {
            let result = try await disk.results(for: identity, retentionDays: retentionDays)
            apply(result.stats)
            return result.resultsBySampleIndex
        } catch {
            errorMessage = error.localizedDescription
            return [:]
        }
    }

    /// Adds compressed score records for frames that have not changed.
    func remember(identity: AnalysisResultCacheIdentity, frames: [AnalyzedFrame]) async {
        guard rememberAnalysisResults else { return }
        let records = Self.records(from: frames)
        guard !records.isEmpty else { return }
        do {
            apply(try await disk.remember(records, for: identity, retentionDays: retentionDays))
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func flushAtTermination(identity: AnalysisResultCacheIdentity, frames: [AnalyzedFrame]) {
        guard rememberAnalysisResults else { return }
        let records = Self.records(from: frames)
        guard !records.isEmpty else { return }
        let disk = self.disk
        let retentionDays = self.retentionDays
        let finished = DispatchSemaphore(value: 0)
        Task.detached(priority: .utility) {
            _ = try? await disk.remember(records, for: identity, retentionDays: retentionDays)
            finished.signal()
        }
        finished.wait()
    }

    private func apply(_ stats: AnalysisResultCacheStats) {
        entryCount = stats.entryCount
        fileSize = stats.fileSize
        errorMessage = nil
    }

    private static func records(from frames: [AnalyzedFrame]) -> [AnalysisResultMemoryRecord] {
        frames.compactMap { frame in
            AnalysisResultCacheCodec.encode(frame).map {
                AnalysisResultMemoryRecord(sampleIndex: frame.id.sampleIndex, compressedResult: $0)
            }
        }.sorted { $0.sampleIndex < $1.sampleIndex }
    }
}

/// Losslessly encodes the expensive per-frame Vision outputs. Thumbnail pixels,
/// tag names, grouping IDs and final ranks are deliberately not persisted.
enum AnalysisResultCacheCodec {
    private static let legacyMagic = Data([0x57, 0x41, 0x46, 0x31]) // WAF1
    private static let magic = Data([0x57, 0x41, 0x46, 0x32]) // WAF2
    private static let maximumPayloadBytes = 8 * 1_024 * 1_024 + 32

    static func encode(_ frame: AnalyzedFrame) -> Data? {
        var raw = Data()
        raw.appendLittleEndian(frame.artScore.bitPattern)

        var flags: UInt8 = 0
        if frame.faceScore != nil { flags |= 1 << 0 }
        if frame.faceDefinitionScore != nil { flags |= 1 << 1 }
        if frame.faceAreaFraction != nil { flags |= 1 << 2 }
        if frame.featurePrint != nil { flags |= 1 << 3 }
        raw.append(flags)
        raw.appendLittleEndian(UInt32(clamping: frame.faceCount))
        if let value = frame.faceScore { raw.appendLittleEndian(value.bitPattern) }
        if let value = frame.faceDefinitionScore { raw.appendLittleEndian(value.bitPattern) }
        if let value = frame.faceAreaFraction { raw.appendLittleEndian(value.bitPattern) }
        if let featurePrint = frame.featurePrint {
            guard featurePrint.count <= maximumPayloadBytes else { return nil }
            raw.appendLittleEndian(UInt32(featurePrint.count))
            raw.append(featurePrint)
        }

        var compressed = [UInt8](repeating: 0, count: raw.count + max(1_024, raw.count / 8))
        let compressedCount = raw.withUnsafeBytes { source in
            compressed.withUnsafeMutableBytes { destination in
                compression_encode_buffer(destination.bindMemory(to: UInt8.self).baseAddress!, destination.count,
                                          source.bindMemory(to: UInt8.self).baseAddress!, source.count,
                                          nil, COMPRESSION_ZLIB)
            }
        }
        guard compressedCount > 0 else { return nil }
        var result = magic
        result.appendLittleEndian(UInt32(raw.count))
        result.append(contentsOf: compressed.prefix(compressedCount))
        return result
    }

    static func decode(_ encoded: Data) -> FrameScoreResult? {
        let compressed: Data.SubSequence
        let expectedRawCount: Int?
        if encoded.count > magic.count, encoded.prefix(magic.count) == magic {
            var offset = magic.count
            guard let rawLength = encoded.readUInt32(at: &offset),
                  let exactLength = Int(exactly: rawLength),
                  (9...maximumPayloadBytes).contains(exactLength) else { return nil }
            expectedRawCount = exactLength
            compressed = encoded.dropFirst(offset)
        } else if encoded.count > legacyMagic.count, encoded.prefix(legacyMagic.count) == legacyMagic {
            expectedRawCount = nil
            compressed = encoded.dropFirst(legacyMagic.count)
        } else {
            return nil
        }

        guard let raw = decompress(compressed, expectedRawCount: expectedRawCount) else { return nil }
        var offset = 0
        guard let artBits = raw.readUInt32(at: &offset), offset < raw.count else { return nil }
        let flags = raw[offset]
        offset += 1
        guard flags & 0xf0 == 0, let faceCountValue = raw.readUInt32(at: &offset),
              let faceCount = Int(exactly: faceCountValue) else { return nil }
        let faceScore = flags & (1 << 0) != 0 ? raw.readFloat(at: &offset) : nil
        if flags & (1 << 0) != 0, faceScore == nil { return nil }
        let definition = flags & (1 << 1) != 0 ? raw.readFloat(at: &offset) : nil
        if flags & (1 << 1) != 0, definition == nil { return nil }
        let area = flags & (1 << 2) != 0 ? raw.readFloat(at: &offset) : nil
        if flags & (1 << 2) != 0, area == nil { return nil }
        var featurePrint: Data?
        if flags & (1 << 3) != 0 {
            guard let length = raw.readUInt32(at: &offset), Int(length) <= maximumPayloadBytes,
                  offset <= raw.count - Int(length) else { return nil }
            featurePrint = Data(raw[offset..<(offset + Int(length))])
            offset += Int(length)
        }
        guard offset == raw.count else { return nil }
        return FrameScoreResult(artScore: Float(bitPattern: artBits), faceScore: faceScore,
                                faceCount: faceCount, faceAreaFraction: area,
                                featurePrint: featurePrint, faceDefinitionScore: definition)
    }

    private static func decompress(_ compressed: Data.SubSequence,
                                   expectedRawCount: Int?) -> Data? {
        var capacity = expectedRawCount ?? min(maximumPayloadBytes, max(64, compressed.count * 2))
        while capacity <= maximumPayloadBytes {
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
                if expectedRawCount != nil || rawCount < capacity || capacity == maximumPayloadBytes {
                    raw.removeSubrange(rawCount..<raw.count)
                    return raw
                }
            } else if expectedRawCount != nil {
                return nil
            }
            guard capacity < maximumPayloadBytes else { return nil }
            capacity = min(maximumPayloadBytes, capacity * 2)
        }
        return nil
    }
}

fileprivate struct AnalysisResultCacheRead: Sendable {
    let resultsBySampleIndex: [Int: Data]
    let stats: AnalysisResultCacheStats
}

/// Append-only cache. Payloads remain compressed until their frame is requested.
actor AnalysisResultDiskCache {
    static let shared = AnalysisResultDiskCache()
    nonisolated static var defaultFileURL: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Waifu", isDirectory: true)
            .appendingPathComponent("analysis-results.db")
    }

    private static let magic = Data([0x57, 0x41, 0x4d, 0x31]) // WAM1
    private let fileURL = AnalysisResultDiskCache.defaultFileURL
    private var isLoaded = false
    private var mappedData: Data?
    private var latestRecordsByIdentity: [AnalysisResultCacheIdentity: [Int: Location]] = [:]
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

    func stats(retentionDays: Int) throws -> AnalysisResultCacheStats {
        try loadIndexIfNeeded()
        try pruneExpired(retentionDays: retentionDays)
        return makeStats()
    }

    func deleteAll() throws -> AnalysisResultCacheStats {
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

    fileprivate func results(for identity: AnalysisResultCacheIdentity, retentionDays: Int) throws -> AnalysisResultCacheRead {
        try loadIndexIfNeeded()
        try pruneExpired(retentionDays: retentionDays)
        guard let data = try readMappedDataIfPresent() else {
            return AnalysisResultCacheRead(resultsBySampleIndex: [:], stats: makeStats())
        }
        var results: [Int: Data] = [:]
        for (sampleIndex, location) in latestRecordsByIdentity[identity] ?? [:] {
            let range = location.payloadOffset..<(location.payloadOffset + location.payloadLength)
            results[sampleIndex] = data[range]
        }
        return AnalysisResultCacheRead(resultsBySampleIndex: results, stats: makeStats())
    }

    func remember(_ records: [AnalysisResultMemoryRecord], for identity: AnalysisResultCacheIdentity,
                  retentionDays: Int) throws -> AnalysisResultCacheStats {
        try loadIndexIfNeeded()
        try pruneExpired(retentionDays: retentionDays)
        guard !records.isEmpty else { return makeStats() }
        guard let pathData = identity.sourcePath.data(using: .utf8), pathData.count <= 1_048_576 else {
            return makeStats()
        }
        try ensureFileExists()
        let existingData = try readMappedDataIfPresent()
        let now = Self.nowMilliseconds
        var appendData = Data()
        let startOffset = Int(knownFileSize)

        for record in records where record.sampleIndex >= 0 && !record.compressedResult.isEmpty {
            let isNewEntry = latestRecordsByIdentity[identity]?[record.sampleIndex] == nil
            if let existing = latestRecordsByIdentity[identity]?[record.sampleIndex],
               existing.payloadLength == record.compressedResult.count, let existingData {
                let range = existing.payloadOffset..<(existing.payloadOffset + existing.payloadLength)
                if existingData[range] == record.compressedResult { continue }
            }
            guard record.compressedResult.count <= 8 * 1_024 * 1_024 else { continue }

            var encoded = Data()
            encoded.appendLittleEndian(UInt32(pathData.count))
            encoded.appendLittleEndian(UInt64(record.sampleIndex))
            encoded.appendLittleEndian(identity.sourceSize)
            encoded.appendLittleEndian(identity.sourceModifiedMilliseconds)
            encoded.appendLittleEndian(identity.analysisWidth)
            encoded.appendLittleEndian(identity.minimumFaceAreaBits)
            encoded.appendLittleEndian(now)
            encoded.appendLittleEndian(UInt32(record.compressedResult.count))
            encoded.append(pathData)
            encoded.append(record.compressedResult)

            let recordOffset = startOffset + appendData.count
            let payloadOffset = recordOffset + encoded.count - record.compressedResult.count
            latestRecordsByIdentity[identity, default: [:]][record.sampleIndex] =
                Location(recordOffset: recordOffset, recordLength: encoded.count,
                         payloadOffset: payloadOffset,
                         payloadLength: record.compressedResult.count,
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
        try FileManager.default.createDirectory(at: fileURL.deletingLastPathComponent(), withIntermediateDirectories: true)
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
        guard data.count >= Self.magic.count, data.prefix(Self.magic.count) == Self.magic else {
            throw AnalysisResultCacheError.invalidHeader(fileURL.path)
        }

        var parsed: [AnalysisResultCacheIdentity: [Int: Location]] = [:]
        var previousPathBytes: Data?
        var previousIdentity: AnalysisResultCacheIdentity?
        var offset = Self.magic.count
        var parsedRecordCount = 0
        var malformedTail = false
        while offset < data.count {
            let recordOffset = offset
            guard let pathLength = data.readUInt32(at: &offset), pathLength <= 1_048_576,
                  let sampleIndex = data.readUInt64(at: &offset), sampleIndex <= UInt64(Int.max),
                  let sourceSize = data.readUInt64(at: &offset),
                  let sourceModified = data.readUInt64(at: &offset),
                  let width = data.readUInt32(at: &offset), width > 0,
                  let minimumFaceAreaBits = data.readUInt32(at: &offset),
                  let savedAt = data.readUInt64(at: &offset),
                  let payloadLength = data.readUInt32(at: &offset), payloadLength <= 8 * 1_024 * 1_024,
                  offset <= data.count - Int(pathLength) else {
                malformedTail = true
                break
            }
            let pathEnd = offset + Int(pathLength)
            let pathBytes = data[offset..<pathEnd]
            offset += Int(pathLength)
            let identity: AnalysisResultCacheIdentity
            if let previousPathBytes, previousPathBytes == pathBytes,
               let previousIdentity,
               previousIdentity.sourceSize == sourceSize,
               previousIdentity.sourceModifiedMilliseconds == sourceModified,
               previousIdentity.analysisWidth == width,
               previousIdentity.minimumFaceAreaBits == minimumFaceAreaBits {
                identity = previousIdentity
            } else {
                guard let path = String(data: pathBytes, encoding: .utf8) else {
                    malformedTail = true
                    break
                }
                identity = AnalysisResultCacheIdentity(sourcePath: path, sourceSize: sourceSize,
                                                       sourceModifiedMilliseconds: sourceModified,
                                                       analysisWidth: width,
                                                       minimumFaceAreaBits: minimumFaceAreaBits)
                previousPathBytes = pathBytes
                previousIdentity = identity
            }
            let payloadOffset = offset
            guard offset <= data.count - Int(payloadLength) else { malformedTail = true; break }
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
        let days = min(max(retentionDays, 1), 36_500)
        let age = UInt64(days) * 86_400_000
        let now = Self.nowMilliseconds
        let cutoff = now > age ? now - age : 0
        var expired: [(AnalysisResultCacheIdentity, Int)] = []
        for (identity, records) in latestRecordsByIdentity {
            for (sampleIndex, location) in records where location.savedAtMilliseconds < cutoff {
                expired.append((identity, sampleIndex))
            }
        }
        guard !expired.isEmpty else { try compactIfNeeded(); return }
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
        let locations = latestRecordsByIdentity.values.flatMap(\.values).sorted { $0.recordOffset < $1.recordOffset }
        compacted.reserveCapacity(Self.magic.count + locations.reduce(0) { $0 + $1.recordLength })
        for location in locations {
            let range = location.recordOffset..<(location.recordOffset + location.recordLength)
            compacted.append(contentsOf: data[range])
        }
        try FileManager.default.createDirectory(at: fileURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try compacted.write(to: fileURL, options: .atomic)
        latestRecordsByIdentity = [:]
        recordCount = 0
        liveRecordCount = 0
        knownFileSize = UInt64(compacted.count)
        mappedData = compacted
        isLoaded = false
        try loadIndexIfNeeded()
    }

    private func makeStats() -> AnalysisResultCacheStats {
        return AnalysisResultCacheStats(entryCount: liveRecordCount, fileSize: Int64(knownFileSize))
    }
}

private enum AnalysisResultCacheError: LocalizedError {
    case invalidHeader(String)
    var errorDescription: String? {
        switch self {
        case .invalidHeader(let path): return "The analysis cache has an invalid file header: \(path)"
        }
    }
}

private extension Data {
    mutating func appendLittleEndian<T: FixedWidthInteger>(_ value: T) {
        var littleEndian = value.littleEndian
        Swift.withUnsafeBytes(of: &littleEndian) { append(contentsOf: $0) }
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

    func readFloat(at offset: inout Int) -> Float? {
        readUInt32(at: &offset).map(Float.init(bitPattern:))
    }
}
