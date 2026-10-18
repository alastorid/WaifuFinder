import Foundation
import AppKit

/// Errors that can occur while preparing a thumbnail for the in-memory cache.
enum ThumbnailStoreError: Error, LocalizedError, Equatable {
    case invalidKey(String)
    case pathTraversalAttempt(String)
    case writeFailed(String)
    case readFailed(String)
    case createDirectoryFailed(String)
    case removeFailed(String)

    var errorDescription: String? {
        switch self {
        case .invalidKey(let key): return "Invalid thumbnail key: \(key)"
        case .pathTraversalAttempt(let key): return "Invalid thumbnail key: \(key)"
        case .writeFailed(let detail): return "Failed to encode thumbnail: \(detail)"
        case .readFailed(let detail): return "Failed to read thumbnail: \(detail)"
        case .createDirectoryFailed(let detail): return "Failed to create thumbnail cache directory: \(detail)"
        case .removeFailed(let detail): return "Failed to clear thumbnail cache: \(detail)"
        }
    }

    static func == (lhs: ThumbnailStoreError, rhs: ThumbnailStoreError) -> Bool {
        switch (lhs, rhs) {
        case (.invalidKey(let l), .invalidKey(let r)),
             (.pathTraversalAttempt(let l), .pathTraversalAttempt(let r)),
             (.writeFailed(let l), .writeFailed(let r)),
             (.readFailed(let l), .readFailed(let r)),
             (.createDirectoryFailed(let l), .createDirectoryFailed(let r)),
             (.removeFailed(let l), .removeFailed(let r)):
            return l == r
        default:
            return false
        }
    }
}

/// Thread-safe, memory-only storage for analysis-resolution JPEG thumbnails.
///
/// Images are compressed before being retained so browsing large videos uses much
/// less memory than keeping every decoded pixel buffer. No cache directory or
/// thumbnail file is created. The optional root URL initializer and URL helpers
/// remain as logical identifiers for older callers; they never touch the filesystem.
final class ThumbnailCache: @unchecked Sendable {
    private struct Address: Hashable {
        let videoStem: String
        let key: String
    }

    private struct Entry {
        let data: Data
        let sequence: UInt64
        var previous: Address?
        var next: Address?
    }

    private let lock = NSLock()
    private var entries: [Address: Entry] = [:]
    // Intrusive links keep cache hits, replacement, and eviction O(1). A large
    // album must not scan or shift its whole cache for every visible thumbnail.
    private var leastRecent: Address?
    private var mostRecent: Address?
    private var cachedByteCount = 0
    private var nextSequence: UInt64 = 0
    private let rootDirectory: URL
    private static let jpegQuality: Float = 0.8
    private let byteLimit: Int
    private static let defaultByteLimit = 64 * 1024 * 1024

    init() {
        byteLimit = Self.defaultByteLimit
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        rootDirectory = appSupport
            .appendingPathComponent("Waifu", isDirectory: true)
            .appendingPathComponent("Thumbnails", isDirectory: true)
    }

    /// Retained for callers that need a stable logical cache namespace in fixtures.
    init(rootDirectory: URL, byteLimit: Int = 64 * 1024 * 1024) {
        self.rootDirectory = rootDirectory
        self.byteLimit = max(0, byteLimit)
    }

    /// Returns a logical per-video cache URL without creating a directory.
    func cacheDirectory(for videoStem: String) async throws -> URL {
        rootDirectory.appendingPathComponent(sanitizePathComponent(videoStem), isDirectory: true)
    }

    /// Encodes a thumbnail as JPEG in memory and stores the bytes under its key.
    func save(image: NSImage, forKey key: String, videoStem: String) async throws {
        let safeKey = try sanitizeKey(key)
        let jpegData = try autoreleasepool {
            guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
                throw ThumbnailStoreError.writeFailed("Image has no CGImage representation")
            }
            return try Self.encodeJPEG(cgImage)
        }
        try store(jpegData, forKey: safeKey, videoStem: videoStem)
    }

    /// Adds already-compressed thumbnail bytes to the bounded LRU.
    func store(_ data: Data, forKey key: String, videoStem: String) throws {
        let safeKey = try sanitizeKey(key)
        guard !data.isEmpty else { throw ThumbnailStoreError.writeFailed("Thumbnail data is empty") }
        let address = Address(videoStem: sanitizePathComponent(videoStem), key: safeKey)
        lock.withLock {
            if let previous = entries[address] {
                unlink(address)
                cachedByteCount -= previous.data.count
            }
            nextSequence &+= 1
            entries[address] = Entry(data: data, sequence: nextSequence, previous: nil, next: nil)
            appendMostRecent(address)
            cachedByteCount += data.count
            evictUntilWithinLimit()
        }
    }

    func load(forKey key: String, videoStem: String) async -> NSImage? {
        guard let data = thumbnailDataSync(forKey: key, videoStem: videoStem) else { return nil }
        return NSImage(data: data)
    }

    /// Async form used by export and other actor-friendly callers.
    func thumbnailData(forKey key: String, videoStem: String, touchLRU: Bool = true,
                       allowPrefixMatch: Bool = true) async -> Data? {
        thumbnailDataSync(forKey: key, videoStem: videoStem, touchLRU: touchLRU,
                          allowPrefixMatch: allowPrefixMatch)
    }

    /// Synchronous read for SwiftUI and the nonisolated export-store bridge.
    func thumbnailDataSync(forKey key: String, videoStem: String, touchLRU: Bool = true,
                           allowPrefixMatch: Bool = true) -> Data? {
        guard let safeKey = try? sanitizeKey(key) else { return nil }
        let stem = sanitizePathComponent(videoStem)
        return lock.withLock {
            let exactAddress = Address(videoStem: stem, key: safeKey)
            if let exact = entries[exactAddress] {
                if touchLRU { touch(exactAddress) }
                return exact.data
            }

            // Grid requests already carry the complete analysis key. A miss is
            // final; only legacy export keys need the prefix fallback below.
            guard allowPrefixMatch else { return nil }

            // Export requests historically identify the sample without the analysis
            // width/run suffix. Pick the newest matching in-memory thumbnail.
            let prefix = safeKey + "-"
            guard let match = entries
                .filter({ $0.key.videoStem == stem && $0.key.key.hasPrefix(prefix) })
                .max(by: { $0.value.sequence < $1.value.sequence }) else { return nil }
            if touchLRU { touch(match.key) }
            return match.value.data
        }
    }

    func clear(for videoStem: String) async throws {
        let stem = sanitizePathComponent(videoStem)
        lock.withLock {
            let removed = entries.keys.filter { $0.videoStem == stem }
            for address in removed {
                unlink(address)
                if let entry = entries.removeValue(forKey: address) {
                    cachedByteCount -= entry.data.count
                }
            }
        }
    }

    func clear() async throws {
        removeAll()
    }

    /// Releases the current in-memory cache synchronously when a new analysis starts.
    func removeAll() {
        lock.withLock {
            entries.removeAll(keepingCapacity: false)
            leastRecent = nil
            mostRecent = nil
            cachedByteCount = 0
        }
    }

    private func touch(_ address: Address) {
        guard mostRecent != address else { return }
        unlink(address)
        appendMostRecent(address)
    }

    /// All link mutations occur while holding `lock`.
    private func unlink(_ address: Address) {
        guard let entry = entries[address] else { return }
        if let previous = entry.previous { entries[previous]?.next = entry.next }
        else { leastRecent = entry.next }
        if let next = entry.next { entries[next]?.previous = entry.previous }
        else { mostRecent = entry.previous }
    }

    private func appendMostRecent(_ address: Address) {
        entries[address]?.previous = mostRecent
        entries[address]?.next = nil
        if let previous = mostRecent { entries[previous]?.next = address }
        else { leastRecent = address }
        mostRecent = address
    }

    private func evictUntilWithinLimit() {
        while cachedByteCount > byteLimit, let oldest = leastRecent {
            unlink(oldest)
            if let removed = entries.removeValue(forKey: oldest) {
                cachedByteCount -= removed.data.count
            }
        }
    }

    static func encodeJPEG(_ image: CGImage) throws -> Data {
        let bitmapRep = NSBitmapImageRep(cgImage: image)
        guard let data = bitmapRep.representation(using: .jpeg, properties: [.compressionFactor: jpegQuality]) else {
            throw ThumbnailStoreError.writeFailed("Failed to encode JPEG")
        }
        return data
    }

    private func sanitizeKey(_ key: String) throws -> String {
        let trimmed = key.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, trimmed.count <= 255 else { throw ThumbnailStoreError.invalidKey(key) }
        if trimmed.contains("/") || trimmed.contains("\\") {
            throw ThumbnailStoreError.pathTraversalAttempt(key)
        }
        for component in trimmed.split(separator: ".", omittingEmptySubsequences: false) {
            if component.isEmpty || component == "." || component == ".." {
                throw ThumbnailStoreError.pathTraversalAttempt(key)
            }
        }
        return trimmed
    }

    private func sanitizePathComponent(_ component: String) -> String {
        var result = component
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "\\", with: "_")
        if result.utf8.count > 255 { result = String(result.prefix(255)) }
        return result.isEmpty ? "unknown" : result
    }
}
