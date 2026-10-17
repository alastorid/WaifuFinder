import AVFoundation
import Foundation

/// Remembers a playlist's content type so AVFoundation can open extensionless
/// or misnamed HLS resources throughout the loaded-media session.
enum VideoAssetFactory {
    private final class Registry: @unchecked Sendable {
        private let lock = NSLock()
        private var mimeTypes: [String: String] = [:]

        func set(_ mimeType: String?, for url: URL) {
            lock.lock()
            defer { lock.unlock() }
            if let mimeType { mimeTypes[url.absoluteString] = mimeType }
            else { mimeTypes.removeValue(forKey: url.absoluteString) }
        }

        func get(for url: URL) -> String? {
            lock.lock()
            defer { lock.unlock() }
            return mimeTypes[url.absoluteString]
        }
    }

    private static let registry = Registry()

    static func setMIMEType(_ mimeType: String?, for url: URL) {
        registry.set(mimeType, for: url)
    }

    static func asset(for url: URL) -> AVURLAsset {
        let options: [String: Any]
        if let mimeType = registry.get(for: url) {
            options = [AVURLAssetOverrideMIMETypeKey: mimeType]
        } else {
            options = [:]
        }
        return AVURLAsset(url: url, options: options)
    }

    static func detectedMIMEType(at url: URL) -> String? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        let data = (try? handle.read(upToCount: 512)) ?? Data()
        let bytes = [UInt8](data)
        if String(decoding: data.prefix(16), as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines.union(CharacterSet(charactersIn: "\u{FEFF}")))
            .hasPrefix("#EXTM3U") {
            return "application/x-mpegURL"
        }
        if bytes.count >= 12, Array(bytes[4..<8]) == Array("ftyp".utf8) {
            let brand = String(decoding: bytes[8..<12], as: UTF8.self)
            if brand == "qt  " { return "video/quicktime" }
            if brand.hasPrefix("3g") { return "video/3gpp" }
            return "video/mp4"
        }
        if bytes.starts(with: [0x1A, 0x45, 0xDF, 0xA3]) {
            let header = String(decoding: data, as: UTF8.self)
            return header.contains("webm") ? "video/webm" : "video/x-matroska"
        }
        if bytes.count >= 12, Array(bytes[0..<4]) == Array("RIFF".utf8),
           Array(bytes[8..<11]) == Array("AVI".utf8) {
            return "video/x-msvideo"
        }
        if bytes.count >= 376, bytes[0] == 0x47, bytes[188] == 0x47 {
            return "video/mp2t"
        }
        if bytes.starts(with: [0x00, 0x00, 0x01, 0xBA]) { return "video/mpeg" }
        return nil
    }
}
