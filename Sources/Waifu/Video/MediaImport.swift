import Foundation

/// Owns imported media for as long as analysis, previews and exports need it.
final class PreparedMedia: Sendable {
    let url: URL
    let directory: URL?
    private let mimeTypeOverride: String?
    init(url: URL, directory: URL? = nil, mimeTypeOverride: String? = nil) {
        self.url = url
        self.directory = directory
        self.mimeTypeOverride = mimeTypeOverride
        if let mimeTypeOverride { VideoAssetFactory.setMIMEType(mimeTypeOverride, for: url) }
    }
    deinit {
        if PhotosMedia.isPhotos(url) { PhotosMedia.shared.release(url) }
        if mimeTypeOverride != nil { VideoAssetFactory.setMIMEType(nil, for: url) }
        if let directory { try? FileManager.default.removeItem(at: directory) }
    }
}

enum MediaImport {
    static func webURL(_ text: String) -> URL? {
        guard let url = URL(string: text.trimmingCharacters(in: .whitespacesAndNewlines)),
              ["http", "https"].contains(url.scheme?.lowercased() ?? ""),
              let host = url.host, !host.isEmpty, url.user == nil, url.password == nil else { return nil }
        return url
    }

    private static func isPlaylist(_ file: URL) throws -> Bool {
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        let prefix = try handle.read(upToCount: 4096) ?? Data()
        return String(decoding: prefix, as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines.union(CharacterSet(charactersIn: "\u{FEFF}")))
            .hasPrefix("#EXTM3U")
    }

    private struct PlaylistResponse {
        let data: Data
        let url: URL
    }

    /// AVAssetReader cannot decode a remote HLS playlist. Materialize the
    /// common, finite MPEG-TS variant as one seekable transport stream, which
    /// AVFoundation can read natively. No third-party media decoder is used.
    private static func downloadTSPlaylist(_ source: URL, to destination: URL) async throws {
        var current = source
        var visited = Set<String>()
        for _ in 0..<6 {
            try Task.checkCancellation()
            guard visited.insert(current.absoluteString).inserted else {
                throw failure("The HLS playlist redirects to itself.")
            }
            let playlist = try await playlistResponse(current)
            let lines = String(decoding: playlist.data, as: UTF8.self)
                .trimmingCharacters(in: .whitespacesAndNewlines.union(CharacterSet(charactersIn: "\u{FEFF}")))
                .components(separatedBy: .newlines)
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            guard lines.first == "#EXTM3U" else {
                throw failure("The URL did not return a valid HLS playlist.")
            }

            if lines.contains(where: { $0.hasPrefix("#EXT-X-STREAM-INF:") }) {
                var variants: [(bandwidth: Int, url: URL)] = []
                for index in lines.indices where lines[index].hasPrefix("#EXT-X-STREAM-INF:") {
                    guard let next = lines.dropFirst(index + 1).first(where: { !$0.isEmpty && !$0.hasPrefix("#") }),
                          let variantURL = URL(string: next, relativeTo: playlist.url)?.absoluteURL,
                          webURL(variantURL.absoluteString) != nil else { continue }
                    let attributes = lines[index].drop(after: "#EXT-X-STREAM-INF:")
                    let bandwidth = attributes.split(separator: ",").first(where: { $0.hasPrefix("BANDWIDTH=") })
                        .flatMap { Int($0.dropFirst("BANDWIDTH=".count)) } ?? 0
                    variants.append((bandwidth, variantURL))
                }
                guard let best = variants.max(by: { $0.bandwidth < $1.bandwidth }) else {
                    throw failure("The HLS master playlist has no supported video variant.")
                }
                current = best.url
                continue
            }

            if lines.contains(where: { $0.hasPrefix("#EXT-X-MAP:") }) {
                throw failure("This HLS playlist uses fragmented MP4 segments. WaifuFinder can currently analyze MPEG-TS HLS playlists.")
            }
            if lines.contains(where: { line in
                guard line.hasPrefix("#EXT-X-KEY:") else { return false }
                return !line.contains("METHOD=NONE")
            }) {
                throw failure("Encrypted HLS playlists are not supported by the native frame reader.")
            }
            if lines.contains(where: { $0.hasPrefix("#EXT-X-BYTERANGE:") }) {
                throw failure("HLS playlists that use byte-range segments are not supported.")
            }
            guard lines.contains("#EXT-X-ENDLIST") else {
                throw failure("Live HLS streams cannot be analyzed as a finite video. Use a completed playlist.")
            }

            let segments = lines.filter { !$0.isEmpty && !$0.hasPrefix("#") }
            guard !segments.isEmpty else { throw failure("The HLS playlist contains no media segments.") }
            guard FileManager.default.createFile(atPath: destination.path, contents: nil) else {
                throw failure("WaifuFinder could not create a temporary file for the HLS stream.")
            }
            let handle = try FileHandle(forWritingTo: destination)
            defer { try? handle.close() }
            for segment in segments {
                try Task.checkCancellation()
                guard let segmentURL = URL(string: segment, relativeTo: playlist.url)?.absoluteURL,
                      webURL(segmentURL.absoluteString) != nil else {
                    throw failure("The HLS playlist contains an invalid segment URL.")
                }
                var request = URLRequest(url: segmentURL)
                request.timeoutInterval = 60
                let (data, response) = try await URLSession.shared.data(for: request)
                guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
                    throw failure("An HLS media segment could not be downloaded.")
                }
                guard isMPEGTransportStream(data) else {
                    throw failure("This HLS playlist contains a segment format other than MPEG-TS.")
                }
                try handle.write(contentsOf: data)
            }
            try handle.close()
            return
        }
        throw failure("The HLS master playlist has too many nested variants.")
    }

    private static func playlistResponse(_ url: URL) async throws -> PlaylistResponse {
        var request = URLRequest(url: url)
        request.timeoutInterval = 30
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw failure("The HLS playlist server returned an unsuccessful HTTP response.")
        }
        guard data.count <= 4 * 1024 * 1024 else {
            throw failure("The HLS playlist is larger than the supported 4 MB limit.")
        }
        guard let finalURL = response.url, webURL(finalURL.absoluteString) != nil else {
            throw failure("The HLS playlist redirected to an unsupported URL.")
        }
        return PlaylistResponse(data: data, url: finalURL)
    }

    private static func isMPEGTransportStream(_ data: Data) -> Bool {
        let bytes = [UInt8](data.prefix(188 * 3))
        guard !bytes.isEmpty, bytes[0] == 0x47 else { return false }
        return stride(from: 188, to: bytes.count, by: 188).allSatisfy { bytes[$0] == 0x47 }
    }

    static func prepare(_ source: URL) async throws -> PreparedMedia {
        try Task.checkCancellation()
        if PhotosMedia.isPhotos(source) { return PreparedMedia(url: source) }
        guard source.isFileURL || webURL(source.absoluteString) != nil else {
            throw failure("Enter a complete HTTP or HTTPS media URL.")
        }
        if source.isFileURL {
            let mimeTypeOverride = VideoAssetFactory.detectedMIMEType(at: source)
            let prepared = PreparedMedia(url: source, mimeTypeOverride: mimeTypeOverride)
            do {
                _ = try await VideoProbe.probe(source)
                try Task.checkCancellation()
                return prepared
            } catch {
                if Task.isCancelled { throw CancellationError() }
                throw unsupportedMedia(error)
            }
        }

        var request = URLRequest(url: source)
        request.timeoutInterval = 30
        let (download, response) = try await URLSession.shared.download(for: request)
        defer { try? FileManager.default.removeItem(at: download) }
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw failure("The media server returned an unsuccessful HTTP response.")
        }
        try Task.checkCancellation()
        if try isPlaylist(download) {
            // AVAssetReader is file-oriented, so turn a finite TS-based HLS
            // playlist into a seekable file before native probing/analysis.
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent("waifu-import-\(UUID())")
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let local = directory.appendingPathComponent("playlist.ts")
            do {
                try await downloadTSPlaylist(response.url ?? source, to: local)
            } catch {
                try? FileManager.default.removeItem(at: directory)
                if Task.isCancelled { throw CancellationError() }
                throw error
            }
            let prepared = PreparedMedia(url: local, directory: directory,
                                         mimeTypeOverride: "video/mp2t")
            do {
                _ = try await VideoProbe.probe(local)
                try Task.checkCancellation()
                return prepared
            } catch {
                if Task.isCancelled { throw CancellationError() }
                throw unsupportedMedia(error)
            }
        }

        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("waifu-import-\(UUID())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let local = directory.appendingPathComponent("download")
        try FileManager.default.moveItem(at: download, to: local)
        let prepared = PreparedMedia(url: local, directory: directory,
                                     mimeTypeOverride: VideoAssetFactory.detectedMIMEType(at: local))
        do {
            // Inspect downloaded bytes with AVFoundation rather than trusting the
            // URL suffix or the server's Content-Type header.
            _ = try await VideoProbe.probe(local)
            try Task.checkCancellation()
            return prepared
        } catch {
            if Task.isCancelled { throw CancellationError() }
            throw unsupportedMedia(error)
        }
    }

    private static func unsupportedMedia(_ error: Error) -> NSError {
        failure("Apple’s built-in video reader could not open this media. Try a video format supported by macOS. \(error.localizedDescription)")
    }

    static func failure(_ message: String) -> NSError {
        NSError(domain: "WaifuFinder.MediaImport", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }
}

private extension String {
    func drop(after prefix: String) -> Substring {
        guard hasPrefix(prefix) else { return dropFirst() }
        return dropFirst(prefix.count)
    }
}
