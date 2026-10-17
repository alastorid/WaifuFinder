import AppKit
import Foundation

/// Opt-in Firefox native messaging bridge. No Firefox profiles or preferences are edited.
enum FirefoxIntegration {
    static let host = "com.alastorid.waifufinder"
    static let extensionID = "firefox@waifufinder.local"
    static let defaults = "mp4, dms, ts, m3u8, mov, m4v, webm, mkv"
    static var directory: URL {
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support/WaifuFinder/Firefox")
    }
    static var manifest: URL {
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support/Mozilla/NativeMessagingHosts/\(host).json")
    }
    static var config: URL { directory.appendingPathComponent("config.json") }
    static var healthFile: URL { directory.appendingPathComponent("health.json") }
    static var registered: Bool { FileManager.default.fileExists(atPath: manifest.path) && FileManager.default.fileExists(atPath: config.path) }
    static var isHealthy: Bool {
        guard registered,
              let data = try? Data(contentsOf: healthFile),
              let status = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              let lastHeartbeat = status["lastHeartbeat"] as? TimeInterval else { return false }
        let age = Date().timeIntervalSince1970 - lastHeartbeat
        return age >= -5 && age <= 150
    }

    static func suffixes(_ text: String) -> [String] {
        Array(Set(text.lowercased().split(whereSeparator: { $0 == "," || $0.isWhitespace })
            .map { $0.trimmingCharacters(in: CharacterSet(charactersIn: ".")) }
            .filter { !$0.isEmpty && $0.allSatisfy { $0.isASCII && ($0.isLetter || $0.isNumber) } })).sorted()
    }

    static func register(suffixText: String) throws {
        let suffixes = suffixes(suffixText)
        guard !suffixes.isEmpty else { throw MediaImport.failure("Enter at least one suffix, such as mp4, dms, ts or m3u8.") }
        guard Bundle.main.bundleIdentifier == host, let executable = Bundle.main.executableURL else {
            throw MediaImport.failure("Register Firefox from the built WaifuFinder.app.")
        }
        let fm = FileManager.default
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)
        try fm.createDirectory(at: manifest.deletingLastPathComponent(), withIntermediateDirectories: true)
        let launcher = directory.appendingPathComponent("waifufinder-firefox")
        let quoted = "'" + executable.path.replacingOccurrences(of: "'", with: "'\\''") + "'"
        try "#!/bin/sh\nexec \(quoted) --firefox-native-host\n".write(to: launcher, atomically: true, encoding: .utf8)
        try fm.setAttributes([.posixPermissions: 0o700], ofItemAtPath: launcher.path)
        let data: [String: Any] = ["name": host, "description": "Send media to WaifuFinder", "path": launcher.path,
                                   "type": "stdio", "allowed_extensions": [extensionID]]
        try JSONSerialization.data(withJSONObject: ["enabled": true, "suffixes": suffixes]).write(to: config, options: .atomic)
        try JSONSerialization.data(withJSONObject: data, options: .prettyPrinted).write(to: manifest, options: .atomic)
    }

    static func unregister() throws {
        // Disable first, so already-running bridge processes also stop forwarding.
        if FileManager.default.fileExists(atPath: config.path) { try FileManager.default.removeItem(at: config) }
        if FileManager.default.fileExists(atPath: manifest.path) { try FileManager.default.removeItem(at: manifest) }
        let launcher = directory.appendingPathComponent("waifufinder-firefox")
        if FileManager.default.fileExists(atPath: launcher.path) { try FileManager.default.removeItem(at: launcher) }
        if FileManager.default.fileExists(atPath: healthFile.path) { try FileManager.default.removeItem(at: healthFile) }
    }

    /// The temporary add-on loader page.
    ///
    /// Written literally: `/usr/bin/open` re-parses its argument as a URL and
    /// percent-encodes the `#` of a scheme-only URL, so this exact string used to
    /// reach Firefox as `about:debugging%23/runtime/this-firefox` — a page name that
    /// does not exist. Firefox's own command line takes the fragment verbatim, and a
    /// running instance picks the request up as a remote command line, so the
    /// launcher inside the bundle is run directly instead of going through `open`.
    static let addOnLoaderURL = "about:debugging#/runtime/this-firefox"

    /// Arguments that ask Firefox for `url` in a new tab.
    static let addOnLoaderArguments = ["-new-tab", addOnLoaderURL]

    /// The runnable launcher inside a Firefox application bundle, when there is one.
    static func launcher(in application: URL) -> URL? {
        let directory = application.appendingPathComponent("Contents/MacOS")
        let names = (try? FileManager.default.contentsOfDirectory(atPath: directory.path))?.sorted() ?? []
        for name in names where name.lowercased().hasPrefix("firefox") {
            let candidate = directory.appendingPathComponent(name)
            guard FileManager.default.isExecutableFile(atPath: candidate.path),
                  (try? candidate.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true else { continue }
            return candidate
        }
        return nil
    }

    /// Opens Firefox's temporary add-on loader and selects the packaged XPI in
    /// Finder so a local install doesn't require navigating to WaifuFinder's bundle.
    static func openExtensionSetup() throws {
        let applicationIDs = ["org.mozilla.firefox", "org.mozilla.firefoxdeveloperedition", "org.mozilla.nightly"]
        guard let firefox = applicationIDs.compactMap({
            NSWorkspace.shared.urlForApplication(withBundleIdentifier: $0)
        }).first else {
            throw MediaImport.failure("Install Firefox, then set up the WaifuFinder add-on again.")
        }
        guard let xpi = Bundle.main.resourceURL?.appendingPathComponent("WaifuFinderFirefox.xpi"),
              FileManager.default.fileExists(atPath: xpi.path) else {
            throw MediaImport.failure("The packaged Firefox add-on is missing. Rebuild WaifuFinder and try again.")
        }
        let process = Process()
        if let launcher = launcher(in: firefox) {
            process.executableURL = launcher
            process.arguments = addOnLoaderArguments
        } else {
            // Nothing runnable in the bundle: `open` still reaches the page, so drop
            // the fragment it would corrupt rather than handing over a dead URL.
            process.executableURL = URL(fileURLWithPath: "/usr/bin/open")
            process.arguments = ["-a", firefox.path, "about:debugging"]
        }
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try process.run()
        NSWorkspace.shared.activateFileViewerSelecting([xpi])
    }

    static func wrappedURL(_ url: URL) -> URL {
        var parts = URLComponents()
        parts.scheme = "waifufinder"; parts.host = "open"
        parts.queryItems = [URLQueryItem(name: "url", value: url.absoluteString)]
        return parts.url!
    }

    static func incomingURL(_ url: URL) -> URL? {
        if url.isFileURL || MediaImport.webURL(url.absoluteString) != nil { return url }
        guard url.scheme == "waifufinder", url.host == "open",
              let parts = URLComponents(url: url, resolvingAgainstBaseURL: false),
              let value = parts.queryItems?.first(where: { $0.name == "url" })?.value else { return nil }
        return MediaImport.webURL(value)
    }

    private static func readExactly(_ count: Int) -> Data? {
        var data = Data()
        while data.count < count {
            guard let chunk = try? FileHandle.standardInput.read(upToCount: count - data.count), !chunk.isEmpty else { return nil }
            data.append(chunk)
        }
        return data
    }

    /// Firefox frames JSON with a native-endian UInt32 byte count.
    static func runNativeHost() -> Never {
        while let header = readExactly(4) {
            let length = header.withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }
            guard length > 0, length <= 1_048_576, let bytes = readExactly(Int(length)),
                  let request = (try? JSONSerialization.jsonObject(with: bytes)) as? [String: Any] else { break }
            let settings = (try? Data(contentsOf: config)).flatMap { try? JSONSerialization.jsonObject(with: $0) } as? [String: Any]
            let suffixes = settings?["suffixes"] as? [String] ?? []
            let command = request["command"] as? String
            if registered && (command == "health" || command == "status") {
                recordHeartbeat()
            }
            var reply: [String: Any] = ["enabled": registered, "suffixes": suffixes, "opened": false]
            if registered, command == "open", let text = request["url"] as? String,
               let url = MediaImport.webURL(text), suffixes.contains(url.pathExtension.lowercased()) {
                let process = Process()
                process.executableURL = URL(fileURLWithPath: "/usr/bin/open")
                process.arguments = ["-b", host, wrappedURL(url).absoluteString]
                process.standardOutput = FileHandle.nullDevice
                process.standardError = FileHandle.nullDevice
                do { try process.run(); process.waitUntilExit(); reply["opened"] = process.terminationStatus == 0 }
                catch { reply["opened"] = false }
            }
            guard let payload = try? JSONSerialization.data(withJSONObject: reply) else { break }
            var size = UInt32(payload.count)
            FileHandle.standardOutput.write(withUnsafeBytes(of: &size) { Data($0) })
            FileHandle.standardOutput.write(payload)
        }
        exit(0)
    }

    private static func recordHeartbeat() {
        let status: [String: TimeInterval] = ["lastHeartbeat": Date().timeIntervalSince1970]
        guard let data = try? JSONSerialization.data(withJSONObject: status) else { return }
        try? data.write(to: healthFile, options: .atomic)
    }
}
