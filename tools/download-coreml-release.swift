import CryptoKit
import Foundation

// Verifies a hosted Core ML bundle: the manifest is fetched first and its pinned
// files are then downloaded and checked against it, so an interrupted or tampered
// download never produces a usable-looking model directory. Native Foundation,
// CryptoKit and curl only — no Python, matching the SeedVR2 and FlashVSR setup.
struct Release: Decodable {
    let repository: String
    let revision: String
    let manifestSHA256: String
    let supportFilesSHA256: [String: String]
    let pathPrefix: String?
}
struct Manifest: Decodable {
    let format: Int
    let algorithm: String
    let validated: Bool
    let fileSHA256: [String: String]
}
func argument(_ name: String) -> String? {
    guard let index = CommandLine.arguments.firstIndex(of: name), index + 1 < CommandLine.arguments.count
    else { return nil }
    return CommandLine.arguments[index + 1]
}
func digest(_ url: URL) throws -> String {
    let file = try FileHandle(forReadingFrom: url)
    defer { try? file.close() }
    var hash = SHA256()
    while try autoreleasepool(invoking: { () throws -> Bool in
        guard let data = try file.read(upToCount: 8 * 1024 * 1024), !data.isEmpty else { return false }
        hash.update(data: data)
        return true
    }) {}
    return hash.finalize().map { String(format: "%02x", $0) }.joined()
}
func require(_ condition: Bool, _ message: String) throws {
    guard condition else {
        throw NSError(domain: "CoreMLSetup", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }
}
func validFile(_ name: String, _ hash: String) -> Bool {
    !name.isEmpty && !name.contains("\\")
        && name.split(separator: "/", omittingEmptySubsequences: false).allSatisfy { !$0.isEmpty && $0 != "." && $0 != ".." }
        && hash.count == 64 && hash.allSatisfy(\.isHexDigit)
}
do {
    guard let releasePath = argument("--release"), let directory = argument("--destination"),
          let algorithm = argument("--algorithm") else {
        throw NSError(
            domain: "CoreMLSetup", code: 1,
            userInfo: [NSLocalizedDescriptionKey: "Supply --release FILE --destination DIRECTORY --algorithm NAME"])
    }
    let release = try JSONDecoder().decode(
        Release.self, from: Data(contentsOf: URL(fileURLWithPath: releasePath)))
    try require(
        release.revision.count == 40 && release.revision.allSatisfy(\.isHexDigit),
        "Invalid pinned release revision")
    let requestedDestination = URL(fileURLWithPath: directory, isDirectory: true).standardizedFileURL
    try FileManager.default.createDirectory(at: requestedDestination, withIntermediateDirectories: true)
    // Canonicalize the existing root before checking paths for new members.
    let destination = requestedDestination.resolvingSymlinksInPath()
    let prefix = release.pathPrefix ?? ""
    try require(prefix.isEmpty || prefix.allSatisfy { $0.isLetter || $0.isNumber || $0 == "-" }, "Invalid release subdirectory")
    let base = argument("--base-url")
        ?? "https://huggingface.co/\(release.repository)/resolve/\(release.revision)\(prefix.isEmpty ? "" : "/" + prefix)"
    let verifyOnly = CommandLine.arguments.contains("--verify-only")
    func fetch(_ name: String, expected: String) throws {
        try require(validFile(name, expected), "Invalid release filename or checksum")
        let target = destination.appendingPathComponent(name)
        if FileManager.default.fileExists(atPath: target.path), try digest(target) == expected {
            print("Verified · \(name)")
            return
        }
        let resolvedRoot = destination.resolvingSymlinksInPath().path + "/"
        try require(target.resolvingSymlinksInPath().path.hasPrefix(resolvedRoot), "Release file escapes destination")
        try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
        try require(!verifyOnly, "Missing or incompatible file: \(name)")
        let partial = destination.appendingPathComponent(name + ".part")
        try require(partial.resolvingSymlinksInPath().path.hasPrefix(resolvedRoot), "Partial file escapes destination")
        if FileManager.default.fileExists(atPath: partial.path), try digest(partial) == expected {
            // A prior run may have finished downloading before setup was interrupted.
            if FileManager.default.fileExists(atPath: target.path) {
                _ = try FileManager.default.replaceItemAt(target, withItemAt: partial)
            } else { try FileManager.default.moveItem(at: partial, to: target) }
            print("Recovered completed download · \(name)")
            return
        }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/curl")
        process.arguments = [
            "--location", "--fail", "--retry", "2", "--connect-timeout", "30", "--continue-at", "-",
            "--output", partial.path, base + "/" + name,
        ]
        print("Downloading · \(name)")
        try process.run()
        process.waitUntilExit()
        try require(process.terminationStatus == 0, "Download failed; rerun setup to resume \(name)")
        guard try digest(partial) == expected else {
            try? FileManager.default.removeItem(at: partial)
            try require(false, "Checksum mismatch: \(name). The existing model was preserved; retry setup.")
            return
        }
        // Keep a previous good file until a complete replacement has been verified.
        if FileManager.default.fileExists(atPath: target.path) {
            _ = try FileManager.default.replaceItemAt(target, withItemAt: partial)
        } else {
            try FileManager.default.moveItem(at: partial, to: target)
        }
    }
    try fetch("manifest.json", expected: release.manifestSHA256)
    let manifest = try JSONDecoder().decode(
        Manifest.self, from: Data(contentsOf: destination.appendingPathComponent("manifest.json")))
    try require(
        manifest.format == 1 && manifest.algorithm == algorithm && manifest.validated
            && manifest.fileSHA256.values.allSatisfy { $0.count == 64 && $0.allSatisfy(\.isHexDigit) },
        "Incompatible \(algorithm) manifest")
    for name in manifest.fileSHA256.keys.sorted() { try fetch(name, expected: manifest.fileSHA256[name]!) }
    for name in release.supportFilesSHA256.keys.sorted() {
        try fetch(name, expected: release.supportFilesSHA256[name]!)
    }
    print("Hosted \(algorithm) release verified. No original checkpoint, Python or conversion required.")
} catch {
    fputs("Core ML setup: \(error.localizedDescription)\n", stderr)
    exit(1)
}
