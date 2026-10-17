import CryptoKit
import Foundation

// Setup uses native Foundation/CryptoKit and curl, without a Python interpreter.
struct Release: Decodable {
    let repository: String
    let revision: String
    let manifestSHA256: String
    let supportFilesSHA256: [String: String]
}
struct Manifest: Decodable {
    let format: Int
    let revision: String
    let precision: String
    let fileSHA256: [String: String]
    let weightMap: [String: Weight]
    struct Weight: Decodable { let file: String }
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
    if !condition {
        throw NSError(domain: "FlashVSRSetup", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }
}
func validFile(_ name: String, _ hash: String) -> Bool {
    !name.isEmpty && !name.contains("/") && !name.contains("..")
        && hash.count == 64 && hash.allSatisfy { $0.isHexDigit }
}
do {
    guard let releasePath = argument("--release"), let directory = argument("--destination") else {
        throw NSError(
            domain: "FlashVSRSetup", code: 1,
            userInfo: [NSLocalizedDescriptionKey: "Supply --release FILE --destination DIRECTORY"])
    }
    let release = try JSONDecoder().decode(
        Release.self, from: Data(contentsOf: URL(fileURLWithPath: releasePath)))
    try require(
        release.revision.count == 40 && release.revision.allSatisfy { $0.isHexDigit },
        "Invalid pinned release revision")
    let destination = URL(fileURLWithPath: directory, isDirectory: true)
    try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
    let base =
        argument("--base-url") ?? "https://huggingface.co/\(release.repository)/resolve/\(release.revision)"
    let verifyOnly = CommandLine.arguments.contains("--verify-only")
    func fetch(_ name: String, expected: String) throws {
        try require(validFile(name, expected), "Invalid release filename or checksum")
        let target = destination.appendingPathComponent(name)
        if FileManager.default.fileExists(atPath: target.path), try digest(target) == expected {
            print("Verified · \(name)")
            return
        }
        try require(!verifyOnly, "Missing or incompatible file: \(name)")
        let partial = destination.appendingPathComponent(name + ".part")
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
        manifest.format == 1 && manifest.revision == "27561b186ded3402d7c975f4fd722e2885b6135f"
            && manifest.precision == "bfloat16" && manifest.weightMap.count == 899
            && manifest.fileSHA256.count == 34
            && Set(manifest.weightMap.values.map(\.file)).union(["prompt.safetensors"])
                == Set(manifest.fileSHA256.keys),
        "Incompatible FlashVSR manifest")
    for name in manifest.fileSHA256.keys.sorted() { try fetch(name, expected: manifest.fileSHA256[name]!) }
    for name in release.supportFilesSHA256.keys.sorted() {
        try fetch(name, expected: release.supportFilesSHA256[name]!)
    }
    print("Converted FlashVSR release verified. No original checkpoints or Python conversion required.")
} catch {
    fputs("FlashVSR setup: \(error.localizedDescription)\n", stderr)
    exit(1)
}
