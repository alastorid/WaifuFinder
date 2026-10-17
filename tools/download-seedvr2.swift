import CryptoKit
import Foundation

// Setup uses native Foundation/CryptoKit and curl, without a Python interpreter.
struct Release: Decodable {
    let repository: String
    let revision: String
    let manifestSHA256: String
    let supportFilesSHA256: [String: String]
    let algorithm: String?
    let fusion: String?
    let pathPrefix: String?
    let vaeFusion: String?
}
struct Manifest: Decodable {
    let format: Int
    let algorithm: String
    let precision: String
    let groupSize: Int
    let sourceRevision: String
    let fileSHA256: [String: String]
    let fusion: String?
    let vaeFusion: String?
    let transformerAliases: [String: String]?
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
        throw NSError(domain: "SeedVR2Setup", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }
}
func validFile(_ name: String, _ hash: String) -> Bool {
    !name.isEmpty && !name.contains("\\")
        && name.split(separator: "/", omittingEmptySubsequences: false).allSatisfy { !$0.isEmpty && $0 != "." && $0 != ".." }
        && hash.count == 64 && hash.allSatisfy { $0.isHexDigit }
}
do {
    guard let releasePath = argument("--release"), let directory = argument("--destination") else {
        throw NSError(
            domain: "SeedVR2Setup", code: 1,
            userInfo: [NSLocalizedDescriptionKey: "Supply --release FILE --destination DIRECTORY"])
    }
    let release = try JSONDecoder().decode(
        Release.self, from: Data(contentsOf: URL(fileURLWithPath: releasePath)))
    try require(
        release.revision.count == 40 && release.revision.allSatisfy { $0.isHexDigit },
        "Invalid pinned release revision")
    let requestedDestination = URL(fileURLWithPath: directory, isDirectory: true).standardizedFileURL
    try FileManager.default.createDirectory(at: requestedDestination, withIntermediateDirectories: true)
    // Resolve the existing root before appending a not-yet-created file. On
    // macOS, resolving /tmp/new-file alone can retain the /tmp alias.
    let destination = requestedDestination.resolvingSymlinksInPath()
    let prefix = release.pathPrefix ?? ""
    try require(prefix.isEmpty || prefix.allSatisfy { $0.isLetter || $0.isNumber || $0 == "-" }, "Invalid release subdirectory")
    let base = argument("--base-url") ?? "https://huggingface.co/\(release.repository)/resolve/\(release.revision)\(prefix.isEmpty ? "" : "/" + prefix)"
    let verifyOnly = CommandLine.arguments.contains("--verify-only")
    func fetch(_ name: String, expected: String) throws {
        try require(validFile(name, expected), "Invalid release filename or checksum")
        let target = destination.appendingPathComponent(name)
        // Existing Hugging Face caches can use symlinks into their blob store.
        // A verified link is safe to read; new writes must remain in the destination.
        if FileManager.default.fileExists(atPath: target.path), try digest(target) == expected {
            print("Verified · \(name)")
            return
        }
        let resolvedRoot = destination.resolvingSymlinksInPath().path + "/"
        try require(target.resolvingSymlinksInPath().path.hasPrefix(resolvedRoot), "Release file escapes destination")
        let resolvedParent = target.deletingLastPathComponent().resolvingSymlinksInPath().path
        try require(resolvedParent == destination.resolvingSymlinksInPath().path
                    || resolvedParent.hasPrefix(resolvedRoot), "Release directory escapes destination")
        try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
        try require(!verifyOnly, "Missing or incompatible file: \(name)")
        let partial = destination.appendingPathComponent(name + ".part")
        try require(partial.resolvingSymlinksInPath().path.hasPrefix(resolvedRoot), "Partial file escapes destination")
        if let reuseDirectory = argument("--reuse-directory") {
            let candidate = URL(fileURLWithPath: reuseDirectory).appendingPathComponent(name).resolvingSymlinksInPath()
            if !FileManager.default.fileExists(atPath: partial.path),
               FileManager.default.fileExists(atPath: candidate.path), try digest(candidate) == expected {
                // Hard-link immutable verified cache files instead of duplicating gigabytes.
                do { try FileManager.default.linkItem(at: candidate, to: partial) }
                catch { try FileManager.default.copyItem(at: candidate, to: partial) }
            }
        }
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
    let algorithm = release.algorithm ?? "seedvr2-3b"
    try require(["seedvr2-3b", "seedvr2-7b"].contains(algorithm), "Unsupported release architecture")
    try require(release.fusion == nil || release.fusion == "packed-projections-v1", "Unsupported release fusion")
    try require(release.vaeFusion == nil || release.vaeFusion == "qkv-packed-v1", "Unsupported release VAE fusion")
    var expectedFiles: Set<String> = [
        "transformer/0.safetensors", "transformer/1.safetensors",
        "transformer/model.safetensors.index.json", "vae/0.safetensors",
        "vae/model.safetensors.index.json", "pos_emb.safetensors",
    ]
    if release.fusion == "packed-projections-v1" {
        expectedFiles.remove("transformer/0.safetensors")
        expectedFiles.remove("transformer/1.safetensors")
        expectedFiles.insert("transformer/root.safetensors")
        for index in 0..<(algorithm == "seedvr2-7b" ? 36 : 32) {
            expectedFiles.insert(String(format: "transformer/block-%03d.safetensors", index))
        }
    }
    try require(
        manifest.format == 1 && manifest.algorithm == algorithm && manifest.fusion == release.fusion && manifest.vaeFusion == release.vaeFusion
            && manifest.precision == "mlx-4bit" && manifest.groupSize == 64
            && manifest.sourceRevision == (algorithm == "seedvr2-7b" ? "22f491ba03647864491520cb803916c21eaad303" : "9b2f6c3c1b1b4548c66f302ec5a4cd3cd9937eb5")
            && Set(manifest.fileSHA256.keys) == expectedFiles
            && manifest.fileSHA256["pos_emb.safetensors"] == "d2e06cadb10f1f149b5aa781185b250eb60f0e64327f12efd17886cecc9c2984",
        "Incompatible SeedVR2 manifest")
    let embeddingOnly = CommandLine.arguments.contains("--only-embedding")
    for name in manifest.fileSHA256.keys.sorted() where !embeddingOnly || name == "pos_emb.safetensors" {
        try fetch(name, expected: manifest.fileSHA256[name]!)
    }
    if !embeddingOnly {
        for name in release.supportFilesSHA256.keys.sorted() {
            try fetch(name, expected: release.supportFilesSHA256[name]!)
        }
        // Shard indices must refer exactly to the verified model files.
        for component in ["transformer", "vae"] {
            let index = try JSONSerialization.jsonObject(with: Data(contentsOf:
                destination.appendingPathComponent(component + "/model.safetensors.index.json"))) as? [String: Any]
            let map = index?["weight_map"] as? [String: String]
            let shards = Set(expectedFiles.filter { $0.hasPrefix(component + "/") && $0.hasSuffix(".safetensors") })
            try require(map?.isEmpty == false && Set(map!.values.map { component + "/" + $0 }) == shards,
                        "Incompatible SeedVR2 shard index")
            if component == "transformer" {
                let aliases = manifest.transformerAliases ?? [:]
                try require(aliases.allSatisfy { alias, canonical in
                    let parts = alias.split(separator: ".")
                    return algorithm == "seedvr2-3b" && parts.count >= 4 && parts[0] == "blocks"
                        && Int(parts[1]).map { (10..<32).contains($0) } == true && parts[2] == "attn"
                        && alias.contains("_txt.") && canonical == alias.replacingOccurrences(of: "_txt.", with: "_vid.")
                        && map?[alias] == nil && map?[canonical] != nil && aliases[canonical] == nil
                }, "Invalid shared-weight alias")
            }
        }
    }
    print("Ready-to-use SeedVR2 release verified. No Python, conversion or re-quantization required.")
} catch {
    fputs("SeedVR2 setup: \(error.localizedDescription)\n", stderr)
    exit(1)
}
