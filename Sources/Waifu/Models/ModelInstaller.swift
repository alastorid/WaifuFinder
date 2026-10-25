import CoreML
import CryptoKit
import Foundation

struct ModelRelease: Decodable, Sendable {
    let repository: String
    let revision: String
    let manifestSHA256: String
    let supportFilesSHA256: [String: String]
    let pathPrefix: String?
    var downloadBaseURL: String? = nil
}

enum ManagedModel: String, CaseIterable, Identifiable, Sendable {
    case seed3 = "seedvr2-3b", seed7 = "seedvr2-7b", flash = "flashvsr", real = "realesrgan", pgt = "pgtformer", tagger = "wd-tagger", safire = "safire", trufor = "trufor"
    var id: String { rawValue }
    var name: String {
        switch self {
        case .seed3: "SeedVR2 3B"
        case .seed7: "SeedVR2 7B"
        case .flash: "FlashVSR 1.1 Tiny"
        case .real: "RealESRGAN x2plus"
        case .pgt: "PGTFormer"
        case .tagger: "WD SwinV2 Tagger"
        case .safire: "SAFIRE"
        case .trufor: "TruFor"
        }
    }
    var detail: String {
        switch self {
        case .seed3: "Video restoration · optimized 4-bit · about 2.4 GB"
        case .seed7: "Video restoration · optimized 4-bit · about 5.14 GB"
        case .flash: "Streaming video restoration · BF16 · 4× · about 3.51 GB"
        case .real: "Frame restoration · Core ML FP16 · about 33 MB download"
        case .pgt: "Temporal face restoration · Core ML · about 381 MB"
        case .tagger: "Frame traits · optimized native GPU · about 225 MB"
        case .safire: "Photos forensics · native Swift/MLX · CC BY 4.0 weights"
        case .trufor: "Photos forensics · native Swift/MLX · GRIP-UNINA nonprofit weights"
        }
    }
    var releaseName: String? {
        switch self {
        case .seed3: "seedvr2-release"
        case .seed7: "seedvr2-7b-release"
        case .flash: "flashvsr-release"
        case .real: "realesrgan-release"
        case .pgt: "pgtformer-release"
        case .tagger: "wd-tagger-release"
        case .safire: "safire-release"
        case .trufor: "trufor-release"
        }
    }
    /// `true` when this model's upstream licence forbids commercial use.
    ///
    /// PGTFormer's non-commercial licence and TruFor's informational/nonprofit
    /// licence travel with the weights; everything else WaifuFinder ships is Apache-2.0,
    /// BSD-3-Clause or MIT. See `WaifuDistribution` for why this is decided at
    /// build time rather than at download time.
    var isNonCommercialRestricted: Bool { self == .pgt || self == .trufor }
    var supportsDownload:Bool { true }
    /// The models this build is licensed to offer.
    ///
    /// The default build is non-commercial and offers every model. A commercial build
    /// drops the restricted ones, so the Settings row, the download button and the
    /// pinned release descriptor are absent rather than present and refused at
    /// download time.
    static var offered: [ManagedModel] {
        allCases.filter { WaifuDistribution.isNonCommercialOnly || !$0.isNonCommercialRestricted }
    }
    var workerName: String? {
        switch self {
        case .seed3, .seed7: "waifu-seedvr2"
        case .flash: "waifu-flashvsr"
        case .pgt: "waifu-pgtformer"
        case .tagger: "waifu-wdtagger"
        case .safire: "waifu-safire"
        case .trufor: "waifu-trufor"
        case .real: nil
        }
    }
    var algorithm: UncoverAlgorithm? {
        switch self {
        case .seed3: .seedVR2
        case .seed7: .seedVR2SevenB
        case .flash: .flashVSR11
        case .real: .realESRGAN
        case .pgt: .pgtFormer
        case .tagger, .safire, .trufor: nil
        }
    }
}

extension Notification.Name {
    static let waifuModelsChanged = Notification.Name("waifuModelsChanged")
}

/// All writes are inside a per-model staging folder; configuration is published last.
actor ModelInstaller {
    static let shared = ModelInstaller()
    private let configurationRoot: URL?
    /// Tests can exercise the real installer without replacing a user's models.
    init(configurationRoot: URL? = nil) { self.configurationRoot = configurationRoot }
    typealias Progress = @Sendable (String, Double?) -> Void

    static func fail(_ message: String) -> NSError {
        NSError(domain: "WaifuModels", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }
    static func safeFile(_ name: String, hash: String) -> Bool {
        !name.isEmpty && !name.contains("\\") && !name.hasPrefix("/")
            && name.split(separator: "/", omittingEmptySubsequences: false).allSatisfy { !$0.isEmpty && $0 != "." && $0 != ".." }
            && hash.count == 64 && hash.allSatisfy(\.isHexDigit)
    }
    static func digest(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hash = SHA256()
        while try autoreleasepool(invoking: { () throws -> Bool in
            try Task.checkCancellation()
            guard let data = try handle.read(upToCount: 8 * 1024 * 1024), !data.isEmpty else { return false }
            hash.update(data: data)
            return true
        }) {}
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }
    static func release(_ model: ManagedModel) throws -> ModelRelease {
        // A commercial build ships no descriptor for a non-commercial model. Saying
        // so plainly beats reporting a corrupt catalog, which would send the user
        // looking for a bug that does not exist.
        if model.isNonCommercialRestricted, !WaifuDistribution.isNonCommercialOnly {
            let terms = model == .trufor ? "informational and nonprofit use" : "non-commercial use"
            throw fail("\(model.name) is not offered in a commercial build of WaifuFinder, because its licence permits \(terms) only.")
        }
        guard let name = model.releaseName,
              let url = Bundle.main.url(forResource: name, withExtension: "json", subdirectory: "ModelReleases") else {
            if model == .safire || model == .trufor { throw fail("The validated \(model.name) checkpoint has not been published in this build’s model catalog yet.") }
            throw fail("The app is missing its model catalog. Reinstall WaifuFinder.")
        }
        return try JSONDecoder().decode(ModelRelease.self, from: Data(contentsOf: url))
    }
    static func checkedFile(_ root: URL, _ name: String, _ hash: String) throws -> URL {
        guard safeFile(name, hash: hash) else { throw fail("Invalid model file in manifest.") }
        let resolvedRoot = root.resolvingSymlinksInPath()
        var ancestor = resolvedRoot
        // Resolving a nonexistent leaf does not resolve its symlinked parents on macOS.
        for component in name.split(separator: "/") {
            ancestor = ancestor.appendingPathComponent(String(component)).resolvingSymlinksInPath()
            guard ancestor.path.hasPrefix(resolvedRoot.path + "/") else {
                throw fail("A model file points outside its folder.")
            }
        }
        return root.appendingPathComponent(name)
    }
    func fetch(_ remote: URL, to target: URL, hash: String, progress: @escaping Progress) async throws {
        try Task.checkCancellation()
        if FileManager.default.fileExists(atPath: target.path), try Self.digest(target) == hash { return }
        progress("Downloading \(target.lastPathComponent)", nil)
        let delegate = ModelDownloadProgress(progress: progress, name: target.lastPathComponent)
        let (temporary, response) = try await URLSession.shared.download(from: remote, delegate: delegate)
        defer { try? FileManager.default.removeItem(at: temporary) }
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw Self.fail("Download failed for \(target.lastPathComponent). Please retry.")
        }
        progress("Verifying \(target.lastPathComponent)", nil)
        guard try Self.digest(temporary) == hash else { throw Self.fail("Checksum verification failed. Please retry the download.") }
        try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
        if FileManager.default.fileExists(atPath: target.path) {
            _ = try FileManager.default.replaceItemAt(target, withItemAt: temporary)
        } else { try FileManager.default.moveItem(at: temporary, to: target) }
    }
    func install(_ model: ManagedModel, root: URL, progress: @escaping Progress) async throws -> URL {
        if let workerName = model.workerName {
            guard let worker = Bundle.main.url(forResource: workerName, withExtension: nil, subdirectory: "Runtime"),
                  FileManager.default.isExecutableFile(atPath: worker.path) else {
                throw Self.fail("This app does not include the \(model.name) runtime. Install the Apple Silicon build.")
            }
        }
        let release = try Self.release(model)
        // A PGTFormer upgrade gets its own directory. Keep the old installation
        // usable until all replacement bytes are verified and activation succeeds.
        let destination = Self.installationDirectory(model, release: release, root: root)
        if (model == .pgt || model == .tagger || model == .safire || model == .trufor), FileManager.default.fileExists(atPath: destination.path) {
            try await configure(model, directory: destination, progress: progress)
            return destination
        }
        let stage = root.appendingPathComponent(".\(model.rawValue)-download", isDirectory: true)
        try FileManager.default.createDirectory(at: stage, withIntermediateDirectories: true)
        let base = URL(string: release.downloadBaseURL ?? "https://huggingface.co/\(release.repository)/resolve/\(release.revision)/\(release.pathPrefix.map { $0 + "/" } ?? "")")!
        try await fetch(base.appendingPathComponent("manifest.json"), to: stage.appendingPathComponent("manifest.json"), hash: release.manifestSHA256, progress: progress)
        let hashes = try Self.hashes(stage)
        let files = hashes.merging(release.supportFilesSHA256) { original, _ in original }
        for (index, name) in files.keys.sorted().enumerated() {
            let target = try Self.checkedFile(stage, name, files[name]!)
            progress("File \(index + 1) of \(files.count) · \(name)", Double(index) / Double(files.count))
            try await fetch(base.appendingPathComponent(name), to: target, hash: files[name]!, progress: { message, fraction in
                progress("File \(index + 1) of \(files.count) · \(message)", (Double(index) + (fraction ?? 0)) / Double(files.count))
            })
        }
        progress("Download complete · setting up \(model.name)", nil)
        // The pinned Core ML bundles contain .mlpackage sources. Compiled Core ML is tied
        // to the local compiler and macOS version, so it is built here on the machine
        // that will run it rather than published.
        if model == .real {
            guard let package = try Self.package(stage, hashes, "RealESRGANValidated.mlpackage") else {
                throw Self.fail("The \(model.name) release does not contain its Core ML package.")
            }
            try Task.checkCancellation()
            progress("Setting up \(model.name) · compiling Core ML model", nil)
            let compiled = try await MLModel.compileModel(at: package)
            defer { try? FileManager.default.removeItem(at: compiled) }
            let target = stage.appendingPathComponent("model.mlmodelc")
            if FileManager.default.fileExists(atPath: target.path) { try FileManager.default.removeItem(at: target) }
            try FileManager.default.copyItem(at: compiled, to: target)
        }
        try Task.checkCancellation()
        // Never delete an existing installation to make room for a replacement.
        guard !FileManager.default.fileExists(atPath: destination.path) else {
            throw Self.fail("This download folder already contains \(model.name). Use Choose Existing… or remove the existing managed model first.")
        }
        try FileManager.default.moveItem(at: stage, to: destination)
        do { try await configure(model, directory: destination, progress: progress) }
        catch {
            try? FileManager.default.moveItem(at: destination, to: stage)
            throw error
        }
        return destination
    }
    static func installationDirectory(_ model: ManagedModel, release: ModelRelease, root: URL) -> URL {
        let name = model == .tagger ? "wd-tagger-native-v2"
            : (model == .pgt || model == .safire || model == .trufor) ? model.rawValue + "-" + String(release.revision.prefix(12)) : model.rawValue
        return root.appendingPathComponent(name, isDirectory: true)
    }
    /// The `fileSHA256` map a release manifest must publish.
    static func hashes(_ directory: URL) throws -> [String: String] {
        let data = try Data(contentsOf: directory.appendingPathComponent("manifest.json"))
        let manifest = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        guard let hashes = manifest?["fileSHA256"] as? [String: String], !hashes.isEmpty else {
            throw fail("Incomplete model manifest.")
        }
        return hashes
    }
    /// The single `.mlpackage` directory a verified manifest publishes, or `nil` when
    /// the release contains none. A package's own `Manifest.json` is what marks it as
    /// present and pinned; the caller gets the directory, because that is what Core ML
    /// compiles and loads.
    static func package(_ root: URL, _ hashes: [String: String], _ name: String) throws -> URL? {
        let marker = name + "/Manifest.json"
        guard let digest = hashes[marker] else { return nil }
        return try checkedFile(root, marker, digest).deletingLastPathComponent()
    }
    func validate(_ model: ManagedModel, directory: URL, progress: @escaping Progress) throws {
        // A legacy RealESRGAN import is a bare compiled model with no manifest.
        // It cannot be checksum-verified, so it is only accepted when the user
        // explicitly chose that folder.
        if model == .real, directory.pathExtension == "mlmodelc" {
            _ = try NativeSpatialUpscaler(modelURL: directory)
            return
        }
        if model == .trufor {
            guard WaifuDistribution.isNonCommercialOnly else { throw Self.fail("TruFor is restricted to informational and nonprofit use.") }
            _ = try TruForConfiguration.validate(directory)
            for (name,hash) in try Self.hashes(directory) {
                progress("Setting up TruFor · verifying \(name)",nil)
                guard try Self.digest(Self.checkedFile(directory,name,hash))==hash else { throw Self.fail("Missing or damaged TruFor file: \(name)") }
            }
            return
        }
        let release = try Self.release(model)
        guard try Self.digest(directory.appendingPathComponent("manifest.json")) == release.manifestSHA256 else {
            throw Self.fail("This folder does not match the supported \(model.name) release.")
        }
        let verifiedFiles = try Self.hashes(directory).merging(release.supportFilesSHA256) { pinned, _ in pinned }
        for (name, hash) in verifiedFiles {
            progress("Setting up \(model.name) · verifying \(name)", nil)
            guard try Self.digest(Self.checkedFile(directory, name, hash)) == hash else { throw Self.fail("Missing or damaged model file: \(name)") }
        }
        // Verified bytes are not a usable installation on their own: a Core ML
        // package that will not compile on this machine is not a working model.
        // The PGTFormer worker does the same for its packages when it loads them.
        if model == .tagger {
            _ = try WDTaggerBackend(directory: directory)
        }
        if model == .safire { _ = try SafireConfiguration.validate(directory) }
        if model == .real {
            _ = try NativeSpatialUpscaler(modelURL: directory.appendingPathComponent("model.mlmodelc"))
        }
    }
    func configure(_ model: ManagedModel, directory: URL, progress: @escaping Progress) async throws {
        // RealESRGAN is verified as a whole bundle — pinned manifest, every pinned
        // file, and the compiled model that will actually run — not just as the
        // compiled form. A bare .mlmodelc falls back to the load check alone.
        progress("Setting up \(model.name)", nil)
        try validate(model, directory: directory, progress: progress)
        if model == .tagger {
            guard let worker = WDTaggerWorker.executableURL else { throw Self.fail("Missing native WD Tagger runtime.") }
            let log = FileManager.default.temporaryDirectory.appendingPathComponent("tagger-setup-\(UUID().uuidString).log")
            defer { try? FileManager.default.removeItem(at: log) }
            try await UncoverProcess().run(executable: worker.path,
                arguments: ["--validate", "--weights", directory.appendingPathComponent("model.safetensors").path],
                log: log, label: "WD Tagger setup")
        }
        if model == .safire || model == .trufor {
            guard let worker = model == .trufor ? TruForConfiguration.executable : SafireConfiguration.executable else { throw Self.fail("Missing native \(model.name) runtime.") }
            let log = FileManager.default.temporaryDirectory.appendingPathComponent("\(model.rawValue)-setup-\(UUID().uuidString).log")
            defer { try? FileManager.default.removeItem(at:log) }
            try await UncoverProcess().run(executable:worker.path,arguments:["--validate","--model",directory.path],log:log,label:"\(model.name) setup")
        }
        try Task.checkCancellation()
        let configRoot = configurationRoot ?? SeedVR2Configuration.configurationURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: configRoot, withIntermediateDirectories: true)
        let configName: String
        switch model {
        case .seed3, .seed7: configName = "backend.json"
        case .flash: configName = "flashvsr.json"
        case .pgt: configName = "pgtformer.json"
        case .real: configName = "realesrgan-model.txt"
        case .tagger: configName = "wd-tagger-model.txt"
        case .safire: configName = "safire-model.txt"
        case .trufor: configName = "trufor-model.txt"
        }
        let configURL = configRoot.appendingPathComponent(configName)
        let previous = try? Data(contentsOf: configURL)
        do {
            if model == .tagger || model == .safire || model == .trufor {
                try Data(directory.path.utf8).write(to: configURL, options: .atomic)
            } else if model == .real {
                let actual = directory.pathExtension == "mlmodelc" ? directory : directory.appendingPathComponent("model.mlmodelc")
                try Data(actual.path.utf8).write(to: configRoot.appendingPathComponent("realesrgan-model.txt"), options: .atomic)
            } else {
                guard let name = model.workerName,
                      let worker = Bundle.main.url(forResource: name, withExtension: nil, subdirectory: "Runtime"),
                      FileManager.default.isExecutableFile(atPath: worker.path) else {
                    throw Self.fail("This app does not include the \(model.name) runtime. Install the Apple Silicon build.")
                }
                var arguments = ["--configure", "--model", directory.path]
                if model == .seed3 || model == .seed7 { arguments += ["--embedding", directory.appendingPathComponent("pos_emb.safetensors").path] }
                let log = FileManager.default.temporaryDirectory.appendingPathComponent("model-setup-\(UUID().uuidString).log")
                defer { try? FileManager.default.removeItem(at: log) }
                progress("Setting up \(model.name) · activating", nil)
                try await UncoverProcess().run(executable: worker.path, arguments: arguments, log: log, label: "Model setup")
            }
        } catch {
            if let previous { try previous.write(to: configURL, options: .atomic) }
            else if FileManager.default.fileExists(atPath: configURL.path) { try FileManager.default.removeItem(at: configURL) }
            throw error
        }
    }

    func diskBytes(_ directory: URL) -> Int64 {
        guard let enumerator = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey], options: []) else { return 0 }
        var total: Int64 = 0
        for case let file as URL in enumerator {
            if let values = try? file.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]), values.isRegularFile == true {
                total += Int64(values.fileSize ?? 0)
            }
        }
        return total
    }
}

private final class ModelDownloadProgress: NSObject, URLSessionDownloadDelegate, Sendable {
    let progress: ModelInstaller.Progress
    let name: String
    let startedAt = ProcessInfo.processInfo.systemUptime
    init(progress: @escaping ModelInstaller.Progress, name: String) { self.progress = progress; self.name = name }
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {}
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        let written = ByteCountFormatter.string(fromByteCount: totalBytesWritten, countStyle: .file)
        let expected = totalBytesExpectedToWrite > 0 ? ByteCountFormatter.string(fromByteCount: totalBytesExpectedToWrite, countStyle: .file) : ""
        let elapsed = max(0.001, ProcessInfo.processInfo.systemUptime - startedAt)
        let speed = Self.formattedSpeed(Double(totalBytesWritten) / elapsed)
        // Keep the rate first so it remains visible if the compact Settings row truncates the filename.
        progress("\(speed) · \(name) · \(written)\(expected.isEmpty ? "" : " / " + expected)", totalBytesExpectedToWrite > 0 ? Double(totalBytesWritten) / Double(totalBytesExpectedToWrite) : nil)
    }

    private static func formattedSpeed(_ bytesPerSecond: Double) -> String {
        let units = ["B/s", "KB/s", "MB/s", "GB/s", "TB/s"]
        var value = max(0, bytesPerSecond)
        var unit = 0
        while value >= 1_000 && unit < units.count - 1 {
            value /= 1_000
            unit += 1
        }
        return String(format: "%.1f %@", value, units[unit])
    }
}
