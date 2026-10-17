import Darwin
import CryptoKit
import Foundation

struct SeedVR2Configuration: Codable, Sendable {
    var python: String
    var model: String
    var helper: String
    var allowUnsafeMemory: Bool
    var nativeExecutable: String? = nil
    var nativeEmbedding: String? = nil
    var nativeMemoryBudgetGB: Double? = nil
    var modelVariant: String? = nil
    var optimizedModels: [String: String]? = nil
    var disconnectedVariants: [String]? = nil

    static var configurationURL: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Waifu/Uncover/backend.json")
    }

    static func load() -> Self? {
        guard let data = try? Data(contentsOf: configurationURL) else { return nil }
        return try? JSONDecoder().decode(Self.self, from: data)
    }

    /// Selection never falls back to an unfused checkpoint or Python.
    func optimized(variant: String) -> Self? {
        guard ["seedvr2-3b", "seedvr2-7b"].contains(variant), nativeExecutable != nil,
              disconnectedVariants?.contains(variant) != true else { return nil }
        let sibling = URL(fileURLWithPath: model).resolvingSymlinksInPath().deletingLastPathComponent()
            .appendingPathComponent(variant == "seedvr2-7b" ? "model-7b-optimized" : "model-optimized").path
        let candidates = [optimizedModels?[variant], model, sibling].compactMap { $0 }
        for path in candidates {
            let directory = URL(fileURLWithPath: path)
            guard let data = try? Data(contentsOf: directory.appendingPathComponent("manifest.json")),
                  let manifest = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  manifest["algorithm"] as? String == variant,
                  manifest["fusion"] as? String == "packed-projections-v1",
                  manifest["vaeFusion"] as? String == "qkv-packed-v1",
                  manifest["precision"] as? String == "mlx-4bit",
                  manifest["groupSize"] as? Int == 64 else { continue }
            var selected = self
            selected.model = path
            selected.modelVariant = variant
            selected.nativeEmbedding = directory.appendingPathComponent("pos_emb.safetensors").path
            return selected
        }
        return nil
    }
}

/// Disk-backed subprocess IO avoids both pipe deadlocks and retaining a model in
/// the app. Cancellation kills the helper's process group, including MLX-Gen.
final class UncoverProcess: @unchecked Sendable {
    private let lock = NSLock()
    private var process: Process?
    private var cancelled = false

    func cancel() {
        lock.withLock {
            cancelled = true
            if let process, process.isRunning {
                kill(-process.processIdentifier, SIGTERM)
                process.terminate()
            }
        }
    }

    func run(executable: String, arguments: [String], log: URL, label: String = "SeedVR2",
             progress: (@Sendable (String) -> Void)? = nil) async throws {
        try await withTaskCancellationHandler {
            try await Task.detached {
                let handle = try FileHandle(forWritingTo: Self.createLog(log))
                defer { try? handle.close() }
                let child = Process()
                child.executableURL = URL(fileURLWithPath: executable)
                child.arguments = arguments
                child.standardOutput = handle
                child.standardError = handle
                var environment = ProcessInfo.processInfo.environment
                environment["HF_HUB_OFFLINE"] = "1"
                environment["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
                child.environment = environment
                try self.lock.withLock {
                    guard !self.cancelled else { throw CancellationError() }
                    self.process = child
                    try child.run()
                }
                var terminationStarted: Date?
                var lastProgress = ""
                while child.isRunning {
                    if let progress, let reading = try? FileHandle(forReadingFrom: log) {
                        let end = (try? reading.seekToEnd()) ?? 0
                        try? reading.seek(toOffset: end > 8192 ? end - 8192 : 0)
                        let lines = String(decoding: (try? reading.readToEnd()) ?? Data(), as: UTF8.self).components(separatedBy: "\n")
                        try? reading.close()
                        if let line = lines.last(where: { $0.hasPrefix("Progress: ") }), line != lastProgress {
                            lastProgress = line; progress(String(line.dropFirst(10)))
                        }
                    }
                    if self.lock.withLock({ self.cancelled }) {
                        if terminationStarted == nil { terminationStarted = Date() }
                        if let terminationStarted, Date().timeIntervalSince(terminationStarted) > 2 {
                            kill(-child.processIdentifier, SIGKILL)
                            kill(child.processIdentifier, SIGKILL)
                        }
                    }
                    try await Task.sleep(nanoseconds: 50_000_000)
                }
                // isRunning is already false. A redundant waitUntilExit on an
                // async task that resumed on another thread can strand the
                // Foundation task run-loop notification, especially for setup.
                if self.lock.withLock({ self.cancelled }) { throw CancellationError() }
                guard child.terminationStatus == 0 else {
                    let reading = try FileHandle(forReadingFrom: log)
                    defer { try? reading.close() }
                    let length = try reading.seekToEnd()
                    try reading.seek(toOffset: length > 6000 ? length - 6000 : 0)
                    let suffix = String(decoding: try reading.readToEnd() ?? Data(), as: UTF8.self)
                    throw UncoverError.backend("\(label) failed (\(child.terminationStatus)). \(suffix)")
                }
            }.value
        } onCancel: { self.cancel() }
    }

    private static func createLog(_ url: URL) throws -> URL {
        guard FileManager.default.createFile(atPath: url.path, contents: nil) else {
            throw UncoverError.backend("Could not create the backend log.")
        }
        return url
    }
}

struct SeedVR2Backend: TemporalVideoUpscaler {
    let configuration: SeedVR2Configuration?
    let variant: String
    init(configuration: SeedVR2Configuration? = SeedVR2Configuration.load(), variant: String = "seedvr2-3b") {
        self.variant = variant
        self.configuration = configuration?.optimized(variant: variant)
    }
    private var modelLabel: String { variant == "seedvr2-7b" ? "SeedVR2 7B" : "SeedVR2 3B" }
    private var nativeMemoryBudgetGB: Double { configuration?.nativeMemoryBudgetGB ?? min(6, Double(ProcessInfo.processInfo.physicalMemory) * 0.5 / 1e9) }
    var cacheIdentity: String {
        let manifest = configuration.flatMap { try? Data(contentsOf: URL(fileURLWithPath: $0.model).appendingPathComponent("manifest.json")) }
        let fingerprint = manifest.map { SHA256.hash(data: $0).description } ?? "legacy"
        let modified = nativeExecutable.flatMap { try? FileManager.default.attributesOfItem(atPath: $0)[.modificationDate] as? Date }?.timeIntervalSince1970 ?? 0
        return "SeedVR2:native-scale-v1:\(variant):\(configuration?.model ?? "missing"):\(nativeExecutable ?? "missing"):\(nativeEmbedding ?? ""):\(nativeMemoryBudgetGB):\(fingerprint):\(modified)"
    }

    private var nativeExecutable: String? {
        let bundled = Bundle.main.bundleURL.appendingPathComponent("Contents/Resources/Runtime/waifu-seedvr2").path
        if FileManager.default.isExecutableFile(atPath: bundled) { return bundled }
        return configuration?.nativeExecutable
    }
    private var nativeEmbedding: String? {
        Bundle.main.url(forResource: "pos_emb", withExtension: "safetensors")?.path ?? configuration?.nativeEmbedding
    }
    func capabilities() async -> UpscalerCapabilities {
        let ready = configuration.map {
            let runtime = FileManager.default.isExecutableFile(atPath: nativeExecutable ?? "")
                    && FileManager.default.fileExists(atPath: nativeEmbedding ?? "")
                    && FileManager.default.fileExists(atPath: URL(fileURLWithPath: nativeExecutable ?? "").deletingLastPathComponent().appendingPathComponent("mlx.metallib").path)
            return runtime && FileManager.default.fileExists(atPath: $0.model + "/transformer/model.safetensors.index.json")
        } ?? false
        return UpscalerCapabilities(available: ready,
            message: ready ? "\(modelLabel) · native Swift MLX upscaling" : "Download or choose optimized \(modelLabel) in Settings → Models.",
            scales: UncoverScale.allCases, minimumFrames: 5, requiresUnsafeMemoryOptIn: false,
            maximumOutputPixels: 16_000_000)
    }

    func reconstruct(_ prepared: UncoverPrepared, allowUnsafeMemory: Bool,
                     progress: @escaping @Sendable (String) -> Void) async throws -> UncoverResult {
        guard let configuration, let executable = nativeExecutable, let embedding = nativeEmbedding,
              await capabilities().available else {
            throw UncoverError.backend("Optimized \(modelLabel) is unavailable. Open Settings → Models to set it up.")
        }
        let arguments = ["--job", prepared.directory.path, "--model", configuration.model,
                         "--embedding", embedding, "--memory-budget-gb", String(nativeMemoryBudgetGB)]
        progress("Reconstructing · \(prepared.manifest.frames.count) frames")
        let started = Date()
        try await UncoverProcess().run(executable: executable, arguments: arguments,
                                       log: prepared.directory.appendingPathComponent("backend.log"))
        try Task.checkCancellation()
        for frame in prepared.manifest.frames {
            try Task.checkCancellation()
            let image = try RegionImages.read(RegionImages.frameURL(prepared.directory, "restored", frame.index))
            guard image.width == prepared.manifest.regionWidth, image.height == prepared.manifest.regionHeight else {
                throw UncoverError.backend("SeedVR2 returned incorrect region dimensions.")
            }
            try RegionImages.write(image, to: RegionImages.frameURL(prepared.directory, "reconstructed", frame.index))
        }
        let elapsed = Date().timeIntervalSince(started)
        let report: [String: Any] = ["spatialModel": "SeedVR2 native", "stagesPerFrame": 0,
            "inputWidth": Int(ceil(Double(prepared.manifest.regionWidth) / Double(prepared.manifest.scale.rawValue))),
            "inputHeight": Int(ceil(Double(prepared.manifest.regionHeight) / Double(prepared.manifest.scale.rawValue))),
            "outputWidth": prepared.manifest.regionWidth, "outputHeight": prepared.manifest.regionHeight,
            "frames": prepared.manifest.frames.count]
        let reportData = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        try reportData.write(to: prepared.directory.appendingPathComponent("spatial-upscale-report.json"), options: .atomic)
        return UncoverResult(prepared: prepared, model: "\(modelLabel) optimized 4-bit · Swift MLX native upscaling", seconds: elapsed)
    }
}
