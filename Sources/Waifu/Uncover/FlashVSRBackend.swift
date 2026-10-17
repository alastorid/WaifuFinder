import Foundation

struct FlashVSRConfiguration: Codable, Sendable {
    let executable: String
    let model: String
    let memoryBudgetGB: Double
    var validBudget: Bool {
        memoryBudgetGB.isFinite && memoryBudgetGB > 0
            && memoryBudgetGB * 1e9 <= Double(ProcessInfo.processInfo.physicalMemory) / 2
    }
    static var configurationURL: URL {
        SeedVR2Configuration.configurationURL.deletingLastPathComponent().appendingPathComponent("flashvsr.json")
    }
    static func load() -> Self? {
        guard let data = try? Data(contentsOf: configurationURL) else { return nil }
        guard let config = try? JSONDecoder().decode(Self.self, from: data), config.validBudget else { return nil }
        return config
    }
}

struct FlashVSRBackend: TemporalVideoUpscaler {
    static let outputScale = 4
    let configuration: FlashVSRConfiguration?
    init(configuration: FlashVSRConfiguration? = FlashVSRConfiguration.load()) { self.configuration = configuration }
    private var executable: String? {
        let bundled = Bundle.main.bundleURL.appendingPathComponent("Contents/Resources/Runtime/waifu-flashvsr").path
        if FileManager.default.isExecutableFile(atPath: bundled) { return bundled }
        return configuration?.executable
    }
    var cacheIdentity: String {
        let manifest = configuration.map { URL(fileURLWithPath: $0.model).appendingPathComponent("manifest.json") }
        let changed = manifest.flatMap { try? $0.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate }?.timeIntervalSince1970 ?? 0
        return "FlashVSR1.1:27561b186ded3402d7c975f4fd722e2885b6135f:BF16:LCSA11:KV3:ADAIN:output4:\(configuration?.model ?? "missing"):\(changed):\(configuration?.memoryBudgetGB ?? 6)"
    }
    func capabilities() async -> UpscalerCapabilities {
        let ready = configuration.map { config in
            guard config.validBudget, let executable,
                  let data = try? Data(contentsOf: URL(fileURLWithPath: config.model).appendingPathComponent("manifest.json")),
                  let manifest = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  manifest["format"] as? Int == 1,
                  manifest["revision"] as? String == "27561b186ded3402d7c975f4fd722e2885b6135f",
                  let weights = manifest["weightMap"] as? [String: [String: Any]], weights.count == 899,
                  let hashes = manifest["fileSHA256"] as? [String: String], hashes.count == 34,
                  Set(weights.values.compactMap { $0["file"] as? String }).union(["prompt.safetensors"]) == Set(hashes.keys),
                  hashes.keys.allSatisfy({ !$0.contains("/") && !$0.contains("..") && FileManager.default.fileExists(atPath: config.model + "/" + $0) }) else { return false }
            return FileManager.default.isExecutableFile(atPath: executable)
                && FileManager.default.fileExists(atPath: URL(fileURLWithPath: executable).deletingLastPathComponent().appendingPathComponent("mlx.metallib").path)
                && FileManager.default.fileExists(atPath: config.model + "/manifest.json")
                && FileManager.default.fileExists(atPath: config.model + "/prompt.safetensors")
        } ?? false
        let safeBudget = configuration.flatMap { $0.validBudget ? $0.memoryBudgetGB : nil } ?? 6
        let area = max(0, Int((safeBudget * 1e9 - 3.6e9) / 12_000))
        return UpscalerCapabilities(available: ready,
            message: ready ? "FlashVSR 1.1 Tiny · native Swift MLX · tuned for 4× super-resolution" : "Download or choose our converted FlashVSR weights in Settings → Models.",
            scales: UncoverScale.allCases, minimumFrames: 5, requiresUnsafeMemoryOptIn: false, maximumPaddedArea: area)
    }
    func reconstruct(_ prepared: UncoverPrepared, allowUnsafeMemory: Bool,
                     progress: @escaping @Sendable (String) -> Void) async throws -> UncoverResult {
        guard let config = configuration, let executable, await capabilities().available else {
            throw UncoverError.backend("Open Settings → Models to set up FlashVSR.")
        }
        progress("FlashVSR 1.1 · \(prepared.manifest.frames.count) frames")
        let started = Date()
        try await UncoverProcess().run(executable: executable,
            arguments: ["--job", prepared.directory.path, "--model", config.model, "--memory-budget-gb", String(config.memoryBudgetGB)],
            log: prepared.directory.appendingPathComponent("backend.log"), label: "FlashVSR 1.1")
        try Task.checkCancellation()
        for frame in prepared.manifest.frames {
            let image = try RegionImages.read(RegionImages.frameURL(prepared.directory, "reconstructed", frame.index))
            guard image.width == prepared.manifest.regionWidth * Self.outputScale,
                  image.height == prepared.manifest.regionHeight * Self.outputScale else {
                throw UncoverError.backend("FlashVSR returned incorrect ROI geometry.")
            }
        }
        return UncoverResult(prepared: prepared, model: "FlashVSR 1.1 Tiny BF16 · Swift MLX · 4× output",
                             seconds: Date().timeIntervalSince(started), outputScale: Self.outputScale)
    }
}
