import Foundation

struct PGTFormerConfiguration: Codable, Sendable {
    let executable: String
    let model: String
    let memoryBudgetGB: Double
    static var configurationURL: URL { SeedVR2Configuration.configurationURL.deletingLastPathComponent().appendingPathComponent("pgtformer.json") }
    static func load() -> Self? {
        guard let data = try? Data(contentsOf:configurationURL), let config = try? JSONDecoder().decode(Self.self,from:data),
              config.memoryBudgetGB.isFinite, config.memoryBudgetGB > 0,
              config.memoryBudgetGB*1e9 <= Double(ProcessInfo.processInfo.physicalMemory)/2 else { return nil }
        return config
    }
}
struct PGTFormerBackend: TemporalVideoUpscaler {
    let configuration: PGTFormerConfiguration?
    init(configuration: PGTFormerConfiguration? = PGTFormerConfiguration.load()) { self.configuration = configuration }
    private var executable: String? {
        let bundled = Bundle.main.bundleURL.appendingPathComponent("Contents/Resources/Runtime/waifu-pgtformer").path
        return FileManager.default.isExecutableFile(atPath:bundled) ? bundled : configuration?.executable
    }
    var cacheIdentity: String {
        let manifest = configuration.flatMap { try? Data(contentsOf:URL(fileURLWithPath:$0.model).appendingPathComponent("manifest.json")) }
        return "PGTFormer:gpu-encoder-decoder-v2:encoder-fp32:bilinear-rgb:window3:center1:\(configuration?.model ?? "missing"):\(configuration?.memoryBudgetGB ?? 6):\(manifest?.base64EncodedString() ?? "missing")"
    }
    func capabilities() async -> UpscalerCapabilities {
        let ready = configuration.map { config in
            guard let executable, FileManager.default.isExecutableFile(atPath:executable),
                  let data = try? Data(contentsOf:URL(fileURLWithPath:config.model).appendingPathComponent("manifest.json")),
                  let metadata = try? JSONSerialization.jsonObject(with:data) as? [String:Any] else { return false }
            return metadata["validated"] as? Bool == true && metadata["algorithm"] as? String == "pgtformer"
                && metadata["upstreamRevision"] as? String == "e78a26cd463182f4a5aa96b8681110da8615d44d"
                && metadata["runtime"] as? String == "cpu-codebook-gpu-decoder-v1"
                && (metadata["optimizations"] as? [String:Any])?["fusedCodeAttention"] as? Bool == true
                && (metadata["splitPackageSHA256"] as? [String:String])?.isEmpty == false
                && FileManager.default.fileExists(atPath:config.model+"/PGTEncoder.mlpackage/Manifest.json")
                && FileManager.default.fileExists(atPath:config.model+"/PGTDecoder.mlpackage/Manifest.json")
        } ?? false
        return UpscalerCapabilities(available:ready,
            message:ready ? "PGTFormer · fused attention · Apple GPU" : "Download the optimized PGTFormer package in Settings → Models.",
            scales:UncoverScale.allCases,minimumFrames:1,requiresUnsafeMemoryOptIn:false,maximumOutputPixels:16_000_000)
    }
    func reconstruct(_ prepared: UncoverPrepared, allowUnsafeMemory: Bool, progress: @escaping @Sendable (String)->Void) async throws -> UncoverResult {
        guard let config = configuration, let executable, await capabilities().available else { throw UncoverError.backend("Open Settings → Models to choose a validated PGTFormer package.") }
        let started = Date()
        progress("Loading PGTFormer")
        var arguments = ["--job",prepared.directory.path,"--model",config.model,"--memory-budget-gb",String(config.memoryBudgetGB)]
        if ProcessInfo.processInfo.environment["WAIFU_PGT_VALIDATE_TENSORS"] == "1" {
            arguments += ["--validation-dir",prepared.directory.appendingPathComponent("validation-tensors").path]
        }
        try await UncoverProcess().run(executable:executable,
            arguments:arguments,
            log:prepared.directory.appendingPathComponent("backend.log"),label:"PGTFormer",progress:progress)
        try Task.checkCancellation()
        for frame in prepared.manifest.frames {
            let image = try RegionImages.read(RegionImages.frameURL(prepared.directory,"reconstructed",frame.index))
            guard image.width == prepared.manifest.regionWidth, image.height == prepared.manifest.regionHeight else { throw UncoverError.backend("PGTFormer returned incorrect ROI dimensions.") }
        }
        return UncoverResult(prepared:prepared,model:"PGTFormer · temporal face restoration · Core ML",seconds:Date().timeIntervalSince(started))
    }
}
