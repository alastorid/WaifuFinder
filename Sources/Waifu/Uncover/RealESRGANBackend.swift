import CoreImage
import Foundation

struct RealESRGANBackend: TemporalVideoUpscaler {
    private static var configurationURL: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Waifu/Uncover/realesrgan-model.txt")
    }

    private var modelPath: String? {
        if let data = try? Data(contentsOf: Self.configurationURL),
           let path = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines),
           !path.isEmpty { return path }
        // Existing installations stored the same model path in backend.json.
        let legacyURL = Self.configurationURL.deletingLastPathComponent().appendingPathComponent("backend.json")
        guard let data = try? Data(contentsOf: legacyURL),
              let fields = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        return fields["spatialUpscaler"] as? String
    }

    var cacheIdentity: String {
        let path = modelPath ?? "missing"
        let modified = (try? FileManager.default.attributesOfItem(atPath: path)[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
        return "RealESRGAN:x2plus-fp16-runtime-v3:straight-rgb:\(path):\(modified)"
    }

    func capabilities() async -> UpscalerCapabilities {
        let ready = modelPath.map { FileManager.default.fileExists(atPath: $0) } ?? false
        return UpscalerCapabilities(available: ready,
            message: ready ? "RealESRGAN x2plus · Core ML" : "Download or choose RealESRGAN in Settings → Models.",
            scales: UncoverScale.allCases, minimumFrames: 1, requiresUnsafeMemoryOptIn: false,
            maximumOutputPixels: 4_000_000)
    }

    func reconstruct(_ prepared: UncoverPrepared, allowUnsafeMemory: Bool,
                     progress: @escaping @Sendable (String) -> Void) async throws -> UncoverResult {
        guard let path = modelPath, await capabilities().available else {
            throw UncoverError.backend("Download or choose RealESRGAN in Settings → Models.")
        }
        let jobLock = try RealESRGANJobLock()
        defer { withExtendedLifetime(jobLock) {} }
        let memory = RealESRGANMemorySampler()
        let started = Date()
        let loadStart = Date()
        let upscaler = try NativeSpatialUpscaler(modelURL: URL(fileURLWithPath: path))
        let loadSeconds = Date().timeIntervalSince(loadStart)
        let context = CIContext(options: [.cacheIntermediates: false])
        // At 1×, run one restoration pass and return to the unchanged ROI size.
        let stages = max(1, prepared.manifest.scale.stages)
        for frame in prepared.manifest.frames {
            try Task.checkCancellation()
            try memory.checkBudget()
            progress("RealESRGAN · frame \(frame.index + 1)/\(prepared.manifest.frames.count)")
            try autoreleasepool {
            var image = try RegionImages.read(RegionImages.frameURL(prepared.directory, "degraded", frame.index))
            for _ in 0..<stages {
                try Task.checkCancellation()
                image = try upscaler.upscale(image)
                try memory.checkBudget()
            }
            if image.width != prepared.manifest.regionWidth || image.height != prepared.manifest.regionHeight {
                image = try RegionImages.resize(image,
                    to: CGSize(width: prepared.manifest.regionWidth, height: prepared.manifest.regionHeight), context: context)
            }
            try RegionImages.write(image, to: RegionImages.frameURL(prepared.directory, "reconstructed", frame.index))
            }
        }
        let report:[String:Any] = ["peakPhysicalFootprintBytes":memory.peakBytes,"memoryBudgetBytes":memory.budget,"memoryScope":"whole application process; sampled every 20 ms; checked between predictions","model":"realesrgan-x2plus","backend":"native Swift Core ML","pythonRuntime":false,"frames":prepared.manifest.frames.count,"scale":prepared.manifest.scale.rawValue,"sourceDownsampling":prepared.manifest.scale.rawValue,"stagesPerFrame":stages,"loadSeconds":loadSeconds,"inferenceSeconds":upscaler.predictionSeconds,"seconds":Date().timeIntervalSince(started),"predictionSizes":upscaler.predictionSizes,"computeUnits":upscaler.selectedComputeUnits,"reusedInputBuffers":true,"singleTileBlendBypass":true]
        try JSONSerialization.data(withJSONObject:report,options:[.sortedKeys,.prettyPrinted]).write(to:prepared.directory.appendingPathComponent("report.json"),options:.atomic)
        return UncoverResult(prepared: prepared, model: "RealESRGAN x2plus · Core ML FP16", seconds: Date().timeIntervalSince(started))
    }
}
