import CoreImage
import Foundation

struct RealESRGANBackend: TemporalVideoUpscaler {
    let variant: RealESRGANVariant
    private let overrideURL: URL?
    init(variant: RealESRGANVariant = .x2plus, modelURL: URL? = nil) {
        self.variant = variant; overrideURL = modelURL
    }
    private var modelPath: String? { (overrideURL ?? variant.modelURL)?.path }
    var cacheIdentity: String { variant.cacheIdentity + (overrideURL?.path ?? "") }

    func capabilities() async -> UpscalerCapabilities {
        let ready = modelPath.map { FileManager.default.fileExists(atPath: $0) } ?? false
        return UpscalerCapabilities(available: ready,
            message: ready ? variant.modelName : "Download or choose RealESRGAN in Settings → Models.",
            scales: UncoverScale.allCases, minimumFrames: 1, requiresUnsafeMemoryOptIn: false,
            maximumOutputPixels: 4_000_000)
    }

    func reconstruct(_ prepared: UncoverPrepared, allowUnsafeMemory: Bool,
                     progress: @escaping @Sendable (String) -> Void) async throws -> UncoverResult {
        guard let path = modelPath, await capabilities().available else {
            throw UncoverError.backend("Download or choose RealESRGAN in Settings → Models.")
        }
        if let request = prepared.manifest.realESRGANImage {
            try request.validate(variant)
            guard prepared.manifest.frames.count == 1 else { throw UncoverError.backend("Photo requests contain one image.") }
            let source = try RegionImages.read(RegionImages.frameURL(prepared.directory,"degraded",0))
            guard source.width == prepared.manifest.regionWidth, source.height == prepared.manifest.regionHeight else { throw UncoverError.backend("Photo input geometry differs.") }
            let started = Date()
            let image = try await RealESRGANImageRuntime.shared.restore(source,variant:variant,modelURL:URL(fileURLWithPath:path))
            try Task.checkCancellation()
            try RegionImages.write(image,to:RegionImages.frameURL(prepared.directory,"reconstructed",0))
            return UncoverResult(prepared:prepared,model:variant.modelName,seconds:Date().timeIntervalSince(started),outputScale:variant.scale)
        }
        let jobLock = try RealESRGANJobLock()
        defer { withExtendedLifetime(jobLock) {} }
        let memory = RealESRGANMemorySampler()
        let started = Date()
        let loadStart = Date()
        let upscaler = try NativeSpatialUpscaler(modelURL: URL(fileURLWithPath: path), variant: variant)
        let nativeImage = prepared.manifest.realESRGANImage
        if let request = nativeImage {
            try request.validate(variant)
            guard prepared.manifest.frames.count == 1 else { throw UncoverError.backend("A photo request must contain one image.") }
        }
        let loadSeconds = Date().timeIntervalSince(loadStart)
        let context = CIContext(options: [.cacheIntermediates: false, .useSoftwareRenderer: true])
        // At 1×, run one restoration pass and return to the unchanged ROI size.
        let stages = nativeImage != nil ? 1 : max(1, (prepared.manifest.scale.stages + (variant == .x4plus ? 1 : 0)) / (variant == .x4plus ? 2 : 1))
        for frame in prepared.manifest.frames {
            try Task.checkCancellation()
            try memory.checkBudget()
            progress("RealESRGAN · frame \(frame.index + 1)/\(prepared.manifest.frames.count)")
            try autoreleasepool {
            var image = try RegionImages.read(RegionImages.frameURL(prepared.directory, "degraded", frame.index))
            if nativeImage != nil {
                guard image.width == prepared.manifest.regionWidth, image.height == prepared.manifest.regionHeight else {
                    throw UncoverError.backend("Photo input dimensions differ from its manifest.")
                }
            }
            for _ in 0..<stages {
                try Task.checkCancellation()
                image = try upscaler.upscale(image)
                try memory.checkBudget()
            }
            if nativeImage == nil && (image.width != prepared.manifest.regionWidth || image.height != prepared.manifest.regionHeight) {
                image = try RegionImages.resize(image,
                    to: CGSize(width: prepared.manifest.regionWidth, height: prepared.manifest.regionHeight), context: context)
            }
            try RegionImages.write(image, to: RegionImages.frameURL(prepared.directory, "reconstructed", frame.index))
            }
        }
        let report:[String:Any] = ["peakPhysicalFootprintBytes":memory.peakBytes,"memoryBudgetBytes":memory.budget,"memoryScope":"whole application process; sampled every 20 ms; checked between predictions","model":"realesrgan-\(variant.rawValue)","backend":"native Swift Core ML","pythonRuntime":false,"frames":prepared.manifest.frames.count,"scale":prepared.manifest.scale.rawValue,"sourceDownsampling":prepared.manifest.scale.rawValue,"stagesPerFrame":stages,"loadSeconds":loadSeconds,"inferenceSeconds":upscaler.predictionSeconds,"seconds":Date().timeIntervalSince(started),"predictionSizes":upscaler.predictionSizes,"computeUnits":upscaler.selectedComputeUnits,"reusedInputBuffers":true,"singleTileBlendBypass":true]
        try JSONSerialization.data(withJSONObject:report,options:[.sortedKeys,.prettyPrinted]).write(to:prepared.directory.appendingPathComponent("report.json"),options:.atomic)
        return UncoverResult(prepared: prepared, model: variant.modelName, seconds: Date().timeIntervalSince(started), outputScale: nativeImage == nil ? 1 : variant.scale)
    }
}

extension RealESRGANVariant {
    var algorithm: UncoverAlgorithm { self == .x2plus ? .realESRGAN : .realESRGANFour }
    var managedModel: ManagedModel { self == .x2plus ? .real : .real4 }
}

struct RealESRGANImageRequest: Codable, Sendable {
    let variant: RealESRGANVariant
    let modelFingerprint: String
    let runtimeRevision: String
    init(_ variant: RealESRGANVariant) {
        self.variant = variant; modelFingerprint = variant.weightsSHA256
        runtimeRevision = RealESRGANVariant.runtimeRevision
    }
    func validate(_ expected: RealESRGANVariant) throws {
        guard variant == expected, modelFingerprint == expected.weightsSHA256,
              runtimeRevision == RealESRGANVariant.runtimeRevision else {
            throw UncoverError.backend("RealESRGAN model identity or runtime revision differs.")
        }
    }
}
