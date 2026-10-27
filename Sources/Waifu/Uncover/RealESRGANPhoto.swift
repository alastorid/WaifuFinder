import AppKit
import Combine
import CryptoKit
import CoreMedia

/// Serial native photo execution and bounded, transient result storage.
actor RealESRGANImageRuntime {
    static let shared = RealESRGANImageRuntime()
    private var models: [RealESRGANVariant: (identity:String, model:NativeSpatialUpscaler)] = [:]
    private var results: [(key:String, image:CGImage)] = []
    private let cacheLimit = 128 * 1024 * 1024

    func restore(_ image: CGImage, variant: RealESRGANVariant, modelURL: URL) throws -> CGImage {
        try Task.checkCancellation()
        let lock = try RealESRGANJobLock()
        defer { withExtendedLifetime(lock) {} }
        let modified = try modelURL.resourceValues(forKeys:[.contentModificationDateKey]).contentModificationDate
        let identity = "\(modelURL.path):\(modified?.timeIntervalSince1970 ?? 0):\(variant.cacheIdentity)"
        if models[variant]?.identity != identity {
            models[variant] = (identity,try NativeSpatialUpscaler(modelURL:modelURL,variant:variant))
        }
        models[variant]!.model.resetDiagnostics()
        return try models[variant]!.model.upscale(image)
    }

    func key(_ input: PreviewEvidenceInput, variant: RealESRGANVariant, revision: String) throws -> String {
        try Task.checkCancellation()
        guard variant.accepts(width:input.image.width,height:input.image.height) else { throw UncoverError.backend("The reconstructed image exceeds the 16,384px / 256 million pixel output limit.") }
        let bytes = try NativeSpatialUpscaler.straightRGBPixels(input.image,convertToSRGB:true)
        let digest = SHA256.hash(data:Data(bytes)).map { String(format:"%02x",$0) }.joined()
        return "\(input.photoURL.absoluteString)|\(input.isOriginal)|\(input.image.width)x\(input.image.height)|\(digest)|\(variant.rawValue)|\(revision)"
    }
    func cached(_ key: String) -> CGImage? { results.first { $0.key == key }?.image }
    func store(_ image: CGImage, key: String) {
        let cost = image.bytesPerRow * image.height
        guard cost <= cacheLimit else { return }
        results.removeAll { $0.key == key }
        while results.reduce(cost,{ $0 + $1.image.bytesPerRow * $1.image.height }) > cacheLimit, !results.isEmpty { results.removeFirst() }
        results.append((key,image))
    }

    func prepare(_ input: PreviewEvidenceInput, variant: RealESRGANVariant) throws -> UncoverPrepared {
        try Task.checkCancellation()
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("waifu-realesrgan-photo-\(UUID().uuidString)")
        do {
            for name in ["degraded","reconstructed"] { try FileManager.default.createDirectory(at:directory.appendingPathComponent(name),withIntermediateDirectories:true) }
            try RegionImages.write(input.image,to:RegionImages.frameURL(directory,"degraded",0))
            let manifest = UncoverManifest(source:input.photoURL,sourceBytes:0,sourceModified:0,
                roi:UncoverROI(x:0,y:0,width:1,height:1),width:input.image.width,height:input.image.height,
                regionX:0,regionY:0,regionWidth:input.image.width,regionHeight:input.image.height,scale:.one,
                frames:[UncoverFrame(index:0,timestamp:UncoverTime(.zero),duration:UncoverTime(CMTime(value:1,timescale:1)))],
                realESRGANImage:RealESRGANImageRequest(variant))
            return UncoverPrepared(directory:directory,manifest:manifest)
        } catch { try? FileManager.default.removeItem(at:directory); throw error }
    }
    func read(_ result: UncoverResult, variant: RealESRGANVariant) throws -> CGImage {
        guard result.model == variant.modelName, result.outputScale == variant.scale else { throw UncoverError.backend("RealESRGAN result identity differs.") }
        let image = try RegionImages.read(RegionImages.frameURL(result.directory,"reconstructed",0))
        guard image.width == result.prepared.manifest.regionWidth * variant.scale,
              image.height == result.prepared.manifest.regionHeight * variant.scale else { throw UncoverError.backend("RealESRGAN output geometry differs.") }
        return image
    }
}

@MainActor
final class RealESRGANPreviewModel: ObservableObject {
    typealias Processor = @Sendable (PreviewEvidenceInput, RealESRGANVariant) async throws -> CGImage
    @Published private(set) var selection: RealESRGANVariant?
    @Published private(set) var loading = false
    @Published private(set) var status = ""
    @Published private(set) var rendered: NSImage?
    private(set) var outputInputID: UUID?
    private var generation = UUID()
    private let processor: Processor?
    init(processor: Processor? = nil) { self.processor = processor }

    func select(_ variant: RealESRGANVariant?) {
        selection = variant; generation = UUID(); loading = false; status = ""
        rendered = nil; outputInputID = nil
    }
    func image(for input: PreviewEvidenceInput?) -> NSImage? {
        guard selection != nil, input?.id == outputInputID else { return nil }
        return rendered
    }
    func load(input: PreviewEvidenceInput?, remote: RemoteClient? = nil) async {
        let token = UUID(); generation = token
        guard let input, let variant = selection else { loading = false; return }
        loading = true; status = "Reconstructing \(variant.label)…"
        defer { if generation == token { loading = false } }
        do {
            let runtime = RealESRGANImageRuntime.shared
            let revision = remote.map { "\($0.remoteIdentity):\(variant.weightsSHA256):\(RealESRGANVariant.runtimeRevision)" } ?? variant.cacheIdentity
            let key = try await runtime.key(input,variant:variant,revision:revision)
            let image: CGImage
            if let cached = await runtime.cached(key) { image = cached }
            else if let processor { image = try await processor(input,variant) }
            else {
                ModelLibrary.shared.acquire(variant.algorithm)
                do {
                    if let remote {
                        let prepared = try await runtime.prepare(input,variant:variant)
                        defer { try? FileManager.default.removeItem(at:prepared.directory) }
                        let result = try await RemoteUncoverBackend(algorithm:variant.algorithm,client:remote).reconstruct(prepared,allowUnsafeMemory:false,progress:{ _ in })
                        image = try await runtime.read(result,variant:variant)
                    } else {
                        guard let url = variant.modelURL else { throw UncoverError.backend("Download RealESRGAN \(variant.rawValue) in Settings → Models.") }
                        image = try await runtime.restore(input.image,variant:variant,modelURL:url)
                    }
                } catch { ModelLibrary.shared.release(variant.algorithm); throw error }
                ModelLibrary.shared.release(variant.algorithm)
            }
            try Task.checkCancellation()
            guard token == generation, selection == variant else { return }
            await runtime.store(image,key:key)
            guard token == generation, selection == variant, !Task.isCancelled else { return }
            rendered = NSImage(cgImage:image,size:NSSize(width:image.width,height:image.height)); outputInputID = input.id
            status = "\(variant.label) · AI reconstruction"
        } catch {
            guard token == generation else { return }
            status = Task.isCancelled || error is CancellationError ? "Cancelled" : error.localizedDescription
        }
    }
}
