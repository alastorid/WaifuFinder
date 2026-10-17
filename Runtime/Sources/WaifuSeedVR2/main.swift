import AppKit
import Darwin
import Foundation
import ImageIO
import CoreImage
import MLX
import SeedVR2MLX
import NativeVideoIO
import UniformTypeIdentifiers

struct Frame: Decodable { let index: Int }
struct Manifest: Decodable {
    let scale: Int
    let regionWidth: Int
    let regionHeight: Int
    let frames: [Frame]
}
func tiles(width: Int, height: Int) -> [Tile] {
    var result: [Tile] = []
    for y in Swift.stride(from: 0, to: height, by: 192) {
        for x in Swift.stride(from: 0, to: width, by: 192) {
            result.append(Tile(x: x, y: y, width: min(256, width - x), height: min(256, height - y)))
            if x + 256 >= width { break }
        }
        if y + 256 >= height { break }
    }
    return result
}
func tileWeight(x: Int, y: Int, tile: Tile, width: Int, height: Int) -> Float {
    func axis(_ position: Int, length: Int, start: Int, total: Int) -> Float {
        let left = start > 0 ? min(1, Float(position + 1) / 64) : 1
        let right = start + length < total ? min(1, Float(length - position) / 64) : 1
        return left * right
    }
    return axis(x, length: tile.width, start: tile.x, total: width)
        * axis(y, length: tile.height, start: tile.y, total: height)
}
struct Chunk { let start: Int; let count: Int; let outputStart: Int; let outputCount: Int }
func padded(_ n: Int) -> Int { ((n - 1 + 3) / 4) * 4 + 1 }
func chunks(_ n: Int) -> [Chunk] {
    let total = padded(n), size = min(29, total)
    let stride = size >= total ? size : size - 9
    var result: [Chunk] = [], output = 0
    while output < total {
        var start = max(0, output - 8)
        let end = min(total, start + size)
        if end - start < size { start = max(0, end - size) }
        let count = end >= total ? total - output : min(stride, total - output)
        result.append(Chunk(start: start, count: padded(end - start), outputStart: output, outputCount: count))
        output += count
    }
    return result
}

if CommandLine.arguments.contains("--benchmark-vae-fusion") {
    SeedVR2VAENormalizationProof.run()
    exit(0)
}
if CommandLine.arguments.contains("--benchmark-vae-convolution") {
    SeedVR2VAEConvolutionProof.run()
    exit(0)
}
if CommandLine.arguments.contains("--benchmark-fusion") {
    SeedVR2FusionProof.run()
    exit(0)
}
if CommandLine.arguments.contains("--export-fused") {
    guard let source = option("--model"), let output = option("--output") else { fail("Supply --model and --output") }
    do { try SeedVR2FusedCheckpoint.export(source: URL(fileURLWithPath: source), destination: URL(fileURLWithPath: output)) }
    catch { fail("Fusion export failed: \(error)") }
    exit(0)
}
if CommandLine.arguments.contains("--configure") {
    guard let model = option("--model"), let embedding = option("--embedding"),
          FileManager.default.fileExists(atPath: model + "/transformer/model.safetensors.index.json"),
          FileManager.default.fileExists(atPath: embedding) else { fail("Native model or embedding is missing.") }
    let directory = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("Waifu/Uncover")
    let url = directory.appendingPathComponent("backend.json")
    var config = (try? Data(contentsOf: url)).flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] } ?? [:]
    config["model"] = model
    do {
        let checkpoint = try SeedVR2VideoCheckpoint(directory: URL(fileURLWithPath: model))
        config["modelVariant"] = checkpoint.variant
        config["disconnectedVariants"] = (config["disconnectedVariants"] as? [String] ?? []).filter { $0 != checkpoint.variant }
        if checkpoint.fusedVAE {
            var models = config["optimizedModels"] as? [String: String] ?? [:]
            models[checkpoint.variant] = URL(fileURLWithPath: model).standardizedFileURL.path
            config["optimizedModels"] = models
        }
    }
    catch { fail("Invalid native checkpoint: \(error)") }
    config["nativeExecutable"] = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL.path
    config["nativeEmbedding"] = embedding
    config["nativeMemoryBudgetGB"] = min(6, Double(ProcessInfo.processInfo.physicalMemory) * 0.5 / 1e9)
    if config["python"] == nil { config["python"] = "" }
    if config["helper"] == nil { config["helper"] = "" }
    if config["allowUnsafeMemory"] == nil { config["allowUnsafeMemory"] = false }
    do {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: config, options: [.prettyPrinted, .sortedKeys]).write(to: url, options: .atomic)
    } catch { fail("Could not configure native inference: \(error)") }
    print("Native Swift MLX configured: \(url.path)")
    exit(0)
}
if CommandLine.arguments.contains("--self-test") {
    for n in [5, 6, 29, 30, 60, 149, 300] {
        let plan = chunks(n)
        precondition(plan.allSatisfy { $0.start % 4 == 0 && ($0.count - 1) % 4 == 0 })
        precondition(plan.reduce(0) { $0 + $1.outputCount } == padded(n))
        var expected = 0
        for chunk in plan { precondition(chunk.outputStart == expected); expected += chunk.outputCount }
    }
    for width in [16, 256, 544, 640, 1920] {
        let height = 544
        let spatial = tiles(width: width, height: height)
        var sums = [Float](repeating: 0, count: width*height)
        for tile in spatial {
            precondition(tile.x % 16 == 0 && tile.y % 16 == 0 && tile.width % 16 == 0 && tile.height % 16 == 0)
            for y in 0..<tile.height { for x in 0..<tile.width {
                sums[(tile.y+y)*width+tile.x+x] += tileWeight(x: x, y: y, tile: tile, width: width, height: height)
            } }
        }
        precondition(sums.allSatisfy { $0 > 0 && $0.isFinite })
    }
    print("Native temporal planning and overlapping tile coverage passed")
    exit(0)
}
setpgid(0, 0)
guard let jobPath = option("--job"), let modelPath = option("--model"), let embeddingPath = option("--embedding") else {
    fail("Usage: waifu-seedvr2 --job DIRECTORY --model DIRECTORY --embedding FILE [--memory-budget-gb 6]")
}
let defaultBudgetGB = min(6, Double(ProcessInfo.processInfo.physicalMemory) * 0.5 / 1e9)
guard let budgetGB = Double(option("--memory-budget-gb") ?? String(defaultBudgetGB)), budgetGB.isFinite,
      budgetGB >= 0.1, budgetGB * 1e9 <= Double(ProcessInfo.processInfo.physicalMemory) * 0.5 else {
    fail("Native memory budget must be finite, positive, and at most half of physical RAM.")
}
let budget = Int(budgetGB * 1_000_000_000)
let lockDirectory = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
    .appendingPathComponent("Waifu/Uncover")
try FileManager.default.createDirectory(at: lockDirectory, withIntermediateDirectories: true)
let lockDescriptor = open(lockDirectory.appendingPathComponent("native.lock").path, O_CREAT | O_RDWR | O_NOFOLLOW, 0o600)
guard lockDescriptor >= 0, flock(lockDescriptor, LOCK_EX | LOCK_NB) == 0 else {
    fail("Another native SeedVR2 job is already running. Wait for it to finish.")
}
SeedVR2Runtime.memoryBudget = budget
Memory.cacheLimit = 64 * 1024 * 1024
Memory.memoryLimit = budget // MLX guideline; the physical monitor enforces the process budget.
let monitor = MemoryMonitor(budget: budget)
let pressure = DispatchSource.makeMemoryPressureSource(eventMask: [.warning, .critical], queue: DispatchQueue(label: "native-pressure"))
pressure.setEventHandler {
    fputs("Native inference stopped because macOS reported memory pressure. Close other heavy apps or reduce the interval.\n", stderr)
    exit(75)
}
pressure.resume()
SeedVR2Runtime.progress = { print($0); fflush(stdout) }
let started = Date()
do {
    let job = URL(fileURLWithPath: jobPath)
    let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: job.appendingPathComponent("manifest.json")))
    guard [1, 2, 4, 8, 16, 32, 64].contains(manifest.scale), manifest.frames.count >= 5,
          manifest.frames.enumerated().allSatisfy({ $0.offset == $0.element.index }) else { fail("Invalid temporal manifest.") }
    let first = try readRGB(job.appendingPathComponent("degraded/000000.png"))
    guard manifest.regionWidth > 0, manifest.regionHeight > 0,
          first.width == (manifest.regionWidth + manifest.scale - 1) / manifest.scale,
          first.height == (manifest.regionHeight + manifest.scale - 1) / manifest.scale else {
        fail("Degraded input does not match the selected ROI dimensions.")
    }
    // SeedVR2 conditions on bicubic-enlarged source pixels and generates at the
    // selected output resolution. Pad only the model canvas, then normalize to the ROI.
    let width = ((first.width * manifest.scale + 15) / 16) * 16
    let height = ((first.height * manifest.scale + 15) / 16) * 16
    // This estimator is a preflight bound, not a claim that MLX's memoryLimit prevents swap.
    let checkpoint = try SeedVR2VideoCheckpoint(directory: URL(fileURLWithPath: modelPath))
    SeedVR2Runtime.disableSharedAttentionFusion = CommandLine.arguments.contains("--no-shared-fusion")
    SeedVR2Runtime.disableVAENormalizationFusion = CommandLine.arguments.contains("--no-vae-norm-fusion")
    SeedVR2Runtime.disableVAECausalFusion = CommandLine.arguments.contains("--no-vae-causal-fusion")
    SeedVR2Runtime.disableVAEPointwiseConv = CommandLine.arguments.contains("--no-vae-pointwise-conv")
    SeedVR2Runtime.disableVAEResidualFusion = CommandLine.arguments.contains("--no-vae-residual-fusion")
    let plan = chunks(manifest.frames.count)
    let spatial = tiles(width: width, height: height)
    let forceResident = CommandLine.arguments.contains("--resident-weights")
    guard !forceResident || (checkpoint.variant == "seedvr2-3b" && ProcessInfo.processInfo.physicalMemory > 8 * 1024 * 1024 * 1024) else {
        fail("Resident weights require 3B and more than 8 GB RAM.")
    }
    guard !forceResident || !CommandLine.arguments.contains("--stream-weights") else { fail("Choose one weight-loading override.") }
    let streamed = CommandLine.arguments.contains("--stream-weights") || checkpoint.variant == "seedvr2-7b"
        || ProcessInfo.processInfo.physicalMemory <= 8 * 1024 * 1024 * 1024
        || (!forceResident && checkpoint.fusedVAE && plan.count * spatial.count == 1)
    let estimated = try checkpoint.estimatedWorkingSet(streamed: streamed) + width * height * min(29, padded(manifest.frames.count)) * 3 * 4
    guard estimated < budget else { fail("This region's estimated working set exceeds the native memory budget. Use a smaller region.") }
    print("Loading native 4-bit SeedVR2 weights"); fflush(stdout)
    let model = try SeedVR2VideoUpscaler(directory: URL(fileURLWithPath: modelPath), embedding: URL(fileURLWithPath: embeddingPath), streamWeights: streamed)
    let total = padded(manifest.frames.count)
    let noise = SeedVR2LatentCreator.noiseLatents(seed: 42, height: height / 8, width: width / 8, frames: (total - 1) / 4 + 1)
    SeedVR2Runtime.materialize(noise)
    let pixels = width * height
    try FileManager.default.createDirectory(at: job.appendingPathComponent("restored"), withIntermediateDirectories: true)
    var weights = [Float](repeating: 0, count: pixels)
    for tile in spatial { for y in 0..<tile.height { for x in 0..<tile.width {
        weights[(tile.y+y)*width+tile.x+x] += tileWeight(x: x, y: y, tile: tile, width: width, height: height)
    } } }
    for (chunkIndex, chunk) in plan.enumerated() {
        let visibleEnd = min(manifest.frames.count, chunk.outputStart + chunk.outputCount)
        let visibleCount = visibleEnd - chunk.outputStart
        var merged = [Float](repeating: 0, count: visibleCount * pixels * 3)
        for (tileIndex, tile) in spatial.enumerated() {
            print("Native temporal chunk \(chunkIndex + 1)/\(plan.count) · tile \(tileIndex + 1)/\(spatial.count)"); fflush(stdout)
            let frames: [MLXArray] = try (0..<chunk.count).map { offset in
                let index = min(manifest.frames.count - 1, chunk.start + offset)
                let input = try readRGB(job.appendingPathComponent(String(format: "degraded/%06d.png", index)))
                guard input.width == first.width && input.height == first.height else { fail("Degraded geometry changed.") }
                let values = resize(input, factor: manifest.scale, rect: tile)
                let image = MLXArray(values, [tile.height, tile.width, 3])
                eval(image)
                return image
            }
            let video = stacked(frames, axis: 0).transposed(3, 0, 1, 2).expandedDimensions(axis: 0)
            let latentStart = chunk.start / 4, latentCount = (chunk.count - 1) / 4 + 1
            let decoded = try model.restore(video, noise: noise[0..., 0..., latentStart ..< (latentStart + latentCount),
                (tile.y/8)..<((tile.y+tile.height)/8), (tile.x/8)..<((tile.x+tile.width)/8)])
            guard decoded.shape == [1, 3, chunk.count, tile.height, tile.width] else {
                fail("Native temporal output geometry mismatch: \(decoded.shape)")
            }
            for index in chunk.outputStart..<visibleEnd {
                let local = index - chunk.start
                let image = decoded[0, 0..., local].transposed(1, 2, 0).asType(.float32).asArray(Float.self)
                guard image.allSatisfy({ $0.isFinite }) else { fail("Native model produced non-finite pixels") }
                for y in 0..<tile.height { for x in 0..<tile.width {
                    let weight = tileWeight(x: x, y: y, tile: tile, width: width, height: height)
                    let destination = ((index-chunk.outputStart)*pixels + (tile.y+y)*width + tile.x+x)*3
                    let source = (y*tile.width+x)*3
                    for c in 0..<3 { merged[destination+c] += image[source+c]*weight }
                } }
            }
            Memory.clearCache()
        }
        for index in chunk.outputStart..<visibleEnd {
            let offset = (index-chunk.outputStart)*pixels*3
            var values = [Float](repeating: 0, count: pixels*3)
            for pixel in 0..<pixels { for c in 0..<3 { values[pixel*3+c] = merged[offset+pixel*3+c]/weights[pixel] } }
            let generatedWidth = first.width * manifest.scale
            let generatedHeight = first.height * manifest.scale
            let image = MLXArray(values, [height, width, 3])[0..<generatedHeight, 0..<generatedWidth]
            try save(image, width: generatedWidth, height: generatedHeight,
                targetWidth: manifest.regionWidth, targetHeight: manifest.regionHeight,
                to: job.appendingPathComponent(String(format: "restored/%06d.png", index)))
        }
        Memory.clearCache()
    }
    var usage = rusage(); getrusage(RUSAGE_SELF, &usage)
    let report: [String: Any] = ["backend": "Native Swift MLX", "mlxSwiftVersion": "0.31.2", "model": checkpoint.variant,
        "inputWidth": first.width, "inputHeight": first.height, "outputWidth": manifest.regionWidth,
        "outputHeight": manifest.regionHeight, "targetRegionWidth": manifest.regionWidth,
        "targetRegionHeight": manifest.regionHeight, "upscalingModel": "SeedVR2 native",
        "inputFrames": manifest.frames.count, "outputFrames": manifest.frames.count, "paddedOutputWidth": width, "paddedOutputHeight": height,
        "temporalPaddedFrames": total, "temporalChunk": min(29, total), "temporalOverlap": plan.count > 1 ? 8 : 0,
        "temporalChunks": plan.count, "spatialTiles": spatial.count, "tileEdge": 256, "tileOverlap": 64, "scale": manifest.scale, "seed": 42, "seconds": Date().timeIntervalSince(started),
        "modelVariant": checkpoint.variant, "fusedMLP": checkpoint.config.fusedMLP, "fusedVAEAttention": checkpoint.fusedVAE,
        "sharedProjectionFusion": checkpoint.config.sharedAttentionFusion && !SeedVR2Runtime.disableSharedAttentionFusion,
        "transformerPrecision": checkpoint.config.halfPrecision ? "float16" : "float32",
        "fusedVAENormalization": checkpoint.fusedVAE && !SeedVR2Runtime.disableVAENormalizationFusion,
        "fusedVAECausalPreparation": !SeedVR2Runtime.disableVAECausalFusion,
        "vaePointwiseMatmul": !SeedVR2Runtime.disableVAEPointwiseConv,
        "fusedVAEBiasResidual": !SeedVR2Runtime.disableVAEResidualFusion,
        "streamedWeights": streamed, "phaseSeconds": SeedVR2Runtime.phaseSeconds,
        "physicalSamplingMilliseconds": 20, "memoryBudgetBytes": budget, "peakPhysicalFootprintBytes": monitor.peakBytes,
        "peakRSSBytes": usage.ru_maxrss, "mlxPeakBytes": Memory.peakMemory, "pythonRuntime": false]
    try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        .write(to: job.appendingPathComponent("report.json"), options: .atomic)
    print("Native reconstruction complete · \(manifest.frames.count) frames · peak physical \(Double(monitor.peakBytes)/1e9) GB")
} catch { fail("Native SeedVR2 failed: \(error)") }
