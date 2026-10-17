import Darwin
import FlashVSRMLX
import Foundation
import MLX
import NativeVideoIO

func selfTest() {
    let attention = FlashSparseAttention()
    for dtype in [DType.float32, .bfloat16] {
        for (h, w, temporal, keyTemporal) in [(1, 2, 3, 3), (2, 1, 1, 4), (2, 2, 1, 4)] {
            let blocks = h * w * temporal
            let keys = h * w * keyTemporal
            let q = MLXRandom.normal([blocks, 128, 1536]).asType(dtype)
            let k = MLXRandom.normal([keys, 128, 1536]).asType(dtype)
            let v = MLXRandom.normal([keys, 128, 1536]).asType(dtype)
            let mask = attention.blockMask(
                q: q, k: k, height: h, width: w, temporalWindows: temporal, topK: max(1, h * w * keys / 2),
                localRange: 1)
            let optimized = attention.apply(q: q, k: k, v: v, mask: mask)
            let reference = FlashSparseAttention.reference(q: q, k: k, v: v, mask: mask)
            let error = abs(optimized - reference).max().item(Float.self)
            precondition(
                error < (dtype == .float32 ? 0.00005 : 0.008), "Sparse online attention mismatch: \(error)")
            print(
                "LCSA sparse/reference parity \(dtype) \(h)×\(w), temporal \(temporal)/\(keyTemporal): \(error)"
            )
        }
    }
}
if CommandLine.arguments.contains("--self-test") {
    selfTest()
    exit(0)
}
if CommandLine.arguments.contains("--configure") {
    guard let model = option("--model"),
        let data = try? Data(contentsOf: URL(fileURLWithPath: model).appendingPathComponent("manifest.json")),
        let manifest = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
        manifest["revision"] as? String == "27561b186ded3402d7c975f4fd722e2885b6135f"
    else { fail("Compatible converted FlashVSR weights are missing") }
    let root = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("Waifu/Uncover")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let config: [String: Any] = [
        "executable": URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL.path,
        "model": URL(fileURLWithPath: model).standardizedFileURL.path,
        "memoryBudgetGB": min(6, Double(ProcessInfo.processInfo.physicalMemory) / 2e9),
    ]
    try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys, .prettyPrinted]).write(
        to: root.appendingPathComponent("flashvsr.json"), options: .atomic)
    print("Native FlashVSR configured; SeedVR2 remains the default.")
    exit(0)
}
setpgid(0, 0)
guard let jobPath = option("--job"), let modelPath = option("--model") else {
    fail("Usage: waifu-flashvsr --job DIRECTORY --model DIRECTORY [--memory-budget-gb 6]")
}
let job = URL(fileURLWithPath: jobPath)
let modelDirectory = URL(fileURLWithPath: modelPath)
guard let budgetGB = Double(option("--memory-budget-gb") ?? "6"), budgetGB.isFinite, budgetGB >= 0.1,
    budgetGB * 1e9 <= Double(ProcessInfo.processInfo.physicalMemory) * 0.5
else { fail("Invalid native memory budget") }
let budget = Int(budgetGB * 1e9)
let lockRoot = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
    .appendingPathComponent("Waifu/Uncover")
try FileManager.default.createDirectory(at: lockRoot, withIntermediateDirectories: true)
// Shared lock forbids two different engines from occupying unified memory together.
let descriptor = open(
    lockRoot.appendingPathComponent("native.lock").path, O_CREAT | O_RDWR | O_NOFOLLOW, 0o600)
guard descriptor >= 0, flock(descriptor, LOCK_EX | LOCK_NB) == 0 else {
    fail("Another native reconstruction is already running")
}
let monitor = MemoryMonitor(budget: budget)
Memory.cacheLimit = 64 * 1024 * 1024
Memory.memoryLimit = budget
let pressure = DispatchSource.makeMemoryPressureSource(
    eventMask: [.warning, .critical], queue: DispatchQueue(label: "flashvsr-pressure"))
pressure.setEventHandler {
    fputs("FlashVSR stopped because macOS reported memory pressure.\n", stderr)
    exit(75)
}
pressure.resume()
struct Frame: Decodable { let index: Int }
struct Manifest: Decodable {
    let scale: Int, regionWidth: Int, regionHeight: Int
    let frames: [Frame]
}
let outputScale = 4
let maximumOutputDimension = 16_384
let maximumOutputPixels = 100_000_000
do {
    let manifest = try JSONDecoder().decode(
        Manifest.self, from: Data(contentsOf: job.appendingPathComponent("manifest.json")))
    guard [1, 2, 4, 8, 16, 32, 64].contains(manifest.scale), manifest.frames.count >= 5, manifest.frames.count <= 300,
        manifest.frames.enumerated().allSatisfy({ $0.offset == $0.element.index }), manifest.regionWidth > 0,
        manifest.regionHeight > 0, manifest.regionWidth <= maximumOutputDimension / outputScale,
        manifest.regionHeight <= maximumOutputDimension / outputScale
    else { fail("FlashVSR requires a supported downsampling factor and 5–300 ordered frames") }
    let first = try readRGB(job.appendingPathComponent("degraded/000000.png"), straightAlpha: true)
    guard first.width == (manifest.regionWidth + manifest.scale - 1) / manifest.scale,
        first.height == (manifest.regionHeight + manifest.scale - 1) / manifest.scale
    else { fail("Degraded geometry mismatch") }
    let outputRegionWidth = manifest.regionWidth * outputScale
    let outputRegionHeight = manifest.regionHeight * outputScale
    guard outputRegionWidth <= maximumOutputPixels / outputRegionHeight else {
        fail("The 4× FlashVSR ROI exceeds the supported output dimensions")
    }
    let width = (outputRegionWidth + 127) / 128 * 128
    let height = (outputRegionHeight + 127) / 128 * 128
    let following = max(0, (manifest.frames.count - 21 + 7) / 8)
    // No spatial DiT tiling: it would alter trained LCSA selection and temporal caches.
    let estimate = 3_600_000_000 + width * height * 12_000
    guard estimate < budget else {
        fail("FlashVSR ROI exceeds its conservative memory budget; reduce the selection")
    }
    let started = Date()
    print("Loading official FlashVSR 1.1 BF16 Tiny weights")
    let model = try FlashVSRVideoUpscaler(directory: modelDirectory)
    let loaded = Date()
    var noise = MLXRandom.normal([1, 6 + following * 2, height / 8, width / 8, 16], key: MLXRandom.key(0))
        .asType(.bfloat16)
    if let reference = option("--reference-noise") {
        guard let supplied = try loadArrays(url: URL(fileURLWithPath: reference))["noise"],
            supplied.shape == noise.shape
        else { fail("Invalid reference noise") }
        noise = supplied.asType(.bfloat16)
    }
    eval(noise)
    let fixture = option("--capture-fixture").map { URL(fileURLWithPath: $0) }
    var captureError: Error?
    if let fixture {
        try FileManager.default.createDirectory(at: fixture, withIntermediateDirectories: true)
        try save(arrays: ["noise": noise], url: fixture.appendingPathComponent("noise.safetensors"))
        model.observer = { name, array in
            do {
                try save(arrays: [name: array], url: fixture.appendingPathComponent(name + ".safetensors"))
            } catch { captureError = error }
        }
        let frames = try (0..<(25 + following * 8)).map { index -> MLXArray in
            let source = try readRGB(
                job.appendingPathComponent(
                    String(format: "degraded/%06d.png", min(index, manifest.frames.count - 1))),
                straightAlpha: true)
            return MLXArray(
                resize(source, toWidth: outputRegionWidth, toHeight: outputRegionHeight,
                       rect: Tile(x: 0, y: 0, width: width, height: height)),
                [height, width, 3]
            ).asType(.bfloat16)
        }
        try save(
            arrays: ["conditioning": stacked(frames, axis: 0)],
            url: fixture.appendingPathComponent("conditioning.safetensors"))
    }
    var firstFrameSeconds: Double?
    guard let repeats = Int(option("--benchmark-repeats") ?? "1"), (1...3).contains(repeats) else {
        fail("Benchmark repeats must be 1–3")
    }
    var runs: [Double] = []
    try FileManager.default.createDirectory(
        at: job.appendingPathComponent("reconstructed"), withIntermediateDirectories: true)
    for run in 0..<repeats {
        let runStarted = Date()
        try model.restore(
            frameCount: manifest.frames.count, width: width, height: height, noise: noise,
            frame: { index in
                let source = try readRGB(
                    job.appendingPathComponent(String(format: "degraded/%06d.png", index)),
                    straightAlpha: true)
                guard source.width == first.width, source.height == first.height else {
                    throw FlashError.invalid("Inconsistent degraded dimensions")
                }
                return MLXArray(
                    resize(source, toWidth: outputRegionWidth, toHeight: outputRegionHeight,
                           rect: Tile(x: 0, y: 0, width: width, height: height)),
                    [height, width, 3])
            },
            emit: { index, image in
                if let captureError { throw captureError }
                guard isFinite(image).all().item(Bool.self) else {
                    throw FlashError.invalid("Non-finite FlashVSR output")
                }
                let cropped = image[0..<outputRegionHeight, 0..<outputRegionWidth]
                try save(
                    cropped, width: outputRegionWidth, height: outputRegionHeight,
                    targetWidth: outputRegionWidth, targetHeight: outputRegionHeight,
                    to: job.appendingPathComponent(String(format: "reconstructed/%06d.png", index)))
                if firstFrameSeconds == nil { firstFrameSeconds = Date().timeIntervalSince(started) }
                if let fixture {
                    let outputURL = fixture.appendingPathComponent(
                        String(format: "output-%06d.safetensors", index))
                    if run > 0 {
                        let previous = try loadArrays(url: outputURL)["output"]!
                        guard abs(previous - image).max().item(Float.self) == 0 else {
                            throw FlashError.invalid("Temporal state leaked between benchmark runs")
                        }
                    } else {
                        try save(arrays: ["output": image], url: outputURL)
                    }
                }
            })
        runs.append(Date().timeIntervalSince(runStarted))
    }
    if let captureError { throw captureError }
    var usage = rusage()
    getrusage(RUSAGE_SELF, &usage)
    let report: [String: Any] = [
        "backend": "Native Swift MLX FlashVSR 1.1", "algorithm": "FlashVSR 1.1 Tiny Streaming",
        "modelRevision": "27561b186ded3402d7c975f4fd722e2885b6135f",
        "scale": outputScale, "sourceDownsampling": manifest.scale, "outputScale": outputScale,
        "inputFrames": manifest.frames.count, "outputFrames": manifest.frames.count,
        "inputWidth": first.width, "inputHeight": first.height,
        "outputWidth": outputRegionWidth, "outputHeight": outputRegionHeight, "paddedWidth": width,
        "paddedHeight": height,
        "modelLoadSeconds": loaded.timeIntervalSince(started), "seconds": Date().timeIntervalSince(started),
        "firstFrameSeconds": firstFrameSeconds!,
        "temporalChunks": following + 1, "temporalPaddedFrames": 25 + following * 8, "lookaheadFrames": 4,
        "kvWindows": 3, "localRange": 11,
        "precision": "bfloat16", "attention": "LCSA Metal sparse online softmax", "colorCorrection": "ADAIN",
        "memoryBudgetBytes": budget, "peakPhysicalFootprintBytes": monitor.peakBytes,
        "peakRSSBytes": usage.ru_maxrss, "mlxPeakBytes": Memory.peakMemory,
        "pythonRuntime": false, "mlxSwiftVersion": "0.31.2", "benchmarkInferenceSeconds": runs,
        "benchmarkRepeats": repeats,
    ]
    try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys, .prettyPrinted]).write(
        to: job.appendingPathComponent("report.json"), options: .atomic)
    print(
        "Native FlashVSR complete · \(manifest.frames.count) frames · \(Double(monitor.peakBytes)/1e9) GB peak physical"
    )
} catch { fail("Native FlashVSR failed: \(error)") }
