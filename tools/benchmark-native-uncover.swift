#!/usr/bin/env swift
import Foundation
import CryptoKit

// Compare the new VAE paths with their original implementations, one worker at a time.
let args = Array(CommandLine.arguments.dropFirst())
func option(_ name: String, _ fallback: String? = nil) -> String? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return fallback }
    return args[i + 1]
}
func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8)); exit(1)
}
if args.contains("--help") {
    print("Usage: swift tools/benchmark-native-uncover.swift --job PREPARED_JOB --output NEW_DIRECTORY [--model 3b|7b] [--pairs 4] [--budget-gb 4] [--worker EXECUTABLE]")
    print("Use a cooled Mac. Jobs run serially in alternating order; summary.json contains timings, memory and PNG byte parity. The output directory must not exist.")
    exit(0)
}
let fm = FileManager.default
let root = URL(fileURLWithPath: #filePath).standardizedFileURL.deletingLastPathComponent().deletingLastPathComponent()
guard let sourcePath = option("--job"), let outputPath = option("--output") else { fail("Supply --job and --output; see --help.") }
let source = URL(fileURLWithPath: sourcePath).standardizedFileURL
let output = URL(fileURLWithPath: outputPath).standardizedFileURL
let variant = option("--model", "3b")!
guard ["3b", "7b"].contains(variant), let pairs = Int(option("--pairs", "4")!), (1...20).contains(pairs),
      let budget = Double(option("--budget-gb", "4")!), budget.isFinite, budget > 0 else { fail("Invalid model, pair count or budget.") }
let worker = option("--worker", root.appendingPathComponent(".native-runtime/waifu-seedvr2").path)!
let model = root.appendingPathComponent(".native-runtime/" + (variant == "3b" ? "model-optimized" : "model-7b-optimized"))
let embedding = root.appendingPathComponent(".native-runtime/pos_emb.safetensors")
guard fm.isExecutableFile(atPath: worker), fm.fileExists(atPath: source.appendingPathComponent("manifest.json").path),
      fm.fileExists(atPath: source.appendingPathComponent("degraded").path),
      fm.fileExists(atPath: model.path), fm.fileExists(atPath: embedding.path) else { fail("Prepared job, native worker or selected model is missing. Run native setup first.") }
guard !fm.fileExists(atPath: output.path) else { fail("Output already exists; choose a new directory.") }
func readJSON(_ url: URL) throws -> [String: Any] {
    guard let value = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else { throw NSError(domain: "Benchmark", code: 1) }
    return value
}
func hashes(_ job: URL) throws -> [String: String] {
    let files = try fm.contentsOfDirectory(at: job.appendingPathComponent("reconstructed"), includingPropertiesForKeys: nil)
        .filter { $0.pathExtension.lowercased() == "png" }
    guard !files.isEmpty else { throw NSError(domain: "No reconstructed PNGs", code: 1) }
    return try Dictionary(uniqueKeysWithValues: files.map { ($0.lastPathComponent, SHA256.hash(data: try Data(contentsOf: $0)).map { String(format: "%02x", $0) }.joined()) })
}
func median(_ values: [Double]) -> Double {
    let v = values.sorted(); return v.count % 2 == 1 ? v[v.count / 2] : (v[v.count / 2 - 1] + v[v.count / 2]) / 2
}
do {
    let manifest = try readJSON(source.appendingPathComponent("manifest.json"))
    guard let frames = manifest["frames"] as? [Any], frames.count >= 5 else { fail("Prepared manifest must contain at least five frames.") }
    try fm.createDirectory(at: output, withIntermediateDirectories: true)
    var records = [[String: Any]](), reductions = [Double](), baselinePeaks = [Double](), fusedPeaks = [Double]()
    var allExact = true
    for pair in 1...pairs {
        let order = pair % 2 == 1 ? ["baseline", "fused"] : ["fused", "baseline"]
        var times = [String: Double](), pixels = [String: [String: String]]()
        for kind in order {
            let job = output.appendingPathComponent("\(variant)-\(kind)-\(pair)")
            try fm.createDirectory(at: job.appendingPathComponent("reconstructed"), withIntermediateDirectories: true)
            for name in ["manifest.json", "degraded"] { try fm.copyItem(at: source.appendingPathComponent(name), to: job.appendingPathComponent(name)) }
            let log = job.appendingPathComponent("worker.log")
            _ = fm.createFile(atPath: log.path, contents: nil)
            let handle = try FileHandle(forWritingTo: log)
            let process = Process(); process.executableURL = URL(fileURLWithPath: worker)
            process.arguments = ["--job", job.path, "--model", model.path, "--embedding", embedding.path, "--memory-budget-gb", String(budget), "--stream-weights"]
            if kind == "baseline" { process.arguments! += ["--no-vae-causal-fusion", "--no-vae-pointwise-conv", "--no-vae-residual-fusion"] }
            process.standardOutput = handle; process.standardError = handle
            print("Pair \(pair)/\(pairs): \(variant) \(kind)")
            try process.run(); process.waitUntilExit(); try handle.close()
            guard process.terminationStatus == 0 else { fail("Worker failed. Inspect the worker.log in the new output directory.") }
            let report = try readJSON(job.appendingPathComponent("report.json"))
            guard let seconds = report["seconds"] as? Double, seconds.isFinite, seconds > 0,
                  let peak = report["peakPhysicalFootprintBytes"] as? Double else { fail("Worker report lacks timing or memory measurements.") }
            times[kind] = seconds; pixels[kind] = try hashes(job)
            guard pixels[kind]!.count == frames.count else { fail("Reconstructed PNG count differs from the prepared manifest.") }
            if kind == "baseline" { baselinePeaks.append(peak) } else { fusedPeaks.append(peak) }
            // Whitelist metrics so the shareable summary never includes source paths.
            var record: [String: Any] = ["pair": pair, "kind": kind, "seconds": seconds, "peakPhysicalFootprintBytes": peak]
            if let phases = report["phaseSeconds"] { record["phaseSeconds"] = phases }
            records.append(record)
        }
        let exact = pixels["baseline"]! == pixels["fused"]!
        allExact = allExact && exact
        reductions.append(100 * (1 - times["fused"]! / times["baseline"]!))
        print("Pair \(pair): PNG byte parity \(exact), time reduction \(reductions.last!)%")
    }
    let summary: [String: Any] = ["model": variant, "frames": frames.count, "pairs": pairs, "memoryBudgetGB": budget,
        "installedRAMBytes": ProcessInfo.processInfo.physicalMemory, "allPNGByteIdentical": allExact,
        "medianPairedTimeReductionPercent": median(reductions), "fasterPairs": reductions.filter { $0 > 0 }.count,
        "medianBaselinePeakBytes": median(baselinePeaks), "medianFusedPeakBytes": median(fusedPeaks), "runs": records]
    try JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted, .sortedKeys]).write(to: output.appendingPathComponent("summary.json"))
    print("Saved summary.json. Review both speed and memory before drawing conclusions.")
    if !allExact { fail("Output parity failed; retain these results for investigation.") }
} catch { fail("Benchmark failed: \(error.localizedDescription)") }
