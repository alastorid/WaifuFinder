// Compile with production Sources/ (excluding App/). Runs real thumbnail
// regeneration/cache decoding, preprocessing, worker IPC and result encoding.
import AppKit
import CryptoKit
import Darwin
import Foundation
import ImageIO

private func processFootprint(_ pid: Int32) -> UInt64 {
    var info = rusage_info_v4()
    let status = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
            proc_pid_rusage(pid, RUSAGE_INFO_V4, $0)
        }
    }
    return status == 0 ? info.ri_phys_footprint : 0
}

private final class AppPeakMemory: @unchecked Sendable {
    private let timer = DispatchSource.makeTimerSource(queue: DispatchQueue(label: "wd.app.memory"))
    private let lock = NSLock()
    private var parent: UInt64 = 0, workers: UInt64 = 0, combined: UInt64 = 0
    init() {
        timer.schedule(deadline: .now(), repeating: .milliseconds(10))
        timer.setEventHandler { [weak self] in
            guard let self else { return }
            var children = [Int32](repeating: 0, count: 16)
            let count = children.withUnsafeMutableBytes {
                proc_listchildpids(getpid(), $0.baseAddress, Int32($0.count))
            }
            // proc_listchildpids returns a PID count, not a byte count.
            let worker = children.prefix(max(0, Int(count))).filter { $0 > 0 }
                .reduce(UInt64(0)) { $0 + processFootprint($1) }
            let own = processFootprint(getpid())
            self.lock.withLock {
                self.parent = max(self.parent, own)
                self.workers = max(self.workers, worker)
                self.combined = max(self.combined, own + worker)
            }
        }
        timer.resume()
    }
    deinit { timer.cancel() }
    var report: [String: UInt64] {
        lock.withLock { ["peakParentBytes": parent, "peakChildrenBytes": workers,
                         "peakSimultaneousCombinedBytes": combined] }
    }
}

private struct EncodedFrame: Sendable {
    let index: Int
    let jpeg: Data
}

private actor BatchSource {
    let cached: [Data]
    let video: URL?
    let width: Int
    var index = 0
    var fetchSeconds = 0.0
    init(cached: [Data], video: URL?, width: Int) {
        self.cached = cached; self.video = video; self.width = width
    }
    func next() async throws -> [EncodedFrame]? {
        guard index < 64 else { return [] }
        let tick = ProcessInfo.processInfo.systemUptime
        var batch: [EncodedFrame] = []
        while index < 64 && batch.count < 8 {
            try Task.checkCancellation()
            let data: Data
            if let video {
                data = try await ThumbnailRegenerator.shared.jpegData(
                    for: video, timeSeconds: 0.05 + Double(index) * 0.125, analysisWidth: width)
            } else {
                data = cached[index]
            }
            batch.append(EncodedFrame(index: index, jpeg: data)); index += 1
        }
        fetchSeconds += ProcessInfo.processInfo.systemUptime - tick
        return batch
    }
}

private actor BatchOutput {
    var data = Data()
    var indices: [Int] = []
    func append(_ results: [TagResult], batch: [EncodedFrame]) throws {
        guard results.count == batch.count else { throw TaggerError.invalidBatch }
        for (frame, result) in zip(batch, results) {
            guard let encoded = TagProbabilityCodec.encode(result.probabilities) else {
                throw TaggerError.invalidProbabilities
            }
            data.append(encoded); indices.append(frame.index)
        }
    }
}

@main struct AppTaggerBenchmark {
    static func main() async throws {
        let args = CommandLine.arguments
        func option(_ name: String) -> String? {
            guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
            return args[i + 1]
        }
        guard let model = option("--model"), let executable = option("--worker"),
              let destination = option("--report") else { throw TaggerError.invalidBatch }
        let video = option("--video").map { URL(fileURLWithPath: $0) }
        let width = Int(option("--width") ?? "512") ?? 512
        var cached: [Data] = []
        if video == nil {
            guard let input = option("--rgba-input") else { throw TaggerError.invalidBatch }
            let bytes = try Data(contentsOf: URL(fileURLWithPath: input))
            let frameBytes = 448 * 448 * 4
            guard bytes.count == 64 * frameBytes else { throw TaggerError.invalidBatch }
            // Construct a compressed cache from the retained corpus before timing.
            cached = try (0..<64).map { i in
                try autoreleasepool {
                    let frame = bytes.subdata(in: (i * frameBytes)..<((i + 1) * frameBytes))
                    let image = CGImage(width: 448, height: 448, bitsPerComponent: 8, bitsPerPixel: 32,
                        bytesPerRow: 448 * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                        provider: CGDataProvider(data: frame as CFData)!, decode: nil,
                        shouldInterpolate: false, intent: .defaultIntent)!
                    return try ThumbnailCache.encodeJPEG(image)
                }
            }
        }
        let memory = AppPeakMemory()
        let backend = try WDTaggerBackend(directory: URL(fileURLWithPath: model),
                                          executableURL: URL(fileURLWithPath: executable))
        @Sendable func images(_ batch: [EncodedFrame]) throws -> [CGImage] {
            try batch.map { frame in
                guard let source = CGImageSourceCreateWithData(frame.jpeg as CFData, nil),
                      let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
                    throw TaggerError.preprocessing
                }
                return image
            }
        }
        let warm = BatchSource(cached: cached, video: video, width: width)
        _ = try await backend.tag(batch: images(try await warm.next()!))
        let thermalBefore = ProcessInfo.processInfo.thermalState.rawValue
        let prefetch = args.contains("--prefetch")
        var times: [Double] = [], fetch: [Double] = [], reference: Data?
        for _ in 0..<3 {
            let source = BatchSource(cached: cached, video: video, width: width)
            let output = BatchOutput()
            let tick = ProcessInfo.processInfo.systemUptime
            if prefetch {
                try await TaggingBatchPipeline.run(next: { try await source.next() }, whenEmpty: { false }) { batch in
                    let results = try await backend.tag(batch: images(batch))
                    try await output.append(results, batch: batch)
                }
            } else {
                while let batch = try await source.next(), !batch.isEmpty {
                    let results = try await backend.tag(batch: images(batch))
                    try await output.append(results, batch: batch)
                }
            }
            times.append(ProcessInfo.processInfo.systemUptime - tick)
            fetch.append(await source.fetchSeconds)
            guard await output.indices == Array(0..<64) else { throw TaggerError.invalidBatch }
            let data = await output.data
            if let reference {
                guard reference == data else { throw TaggerError.invalidProbabilities }
            } else { reference = data }
        }
        let report: [String: Any] = [
            "prefetch": prefetch, "input": video == nil ? "in-memory-jpeg-corpus" : "regenerated-video-thumbnails",
            "analysisWidth": width, "frames": 64, "batch": 8, "rounds": 3,
            "sweepSeconds": times, "fetchSeconds": fetch, "appPipelineFPS": 64 / times.sorted()[1],
            "memory": memory.report, "worker": executable,
            "outputsStableAcrossRounds": true,
            "outputSHA256": SHA256.hash(data: reference!).map { String(format: "%02x", $0) }.joined(),
            "thermalBefore": thermalBefore, "thermalAfter": ProcessInfo.processInfo.thermalState.rawValue,
            "method": "Production thumbnail retrieval/regeneration, JPEG decode, CI preprocessing, native worker, vocabulary and probability encoding; ordered 64-frame sweeps. Warmup excluded. Combined footprint sums this parent and its direct children at each 10 ms sample; shared mappings can be accounted in both processes. OS codec/XPC service memory is excluded. No GPU trace.",
        ]
        guard memory.report["peakChildrenBytes", default: 0] > 0 else {
            throw NSError(domain: "Worker memory sampler did not observe a child", code: 1)
        }
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: URL(fileURLWithPath: destination))
        print("\(64 / times.sorted()[1]) app pipeline fps, prefetch=\(prefetch)")
    }
}
