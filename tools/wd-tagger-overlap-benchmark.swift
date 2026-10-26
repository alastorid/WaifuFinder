// Standalone native adapter benchmark; link with the production tagger sources.
import AppKit
import CryptoKit
import Foundation

// Only unrelated app configuration helpers are replaced in this standalone tool.
enum ModelInstaller {
    static func fail(_ text: String) -> Error { NSError(domain: "WD benchmark", code: 1, userInfo: [NSLocalizedDescriptionKey: text]) }
}
enum PhotosMedia { static let analysisPixelSize = 512 }

private final class GPUSamples: @unchecked Sendable {
    private let timer = DispatchSource.makeTimerSource(queue: DispatchQueue(label: "wd.gpu.samples"))
    private let lock = NSLock()
    private var values: [Int] = []
    init() {
        timer.schedule(deadline: .now(), repeating: .milliseconds(250))
        timer.setEventHandler { [weak self] in
            let command = Process(), pipe = Pipe()
            command.executableURL = URL(fileURLWithPath: "/usr/sbin/ioreg")
            command.arguments = ["-r", "-d", "1", "-c", "AGXAccelerator"]
            command.standardOutput = pipe
            do {
                try command.run()
                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                command.waitUntilExit()
                let text = String(decoding: data, as: UTF8.self)
                let pattern = #""Device Utilization %"=(\d+)"#
                if let regex = try? NSRegularExpression(pattern: pattern),
                   let match = regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)),
                   let range = Range(match.range(at: 1), in: text), let value = Int(text[range]) {
                    self?.lock.withLock { self?.values.append(value) }
                }
            } catch { }
        }
        timer.resume()
    }
    deinit { timer.cancel() }
    var samples: [Int] { lock.withLock { values } }
}
private actor Source {
    let images: [CGImage]
    var position = 0
    init(_ images: [CGImage]) { self.images = images }
    func next() -> [CGImage]? {
        guard position < images.count else { return nil }
        let end = min(position + 8, images.count)
        defer { position = end }
        return Array(images[position..<end])
    }
}
private actor Results {
    var bytes = Data()
    func append(_ rows: [TagResult]) throws {
        for row in rows {
            guard let encoded = row.encodedProbabilities else { throw TaggerError.invalidProbabilities }
            bytes.append(encoded)
        }
    }
}
@main struct WDOverlapBenchmark {
    static func main() async throws {
        let args = CommandLine.arguments
        func option(_ key: String) -> String? { args.firstIndex(of: key).flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil } }
        guard let model = option("--model"), let worker = option("--worker"), let report = option("--report") else { throw TaggerError.invalidBatch }
        let depth = Int(option("--depth") ?? "1")!
        let rounds = Int(option("--rounds") ?? "3")!
        let images = try (0..<64).map { seed -> CGImage in
            let width = seed % 2 == 0 ? 512 : 384, height = seed % 2 == 0 ? 384 : 512
            var bytes = [UInt8](repeating: 255, count: width * height * 4)
            for i in 0..<(width * height) {
                bytes[i * 4] = UInt8((i % width + seed * 7) % 256)
                bytes[i * 4 + 1] = UInt8((i / width + seed * 11) % 256)
                bytes[i * 4 + 2] = UInt8((i / 37 + seed * 13) % 256)
            }
            guard let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                provider: CGDataProvider(data: Data(bytes) as CFData)!, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else { throw TaggerError.preprocessing }
            return image
        }
        let backend = try WDTaggerBackend(directory: URL(fileURLWithPath: model), executableURL: URL(fileURLWithPath: worker))
        _ = try await backend.tag(batch: Array(images.prefix(8)))
        var seconds: [Double] = [], hashes: [String] = []
        let thermalBefore = ProcessInfo.processInfo.thermalState.rawValue
        let gpu = GPUSamples()
        for _ in 0..<rounds {
            let source = Source(images), output = Results()
            let started = ContinuousClock.now
            try await TaggingBatchPipeline.runPipelined(maximumInFlight: depth,
                next: { await source.next() }, whenEmpty: { false },
                process: { try await backend.tag(batch: $0) }, commit: { try await output.append($0) })
            let elapsed = started.duration(to: .now)
            seconds.append(Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18)
            let data = await output.bytes
            hashes.append(SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined())
        }
        guard Set(hashes).count == 1 else { throw TaggerError.invalidProbabilities }
        let samples = gpu.samples
        let result: [String: Any] = ["depth": depth, "framesPerRound": 64, "seconds": seconds,
            "medianFPS": 64 / seconds.sorted()[seconds.count / 2], "outputSHA256": hashes[0],
            "gpuUtilizationSamples": samples, "gpuUtilizationMean": samples.isEmpty ? 0 : Double(samples.reduce(0,+)) / Double(samples.count),
            "thermalBefore": thermalBefore, "thermalAfter": ProcessInfo.processInfo.thermalState.rawValue,
            "scope": "Native synthetic portrait/landscape images; real production Core Image preparation, WDT4 worker, quantized results and ordered commit. ioreg samples describe whole-device GPU activity, not this process alone. No interactive UI." ]
        try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: report))
        print("depth \(depth): \(result["medianFPS"]!) fps, GPU mean \(result["gpuUtilizationMean"]!)%")
    }
}
