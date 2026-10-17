import AppKit
import CoreML
import Darwin
import Foundation

private func footprint() -> UInt64 {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let status = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return status == KERN_SUCCESS ? info.phys_footprint : 0
}

private final class PeakMemory: @unchecked Sendable {
    private let lock = NSLock()
    private var peak: UInt64 = 0
    private let timer = DispatchSource.makeTimerSource(queue: DispatchQueue(label: "tagger.benchmark.memory"))
    init() {
        timer.schedule(deadline: .now(), repeating: .milliseconds(10))
        timer.setEventHandler { [weak self] in
            guard let self else { return }
            let value = footprint()
            self.lock.lock(); self.peak = max(self.peak, value); self.lock.unlock()
        }
        timer.resume()
    }
    deinit { timer.cancel() }
    var bytes: UInt64 { lock.lock(); defer { lock.unlock() }; return max(peak, footprint()) }
}

@main struct TaggerBenchmark {
    static func main() async throws {
        let arguments = CommandLine.arguments
        func option(_ name: String) -> String? {
            guard let index = arguments.firstIndex(of: name), index + 1 < arguments.count else { return nil }
            return arguments[index + 1]
        }
        guard let directory = option("--model"), let report = option("--report") else {
            throw TaggerError.invalidModel("Supply --model DIRECTORY and --report FILE")
        }
        let repeats = max(3, Int(option("--repeats") ?? "7") ?? 7)
        let config = WDTaggerConfiguration(directory: URL(fileURLWithPath: directory))
        let compiled = config.modelURL.pathExtension == "mlmodelc" ? config.modelURL : try await MLModel.compileModel(at: config.modelURL)
        defer { if compiled != config.modelURL { try? FileManager.default.removeItem(at: compiled) } }
        let fixture = URL(fileURLWithPath: "Tests/Fixtures/WDTagger/rgb-gradient.png")
        let original = CGImageSourceCreateWithURL(fixture as CFURL, nil)!
        let image = CGImageSourceCreateImageAtIndex(original, 0, nil)!
        let context = CGContext(data: nil, width: 1920, height: 1080, bitsPerComponent: 8, bytesPerRow: 0,
                                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        context.draw(image, in: CGRect(x: 0, y: 0, width: 1920, height: 1080))
        let images = Array(repeating: context.makeImage()!, count: 8)
        let monitor = PeakMemory(), start = Date()
        let beforeLoadFootprint = footprint()
        let unitsName = option("--units") ?? "cpuAndGPU"
        let units: MLComputeUnits
        switch unitsName {
        case "cpuAndGPU": units = .cpuAndGPU
        case "all": units = .all
        default: throw TaggerError.invalidModel("--units must be cpuAndGPU or all")
        }
        let caching = !arguments.contains("--no-intermediate-cache")
        let backend = try WDTaggerBackend(modelURL: compiled, labelsURL: config.labelsURL,
                                          computeUnits: units, cacheIntermediates: caching)
        let load = Date().timeIntervalSince(start)
        let afterLoadFootprint = footprint()
        _ = try await backend.tag(batch: images)
        let afterWarmupFootprint = footprint()
        var seconds: [Double] = []
        var checksum: Double = 0
        for _ in 0..<repeats {
            let tick = Date()
            let result = try await backend.tag(batch: images)
            seconds.append(Date().timeIntervalSince(tick))
            checksum = result[0].probabilities.reduce(0) { $0 + Double($1) }
        }
        let median = seconds.sorted()[seconds.count / 2]
        let referenceData = try Data(contentsOf: fixture.deletingLastPathComponent().appendingPathComponent("original-bgr-probabilities.bin"))
        let expected = referenceData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        let actual = try await backend.tag(batch: [image])[0].probabilities
        let expectedTop = Set(expected.indices.sorted { expected[$0] > expected[$1] }.prefix(15))
        let actualTop = Set(actual.indices.sorted { actual[$0] > actual[$1] }.prefix(15))
        let maximumError = zip(expected, actual).map { abs($0 - $1) }.max()!
        let output: [String: Any] = ["units": unitsName, "cacheIntermediates": caching, "repeats": repeats, "batch": 8, "imageSize": [1920, 1080],
            "loadSeconds": load, "beforeLoadFootprintBytes": beforeLoadFootprint,
            "afterLoadFootprintBytes": afterLoadFootprint, "afterWarmupFootprintBytes": afterWarmupFootprint, "batchSeconds": seconds, "medianBatchSeconds": median,
            "framesPerSecond": 8 / median, "peakPhysicalFootprintBytes": monitor.bytes,
            "finalPhysicalFootprintBytes": footprint(), "probabilityChecksum": checksum,
            "thermalState": ProcessInfo.processInfo.thermalState.rawValue,
            "referenceMaximumAbsoluteError": maximumError, "referenceTop15Overlap": expectedTop.intersection(actualTop).count]
        try JSONSerialization.data(withJSONObject: output, options: [.prettyPrinted, .sortedKeys])
            .write(to: URL(fileURLWithPath: report))
        print(String(format: "%.2f frames/s; %.1f MB peak footprint", 8 / median, Double(monitor.bytes) / 1_000_000))
    }
}
