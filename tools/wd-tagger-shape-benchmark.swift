// Native benchmark for experimental fixed batch shapes. Compile with the
// production WDTaggerBackend, TaggerBackend and TagVocabulary sources, using -O.
// --images is a JSON array of local image paths. Models and inputs stay local.
import AppKit
import CoreML
import Darwin
import Foundation
import ImageIO

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
    private let timer = DispatchSource.makeTimerSource(queue: DispatchQueue(label: "tagger.shape.memory"))
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

@main struct ShapeBenchmark {
    static func main() throws {
        let arguments = CommandLine.arguments
        func option(_ name: String) -> String? {
            guard let index = arguments.firstIndex(of: name), index + 1 < arguments.count else { return nil }
            return arguments[index + 1]
        }
        if arguments.contains("--thermal") {
            print(ProcessInfo.processInfo.thermalState.rawValue)
            return
        }
        if let modelPath = option("--model"), let destination = option("--compile-to") {
            let compiled = try MLModel.compileModel(at: URL(fileURLWithPath: modelPath))
            defer { try? FileManager.default.removeItem(at: compiled) }
            try FileManager.default.copyItem(at: compiled, to: URL(fileURLWithPath: destination))
            return
        }
        guard let modelPath = option("--model"),
              let reportPath = option("--report"), let labelsPath = option("--labels"),
              (option("--images") != nil) != (option("--tensor-input") != nil) else {
            throw TaggerError.invalidModel("Supply --model, --labels, --report and exactly one of --images or --tensor-input")
        }
        let urls: [URL]
        if let imagesPath = option("--images") {
            urls = try JSONDecoder().decode([String].self, from: Data(contentsOf: URL(fileURLWithPath: imagesPath)))
                .map { URL(fileURLWithPath: $0) }
        } else { urls = [] }
        let streamedInput = arguments.contains("--stream-input")
        let tensorFile = try streamedInput ? option("--tensor-input").map { try FileHandle(forReadingFrom: URL(fileURLWithPath: $0)) } : nil
        defer { try? tensorFile?.close() }
        let tensorInput = try (streamedInput ? nil : option("--tensor-input")).map {
            try Data(contentsOf: URL(fileURLWithPath: $0), options: .mappedIfSafe)
        }
        let tensorRowBytes = 3 * 448 * 448 * MemoryLayout<Float>.size
        if let tensorInput, tensorInput.count % tensorRowBytes != 0 { throw TaggerError.invalidBatch }
        let fileBytes = try tensorFile?.seekToEnd()
        if let fileBytes, fileBytes % UInt64(tensorRowBytes) != 0 { throw TaggerError.invalidBatch }
        let frameCount = fileBytes.map { Int($0) / tensorRowBytes } ?? tensorInput.map { $0.count / tensorRowBytes } ?? urls.count
        guard frameCount > 0 else { throw TaggerError.invalidBatch }
        let repeats = max(1, Int(option("--repeats") ?? "5") ?? 5)
        let modelURL = URL(fileURLWithPath: modelPath)
        let compileStart = Date()
        let compiled = modelURL.pathExtension == "mlmodelc" ? modelURL : try MLModel.compileModel(at: modelURL)
        let compileSeconds = Date().timeIntervalSince(compileStart)
        defer { if compiled != modelURL { try? FileManager.default.removeItem(at: compiled) } }
        let monitor = PeakMemory()
        let config = MLModelConfiguration()
        let units = option("--units") ?? "cpuAndGPU"
        guard ["cpuAndGPU", "all"].contains(units) else { throw TaggerError.invalidModel("Unknown compute units") }
        config.computeUnits = units == "all" ? .all : .cpuAndGPU
        if arguments.contains("--fast-prediction") {
            config.optimizationHints.specializationStrategy = .fastPrediction
        }
        config.allowLowPrecisionAccumulationOnGPU = arguments.contains("--low-precision-accumulation")
        let beforeLoad = footprint(), loadStart = Date()
        let model = try MLModel(contentsOf: compiled, configuration: config)
        let loadSeconds = Date().timeIntervalSince(loadStart)
        let afterLoad = footprint()
        let feature = model.modelDescription.inputDescriptionsByName.first!
        let shape = feature.value.multiArrayConstraint!.shape.map(\.intValue)
        guard shape.count == 4, Array(shape.dropFirst()) == [3, 448, 448],
              feature.value.multiArrayConstraint!.dataType == .float32 else {
            throw TaggerError.invalidModel("Unexpected input shape/type")
        }
        let batch = shape[0], plane = 448 * 448, rowSize = 3 * 448 * 448
        let input = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        let pointer = input.dataPointer.assumingMemoryBound(to: Float.self)
        let provider = try MLDictionaryFeatureProvider(dictionary: [feature.key: MLFeatureValue(multiArray: input)])
        let outputName = model.modelDescription.outputDescriptionsByName.first!.key
        let vocabulary = try TagVocabulary.load(from: URL(fileURLWithPath: labelsPath))
        // One context for the whole run, matching the production actor.
        let context = CIContext(options: [.workingColorSpace: NSNull(), .cacheIntermediates: true])
        var rgba = [UInt8](repeating: 0, count: plane * 4)
        var probabilities = [Float]()
        var tensors = Data()
        var sweepSeconds = [Double](), stageSeconds = [[String: Double]]()
        var checksum = 0.0
        let thermalBefore = ProcessInfo.processInfo.thermalState.rawValue
        var afterWarmup: UInt64 = 0
        for sweep in -1..<repeats {
            let start = Date()
            var decode = 0.0, prepare = 0.0, predict = 0.0, decodeOutput = 0.0
            for offset in stride(from: 0, to: frameCount, by: batch) {
                if sweep == -1, offset > 0 { break }
                try autoreleasepool {
                    let count = min(batch, frameCount - offset)
                    var tick = Date()
                    let images = try (tensorInput == nil && tensorFile == nil ? Array(urls[offset..<(offset + count)]) : []).map { url -> CGImage in
                        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
                              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
                            throw TaggerError.preprocessing
                        }
                        return image
                    }
                    decode += Date().timeIntervalSince(tick)
                    tick = Date()
                    if let tensorFile {
                        try tensorFile.seek(toOffset: UInt64(offset * tensorRowBytes))
                        var data = Data(capacity: count * tensorRowBytes)
                        while data.count < count * tensorRowBytes {
                            guard let part = try tensorFile.read(upToCount: count * tensorRowBytes - data.count), !part.isEmpty else { throw TaggerError.invalidBatch }
                            data.append(part)
                        }
                        data.withUnsafeBytes { UnsafeMutableRawPointer(pointer).copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
                    } else if let tensorInput {
                        tensorInput.withUnsafeBytes { bytes in
                            UnsafeMutableRawPointer(pointer).copyMemory(
                                from: bytes.baseAddress!.advanced(by: offset * tensorRowBytes),
                                byteCount: count * tensorRowBytes)
                        }
                    }
                    for (row, image) in images.enumerated() {
                        try WDTaggerBackend.renderRGBA(image, context: context, into: &rgba)
                        for pixel in 0..<plane {
                            pointer[row * rowSize + pixel] = Float(rgba[pixel * 4])
                            pointer[row * rowSize + plane + pixel] = Float(rgba[pixel * 4 + 1])
                            pointer[row * rowSize + 2 * plane + pixel] = Float(rgba[pixel * 4 + 2])
                        }
                        if sweep == 0, option("--tensor-output") != nil {
                            tensors.append(contentsOf: UnsafeRawBufferPointer(start: pointer.advanced(by: row * rowSize), count: rowSize * 4))
                        }
                    }
                    if count < batch {
                        for row in count..<batch {
                            pointer.advanced(by: row * rowSize).update(from: pointer.advanced(by: (count - 1) * rowSize), count: rowSize)
                        }
                    }
                    prepare += Date().timeIntervalSince(tick)
                    tick = Date()
                    let prediction = try model.prediction(from: provider)
                    _ = prediction.featureValue(for: outputName)?.multiArrayValue?[0].floatValue
                    predict += Date().timeIntervalSince(tick)
                    tick = Date()
                    guard let output = prediction.featureValue(for: outputName)?.multiArrayValue,
                          output.dataType == .float32, output.shape.map(\.intValue) == [batch, TagVocabulary.headCount] else {
                        throw TaggerError.invalidProbabilities
                    }
                    let source = output.dataPointer.assumingMemoryBound(to: Float.self)
                    let rowStride = output.strides[0].intValue, columnStride = output.strides[1].intValue
                    for row in 0..<count {
                        let start = source.advanced(by: row * rowStride)
                        let scores = columnStride == 1
                            ? Array(UnsafeBufferPointer(start: start, count: TagVocabulary.headCount))
                            : (0..<TagVocabulary.headCount).map { start[$0 * columnStride] }
                        let result = try vocabulary.result(probabilities: scores)
                        checksum += Double(result.generalScore)
                        if sweep == 0 { probabilities.append(contentsOf: scores) }
                    }
                    decodeOutput += Date().timeIntervalSince(tick)
                }
            }
            if sweep == -1 { afterWarmup = footprint() }
            else {
                sweepSeconds.append(Date().timeIntervalSince(start))
                stageSeconds.append(["decode": decode, "prepare": prepare, "prediction": predict, "result": decodeOutput])
            }
        }
        let median = sweepSeconds.sorted()[sweepSeconds.count / 2]
        var report: [String: Any] = [
            "batch": batch, "frames": frameCount, "repeats": repeats, "computeUnits": units,
            "inputMode": streamedInput ? "streamed-tensors" : tensorInput == nil ? "images" : "retained-tensors",
            "fastPrediction": arguments.contains("--fast-prediction"),
            "lowPrecisionAccumulation": config.allowLowPrecisionAccumulationOnGPU,
            "compileSeconds": compileSeconds, "loadSeconds": loadSeconds,
            "beforeLoadFootprintBytes": beforeLoad, "afterLoadFootprintBytes": afterLoad,
            "afterWarmupFootprintBytes": afterWarmup, "peakPhysicalFootprintBytes": monitor.bytes,
            "finalPhysicalFootprintBytes": footprint(), "sweepSeconds": sweepSeconds,
            "stageSeconds": stageSeconds, "medianSweepSeconds": median,
            "framesPerSecond": Double(frameCount) / median, "checksum": checksum,
            "thermalBefore": thermalBefore, "thermalAfter": ProcessInfo.processInfo.thermalState.rawValue
        ]
        if let output = option("--probability-output") {
            try probabilities.withUnsafeBytes { try Data($0).write(to: URL(fileURLWithPath: output)) }
        }
        if let output = option("--tensor-output") {
            try tensors.write(to: URL(fileURLWithPath: output))
            report["tensorShape"] = [frameCount, 3, 448, 448]
        }
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: URL(fileURLWithPath: reportPath))
        print(String(format: "batch %d: %.2f frames/s, %.1f MB peak", batch, Double(frameCount) / median, Double(monitor.bytes) / 1e6))
    }
}
