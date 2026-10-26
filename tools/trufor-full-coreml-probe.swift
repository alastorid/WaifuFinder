import CoreML
import CoreGraphics
import Darwin
import Foundation
import ImageIO

private enum ProbeError: Error, CustomStringConvertible {
    case invalid(String)
    var description: String { if case .invalid(let text) = self { return text }; return "invalid probe input" }
}

private func sysctlString(_ name: String) -> String {
    var size = 0
    guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 1 else { return "unknown" }
    var value = [CChar](repeating: 0, count: size)
    let status = value.withUnsafeMutableBufferPointer { buffer in
        sysctlbyname(name, buffer.baseAddress, &size, nil, 0)
    }
    return status == 0 ? String(cString: value) : "unknown"
}

private func footprint() -> UInt64 {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let status = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return status == KERN_SUCCESS ? UInt64(info.phys_footprint) : 0
}

private final class PeakFootprint: @unchecked Sendable {
    private let lock = NSLock()
    private var peak: UInt64 = 0
    private let timer = DispatchSource.makeTimerSource(queue: DispatchQueue(label: "trufor-coreml-peak"))
    init() {
        timer.schedule(deadline: .now(), repeating: .milliseconds(5))
        timer.setEventHandler { [weak self] in self?.sample() }
        timer.resume()
    }
    private func sample() {
        let value = footprint()
        lock.lock(); peak = max(peak, value); lock.unlock()
    }
    func value() -> UInt64 {
        sample(); lock.lock(); defer { lock.unlock() }; return peak
    }
    deinit { timer.cancel() }
}

private func decode(_ url: URL) throws -> (image: CGImage, width: Int, height: Int) {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        throw ProbeError.invalid("Could not decode PNG input")
    }
    return (image, image.width, image.height)
}

private func makeProvider(_ image: (image: CGImage, width: Int, height: Int), name: String) throws -> MLDictionaryFeatureProvider {
    let width = image.width, height = image.height, plane = width * height
    var rgba = [UInt8](repeating: 0, count: plane * 4)
    let rendered = rgba.withUnsafeMutableBytes { storage -> Bool in
        guard let context = CGContext(data: storage.baseAddress, width: width, height: height,
                                      bitsPerComponent: 8, bytesPerRow: width * 4,
                                      space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return false }
        context.draw(image.image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return true
    }
    guard rendered, stride(from: 3, to: rgba.count, by: 4).allSatisfy({ rgba[$0] == 255 }) else {
        throw ProbeError.invalid("Input must be opaque sRGB")
    }
    let array = try MLMultiArray(shape: [1, 3, height, width].map(NSNumber.init(value:)), dataType: .float32)
    let target = array.dataPointer.assumingMemoryBound(to: Float.self)
    for position in 0..<plane {
        let source = position * 4
        target[position] = Float(rgba[source]) * (1.0 / 256.0)
        target[plane + position] = Float(rgba[source + 1]) * (1.0 / 256.0)
        target[2 * plane + position] = Float(rgba[source + 2]) * (1.0 / 256.0)
    }
    return try MLDictionaryFeatureProvider(dictionary: [name: MLFeatureValue(multiArray: array)])
}

private func computeUnit(_ name: String) throws -> MLComputeUnits {
    switch name {
    case "all": return .all
    case "ane": return .cpuAndNeuralEngine
    case "gpu": return .cpuAndGPU
    case "cpu": return .cpuOnly
    default: throw ProbeError.invalid("Policy must be all, ane, gpu, or cpu")
    }
}

private func deviceName(_ device: MLComputeDevice) -> String {
    switch device {
    case .cpu: return "CPU"
    case .gpu: return "GPU"
    case .neuralEngine: return "ANE"
    @unknown default: return "unknown"
    }
}

private func scoreValue(_ array: MLMultiArray) throws -> Float {
    guard array.count == 1 else { throw ProbeError.invalid("Unexpected score tensor shape") }
    switch array.dataType {
    case .float32: return array.dataPointer.assumingMemoryBound(to: Float.self)[0]
    case .float16: return Float(array.dataPointer.assumingMemoryBound(to: Float16.self)[0])
    default: throw ProbeError.invalid("Unexpected score tensor type: \(array.dataType)")
    }
}

private func tensorDiagnostics(_ array: MLMultiArray) -> [String: Any] {
    var finite = 0
    var minimum = Float.infinity
    var maximum = -Float.infinity
    var smallValues: [String] = []
    switch array.dataType {
    case .float32:
        let values = array.dataPointer.assumingMemoryBound(to: Float.self)
        for index in 0..<array.count {
            let value = values[index]
            if value.isFinite {
                finite += 1; minimum = min(minimum, value); maximum = max(maximum, value)
            }
            if array.count <= 16 { smallValues.append(String(value)) }
        }
    case .float16:
        let values = array.dataPointer.assumingMemoryBound(to: Float16.self)
        for index in 0..<array.count {
            let value = Float(values[index])
            if value.isFinite { finite += 1; minimum = min(minimum, value); maximum = max(maximum, value) }
            if array.count <= 16 { smallValues.append(String(value)) }
        }
    default:
        return ["count": array.count, "dtype": String(describing: array.dataType), "finiteCount": 0]
    }
    var result: [String: Any] = ["count": array.count, "dtype": String(describing: array.dataType), "finiteCount": finite,
                                 "minimum": minimum.isFinite ? String(minimum) : "nonfinite",
                                 "maximum": maximum.isFinite ? String(maximum) : "nonfinite"]
    if array.count <= 16 { result["values"] = smallValues }
    return result
}

private func preferredDevices(_ plan: MLComputePlan?) -> [String: Int] {
    guard let plan else { return [:] }
    guard case .program(let program) = plan.modelStructure else { return [:] }
    var counts: [String: Int] = [:]
    func visit(_ block: MLModelStructure.Program.Block) {
        for operation in block.operations {
            let preferred = plan.deviceUsage(for: operation).map { deviceName($0.preferred) } ?? "unknown"
            counts[preferred, default: 0] += 1
            for nested in operation.blocks { visit(nested) }
        }
    }
    for function in program.functions.values { visit(function.block) }
    return counts
}

@main
private struct FullTruForCoreMLProbe {
    static func main() async {
        do {
            let args = CommandLine.arguments
            guard (3...5).contains(args.count) else {
                throw ProbeError.invalid("Usage: trufor-full-coreml-probe MODEL.mlpackage|mlmodelc IMAGE.png [all|ane|gpu|cpu] [samples]")
            }
            let modelURL = URL(fileURLWithPath: args[1])
            let image = try decode(URL(fileURLWithPath: args[2]))
            let policy = args.count >= 4 ? args[3] : "all"
            let samples = args.count >= 5 ? (Int(args[4]) ?? 4) : 4
            guard samples >= 2 else { throw ProbeError.invalid("Use at least two prediction samples") }
            let sampler = PeakFootprint()
            let configuration = MLModelConfiguration()
            configuration.computeUnits = try computeUnit(policy)
            fputs("compiling/loading Core ML model\n", stderr)
            let compileStart = DispatchTime.now().uptimeNanoseconds
            let compiled = modelURL.pathExtension == "mlpackage"
                ? try await MLModel.compileModel(at: modelURL) : modelURL
            let compileMS = Double(DispatchTime.now().uptimeNanoseconds - compileStart) / 1_000_000
            let loadStart = DispatchTime.now().uptimeNanoseconds
            let model = try MLModel(contentsOf: compiled, configuration: configuration)
            let loadMS = Double(DispatchTime.now().uptimeNanoseconds - loadStart) / 1_000_000
            fputs("Core ML model loaded\n", stderr)
            let plan: MLComputePlan?
            if ProcessInfo.processInfo.environment["WAIFU_COREML_SKIP_PLAN"] == "1" {
                plan = nil
            } else {
                plan = try await MLComputePlan.load(contentsOf: compiled, configuration: configuration)
                fputs("Core ML compute plan loaded\n", stderr)
            }
            guard let inputName = model.modelDescription.inputDescriptionsByName.keys.sorted().first,
                  let scoreName = model.modelDescription.outputDescriptionsByName["score"] != nil
                    ? "score" : model.modelDescription.outputDescriptionsByName.keys.sorted().first else {
                throw ProbeError.invalid("Could not resolve Core ML model inputs and outputs")
            }
            let provider = try makeProvider(image, name: inputName)
            var times: [Double] = []
            var score: Float = 0
            var diagnostics: [String: Any] = [:]
            for _ in 0..<samples {
                fputs("predicting\n", stderr)
                let start = DispatchTime.now().uptimeNanoseconds
                let result = try await model.prediction(from: provider)
                times.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000)
                guard let value = result.featureValue(for: scoreName)?.multiArrayValue else {
                    throw ProbeError.invalid("Core ML did not return a score tensor")
                }
                score = try scoreValue(value)
                if diagnostics.isEmpty {
                    for output in model.modelDescription.outputDescriptionsByName.keys.sorted() {
                        if let tensor = result.featureValue(for: output)?.multiArrayValue {
                            diagnostics[output] = tensorDiagnostics(tensor)
                        }
                    }
                }
            }
            let warm = Array(times.dropFirst()).sorted()
            let p50 = warm[warm.count / 2]
            let p95 = warm[min(warm.count - 1, max(0, Int(ceil(Double(warm.count) * 0.95)) - 1))]
            let report: [String: Any] = [
                "backend": "full TruFor Core ML FP16",
                "chip": sysctlString("machdep.cpu.brand_string"),
                "macOS": ProcessInfo.processInfo.operatingSystemVersionString,
                "policy": policy,
                "inputShape": [1, 3, image.height, image.width],
                "compileMilliseconds": compileMS,
                "loadMilliseconds": loadMS,
                "firstPredictionMilliseconds": times[0],
                "warmPredictionMilliseconds": Array(times.dropFirst()),
                "warmP50Milliseconds": p50,
                "warmP95Milliseconds": p95,
                "score": String(score),
                "scoreFinite": score.isFinite,
                "physicalPeakBytes": sampler.value(),
                "preferredOperationCounts": preferredDevices(plan),
                "outputNames": model.modelDescription.outputDescriptionsByName.keys.sorted(),
                "outputDiagnostics": diagnostics,
            ]
            let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            print(String(decoding: data, as: UTF8.self))
        } catch {
            fputs("Full Core ML probe: \(error)\n", stderr)
            exit(1)
        }
    }
}
