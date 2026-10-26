import CoreML
import CoreGraphics
import Darwin
import Foundation
import ImageIO

private let side = 546
private let centerOffset = 17
private let comparedSide = 512
private let tileOriginX = 512
private let tileOriginY = 512

private enum ProbeError: Error, CustomStringConvertible {
    case invalid(String)
    var description: String {
        if case .invalid(let text) = self { return text }
        return "invalid probe input"
    }
}

private func physicalFootprint() -> UInt64 {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let status = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return status == KERN_SUCCESS ? UInt64(info.phys_footprint) : 0
}

private func sysctlString(_ name: String) -> String {
    var size = 0
    guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 1 else { return "unknown" }
    var value = [CChar](repeating: 0, count: size)
    let result = value.withUnsafeMutableBufferPointer { buffer in
        sysctlbyname(name, buffer.baseAddress, &size, nil, 0)
    }
    return result == 0 ? String(cString: value) : "unknown"
}

private final class FootprintSampler: @unchecked Sendable {
    private let lock = NSLock()
    private var peak: UInt64 = 0
    private let timer = DispatchSource.makeTimerSource(queue: DispatchQueue(label: "trufor-coreml-memory"))
    init() {
        timer.schedule(deadline: .now(), repeating: .milliseconds(5))
        timer.setEventHandler { [weak self] in self?.sample() }
        timer.resume()
    }
    func sample() {
        let current = physicalFootprint()
        lock.lock(); peak = max(peak, current); lock.unlock()
    }
    func peakBytes() -> UInt64 {
        sample()
        lock.lock(); defer { lock.unlock() }
        return peak
    }
    deinit { timer.cancel() }
}

private func decodeRGBA(_ url: URL) throws -> (bytes: [UInt8], width: Int, height: Int) {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        throw ProbeError.invalid("Could not decode input image")
    }
    var bytes = [UInt8](repeating: 0, count: image.width * image.height * 4)
    let rendered = bytes.withUnsafeMutableBytes { storage -> Bool in
        guard let context = CGContext(data: storage.baseAddress, width: image.width, height: image.height,
                                      bitsPerComponent: 8, bytesPerRow: image.width * 4,
                                      space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return false }
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        return true
    }
    guard rendered, stride(from: 3, to: bytes.count, by: 4).allSatisfy({ bytes[$0] == 255 }) else {
        throw ProbeError.invalid("Input must render as opaque sRGB")
    }
    return (bytes, image.width, image.height)
}

private func makeInput(_ image: (bytes: [UInt8], width: Int, height: Int),
                       rgbName: String, validName: String) throws -> MLDictionaryFeatureProvider {
    let rgb = try MLMultiArray(shape: [1, 3, side, side].map { NSNumber(value: $0) }, dataType: .float32)
    let valid = try MLMultiArray(shape: [1, 1, side, side].map { NSNumber(value: $0) }, dataType: .float32)
    let rgbBase = rgb.dataPointer.assumingMemoryBound(to: Float.self)
    let validBase = valid.dataPointer.assumingMemoryBound(to: Float.self)
    let plane = side * side
    for y in 0..<side {
        let sourceY = tileOriginY + y - centerOffset
        for x in 0..<side {
            let sourceX = tileOriginX + x - centerOffset
            let position = y * side + x
            if sourceX >= 0 && sourceY >= 0 && sourceX < image.width && sourceY < image.height {
                let source = (sourceY * image.width + sourceX) * 4
                rgbBase[position] = Float(image.bytes[source]) * (1.0 / 256.0)
                rgbBase[plane + position] = Float(image.bytes[source + 1]) * (1.0 / 256.0)
                rgbBase[2 * plane + position] = Float(image.bytes[source + 2]) * (1.0 / 256.0)
                validBase[position] = 1
            } else {
                rgbBase[position] = 0; rgbBase[plane + position] = 0; rgbBase[2 * plane + position] = 0
                validBase[position] = 0
            }
        }
    }
    return try MLDictionaryFeatureProvider(dictionary: [
        rgbName: MLFeatureValue(multiArray: rgb), validName: MLFeatureValue(multiArray: valid)
    ])
}

private struct EvidenceReference {
    let bytes: Data
    let width: Int
    let height: Int
    let score: Float
    let noiseOffset: Int

    init(url: URL) throws {
        let loaded = try Data(contentsOf: url, options: .mappedIfSafe)
        func word(_ offset: Int) -> UInt32 {
            loaded.withUnsafeBytes { raw in raw.loadUnaligned(fromByteOffset: offset, as: UInt32.self) }
        }
        guard loaded.count >= 20, String(data: loaded[0..<4], encoding: .ascii) == "TFM1" else {
            throw ProbeError.invalid("Reference is not a TFM1 evidence file")
        }
        let widthValue = Int(word(4)), heightValue = Int(word(8))
        let mapCount = Int(word(12))
        guard mapCount == 3, widthValue >= comparedSide, heightValue >= comparedSide,
              loaded.count == 20 + widthValue * heightValue * mapCount * MemoryLayout<Float>.size else {
            throw ProbeError.invalid("Reference evidence dimensions or map layout are invalid")
        }
        self.bytes = loaded
        self.width = widthValue
        self.height = heightValue
        self.score = Float(bitPattern: word(16))
        self.noiseOffset = 20 + 2 * widthValue * heightValue * MemoryLayout<Float>.size
    }

    func noiseprint(x: Int, y: Int) -> Float {
        let offset = noiseOffset + (y * width + x) * MemoryLayout<Float>.size
        let bits = bytes.withUnsafeBytes { raw in raw.loadUnaligned(fromByteOffset: offset, as: UInt32.self) }
        return Float(bitPattern: bits)
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

private func collectOperations(_ block: MLModelStructure.Program.Block, plan: MLComputePlan,
                               into rows: inout [[String: Any]], counts: inout [String: Int],
                               cost: inout [String: Double]) {
    for operation in block.operations {
        let usage = plan.deviceUsage(for: operation)
        let preferred = usage.map { deviceName($0.preferred) } ?? "unavailable"
        counts[preferred, default: 0] += 1
        let weight = plan.estimatedCost(of: operation)?.weight ?? 1
        cost[preferred, default: 0] += weight
        rows.append([
            "operator": operation.operatorName,
            "name": operation.outputs.first?.name ?? "",
            "preferredDevice": preferred,
            "supportedDevices": usage?.supported.map(deviceName) ?? [],
            "estimatedCostWeight": weight
        ])
        for nested in operation.blocks {
            collectOperations(nested, plan: plan, into: &rows, counts: &counts, cost: &cost)
        }
    }
}

private func computeUnits(_ name: String) throws -> MLComputeUnits {
    switch name {
    case "all": return .all
    case "ane": return .cpuAndNeuralEngine
    case "gpu": return .cpuAndGPU
    case "cpu": return .cpuOnly
    default: throw ProbeError.invalid("Policy must be all, ane, gpu, or cpu")
    }
}

@main
private struct TruForCoreMLProbe {
    static func main() async {
        do {
            let args = CommandLine.arguments
            guard args.count == 6 || args.count == 7 else {
                throw ProbeError.invalid("Usage: trufor-coreml-probe MODEL.mlmodelc IMAGE.png REFERENCE.tfm1 all|ane|gpu|cpu fp32|fp16|mixed [samples]")
            }
            let modelURL = URL(fileURLWithPath: args[1])
            let image = try decodeRGBA(URL(fileURLWithPath: args[2]))
            let reference = try EvidenceReference(url: URL(fileURLWithPath: args[3]))
            guard image.width == reference.width && image.height == reference.height else {
                throw ProbeError.invalid("Image and TFM1 reference dimensions differ")
            }
            let policyName = args[4]
            let precision = args[5].lowercased()
            guard ["fp32", "fp16", "mixed"].contains(precision) else {
                throw ProbeError.invalid("Precision must be fp32, fp16, or mixed")
            }
            let sampleCount = args.count == 7 ? (Int(args[6]) ?? 6) : 6
            guard sampleCount >= 2 else { throw ProbeError.invalid("Use at least two prediction samples") }

            let sampler = FootprintSampler()
            let configuration = MLModelConfiguration()
            configuration.computeUnits = try computeUnits(policyName)
            var modelCompileMilliseconds: Double = 0
            let compiledModelURL: URL
            if modelURL.pathExtension == "mlpackage" {
                let compileStart = DispatchTime.now().uptimeNanoseconds
                compiledModelURL = try await MLModel.compileModel(at: modelURL)
                modelCompileMilliseconds = Double(DispatchTime.now().uptimeNanoseconds - compileStart) / 1_000_000
            } else {
                compiledModelURL = modelURL
            }
            let loadStart = DispatchTime.now().uptimeNanoseconds
            let model = try MLModel(contentsOf: compiledModelURL, configuration: configuration)
            let loadMilliseconds = Double(DispatchTime.now().uptimeNanoseconds - loadStart) / 1_000_000
            let planStart = DispatchTime.now().uptimeNanoseconds
            let plan = try await MLComputePlan.load(contentsOf: compiledModelURL, configuration: configuration)
            let planMilliseconds = Double(DispatchTime.now().uptimeNanoseconds - planStart) / 1_000_000

            let inputs = model.modelDescription.inputDescriptionsByName
            let inputNames = inputs.keys.sorted()
            guard inputNames.count == 2,
                  let rgbName = inputNames.first(where: { inputs[$0]?.multiArrayConstraint?.shape.count == 4 && inputs[$0]?.multiArrayConstraint?.shape[1].intValue == 3 }),
                  let validName = inputNames.first(where: { $0 != rgbName && inputs[$0]?.multiArrayConstraint?.shape.count == 4 }) else {
                throw ProbeError.invalid("Compiled model does not expose the expected RGB and validity inputs")
            }
            guard let outputName = model.modelDescription.outputDescriptionsByName.keys.sorted().first else {
                throw ProbeError.invalid("Compiled model has no output")
            }
            let provider = try makeInput(image, rgbName: rgbName, validName: validName)

            var operations: [[String: Any]] = []
            var preferredCounts: [String: Int] = [:]
            var estimatedCost: [String: Double] = [:]
            if case .program(let program) = plan.modelStructure {
                for function in program.functions.values {
                    collectOperations(function.block, plan: plan, into: &operations,
                                      counts: &preferredCounts, cost: &estimatedCost)
                }
            }

            var milliseconds: [Double] = []
            var maxAbsoluteError: Double = 0
            var squaredError: Double = 0
            var squaredReference: Double = 0
            var compared = 0
            for sample in 0..<sampleCount {
                let start = DispatchTime.now().uptimeNanoseconds
                let result = try await model.prediction(from: provider)
                let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
                milliseconds.append(elapsed)
                guard let output = result.featureValue(for: outputName)?.multiArrayValue,
                      output.dataType == .float32, output.shape.count == 4,
                      output.shape[2].intValue == side, output.shape[3].intValue == side else {
                    throw ProbeError.invalid("Core ML returned an unexpected output shape or precision")
                }
                if sample == sampleCount - 1 {
                    let outputBase = output.dataPointer.assumingMemoryBound(to: Float.self)
                    let strides = output.strides.map(\.intValue)
                    for y in 0..<comparedSide {
                        for x in 0..<comparedSide {
                            let actual = Double(outputBase[(centerOffset + y) * strides[2] + (centerOffset + x) * strides[3]])
                            let expected = Double(reference.noiseprint(x: tileOriginX + x, y: tileOriginY + y))
                            let error = actual - expected
                            maxAbsoluteError = max(maxAbsoluteError, abs(error))
                            squaredError += error * error
                            squaredReference += expected * expected
                            compared += 1
                        }
                    }
                }
            }
            let warm = Array(milliseconds.dropFirst())
            let sortedWarm = warm.sorted()
            let p50 = sortedWarm[sortedWarm.count / 2]
            let p95 = sortedWarm[min(sortedWarm.count - 1, max(0, Int(ceil(Double(sortedWarm.count) * 0.95)) - 1))]
            let report: [String: Any] = [
                "backend": "Core ML",
                "chip": sysctlString("machdep.cpu.brand_string"),
                "macOS": ProcessInfo.processInfo.operatingSystemVersionString,
                "policy": policyName,
                "precision": precision,
                "modelLoadMilliseconds": loadMilliseconds,
                "modelCompileMilliseconds": modelCompileMilliseconds,
                "computePlanLoadMilliseconds": planMilliseconds,
                "firstPredictionMilliseconds": milliseconds[0],
                "warmPredictionMilliseconds": Array(warm),
                "warmP50Milliseconds": p50,
                "warmP95Milliseconds": p95,
                "predictionSampleCount": sampleCount,
                "inputShape": [1, 3, side, side],
                "physicalPeakBytes": sampler.peakBytes(),
                "referenceScore": reference.score,
                "comparedNoiseprintPixels": compared,
                "noiseprintMaxAbsoluteError": maxAbsoluteError,
                "noiseprintRelativeRMS": sqrt(squaredError / max(squaredReference, 1e-30)),
                "preferredOperationCounts": preferredCounts,
                "estimatedCostWeightByPreferredDevice": estimatedCost,
                "operations": operations,
                "placementEvidence": "Public MLComputePlan deviceUsage and estimatedCost; preferred device is a placement recommendation, not a runtime trace."
            ]
            let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            print(String(decoding: data, as: UTF8.self))
        } catch {
            fputs("Core ML probe: \(error)\n", stderr)
            exit(1)
        }
    }
}
