import Foundation
import CoreML
import MLX

/// Native Swift execution of the complete converted original PGTFormer graph.
/// No interpreter, detection, alternate model or app-backend registration.
func validateCoreMLPGT(package: URL, fixture: URL, report: URL) throws {
    let arrays = try loadArrays(url: fixture)
    guard let input = arrays["frames"], input.shape == [3,3,512,512],
          let reference = arrays["restored"], reference.shape == [1,3,512,512] else {
        throw NSError(domain: "PGTValidation", code: 1)
    }
    let start = Date()
    let compiled = try MLModel.compileModel(at: package)
    defer { try? FileManager.default.removeItem(at: compiled) }
    let compileSeconds = Date().timeIntervalSince(start)
    let tensor = try MLMultiArray(shape: [3,3,512,512], dataType: .float32)
    let values = input.asArray(Float.self)
    values.withUnsafeBufferPointer { source in
        tensor.dataPointer.assumingMemoryBound(to: Float.self).update(from: source.baseAddress!, count: values.count)
    }
    let features = try MLDictionaryFeatureProvider(dictionary: ["frames": tensor])
    var measurements: [[String: Any]] = []
    for (label, units) in [("cpuOnly", MLComputeUnits.cpuOnly), ("cpuAndGPU", .cpuAndGPU)] {
        let configuration = MLModelConfiguration(); configuration.computeUnits = units
        let loadStart = Date(), model = try MLModel(contentsOf: compiled, configuration: configuration)
        let loadSeconds = Date().timeIntervalSince(loadStart)
        var durations: [Double] = []
        var maximum: Float = 0, average: Float = 0
        for _ in 0..<3 {
            let runStart = Date()
            let prediction = try model.prediction(from: features)
            durations.append(Date().timeIntervalSince(runStart))
            guard let restored = prediction.featureValue(for: "restored")?.multiArrayValue,
                  restored.shape.map(\.intValue) == [1,3,512,512] else { throw NSError(domain: "PGTValidation", code: 2) }
            // Read through indexed strides; don't assume Core ML outputs are contiguous.
            var output = [Float](); output.reserveCapacity(restored.count)
            for channel in 0..<3 { for y in 0..<512 { for x in 0..<512 {
                output.append(restored[[0,NSNumber(value: channel),NSNumber(value:y),NSNumber(value:x)]].floatValue)
            } } }
            let difference = abs(MLXArray(output).reshaped([1,3,512,512]) - reference)
            maximum = max(difference).item(Float.self); average = mean(difference).item(Float.self)
            guard maximum.isFinite, average.isFinite, maximum <= 0.02, average <= 0.002 else {
                throw NSError(domain: "PGTValidation", code: 3, userInfo: [NSLocalizedDescriptionKey: "PGT pretrained pixel parity failed"])
            }
        }
        measurements.append(["computeUnits":label,"loadSeconds":loadSeconds,"inferenceSeconds":durations,
            "maximumAbsoluteError":maximum,"meanAbsoluteError":average,"passed":true])
    }
    let result: [String: Any] = ["scope":"Complete pretrained PGTFormer graph, one three-frame reference fixture",
        "backend":"Native Swift Core ML", "hardware":"Apple M3 / 16 GB", "compileSeconds":compileSeconds,
        "applicationIntegrated":false,"measurements":measurements]
    try JSONSerialization.data(withJSONObject: result, options:[.prettyPrinted,.sortedKeys]).write(to:report, options:.atomic)
    print("Complete pretrained PGTFormer native Swift parity: PASS")
}
