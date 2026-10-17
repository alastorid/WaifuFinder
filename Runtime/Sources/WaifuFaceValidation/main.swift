import Foundation
import MLX
import FaceRestorationMLX

// Explicit developer validation executable. Never registered as an app backend.
do {
    if CommandLine.arguments.count == 5, CommandLine.arguments[1] == "--coreml-pgt" {
        try validateCoreMLPGT(package: URL(fileURLWithPath: CommandLine.arguments[2]),
            fixture: URL(fileURLWithPath: CommandLine.arguments[3]),
            report: URL(fileURLWithPath: CommandLine.arguments[4]))
        exit(0)
    }
    guard CommandLine.arguments.count == 3 else {
        throw NSError(domain: "WaifuFaceValidation", code: 1,
            userInfo: [NSLocalizedDescriptionKey: "Usage: waifu-face-validation FIXTURE_DIRECTORY REPORT_JSON"])
    }
    let root = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
    let fixtures = try loadArrays(url: root.appendingPathComponent("fixtures.safetensors"))
    let weights = try loadArrays(url: root.appendingPathComponent("pgt-layer.safetensors"))
    let layer = try PGTCodeFormerLayer(weights: weights)
    var results: [[String: Any]] = []
    func check(_ name: String, _ actual: MLXArray, _ expected: MLXArray, maxError: Float, meanError: Float) throws {
        guard actual.shape == expected.shape else { throw FaceReferenceError.incompatibleInput }
        let difference = abs(actual.asType(.float32) - expected.asType(.float32))
        let maximum = max(difference).item(Float.self), average = mean(difference).item(Float.self)
        let passed = maximum.isFinite && average.isFinite && maximum <= maxError && average <= meanError
        results.append(["component": name, "maximumAbsoluteError": maximum, "meanAbsoluteError": average,
                        "maximumErrorLimit": maxError, "meanErrorLimit": meanError, "passed": passed])
    }
    for tokens in [37,3072] {
        let prefix = "attention.\(tokens)."
        try check(prefix, layer(fixtures[prefix+"input"]!, positions: fixtures[prefix+"positions"]!),
                  fixtures[prefix+"output"]!, maxError: 1e-4, meanError: 1e-5)
    }
    try check("adain", pgtAdaptiveInstanceNormalize(fixtures["adain.content"]!, style: fixtures["adain.style"]!),
              fixtures["adain.output"]!, maxError: 1e-5, meanError: 1e-6)
    // Strict loader must reject missing, extraneous and incompatible parameters.
    var invalid = weights; invalid.removeValue(forKey: "norm1.weight")
    do { _ = try PGTCodeFormerLayer(weights: invalid); throw FaceReferenceError.incompatibleInput }
    catch FaceReferenceError.incompatibleWeights { }
    invalid = weights; invalid["unexpected"] = MLXArray.zeros([1])
    do { _ = try PGTCodeFormerLayer(weights: invalid); throw FaceReferenceError.incompatibleInput }
    catch FaceReferenceError.incompatibleWeights { }
    invalid = weights; invalid["norm1.weight"] = MLXArray.zeros([511])
    do { _ = try PGTCodeFormerLayer(weights: invalid); throw FaceReferenceError.incompatibleInput }
    catch FaceReferenceError.incompatibleWeights { }
    let passed = results.allSatisfy { $0["passed"] as? Bool == true }
    let report: [String: Any] = ["scope": "Original-weight components only", "completeModelsValidated": false,
        "backend": "MLX Swift / Metal", "passed": passed, "checks": results]
    try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted,.sortedKeys])
        .write(to: URL(fileURLWithPath: CommandLine.arguments[2]), options: .atomic)
    print("Face component reference parity: \(passed ? "PASS" : "FAIL") (\(results.count) checks)")
    if !passed { exit(1) }
} catch {
    fputs("Face validation failed: \(error)\n", stderr); exit(1)
}
