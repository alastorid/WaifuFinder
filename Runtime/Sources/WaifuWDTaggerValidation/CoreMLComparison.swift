import CoreML
import Foundation
import MLX
import WDTaggerMLX

/// Interleaves complete predictions to reduce drift between implementations.
/// Both models stay resident, so this mode cannot compare process footprint.
func compareCoreML(
  weights: URL, modelURL: URL, inputsURL: URL, rounds: Int,
  tile: WDMetal.GEMMTile, compactBias: Bool, probabilities: String?, includeOriginal: Bool = false,
  fusedQKV: Bool = false, currentReference: Bool = false,
  gemmOverrides: [WDNativeTagger.GEMMOperation: [WDMetal.GEMMTile]] = [:], attentionQueryRows: Int = 32,
  coremlUnits: String = "cpuAndGPU", batchSize: Int
) throws -> [String: Any] {
  let batch = batchSize
  let row = 3 * 448 * 448
  let inputs = try Data(contentsOf: inputsURL, options: .mappedIfSafe)
  guard batch > 0, inputs.count > 0, inputs.count % (row * 4) == 0,
    inputs.count / (row * 4) % batch == 0, rounds > 0
  else {
    throw WDNativeError.invalid("Input corpus must contain whole RGB Float32 frames in complete batches")
  }
  let frames = inputs.count / (row * 4)
  // Pin the released v3 control even if future production defaults change.
  let base = currentReference ? try WDNativeTagger(
    weights: weights, gemmTile: .baseline, simdNorm: false,
    compactBias: false, fusedQKV: true) : nil
  let native = try base?.variant(gemmTile: tile, fusedQKV: fusedQKV, compactingBias: compactBias,
      gemmOverrides: gemmOverrides, attentionQueryRows: attentionQueryRows)
    ?? WDNativeTagger(weights: weights, gemmTile: tile, compactBias: compactBias, fusedQKV: fusedQKV,
      gemmOverrides: gemmOverrides, attentionQueryRows: attentionQueryRows)
  let original = currentReference ? base : includeOriginal ? native.variant(gemmTile: .baseline, fusedQKV: false, gemmOverrides: [:], attentionQueryRows: 32) : nil
  let configuration = MLModelConfiguration()
  switch coremlUnits {
  case "all": configuration.computeUnits = .all
  case "cpuAndGPU": configuration.computeUnits = .cpuAndGPU
  case "cpuAndNeuralEngine": configuration.computeUnits = .cpuAndNeuralEngine
  default: throw WDNativeError.invalid("--coreml-units must be all, cpuAndGPU, or cpuAndNeuralEngine")
  }
  configuration.allowLowPrecisionAccumulationOnGPU = false
  let coreml = try MLModel(contentsOf: modelURL, configuration: configuration)
  guard let inputName = coreml.modelDescription.inputDescriptionsByName.keys.first,
    let outputName = coreml.modelDescription.outputDescriptionsByName.keys.first
  else { throw WDNativeError.invalid("Missing Core ML features") }
  guard let inputShape = coreml.modelDescription.inputDescriptionsByName[inputName]?
      .multiArrayConstraint?.shape.map(\.intValue),
    inputShape == [batch, 3, 448, 448]
  else {
    throw WDNativeError.invalid("Core ML model input shape does not match --batch-size \(batch)")
  }
  let array = try MLMultiArray(shape: [batch, 3, 448, 448].map(NSNumber.init(value:)), dataType: .float32)
  guard array.strides.map(\.intValue) == [row, 448 * 448, 448, 1] else {
    throw WDNativeError.invalid("Noncontiguous Core ML input")
  }
  let provider = try MLDictionaryFeatureProvider(dictionary: [
    inputName: MLFeatureValue(multiArray: array)
  ])
  var records: [[String: Any]] = []
  var nativeSweeps: [Double] = []
  var coremlSweeps: [Double] = []
  var nativeProbabilities = Data()
  var coremlProbabilities = Data()
  var coremlOutputStrides: [Int]?
  var originalProbabilities = Data()
  var originalSweeps: [Double] = []
  let thermalBefore = ProcessInfo.processInfo.thermalState.rawValue
  for round in -1..<rounds {
    var nativeTotal = 0.0
    var coremlTotal = 0.0
    var originalTotal = 0.0
    for offset in stride(from: 0, to: frames, by: batch) {
      if round == -1 && offset > 0 { break }
      try autoreleasepool {
        let x = inputs.withUnsafeBytes { bytes -> MLXArray in
          let slice = UnsafeRawBufferPointer(
            rebasing: bytes[(offset * row * 4)..<((offset + batch) * row * 4)])
          do {
            try array.withUnsafeMutableBytes { buffer, strides in
              guard strides == [row, 448 * 448, 448, 1],
                let destination = buffer.baseAddress, let source = slice.baseAddress
              else { throw WDNativeError.invalid("Unexpected Core ML input buffer layout") }
              destination.copyMemory(from: source, byteCount: slice.count)
            }
          } catch {
            preconditionFailure("Unable to populate Core ML input: \(error)")
          }
          if currentReference {
            let floats = slice.assumingMemoryBound(to: Float.self)
            let plane = 448 * 448
            var rgba = [UInt8](repeating: 255, count: batch * plane * 4)
            for b in 0..<batch { for pixel in 0..<plane { for c in 0..<3 {
              let value = floats[(b * 3 + c) * plane + pixel]
              precondition(value.isFinite && value >= 0 && value <= 255 && value.rounded() == value)
              rgba[(b * plane + pixel) * 4 + c] = UInt8(value)
            } } }
            return MLXArray(rgba, [batch, 448, 448, 4])
          }
          return MLXArray(slice, [batch, 3, 448, 448], type: Float.self)
        }
        eval(x)
        let nativeFirst = (max(round, 0) + offset / batch) % 2 == 0
        var y: MLXArray?
        var result: (any MLFeatureProvider)?
        var nativeTime = 0.0
        var coremlTime = 0.0
        var originalY: MLXArray?
        var originalTime = 0.0
        func runNative() {
          let start = ProcessInfo.processInfo.systemUptime
          let output = native.predict(x)
          eval(output)
          nativeTime = ProcessInfo.processInfo.systemUptime - start
          y = output
        }
        func runCoreML() throws {
          let start = ProcessInfo.processInfo.systemUptime
          result = try coreml.prediction(from: provider)
          guard let ready = result?.featureValue(for: outputName)?.multiArrayValue else {
            throw WDNativeError.invalid("Missing Core ML output")
          }
          _ = ready[0].floatValue
          coremlTime = ProcessInfo.processInfo.systemUptime - start
        }
        func runOriginal() {
          let start = ProcessInfo.processInfo.systemUptime
          let output = original!.predict(x)
          eval(output)
          originalTime = ProcessInfo.processInfo.systemUptime - start
          originalY = output
        }
        var order = nativeFirst ? [0, 2] : [2, 0]
        if original != nil {
          let permutations = [
            [0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0],
          ]
          order = permutations[(max(round, 0) * 8 + offset / batch) % permutations.count]
          for implementation in order {
            if implementation == 0 {
              runNative()
            } else if implementation == 1 {
              runOriginal()
            } else {
              try runCoreML()
            }
          }
        } else if nativeFirst {
          runNative()
          try runCoreML()
        } else {
          try runCoreML()
          runNative()
        }
        if round >= 0 {
          nativeTotal += nativeTime
          coremlTotal += coremlTime
          originalTotal += originalTime
          records.append([
            "round": round, "offset": offset, "nativeFirst": order.first == 0,
            "nativeSeconds": nativeTime, "coremlSeconds": coremlTime,
            "coremlOverNativeTime": coremlTime / nativeTime,
            "originalSeconds": originalTime, "executionOrder": order,
            "thermal": ProcessInfo.processInfo.thermalState.rawValue,
          ])
        }
        // Materialize only after both predictions, outside their timers.
        if round == 0 {
          guard let output = result?.featureValue(for: outputName)?.multiArrayValue,
            output.dataType == .float32,
            output.shape.map(\.intValue) == [batch, 10861], let y, y.shape == [batch, 10861]
          else { throw WDNativeError.invalid("Invalid paired classifier output") }
          y.asArray(Float.self).withUnsafeBytes { nativeProbabilities.append(contentsOf: $0) }
          if let originalY {
            originalY.asArray(Float.self).withUnsafeBytes {
              originalProbabilities.append(contentsOf: $0)
            }
          }
          let rowStride = output.strides[0].intValue
          let columnStride = output.strides[1].intValue
          coremlOutputStrides = [rowStride, columnStride]
          let values = try output.withUnsafeBytes { buffer -> [Float] in
            guard let baseAddress = buffer.baseAddress else {
              throw WDNativeError.invalid("Missing Core ML output bytes")
            }
            let base = baseAddress.assumingMemoryBound(to: Float.self)
            return (0..<batch).flatMap { row in
              (0..<10861).map { column in base[row * rowStride + column * columnStride] }
            }
          }
          values.withUnsafeBytes { coremlProbabilities.append(contentsOf: $0) }
        }
      }
    }
    if round >= 0 {
      nativeSweeps.append(nativeTotal)
      coremlSweeps.append(coremlTotal)
      if original != nil { originalSweeps.append(originalTotal) }
      print(
        "Paired round \(round): native \(Double(frames) / nativeTotal), Core ML \(Double(frames) / coremlTotal) fps"
      )
    }
  }
  if let probabilities {
    try nativeProbabilities.write(to: URL(fileURLWithPath: probabilities + "-native.bin"))
    try coremlProbabilities.write(to: URL(fileURLWithPath: probabilities + "-coreml.bin"))
    if original != nil {
      try originalProbabilities.write(to: URL(fileURLWithPath: probabilities + "-original.bin"))
    }
  }
  let medianNative = nativeSweeps.sorted()[rounds / 2]
  let medianCoreML = coremlSweeps.sorted()[rounds / 2]
  var report: [String: Any] = [
    "native": true, "batch": batch, "frames": frames, "rounds": rounds,
    "implementation": WDNativeTagger.implementationVersion,
    "optimizationProfile": try native.resolvedConfiguration(batch: batch).diagnostics,
    "attentionQueryRows": attentionQueryRows,
    "qkvBackend": !fusedQKV || (0..<4).contains { native.selectedTile(.qkv, stage: $0, batch: batch) == .library }
      ? "includes-unfused" : "native-fused",
    "stageTiles": Dictionary(uniqueKeysWithValues: gemmOverrides.map { ($0.key.rawValue, $0.value.map(\.rawValue)) }),
    "gemmTile": tile.rawValue, "compactAttentionBias": compactBias,
    "fusedQKV": fusedQKV,
    "reference": currentReference ? "current-rgba-fused-v3" : "unfused-float32-v1",
    "nativeInputLayout": currentReference ? "RGBA-uint8" : "RGB-planar-float32",
    "coremlOutputStrides": coremlOutputStrides ?? [],
    "nativePredictionFPS": Double(frames) / medianNative,
    "coremlPredictionFPS": Double(frames) / medianCoreML,
    "coremlComputeUnits": coremlUnits,
    "medianSweepSpeedRatio": medianCoreML / medianNative,
    "nativeSweepSeconds": nativeSweeps, "coremlSweepSeconds": coremlSweeps,
    "thermalBefore": thermalBefore, "thermalAfter": ProcessInfo.processInfo.thermalState.rawValue,
    "pairs": records,
    "executionOrderLegend": ["candidate-native", "original-native", "coreml"],
    "method":
      "Balanced adjacent full-model batch-\(batch) predictions; same inputs; Core ML completion included (first output scalar touched inside timer); input preparation and bulk extraction excluded. Original/candidate share immutable native weights. Three-way mode balances all six execution permutations.",
    "limitations":
      "Both models remain resident, so footprint differs from standalone runs. Shared hardware state and background work still vary. This is not a performance ceiling or sustained-video measurement.",
  ]
  if original != nil {
    let medianOriginal = originalSweeps.sorted()[rounds / 2]
    report["originalPredictionFPS"] = Double(frames) / medianOriginal
    report["originalGEMMTile"] = WDMetal.GEMMTile.baseline.rawValue
    report["originalFusedQKV"] = currentReference
    report["originalSweepSeconds"] = originalSweeps
    report["nativeOverOriginalSpeedRatio"] = medianOriginal / medianNative
    report["nativeOriginalProbabilitiesBitwiseEqual"] = nativeProbabilities == originalProbabilities
  }
  return report
}
