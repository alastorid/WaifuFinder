import CoreML
import Foundation

private enum CoreMLActivationProbeError: Error {
  case invalid(String)
}

private struct ActivationDiff {
  var count = 0
  var maximum: Float = 0
  var maximumIndex = 0
  var absoluteSum = 0.0
  var squareSum = 0.0

  mutating func append(_ lhs: Float, _ rhs: Float, index: Int) {
    let difference = abs(lhs - rhs)
    if difference > maximum {
      maximum = difference
      maximumIndex = index
    }
    absoluteSum += Double(difference)
    squareSum += Double(difference) * Double(difference)
    count += 1
  }

  var json: [String: Any] {
    [
      "count": count,
      "maximumAbsoluteError": maximum,
      "maximumLinearIndex": maximumIndex,
      "meanAbsoluteError": count == 0 ? 0 : absoluteSum / Double(count),
      "rootMeanSquareError": count == 0 ? 0 : sqrt(squareSum / Double(count)),
    ]
  }
}

private let wdCoreMLActivationTapNames = [
  "scaled_dot_product_attention_cast_fp16",
  "layer_norm_1_cast_fp16",
  "add_1_cast_fp16",
  "add_4_cast_fp16",
  "add_9_cast_fp16",
  "add_54_cast_fp16",
  "add_58_cast_fp16",
]

private func readValues(_ array: MLMultiArray) throws -> [Float] {
  let shape = array.shape.map(\.intValue)
  let strides = array.strides.map(\.intValue)
  guard !shape.isEmpty, shape.count == strides.count, array.count > 0 else {
    throw CoreMLActivationProbeError.invalid("Invalid activation array shape/strides")
  }
  return try array.withUnsafeBytes { bytes in
    guard let baseAddress = bytes.baseAddress else {
      throw CoreMLActivationProbeError.invalid("Missing activation bytes")
    }
    var values = Array(repeating: Float.zero, count: array.count)
    var coordinates = Array(repeating: 0, count: shape.count)
    var offset = 0
    switch array.dataType {
    case .float16:
      let pointer = baseAddress.assumingMemoryBound(to: Float16.self)
      for index in values.indices {
        values[index] = Float(pointer[offset])
        for axis in stride(from: shape.count - 1, through: 0, by: -1) {
          if coordinates[axis] + 1 < shape[axis] {
            coordinates[axis] += 1
            offset += strides[axis]
            break
          }
          coordinates[axis] = 0
          offset -= (shape[axis] - 1) * strides[axis]
        }
      }
    case .float32:
      let pointer = baseAddress.assumingMemoryBound(to: Float.self)
      for index in values.indices {
        values[index] = pointer[offset]
        for axis in stride(from: shape.count - 1, through: 0, by: -1) {
          if coordinates[axis] + 1 < shape[axis] {
            coordinates[axis] += 1
            offset += strides[axis]
            break
          }
          coordinates[axis] = 0
          offset -= (shape[axis] - 1) * strides[axis]
        }
      }
    default:
      throw CoreMLActivationProbeError.invalid("Unsupported activation dtype: \(array.dataType)")
    }
    return values
  }
}

private func compare(_ lhs: MLMultiArray, _ rhs: MLMultiArray) throws -> (ActivationDiff, [Float], [Float]) {
  guard lhs.shape.map(\.intValue) == rhs.shape.map(\.intValue) else {
    throw CoreMLActivationProbeError.invalid("Activation shapes differ")
  }
  let a = try readValues(lhs)
  let b = try readValues(rhs)
  var difference = ActivationDiff()
  for index in a.indices { difference.append(a[index], b[index], index: index) }
  return (difference, a, b)
}

private func compareReference(_ actual: [Float], reference: [Float]) throws -> ActivationDiff {
  guard actual.count == reference.count else {
    throw CoreMLActivationProbeError.invalid("Saved Core ML output reference has the wrong length")
  }
  var difference = ActivationDiff()
  for index in actual.indices { difference.append(actual[index], reference[index], index: index) }
  return difference
}

private func tagAgreement(_ gpu: [Float], _ ane: [Float]) throws -> [String: Any] {
  let heads = 10_861
  guard gpu.count == ane.count, gpu.count % heads == 0 else {
    throw CoreMLActivationProbeError.invalid("Final classifier outputs must be [batch,10861]")
  }
  let batch = gpu.count / heads
  var thresholdChanges = 0
  var ratingArgmaxChanges = 0
  var minimumTop15Overlap = 15
  for row in 0..<batch {
    let start = row * heads
    for head in 4..<heads {
      if (gpu[start + head] > 0.35) != (ane[start + head] > 0.35) {
        thresholdChanges += 1
      }
    }
    let gpuRating = (0..<4).max { gpu[start + $0] < gpu[start + $1] } ?? 0
    let aneRating = (0..<4).max { ane[start + $0] < ane[start + $1] } ?? 0
    if gpuRating != aneRating { ratingArgmaxChanges += 1 }
    let gpuTop = (0..<heads).sorted { gpu[start + $0] > gpu[start + $1] }.prefix(15)
    let aneTop = (0..<heads).sorted { ane[start + $0] > ane[start + $1] }.prefix(15)
    let overlap = Set(gpuTop).intersection(Set(aneTop)).count
    minimumTop15Overlap = min(minimumTop15Overlap, overlap)
  }
  return [
    "nonRatingTagThresholdChangesAt035": thresholdChanges,
    "ratingArgmaxChangedFrames": ratingArgmaxChanges,
    "minimumTop15Overlap": minimumTop15Overlap,
  ]
}

func compareCoreMLActivations(
  modelURL: URL, inputsURL: URL, batches: Int, reportURL: URL,
  cpuGPUReferenceURL: URL?, cpuANEReferenceURL: URL?
) throws -> [String: Any] {
  let batch = 8
  let frameValues = 3 * 448 * 448
  let inputFrameBytes = frameValues * MemoryLayout<Float>.size
  let source = try Data(contentsOf: inputsURL, options: .mappedIfSafe)
  guard (1...8).contains(batches), source.count >= batches * batch * inputFrameBytes else {
    throw CoreMLActivationProbeError.invalid("Need 1...8 batch-8 groups of RGB Float32 input")
  }

  let loadStart = ProcessInfo.processInfo.systemUptime
  let gpuConfiguration = MLModelConfiguration()
  gpuConfiguration.computeUnits = .cpuAndGPU
  let gpuModel = try MLModel(contentsOf: modelURL, configuration: gpuConfiguration)
  let gpuLoadSeconds = ProcessInfo.processInfo.systemUptime - loadStart
  let aneLoadStart = ProcessInfo.processInfo.systemUptime
  let aneConfiguration = MLModelConfiguration()
  aneConfiguration.computeUnits = .cpuAndNeuralEngine
  let aneModel = try MLModel(contentsOf: modelURL, configuration: aneConfiguration)
  let aneLoadSeconds = ProcessInfo.processInfo.systemUptime - aneLoadStart

  guard let inputName = gpuModel.modelDescription.inputDescriptionsByName.keys.first,
    Set(gpuModel.modelDescription.outputDescriptionsByName.keys)
      == Set(aneModel.modelDescription.outputDescriptionsByName.keys)
  else { throw CoreMLActivationProbeError.invalid("Tapped model input/output descriptions differ") }
  let outputNames = ["output"] + wdCoreMLActivationTapNames
  guard outputNames.allSatisfy({ gpuModel.modelDescription.outputDescriptionsByName[$0] != nil }) else {
    throw CoreMLActivationProbeError.invalid("Tapped Core ML model is missing one or more activation outputs")
  }

  let gpuArray = try MLMultiArray(shape: [8, 3, 448, 448], dataType: .float32)
  let aneArray = try MLMultiArray(shape: [8, 3, 448, 448], dataType: .float32)
  let expectedStrides = [frameValues, 448 * 448, 448, 1]
  guard gpuArray.strides.map(\.intValue) == expectedStrides,
    aneArray.strides.map(\.intValue) == expectedStrides
  else { throw CoreMLActivationProbeError.invalid("Tapped model input arrays are not contiguous RGB tensors") }
  let gpuProvider = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(multiArray: gpuArray)])
  let aneProvider = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(multiArray: aneArray)])

  var perTap: [String: [String: Any]] = [:]
  var finalGPU: [Float] = []
  var finalANE: [Float] = []
  var finalReferenceMetrics: [String: Any] = [:]
  for index in 0..<batches {
    let inputOffset = index * batch * inputFrameBytes
    let inputSlice = source.subdata(in: inputOffset..<(inputOffset + batch * inputFrameBytes))
    for array in [gpuArray, aneArray] {
      try array.withUnsafeMutableBytes { buffer, strides in
        guard strides == expectedStrides, let destination = buffer.baseAddress else {
          throw CoreMLActivationProbeError.invalid("Core ML changed the input layout while writing")
        }
        inputSlice.withUnsafeBytes { bytes in
          destination.copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
        }
      }
    }

    let gpuResult = try gpuModel.prediction(from: gpuProvider)
    let aneResult = try aneModel.prediction(from: aneProvider)
    for name in outputNames {
      guard let gpuOutput = gpuResult.featureValue(for: name)?.multiArrayValue,
        let aneOutput = aneResult.featureValue(for: name)?.multiArrayValue
      else { throw CoreMLActivationProbeError.invalid("Missing tapped output \(name)") }
      let (difference, gpuValues, aneValues) = try compare(gpuOutput, aneOutput)
      var entry = perTap[name] ?? ["name": name, "shape": gpuOutput.shape.map(\.intValue),
        "stridesCPUAndGPU": gpuOutput.strides.map(\.intValue),
        "stridesCPUAndNeuralEngine": aneOutput.strides.map(\.intValue),
        "dtypeCPUAndGPU": String(describing: gpuOutput.dataType),
        "dtypeCPUAndNeuralEngine": String(describing: aneOutput.dataType)]
      var combined = (entry["difference"] as? [String: Any]) ?? ActivationDiff().json
      let oldCount = combined["count"] as? Int ?? 0
      let oldMax = combined["maximumAbsoluteError"] as? Float ?? 0
      let new = difference.json
      let newCount = new["count"] as? Int ?? 0
      let newMax = new["maximumAbsoluteError"] as? Float ?? 0
      let newMaxIndex = new["maximumLinearIndex"] as? Int ?? 0
      let oldSum = (combined["meanAbsoluteError"] as? Double ?? 0) * Double(oldCount)
      let oldRMS = (combined["rootMeanSquareError"] as? Double ?? 0)
      let newSum = (new["meanAbsoluteError"] as? Double ?? 0) * Double(newCount)
      let newSquareSum = (new["rootMeanSquareError"] as? Double ?? 0)
      combined["count"] = oldCount + newCount
      if newMax > oldMax {
        combined["maximumAbsoluteError"] = newMax
        combined["maximumLinearIndex"] = newMaxIndex
        combined["maximumBatch"] = index
      }
      combined["meanAbsoluteError"] = (oldSum + newSum) / Double(oldCount + newCount)
      combined["rootMeanSquareError"] = sqrt(
        (oldRMS * oldRMS * Double(oldCount) + newSquareSum * newSquareSum * Double(newCount))
          / Double(oldCount + newCount))
      entry["difference"] = combined
      perTap[name] = entry

      if name == "output" {
        finalGPU += gpuValues
        finalANE += aneValues
      }
    }
  }

  if let url = cpuGPUReferenceURL {
    let data = try Data(contentsOf: url, options: .mappedIfSafe)
    let count = batch * 10_861
    guard data.count >= count * MemoryLayout<Float>.size else {
      throw CoreMLActivationProbeError.invalid("Saved CPU+GPU probability reference is incomplete")
    }
    let reference = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self).prefix(count)) }
    finalReferenceMetrics["cpuAndGPU"] = try compareReference(
      Array(finalGPU.prefix(count)), reference: reference).json
  }
  if let url = cpuANEReferenceURL {
    let data = try Data(contentsOf: url, options: .mappedIfSafe)
    let count = batch * 10_861
    guard data.count >= count * MemoryLayout<Float>.size else {
      throw CoreMLActivationProbeError.invalid("Saved CPU+ANE probability reference is incomplete")
    }
    let reference = data.withUnsafeBytes { bytes in
      Array(bytes.bindMemory(to: Float.self).prefix(count))
    }
    finalReferenceMetrics["cpuAndNeuralEngine"] = try compareReference(
      Array(finalANE.prefix(count)), reference: reference).json
  }

  return [
    "device": "Mac \(ProcessInfo.processInfo.operatingSystemVersionString)",
    "batchSize": batch, "batches": batches, "frames": batch * batches,
    "computeUnits": ["cpuAndGPU", "cpuAndNeuralEngine"],
    "modelLoadSeconds": ["cpuAndGPU": gpuLoadSeconds, "cpuAndNeuralEngine": aneLoadSeconds],
    "outputOrder": outputNames,
    "activationComparisons": perTap,
    "tagAgreementOnTappedModel": try tagAgreement(finalGPU, finalANE),
    "tappedModelFinalOutputVsPreviousModel": finalReferenceMetrics,
  ]
}
