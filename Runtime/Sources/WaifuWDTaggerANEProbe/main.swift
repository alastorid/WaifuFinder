import Darwin
import Foundation
import MLX
import WDTaggerANE
import WDTaggerMLX

typealias ANEProgram = OpaquePointer
typealias ANETensor = OpaquePointer

private func option(_ name: String, default value: String) -> String {
  guard let index = CommandLine.arguments.firstIndex(of: name), index + 1 < CommandLine.arguments.count
  else { return value }
  return CommandLine.arguments[index + 1]
}

private func fail(_ message: String, _ error: String? = nil) -> Never {
  let detail = error.map { "\n\($0)" } ?? ""
  FileHandle.standardError.write(Data("WD Tagger ANE probe: \(message)\(detail)\n".utf8))
  exit(1)
}

private func makeSurface(elements: Int, allocationBytes: Int) -> ANETensor {
  var error = [CChar](repeating: 0, count: 1024)
  let surface = error.withUnsafeMutableBufferPointer { buffer in
    WDANETensorCreateF16(elements, allocationBytes, buffer.baseAddress, buffer.count)
  }
  guard let surface else {
    fail("IOSurface allocation failed", String(cString: error))
  }
  return surface
}

private func write(_ values: [UInt16], to tensor: ANETensor) {
  var error = [CChar](repeating: 0, count: 1024)
  let ok = values.withUnsafeBufferPointer { buffer in
    error.withUnsafeMutableBufferPointer { errorBuffer in
      WDANETensorWriteF16(tensor, buffer.baseAddress!, buffer.count,
                          errorBuffer.baseAddress, errorBuffer.count)
    }
  }
  if !ok { fail("IOSurface write failed", String(cString: error)) }
}

private func evaluate(_ program: ANEProgram, inputs: [ANETensor], output: ANETensor) {
  var error = [CChar](repeating: 0, count: 1024)
  let inputOK = inputs.withUnsafeBufferPointer { inputBuffer in
    let outputs = [output]
    return outputs.withUnsafeBufferPointer { outputBuffer in
      error.withUnsafeMutableBufferPointer { errorBuffer in
        WDANEEvaluate(program, inputBuffer.baseAddress!, inputBuffer.count,
                      outputBuffer.baseAddress!, outputBuffer.count,
                      errorBuffer.baseAddress, errorBuffer.count)
      }
    }
  }
  if !inputOK { fail("private ANE evaluation failed", String(cString: error)) }
}

private func machineIdentifier() -> String {
  var size = 0
  guard sysctlbyname("hw.model", nil, &size, nil, 0) == 0, size > 1 else { return "unknown" }
  var bytes = [CChar](repeating: 0, count: size)
  guard sysctlbyname("hw.model", &bytes, &size, nil, 0) == 0 else { return "unknown" }
  return String(cString: bytes)
}

private func makeWeightBlob(_ values: [UInt16]) -> Data {
  let byteCount = UInt32(values.count * MemoryLayout<UInt16>.size).littleEndian
  let dataOffset = UInt32(128).littleEndian
  var header = [UInt8](repeating: 0, count: 128)
  header[0] = 1
  header[4] = 2
  header[64] = 0xEF; header[65] = 0xBE; header[66] = 0xAD; header[67] = 0xDE
  header[68] = 1
  withUnsafeBytes(of: byteCount) { header.replaceSubrange(72..<76, with: $0) }
  withUnsafeBytes(of: dataOffset) { header.replaceSubrange(80..<84, with: $0) }
  var blob = Data(header)
  values.withUnsafeBytes { blob.append(contentsOf: $0) }
  return blob
}

private func report(_ value: [String: Any]) {
  guard let data = try? JSONSerialization.data(withJSONObject: value,
                                               options: [.prettyPrinted, .sortedKeys]) else {
    fail("could not encode probe report")
  }
  if let reportPath = CommandLine.arguments.firstIndex(of: "--report")
    .flatMap({ $0 + 1 < CommandLine.arguments.count ? CommandLine.arguments[$0 + 1] : nil }) {
    do {
      try FileManager.default.createDirectory(
        at: URL(fileURLWithPath: reportPath).deletingLastPathComponent(),
        withIntermediateDirectories: true)
      try data.write(to: URL(fileURLWithPath: reportPath))
    } catch {
      fail("could not save probe report to \(reportPath): \(error)")
    }
  }
  FileHandle.standardOutput.write(data)
  FileHandle.standardOutput.write(Data("\n".utf8))
}

private func runAttentionProbe() throws {
  guard WDANEIsAvailable() else { fail("Apple's private ANE runtime classes are unavailable") }
  let stage = Int(option("--stage", default: "1")) ?? 0
  let block = Int(option("--block", default: "1")) ?? 0
  let iterations = Int(option("--iterations", default: "10")) ?? 0
  let selectedWindow = Int(option("--window", default: "0")) ?? 0
  let shifted = CommandLine.arguments.contains("--shifted")
  guard (1...4).contains(stage), (0..<([2, 2, 18, 2][stage - 1])).contains(block),
        (1...100).contains(iterations) else {
    fail("attention probe requires --stage 1...4, a valid --block and --iterations 1...100")
  }
  guard shifted == (block % 2 == 1 && stage != 4) else {
    fail("--shifted must match the selected SwinV2 block (odd blocks in stages 1-3 only)")
  }

  let channels = [128, 256, 512, 1024][stage - 1]
  let heads = channels / 32
  let side = [112, 56, 28, 14][stage - 1]
  let windows = (side / 14) * (side / 14)
  guard (0..<windows).contains(selectedWindow) else { fail("--window is outside this stage's Swin window grid") }
  let prefix = "layers.\(stage - 1).blocks.\(block).attn"
  let weightsURL = URL(fileURLWithPath: option(
    "--weights", default: "models/published/WD-SwinV2-Tagger-v3-Swift-MLX/model.safetensors"))
  let tensors = try loadArrays(url: weightsURL, stream: .cpu)
  guard let relativeBias = tensors[prefix + "._sdpa_bias"],
        relativeBias.size == heads * 196 * 196,
        let queryScale = tensors[prefix + "._sdpa_query_scale"], queryScale.size == heads else {
    fail("prepared checkpoint lacks the SwinV2 attention bias or cosine scale for \(prefix)")
  }

  let generated = (0..<(196 * 3 * channels)).map {
    Float(sin(Double(($0 * 13 + 7) % 65521))) * 0.25
  }
  let packed = MLXArray(generated, [196, 3 * channels]).asType(.float16)
  let qkv = WDMetal.qkv(packed, scale: queryScale.reshaped(-1), heads: heads)
  eval(qkv)
  let qValues = qkv[0].asArray(Float16.self).map(\.bitPattern)
  let kValues = qkv[1].asArray(Float16.self).map(\.bitPattern)
  let vValues = qkv[2].asArray(Float16.self).map(\.bitPattern)
  let biasValues = relativeBias.asArray(Float16.self)
  let regionValues = WDNativeTagger.regions(side: side).asArray(Int32.self)
  let biasElements = heads * 196 * 196
  let vectorElements = heads * 196 * 32
  let packedWidth = 32 * 3
  let packedElements = heads * 196 * packedWidth
  let allocationBytes = packedElements * MemoryLayout<UInt16>.size
  let packedInput = makeSurface(elements: packedElements, allocationBytes: allocationBytes)
  let out = makeSurface(elements: vectorElements, allocationBytes: allocationBytes)
  defer {
    WDANETensorRelease(packedInput); WDANETensorRelease(out)
  }

  func combinedBias(_ window: Int) -> [UInt16] {
    var values = [UInt16](repeating: 0, count: biasElements)
    for head in 0..<heads {
      for query in 0..<196 {
        let queryRegion = regionValues[window * 196 + query]
        for key in 0..<196 {
          let index = (head * 196 + query) * 196 + key
          let regionMask: Float = shifted && queryRegion != regionValues[window * 196 + key] ? -100 : 0
          values[index] = Float16(Float(biasValues[index]) + regionMask).bitPattern
        }
      }
    }
    return values
  }
  let biasBlob = makeWeightBlob(combinedBias(selectedWindow))
  let mil = """
  program(1.3)
  [buildInfo = dict<string, string>({
    {"coremlc-component-MIL", "3510.2.1"},
    {"coremlc-version", "3505.4.1"},
    {"coremltools-component-milinternal", ""},
    {"coremltools-version", "9.0"}
  })]
  {
    func main<ios18>(tensor<fp16, [1,\(heads),196,\(packedWidth)]> x) {
      tensor<int32, [4]> q_begin = const()[name=string("q_begin"), val=tensor<int32, [4]>([0,0,0,0])];
      tensor<int32, [4]> k_begin = const()[name=string("k_begin"), val=tensor<int32, [4]>([0,0,0,32])];
      tensor<int32, [4]> v_begin = const()[name=string("v_begin"), val=tensor<int32, [4]>([0,0,0,64])];
      tensor<int32, [4]> qkv_size = const()[name=string("qkv_size"), val=tensor<int32, [4]>([1,\(heads),196,32])];
      tensor<fp16, [1,\(heads),196,32]> q = slice_by_size(x=x, begin=q_begin, size=qkv_size)[name=string("q")];
      tensor<fp16, [1,\(heads),196,32]> k = slice_by_size(x=x, begin=k_begin, size=qkv_size)[name=string("k")];
      tensor<fp16, [1,\(heads),196,32]> v = slice_by_size(x=x, begin=v_begin, size=qkv_size)[name=string("v")];
      tensor<fp16, [1,\(heads),196,196]> bias = const()[name=string("relative_bias_and_shift_mask"), val=tensor<fp16, [1,\(heads),196,196]>(BLOBFILE(path=string("@model_path/weights/bias.bin"), offset=uint64(64)))];
      bool tx = const()[name=string("tx"), val=bool(false)];
      bool ty = const()[name=string("ty"), val=bool(true)];
      bool noTranspose = const()[name=string("no_transpose"), val=bool(false)];
      fp16 scale = const()[name=string("scale"), val=fp16(0.176776695)];
      int32 sax = const()[name=string("sax"), val=int32(-1)];
      tensor<fp16, [1,\(heads),196,196]> raw = matmul(
        transpose_x=tx, transpose_y=ty, x=q, y=k)[name=string("qk")];
      tensor<fp16, [1,\(heads),196,196]> scaled = mul(
        x=raw, y=scale)[name=string("cosine_scale")];
      tensor<fp16, [1,\(heads),196,196]> masked = add(
        x=scaled, y=bias)[name=string("relative_bias_and_shift_mask")];
      tensor<fp16, [1,\(heads),196,196]> probabilities = softmax(
        axis=sax, x=masked)[name=string("masked_softmax_on_ane")];
      tensor<fp16, [1,\(heads),196,32]> attended = matmul(
        transpose_x=noTranspose, transpose_y=noTranspose,
        x=probabilities, y=v)[name=string("attention_times_value")];
    } -> (attended);
  }
  """
  var compileError = [CChar](repeating: 0, count: 1024)
  let compileStart = ProcessInfo.processInfo.systemUptime
  let maybeProgram = mil.withCString { source in
    "@model_path/weights/bias.bin".withCString { path in
      biasBlob.withUnsafeBytes { blobBytes in
        compileError.withUnsafeMutableBufferPointer { errorBuffer in
          WDANECompileMILWithOneWeightBlob(
            source, path, blobBytes.baseAddress!.assumingMemoryBound(to: UInt8.self), biasBlob.count,
            errorBuffer.baseAddress, errorBuffer.count)
        }
      }
    }
  }
  guard let program = maybeProgram else {
    fail("explicit masked-attention MIL compilation/load failed", String(cString: compileError))
  }
  defer { WDANEReleaseProgram(program) }
  let compileSeconds = ProcessInfo.processInfo.systemUptime - compileStart
  let cacheHitsBeforeRepeat = WDANEProgramCacheHitCount()
  let cacheLookupStart = ProcessInfo.processInfo.systemUptime
  let cachedProgram = mil.withCString { source in
    "@model_path/weights/bias.bin".withCString { path in
      biasBlob.withUnsafeBytes { blobBytes in
        compileError.withUnsafeMutableBufferPointer { errorBuffer in
          WDANECompileMILWithOneWeightBlob(
            source, path, blobBytes.baseAddress!.assumingMemoryBound(to: UInt8.self), biasBlob.count,
            errorBuffer.baseAddress, errorBuffer.count)
        }
      }
    }
  }
  let identicalProgramCacheLookupSeconds = ProcessInfo.processInfo.systemUptime - cacheLookupStart
  guard let cachedProgram else {
    fail("identical MIL cache lookup failed", String(cString: compileError))
  }
  WDANEReleaseProgram(cachedProgram)
  let identicalProgramCacheHit = WDANEProgramCacheHitCount() == cacheHitsBeforeRepeat + 1
  var maximumOutputError: Float = 0
  var meanOutputError = 0.0
  var timedANE: [Double] = []
  var timedMetal: [Double] = []
  var combinedInput = [UInt16](repeating: 0, count: packedElements)
  for head in 0..<heads {
    for query in 0..<196 {
      let packedRow = (head * 196 + query) * packedWidth
      let vectorRow = (head * 196 + query) * 32
      combinedInput.replaceSubrange(packedRow..<(packedRow + 32), with: qValues[vectorRow..<(vectorRow + 32)])
      combinedInput.replaceSubrange((packedRow + 32)..<(packedRow + 64), with: kValues[vectorRow..<(vectorRow + 32)])
      combinedInput.replaceSubrange((packedRow + 64)..<(packedRow + 96), with: vValues[vectorRow..<(vectorRow + 32)])
    }
  }
  write(combinedInput, to: packedInput)
  let outputTensors = [out]
  func evaluateANE() {
    var error = [CChar](repeating: 0, count: 1024)
    let ok = [packedInput].withUnsafeBufferPointer { inputBuffer in
      outputTensors.withUnsafeBufferPointer { outputBuffer in
        error.withUnsafeMutableBufferPointer { errorBuffer in
          WDANEEvaluate(program, inputBuffer.baseAddress!, inputBuffer.count,
                        outputBuffer.baseAddress!, outputBuffer.count,
                        errorBuffer.baseAddress, errorBuffer.count)
        }
      }
    }
    if !ok { fail("direct ANE masked-softmax evaluation failed", String(cString: error)) }
  }
  func readSurface(_ surface: ANETensor, count: Int) -> [UInt16] {
    var values = [UInt16](repeating: 0, count: count)
    var error = [CChar](repeating: 0, count: 1024)
    let ok = values.withUnsafeMutableBufferPointer { outputBuffer in
      error.withUnsafeMutableBufferPointer { errorBuffer in
        WDANETensorReadF16(surface, outputBuffer.baseAddress!, outputBuffer.count,
                           errorBuffer.baseAddress, errorBuffer.count)
      }
    }
    if !ok { fail("cannot read direct ANE attention output", String(cString: error)) }
    return values
  }
  let regionWindow = MLXArray(Array(regionValues[(selectedWindow * 196)..<((selectedWindow + 1) * 196)]))
  evaluateANE()
  let aneOutput = readSurface(out, count: vectorElements)
  let metalOutput = WDMetal.flash(qkv[0], qkv[1], qkv[2], bias: relativeBias,
                                  regions: regionWindow, windows: 1, shifted: shifted,
                                  queryRows: 16)
  eval(metalOutput)
  let gpuValues = metalOutput.reshaped(196, heads, 32).transposed(1, 0, 2)
    .asArray(Float16.self).map(Float.init)
  let outputErrors = zip(aneOutput, gpuValues).map { abs(Float(Float16(bitPattern: $0.0)) - $0.1) }
  maximumOutputError = outputErrors.max() ?? 0
  meanOutputError = Double(outputErrors.reduce(0, +)) / Double(max(1, outputErrors.count))
  guard maximumOutputError.isFinite, maximumOutputError <= 0.01 else {
    fail("masked-attention output exceeded the 0.01 FP16 probe tolerance: \(maximumOutputError)")
  }
  for _ in 0..<3 {
    let warm = WDMetal.flash(qkv[0], qkv[1], qkv[2], bias: relativeBias,
                             regions: regionWindow, windows: 1, shifted: shifted,
                             queryRows: 16)
    eval(warm)
  }
  for _ in 0..<iterations {
    let start = ProcessInfo.processInfo.systemUptime
    let warm = WDMetal.flash(qkv[0], qkv[1], qkv[2], bias: relativeBias,
                             regions: regionWindow, windows: 1, shifted: shifted,
                             queryRows: 16)
    eval(warm)
    timedMetal.append(ProcessInfo.processInfo.systemUptime - start)
  }
  for _ in 0..<5 { evaluateANE() }
  for _ in 0..<iterations {
    let start = ProcessInfo.processInfo.systemUptime
    evaluateANE()
    timedANE.append(ProcessInfo.processInfo.systemUptime - start)
  }
  func median(_ samples: [Double]) -> Double {
    let sorted = samples.sorted()
    return sorted.isEmpty ? 0 : sorted[sorted.count / 2]
  }
  let result: [String: Any] = [
    "backend": "direct-ane-explicit-swinv2-attention-probe",
    "deviceModel": machineIdentifier(),
    "os": ProcessInfo.processInfo.operatingSystemVersionString,
    "stage": stage, "block": block, "heads": heads, "window": selectedWindow, "windowCountTested": 1,
    "shiftedWindowMask": shifted,
    "operation": "QK^T * 1/sqrt(32) + prepared relative bias + shifted region -100 mask + MIL softmax + probabilities @ V",
    "allAttentionArithmeticRunsInsideOneANEProgram": true,
    "maskAndRelativeBiasAreStaticShapeInputs": true,
    "intermediateScoresReturnToHostOrGPU": false,
    "softmaxExecutesInsideANEAndFeedsValueMatmul": true,
    "compileAndLoadSeconds": compileSeconds,
    "identicalProgramCacheHitVerified": identicalProgramCacheHit,
    "identicalProgramCacheLookupMilliseconds": identicalProgramCacheLookupSeconds * 1000,
    "cachedProgramEntryCount": WDANEProgramCacheEntryCount(),
    "maximumAbsoluteAttendedOutputErrorAgainstFlash32": maximumOutputError,
    "meanAbsoluteAttendedOutputErrorAgainstFlash32": meanOutputError,
    "aneTimedEvaluationMedianMilliseconds": median(timedANE) * 1000,
    "metalFlash32MedianMilliseconds": median(timedMetal) * 1000,
    "timedIterationCount": timedANE.count,
    "measurementCaveat": "One fixed 196-token window at a time with synthetic QKV formed by the production cosine-normalization kernel and the checkpoint's learned query scale. This is a building-block result, not a full stage or end-to-end speedup.",
  ]
  report(result)
}

private func runFullBlockProbe() throws {
  guard WDANEIsAvailable() else { fail("Apple's private ANE runtime classes are unavailable") }
  let iterations = Int(option("--iterations", default: "10")) ?? 0
  guard (1...100).contains(iterations) else { fail("full-block probe requires --iterations 1...100") }
  let stage = Int(option("--stage", default: "4")) ?? 0
  let block = Int(option("--block", default: "0")) ?? 0
  let selectedWindow = Int(option("--window", default: "0")) ?? 0
  guard (1...4).contains(stage), (0..<([2, 2, 18, 2][stage - 1])).contains(block) else {
    fail("full-block probe requires a valid --stage 1...4 and --block")
  }
  let channels = [128, 256, 512, 1024][stage - 1]
  let heads = channels / 32, hidden = channels * 4, rows = 196
  let side = [112, 56, 28, 14][stage - 1]
  let windowsPerSide = side / 14
  let windowCount = windowsPerSide * windowsPerSide
  guard (0..<windowCount).contains(selectedWindow) else {
    fail("--window is outside this stage's Swin window grid")
  }
  let shifted = block % 2 == 1 && stage != 4
  let tensorsURL = URL(fileURLWithPath: option(
    "--weights", default: "models/published/WD-SwinV2-Tagger-v3-Swift-MLX/model.safetensors"))
  let tensors = try loadArrays(url: tensorsURL, stream: .cpu)
  let blockPrefix = "layers.\(stage - 1).blocks.\(block)"
  let attentionPrefix = blockPrefix + ".attn"
  func tensor(_ name: String, count: Int? = nil) -> MLXArray {
    guard let value = tensors[name], count == nil || value.size == count! else {
      fail("prepared checkpoint lacks the expected full-block tensor \(name)")
    }
    return value
  }
  let qkvWeight = tensor(attentionPrefix + ".qkv.weight")
  let qkvBias = tensor(attentionPrefix + "._sdpa_qkv_bias", count: 3 * channels)
  let queryScale = tensor(attentionPrefix + "._sdpa_query_scale", count: heads)
  let relativeBias = tensor(attentionPrefix + "._sdpa_bias", count: heads * rows * rows)
  let relativeBiasValues = relativeBias.asType(.float16).asArray(Float16.self)
  let regionValues = WDNativeTagger.regions(side: side).asArray(Int32.self)
  var combinedBias = [UInt16](repeating: 0, count: heads * rows * rows)
  for head in 0..<heads {
    for query in 0..<rows {
      let queryRegion = regionValues[selectedWindow * rows + query]
      for key in 0..<rows {
        let index = (head * rows + query) * rows + key
        let mask: Float = shifted && queryRegion != regionValues[selectedWindow * rows + key] ? -100 : 0
        combinedBias[index] = Float16(Float(relativeBiasValues[index]) + mask).bitPattern
      }
    }
  }
  let milRelativeBias = MLXArray(combinedBias.map { Float16(bitPattern: $0) }, [heads, rows, rows])
  let projectionWeight = tensor(attentionPrefix + ".proj.weight")
  let projectionBias = tensor(attentionPrefix + ".proj.bias", count: channels)
  let norm1Weight = tensor(blockPrefix + ".norm1.weight", count: channels)
  let norm1Bias = tensor(blockPrefix + ".norm1.bias", count: channels)
  let fc1Weight = tensor(blockPrefix + ".mlp.fc1.weight")
  let fc1Bias = tensor(blockPrefix + ".mlp.fc1.bias", count: hidden)
  let fc2Weight = tensor(blockPrefix + ".mlp.fc2.weight")
  let fc2Bias = tensor(blockPrefix + ".mlp.fc2.bias", count: channels)
  let norm2Weight = tensor(blockPrefix + ".norm2.weight", count: channels)
  let norm2Bias = tensor(blockPrefix + ".norm2.bias", count: channels)
  let boundTensors: [(String, String, MLXArray)] = [
    ("qkv_weight", "[\(3 * channels),\(channels),1,1]", qkvWeight),
    ("qkv_bias", "[1,\(3 * channels),1,1]", qkvBias),
    ("query_scale", "[1,\(heads),1,1]", queryScale),
    ("relative_bias", "[1,\(heads),\(rows),\(rows)]", milRelativeBias),
    ("projection_weight", "[\(channels),\(channels),1,1]", projectionWeight),
    ("projection_bias", "[1,\(channels),1,1]", projectionBias),
    ("norm1_weight", "[1,\(channels),1,1]", norm1Weight),
    ("norm1_bias", "[1,\(channels),1,1]", norm1Bias),
    ("fc1_weight", "[\(hidden),\(channels),1,1]", fc1Weight),
    ("fc1_bias", "[1,\(hidden),1,1]", fc1Bias),
    ("fc2_weight", "[\(channels),\(hidden),1,1]", fc2Weight),
    ("fc2_bias", "[1,\(channels),1,1]", fc2Bias),
    ("norm2_weight", "[1,\(channels),1,1]", norm2Weight),
    ("norm2_bias", "[1,\(channels),1,1]", norm2Bias),
  ]
  let weightBlobs = boundTensors.map { $0.2.asType(.float16).asArray(Float16.self).map(\.bitPattern) }
    .map(makeWeightBlob)
  let milConstants = boundTensors.enumerated().map { index, item in
    """
        tensor<fp16, \(item.1)> \(item.0) = const()[name=string("\(item.0)"), val=tensor<fp16, \(item.1)>(BLOBFILE(path=string("@model_path/weights/block_\(index).bin"), offset=uint64(64)))];
    """
  }.joined(separator: "\n")
  func normMIL(_ name: String, value: String, residual: String, weight: String,
               bias: String, output: String) -> String {
    """
        tensor<int32, [1]> \(name)_axes = const()[name=string("\(name)_axes"), val=tensor<int32, [1]>([1])];
        bool \(name)_keep_dims = const()[name=string("\(name)_keep_dims"), val=bool(true)];
        tensor<fp16, [1,1,1,\(rows)]> \(name)_mean = reduce_mean(x=\(value), axes=\(name)_axes, keep_dims=\(name)_keep_dims)[name=string("\(name)_mean")];
        tensor<fp16, [1,\(channels),1,\(rows)]> \(name)_centered = sub(x=\(value), y=\(name)_mean)[name=string("\(name)_centered")];
        tensor<fp16, [1,\(channels),1,\(rows)]> \(name)_squared = mul(x=\(name)_centered, y=\(name)_centered)[name=string("\(name)_squared")];
        tensor<fp16, [1,1,1,\(rows)]> \(name)_variance = reduce_mean(x=\(name)_squared, axes=\(name)_axes, keep_dims=\(name)_keep_dims)[name=string("\(name)_variance")];
        fp16 \(name)_epsilon = const()[name=string("\(name)_epsilon"), val=fp16(0.00001)];
        tensor<fp16, [1,1,1,\(rows)]> \(name)_variance_eps = add(x=\(name)_variance, y=\(name)_epsilon)[name=string("\(name)_variance_eps")];
        fp16 \(name)_negative_half = const()[name=string("\(name)_negative_half"), val=fp16(-0.5)];
        tensor<fp16, [1,1,1,\(rows)]> \(name)_rstd = pow(x=\(name)_variance_eps, y=\(name)_negative_half)[name=string("\(name)_rstd")];
        tensor<fp16, [1,\(channels),1,\(rows)]> \(name)_normalized = mul(x=\(name)_centered, y=\(name)_rstd)[name=string("\(name)_normalized")];
        tensor<fp16, [1,\(channels),1,\(rows)]> \(name)_affine = mul(x=\(name)_normalized, y=\(weight))[name=string("\(name)_affine")];
        tensor<fp16, [1,\(channels),1,\(rows)]> \(name)_biased = add(x=\(name)_affine, y=\(bias))[name=string("\(name)_biased")];
        tensor<fp16, [1,\(channels),1,\(rows)]> \(output) = add(x=\(name)_biased, y=\(residual))[name=string("\(output)")];
    """
  }
  let mil = """
  program(1.3)
  [buildInfo = dict<string, string>({
    {"coremlc-component-MIL", "3510.2.1"},
    {"coremlc-version", "3505.4.1"},
    {"coremltools-component-milinternal", ""},
    {"coremltools-version", "9.0"}
  })]
  {
    func main<ios18>(tensor<fp16, [1,\(channels),1,\(rows)]> x) {
      string pt = const()[name=string("pt"), val=string("valid")];
      tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
      tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
      tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
      int32 gr = const()[name=string("gr"), val=int32(1)];
  \(milConstants)
      tensor<fp16, [1,\(3 * channels),1,\(rows)]> qkv_projection = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=qkv_weight, x=x)[name=string("qkv_projection")];
      tensor<fp16, [1,\(3 * channels),1,\(rows)]> qkv = add(x=qkv_projection, y=qkv_bias)[name=string("qkv_bias_add")];
      tensor<int32, [4]> q_begin = const()[name=string("q_begin"), val=tensor<int32, [4]>([0,0,0,0])];
      tensor<int32, [4]> k_begin = const()[name=string("k_begin"), val=tensor<int32, [4]>([0,\(channels),0,0])];
      tensor<int32, [4]> v_begin = const()[name=string("v_begin"), val=tensor<int32, [4]>([0,\(2 * channels),0,0])];
      tensor<int32, [4]> q_size = const()[name=string("q_size"), val=tensor<int32, [4]>([1,\(channels),1,\(rows)])];
      tensor<fp16, [1,\(channels),1,\(rows)]> q_flat = slice_by_size(x=qkv, begin=q_begin, size=q_size)[name=string("q_flat")];
      tensor<fp16, [1,\(channels),1,\(rows)]> k_flat = slice_by_size(x=qkv, begin=k_begin, size=q_size)[name=string("k_flat")];
      tensor<fp16, [1,\(channels),1,\(rows)]> v_flat = slice_by_size(x=qkv, begin=v_begin, size=q_size)[name=string("v_flat")];
      tensor<int32, [4]> head_shape = const()[name=string("head_shape"), val=tensor<int32, [4]>([1,\(heads),32,\(rows)])];
      tensor<int32, [4]> transpose_perm = const()[name=string("transpose_perm"), val=tensor<int32, [4]>([0,1,3,2])];
      tensor<fp16, [1,\(heads),32,\(rows)]> q_head = reshape(shape=head_shape, x=q_flat)[name=string("q_head")];
      tensor<fp16, [1,\(heads),\(rows),32]> q = transpose(perm=transpose_perm, x=q_head)[name=string("q")];
      tensor<fp16, [1,\(heads),32,\(rows)]> k_head = reshape(shape=head_shape, x=k_flat)[name=string("k_head")];
      tensor<fp16, [1,\(heads),\(rows),32]> k = transpose(perm=transpose_perm, x=k_head)[name=string("k")];
      tensor<fp16, [1,\(heads),32,\(rows)]> v_head = reshape(shape=head_shape, x=v_flat)[name=string("v_head")];
      tensor<fp16, [1,\(heads),\(rows),32]> v = transpose(perm=transpose_perm, x=v_head)[name=string("v")];
      tensor<int32, [1]> qnorm_axes = const()[name=string("qnorm_axes"), val=tensor<int32, [1]>([3])];
      bool qnorm_keep = const()[name=string("qnorm_keep"), val=bool(true)];
      tensor<fp16, [1,\(heads),\(rows),32]> q_square = mul(x=q, y=q)[name=string("q_square")];
      tensor<fp16, [1,\(heads),\(rows),1]> q_norm_sq = reduce_sum(x=q_square, axes=qnorm_axes, keep_dims=qnorm_keep)[name=string("q_norm_sq")];
      fp16 cosine_epsilon = const()[name=string("cosine_epsilon"), val=fp16(0.000001)];
      tensor<fp16, [1,\(heads),\(rows),1]> q_norm_safe = add(x=q_norm_sq, y=cosine_epsilon)[name=string("q_norm_safe")];
      fp16 negative_half = const()[name=string("negative_half"), val=fp16(-0.5)];
      tensor<fp16, [1,\(heads),\(rows),1]> q_rstd = pow(x=q_norm_safe, y=negative_half)[name=string("q_rstd")];
      tensor<fp16, [1,\(heads),\(rows),32]> q_unit = mul(x=q, y=q_rstd)[name=string("q_unit")];
      tensor<fp16, [1,\(heads),\(rows),32]> q_scaled = mul(x=q_unit, y=query_scale)[name=string("q_scaled_by_learned_scale")];
      tensor<fp16, [1,\(heads),\(rows),32]> k_square = mul(x=k, y=k)[name=string("k_square")];
      tensor<fp16, [1,\(heads),\(rows),1]> k_norm_sq = reduce_sum(x=k_square, axes=qnorm_axes, keep_dims=qnorm_keep)[name=string("k_norm_sq")];
      tensor<fp16, [1,\(heads),\(rows),1]> k_norm_safe = add(x=k_norm_sq, y=cosine_epsilon)[name=string("k_norm_safe")];
      tensor<fp16, [1,\(heads),\(rows),1]> k_rstd = pow(x=k_norm_safe, y=negative_half)[name=string("k_rstd")];
      tensor<fp16, [1,\(heads),\(rows),32]> k_unit = mul(x=k, y=k_rstd)[name=string("k_unit")];
      bool tx = const()[name=string("tx"), val=bool(false)];
      bool ty = const()[name=string("ty"), val=bool(true)];
      fp16 attention_scale = const()[name=string("attention_scale"), val=fp16(0.176776695)];
      tensor<fp16, [1,\(heads),\(rows),\(rows)]> qk = matmul(transpose_x=tx, transpose_y=ty, x=q_scaled, y=k_unit)[name=string("cosine_qk")];
      tensor<fp16, [1,\(heads),\(rows),\(rows)]> scaled_qk = mul(x=qk, y=attention_scale)[name=string("attention_scale_apply")];
      tensor<fp16, [1,\(heads),\(rows),\(rows)]> biased_qk = add(x=scaled_qk, y=relative_bias)[name=string("learned_relative_position_bias")];
      int32 softmax_axis = const()[name=string("softmax_axis"), val=int32(-1)];
      tensor<fp16, [1,\(heads),\(rows),\(rows)]> probabilities = softmax(axis=softmax_axis, x=biased_qk)[name=string("on_ane_softmax")];
      bool no_transpose = const()[name=string("no_transpose"), val=bool(false)];
      tensor<fp16, [1,\(heads),\(rows),32]> attended = matmul(transpose_x=no_transpose, transpose_y=no_transpose, x=probabilities, y=v)[name=string("attention_times_value")];
      tensor<fp16, [1,\(heads),32,\(rows)]> attended_head = transpose(perm=transpose_perm, x=attended)[name=string("attended_head")];
      tensor<int32, [4]> channel_shape = const()[name=string("channel_shape"), val=tensor<int32, [4]>([1,\(channels),1,\(rows)])];
      tensor<fp16, [1,\(channels),1,\(rows)]> attended_channels = reshape(shape=channel_shape, x=attended_head)[name=string("attended_channels")];
      tensor<fp16, [1,\(channels),1,\(rows)]> projection = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=projection_weight, x=attended_channels)[name=string("output_projection")];
      tensor<fp16, [1,\(channels),1,\(rows)]> projection_biased = add(x=projection, y=projection_bias)[name=string("output_projection_bias")];
  \(normMIL("norm1", value: "projection_biased", residual: "x", weight: "norm1_weight", bias: "norm1_bias", output: "after_attention_norm"))
      tensor<fp16, [1,\(hidden),1,\(rows)]> fc1 = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=fc1_weight, x=after_attention_norm)[name=string("mlp_fc1")];
      tensor<fp16, [1,\(hidden),1,\(rows)]> fc1_biased = add(x=fc1, y=fc1_bias)[name=string("mlp_fc1_bias")];
      tensor<fp16, [1,\(hidden),1,\(rows)]> gelu_square = mul(x=fc1_biased, y=fc1_biased)[name=string("gelu_square")];
      tensor<fp16, [1,\(hidden),1,\(rows)]> gelu_cube = mul(x=gelu_square, y=fc1_biased)[name=string("gelu_cube")];
      fp16 gelu_cubic = const()[name=string("gelu_cubic"), val=fp16(0.044715)];
      fp16 gelu_scale = const()[name=string("gelu_scale"), val=fp16(0.79788458)];
      fp16 half = const()[name=string("half"), val=fp16(0.5)];
      fp16 one = const()[name=string("one"), val=fp16(1.0)];
      tensor<fp16, [1,\(hidden),1,\(rows)]> gelu_scaled_cube = mul(x=gelu_cubic, y=gelu_cube)[name=string("gelu_scaled_cube")];
      tensor<fp16, [1,\(hidden),1,\(rows)]> gelu_argument = add(x=fc1_biased, y=gelu_scaled_cube)[name=string("gelu_argument")];
      tensor<fp16, [1,\(hidden),1,\(rows)]> gelu_tanh_input = mul(x=gelu_argument, y=gelu_scale)[name=string("gelu_tanh_input")];
      tensor<fp16, [1,\(hidden),1,\(rows)]> gelu_tanh = tanh(x=gelu_tanh_input)[name=string("gelu_tanh")];
      tensor<fp16, [1,\(hidden),1,\(rows)]> gelu_one_plus_tanh = add(x=gelu_tanh, y=one)[name=string("gelu_one_plus_tanh")];
      tensor<fp16, [1,\(hidden),1,\(rows)]> gelu_half_input = mul(x=fc1_biased, y=half)[name=string("gelu_half_input")];
      tensor<fp16, [1,\(hidden),1,\(rows)]> gelu = mul(x=gelu_half_input, y=gelu_one_plus_tanh)[name=string("tanh_gelu")];
      tensor<fp16, [1,\(channels),1,\(rows)]> fc2 = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=fc2_weight, x=gelu)[name=string("mlp_fc2")];
      tensor<fp16, [1,\(channels),1,\(rows)]> fc2_biased = add(x=fc2, y=fc2_bias)[name=string("mlp_fc2_bias")];
  \(normMIL("norm2", value: "fc2_biased", residual: "after_attention_norm", weight: "norm2_weight", bias: "norm2_bias", output: "block_output"))
    } -> (block_output);
  }
  """
  var compileError = [CChar](repeating: 0, count: 1024)
  let compileStart = ProcessInfo.processInfo.systemUptime
  let weightPaths = weightBlobs.indices.map { "@model_path/weights/block_\($0).bin" }
  let weightSizes = weightBlobs.map(\.count)
  func withPathPointers(_ index: Int, _ pointers: [UnsafePointer<CChar>?],
                        _ body: ([UnsafePointer<CChar>?]) -> ANEProgram?) -> ANEProgram? {
    guard index < weightPaths.count else { return body(pointers) }
    return weightPaths[index].withCString { path in
      var next = pointers; next[index] = path
      return withPathPointers(index + 1, next, body)
    }
  }
  func withWeightPointers(_ index: Int, _ pointers: [UnsafePointer<UInt8>?],
                          _ body: ([UnsafePointer<UInt8>?]) -> ANEProgram?) -> ANEProgram? {
    guard index < weightBlobs.count else { return body(pointers) }
    return weightBlobs[index].withUnsafeBytes { bytes in
      var next = pointers; next[index] = bytes.baseAddress!.assumingMemoryBound(to: UInt8.self)
      return withWeightPointers(index + 1, next, body)
    }
  }
  let program = mil.withCString { source in
    withPathPointers(0, [UnsafePointer<CChar>?](repeating: nil, count: weightPaths.count)) { paths in
      withWeightPointers(0, [UnsafePointer<UInt8>?](repeating: nil, count: weightBlobs.count)) { bytes in
        let validPaths = paths.compactMap { $0 }
        let validBytes = bytes.compactMap { $0 }
        return validPaths.withUnsafeBufferPointer { pathBuffer in
          validBytes.withUnsafeBufferPointer { byteBuffer in
            weightSizes.withUnsafeBufferPointer { sizeBuffer in
              compileError.withUnsafeMutableBufferPointer { errorBuffer in
                WDANECompileMILWithWeightBlobs(source, pathBuffer.baseAddress!, byteBuffer.baseAddress!,
                  sizeBuffer.baseAddress!, weightBlobs.count, errorBuffer.baseAddress, errorBuffer.count)
              }
            }
          }
        }
      }
    }
  }
  guard let program else { fail("complete stage-4 SwinV2 block MIL compile/load failed", String(cString: compileError)) }
  defer { WDANEReleaseProgram(program) }
  let compileSeconds = ProcessInfo.processInfo.systemUptime - compileStart
  let inputValues = MLXArray((0..<(rows * channels)).map {
    Float16(Float(sin(Double(($0 * 19 + 3) % 65521))) * 0.25)
  }, [rows, channels])
  var inputBits = [UInt16](repeating: 0, count: rows * channels)
  let inputRowMajor = inputValues.asArray(Float16.self)
  for row in 0..<rows {
    for channel in 0..<channels { inputBits[channel * rows + row] = inputRowMajor[row * channels + channel].bitPattern }
  }
  let ioBytes = rows * channels * MemoryLayout<UInt16>.size
  let inputSurface = makeSurface(elements: rows * channels, allocationBytes: ioBytes)
  let outputSurface = makeSurface(elements: rows * channels, allocationBytes: ioBytes)
  defer { WDANETensorRelease(inputSurface); WDANETensorRelease(outputSurface) }
  write(inputBits, to: inputSurface)
  func evaluateANE() { evaluate(program, inputs: [inputSurface], output: outputSurface) }
  let metalInput = inputValues
  let regionWindow = MLXArray(Array(regionValues[(selectedWindow * rows)..<((selectedWindow + 1) * rows)]))
  func metalBlock(_ values: MLXArray) -> MLXArray {
    let qkv = WDMetal.projectQKV(values, weight: qkvWeight, bias: qkvBias,
                                 scale: queryScale.reshaped(-1), heads: heads)
    let attended = WDMetal.flash(qkv[0], qkv[1], qkv[2], bias: relativeBias,
                                 regions: regionWindow, windows: 1, shifted: shifted, queryRows: 16)
    let projected = WDMetal.gemm(attended, weight: projectionWeight, bias: projectionBias)
    let afterAttention = WDMetal.normalize(projected, residual: values, weight: norm1Weight,
      bias: norm1Bias, route: MLXArray([Int32(0)]), spatial: rows, identityRoute: true)
    let hiddenValues = WDMetal.mlpGelu(afterAttention, weight: fc1Weight, bias: fc1Bias)
    let fc2Values = WDMetal.gemm(hiddenValues, weight: fc2Weight, bias: fc2Bias)
    return WDMetal.normalize(fc2Values, residual: afterAttention, weight: norm2Weight,
      bias: norm2Bias, route: MLXArray([Int32(0)]), spatial: rows, identityRoute: true)
  }
  let gpuGraph = compile { values in metalBlock(values) }
  var metalOutput = gpuGraph(metalInput)
  eval(metalOutput)
  evaluateANE()
  var aneBits = [UInt16](repeating: 0, count: rows * channels)
  var readError = [CChar](repeating: 0, count: 1024)
  let readOK = aneBits.withUnsafeMutableBufferPointer { values in
    readError.withUnsafeMutableBufferPointer { errorBuffer in
      WDANETensorReadF16(outputSurface, values.baseAddress!, values.count,
                         errorBuffer.baseAddress, errorBuffer.count)
    }
  }
  if !readOK { fail("cannot read complete-block output", String(cString: readError)) }
  let expectedRowMajor = metalOutput.asArray(Float16.self)
  var expected = [Float](repeating: 0, count: rows * channels)
  for row in 0..<rows {
    for channel in 0..<channels { expected[channel * rows + row] = Float(expectedRowMajor[row * channels + channel]) }
  }
  let differences = zip(aneBits, expected).map { abs(Float(Float16(bitPattern: $0.0)) - $0.1) }
  guard differences.allSatisfy(\.isFinite), (differences.max() ?? 0) <= 0.02 else {
    fail("complete-block output exceeded the 0.02 FP16 probe tolerance: \(differences.max() ?? .infinity)")
  }
  for _ in 0..<3 { evaluateANE(); metalOutput = gpuGraph(metalInput); eval(metalOutput) }
  var aneTimes: [Double] = [], metalTimes: [Double] = []
  for _ in 0..<iterations {
    var start = ProcessInfo.processInfo.systemUptime
    evaluateANE(); aneTimes.append(ProcessInfo.processInfo.systemUptime - start)
    start = ProcessInfo.processInfo.systemUptime
    metalOutput = gpuGraph(metalInput); eval(metalOutput)
    metalTimes.append(ProcessInfo.processInfo.systemUptime - start)
  }
  func median(_ values: [Double]) -> Double {
    let sorted = values.sorted(); return sorted[sorted.count / 2]
  }
  report([
    "backend": "direct-ane-complete-swinv2-block-probe",
    "deviceModel": machineIdentifier(),
    "os": ProcessInfo.processInfo.operatingSystemVersionString,
    "stage": stage, "block": block, "window": selectedWindow, "batch": 1, "windows": 1,
    "channels": channels, "heads": heads,
    "operation": "QKV projection, cosine Q/K normalization, learned per-head query scale, relative bias, on-ANE softmax, attention-times-V, output projection, norm1/residual, tanh-GELU MLP, norm2/residual",
    "completeTransformerBlockInOneANEProgram": true,
    "allIntermediateBlockActivationsStayOnANE": true,
    "shiftedWindowMask": shifted,
    "compileAndLoadSeconds": compileSeconds,
    "staticFP16WeightBytesIncludingBlobHeaders": weightBlobs.reduce(0) { $0 + $1.count },
    "maximumAbsoluteErrorAgainstWDMetalBlock": differences.max() ?? 0,
    "meanAbsoluteErrorAgainstWDMetalBlock": Double(differences.reduce(0, +)) / Double(max(1, differences.count)),
    "aneMedianMilliseconds": median(aneTimes) * 1000,
    "metalMedianMilliseconds": median(metalTimes) * 1000,
    "timedIterations": iterations,
    "measurementCaveat": "One synthetic input window for stage 4 block 0 at batch 1. This validates a complete block graph, not whole-stage or end-to-end tagger throughput or natural-image accuracy."
  ])
}

private func runFusedMLPProbe() throws {
  guard WDANEIsAvailable() else { fail("Apple's private ANE runtime classes are unavailable") }
  let stage = Int(option("--stage", default: "3")) ?? 0
  let batch = Int(option("--batch", default: "1")) ?? 0
  let iterations = Int(option("--iterations", default: "10")) ?? 0
  let blockCount = Int(option("--blocks", default: "2")) ?? 0
  let paddingMultiple = max(0, Int(option("--pad-rows-to-multiple", default: "0")) ?? 0)
  let includePostNorm = !CommandLine.arguments.contains("--omit-postnorm")
  let firstProjectionOnly = CommandLine.arguments.contains("--fc1-only")
  guard (1...4).contains(stage), (1...8).contains(batch), (1...100).contains(iterations),
        (1...2).contains(blockCount) else {
    fail("fused MLP probe requires --stage 1...4, --batch 1...8, --iterations 1...100")
  }
  let channels = [128, 256, 512, 1024][stage - 1]
  let hidden = channels * 4
  let rowsPerImage = [12544, 3136, 784, 196][stage - 1]
  let fullRows = rowsPerImage * batch
  let rows = paddingMultiple > 0
    ? ((fullRows + paddingMultiple - 1) / paddingMultiple) * paddingMultiple : fullRows
  let tensorsURL = URL(fileURLWithPath: option(
    "--weights", default: "models/published/WD-SwinV2-Tagger-v3-Swift-MLX/model.safetensors"))
  let tensors = try loadArrays(url: tensorsURL, stream: .cpu)
  var blockWeights: [(MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray)] = []
  var blobTensors: [[UInt16]] = []
  var milConstants = ""
  var milOperations = ""
  var previous = "x"
  for block in 0..<blockCount {
    let prefix = "layers.\(stage - 1).blocks.\(block).mlp"
    guard let w1 = tensors[prefix + ".fc1.weight"], w1.shape == [hidden, channels],
          let b1 = tensors[prefix + ".fc1.bias"], b1.size == hidden,
          let w2 = tensors[prefix + ".fc2.weight"], w2.shape == [channels, hidden],
          let b2 = tensors[prefix + ".fc2.bias"], b2.size == channels,
          let normWeight = tensors["layers.\(stage - 1).blocks.\(block).norm2.weight"], normWeight.size == channels,
          let normBias = tensors["layers.\(stage - 1).blocks.\(block).norm2.bias"], normBias.size == channels else {
      fail("prepared checkpoint lacks the fixed MLP weights for \(prefix)")
    }
    blockWeights.append((w1, b1, w2, b2, normWeight, normBias))
    let values = [w1, b1, w2, b2, normWeight, normBias].map { $0.asArray(Float16.self).map(\.bitPattern) }
    let names = ["w1", "b1", "w2", "b2", "norm_weight", "norm_bias"]
    for (index, name) in names.enumerated() {
      if firstProjectionOnly && index > 1 { continue }
      let tensorIndex = blobTensors.count
      blobTensors.append(values[index])
      let firstLayer = name == "w1" || name == "b1"
      let outputChannels = firstLayer ? hidden : channels
      let inputChannels = firstLayer ? channels : hidden
      let shape = name == "w1" || name == "w2"
        ? "[\(outputChannels),\(inputChannels),1,1]" : "[1,\(outputChannels),1,1]"
      milConstants += """
          tensor<fp16, \(shape)> b\(block)_\(name) = const()[name=string("b\(block)_\(name)"), val=tensor<fp16, \(shape)>(BLOBFILE(path=string("@model_path/weights/mlp_\(tensorIndex).bin"), offset=uint64(64)))];
        """
    }
    let hiddenShape = "[1,\(hidden),1,\(rows)]"
    let channelShape = "[1,\(channels),1,\(rows)]"
    if firstProjectionOnly {
      guard block == 0 else { break }
      milOperations += """
        tensor<fp16, \(hiddenShape)> b\(block)_fc1 = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=b\(block)_w1, x=x)[name=string("block\(block)_fc1")];
        tensor<fp16, \(hiddenShape)> b\(block)_biased = add(x=b\(block)_fc1, y=b\(block)_b1)[name=string("block\(block)_bias1")];
      """
      previous = "b\(block)_biased"
      break
    }
    milOperations += """
        tensor<fp16, \(hiddenShape)> b\(block)_fc1 = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=b\(block)_w1, x=\(previous))[name=string("block\(block)_fc1")];
        tensor<fp16, \(hiddenShape)> b\(block)_biased = add(x=b\(block)_fc1, y=b\(block)_b1)[name=string("block\(block)_bias1")];
        tensor<fp16, \(hiddenShape)> b\(block)_square = mul(x=b\(block)_biased, y=b\(block)_biased)[name=string("block\(block)_gelu_square")];
        tensor<fp16, \(hiddenShape)> b\(block)_cube = mul(x=b\(block)_square, y=b\(block)_biased)[name=string("block\(block)_gelu_cube")];
        tensor<fp16, \(hiddenShape)> b\(block)_scaled_cube = mul(x=b\(block)_cube, y=c044)[name=string("block\(block)_gelu_cubic_term")];
        tensor<fp16, \(hiddenShape)> b\(block)_argument = add(x=b\(block)_biased, y=b\(block)_scaled_cube)[name=string("block\(block)_gelu_argument")];
        tensor<fp16, \(hiddenShape)> b\(block)_tanh_argument = mul(x=b\(block)_argument, y=c079)[name=string("block\(block)_gelu_tanh_argument")];
        tensor<fp16, \(hiddenShape)> b\(block)_tanh = tanh(x=b\(block)_tanh_argument)[name=string("block\(block)_gelu_tanh")];
        tensor<fp16, \(hiddenShape)> b\(block)_one_plus_tanh = add(x=b\(block)_tanh, y=one)[name=string("block\(block)_gelu_one_plus_tanh")];
        tensor<fp16, \(hiddenShape)> b\(block)_half_input = mul(x=b\(block)_biased, y=half)[name=string("block\(block)_gelu_half_input")];
        tensor<fp16, \(hiddenShape)> b\(block)_gelu = mul(x=b\(block)_half_input, y=b\(block)_one_plus_tanh)[name=string("block\(block)_gelu")];
        tensor<fp16, \(channelShape)> b\(block)_fc2 = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=b\(block)_w2, x=b\(block)_gelu)[name=string("block\(block)_fc2")];
        tensor<fp16, \(channelShape)> b\(block)_out = add(x=b\(block)_fc2, y=b\(block)_b2)[name=string("block\(block)_bias2")];
      """
    if includePostNorm {
      milOperations += """
        tensor<int32, [1]> b\(block)_norm_axes = const()[name=string("block\(block)_norm_axes"), val=tensor<int32, [1]>([1])];
        bool b\(block)_keep_dims = const()[name=string("block\(block)_keep_dims"), val=bool(true)];
        tensor<fp16, [1,1,1,\(rows)]> b\(block)_mean = reduce_mean(x=b\(block)_out, axes=b\(block)_norm_axes, keep_dims=b\(block)_keep_dims)[name=string("block\(block)_norm_mean")];
        tensor<fp16, \(channelShape)> b\(block)_centered = sub(x=b\(block)_out, y=b\(block)_mean)[name=string("block\(block)_norm_centered")];
        tensor<fp16, \(channelShape)> b\(block)_variance_sq = mul(x=b\(block)_centered, y=b\(block)_centered)[name=string("block\(block)_norm_variance_sq")];
        tensor<fp16, [1,1,1,\(rows)]> b\(block)_variance = reduce_mean(x=b\(block)_variance_sq, axes=b\(block)_norm_axes, keep_dims=b\(block)_keep_dims)[name=string("block\(block)_norm_variance")];
        fp16 b\(block)_norm_eps = const()[name=string("block\(block)_norm_eps"), val=fp16(0.00001)];
        tensor<fp16, [1,1,1,\(rows)]> b\(block)_variance_eps = add(x=b\(block)_variance, y=b\(block)_norm_eps)[name=string("block\(block)_norm_variance_eps")];
        fp16 b\(block)_negative_half = const()[name=string("block\(block)_negative_half"), val=fp16(-0.5)];
        tensor<fp16, [1,1,1,\(rows)]> b\(block)_rstd = pow(x=b\(block)_variance_eps, y=b\(block)_negative_half)[name=string("block\(block)_norm_rstd")];
        tensor<fp16, \(channelShape)> b\(block)_normalized = mul(x=b\(block)_centered, y=b\(block)_rstd)[name=string("block\(block)_norm_normalized")];
        tensor<fp16, \(channelShape)> b\(block)_affine = mul(x=b\(block)_normalized, y=b\(block)_norm_weight)[name=string("block\(block)_norm_affine")];
        tensor<fp16, \(channelShape)> b\(block)_norm = add(x=b\(block)_affine, y=b\(block)_norm_bias)[name=string("block\(block)_norm")];
        tensor<fp16, \(channelShape)> b\(block)_residual = add(x=b\(block)_norm, y=\(previous))[name=string("block\(block)_residual")];
      """
    } else {
      milOperations += """
        tensor<fp16, \(channelShape)> b\(block)_residual = add(x=b\(block)_out, y=\(previous))[name=string("block\(block)_residual_without_norm")];
      """
    }
    previous = "b\(block)_residual"
  }
  let weightBlobs = blobTensors.map(makeWeightBlob)
  let mil = """
  program(1.3)
  [buildInfo = dict<string, string>({
    {"coremlc-component-MIL", "3510.2.1"},
    {"coremlc-version", "3505.4.1"},
    {"coremltools-component-milinternal", ""},
    {"coremltools-version", "9.0"}
  })]
  {
    func main<ios18>(tensor<fp16, [1,\(channels),1,\(rows)]> x) {
      string pt = const()[name=string("pt"), val=string("valid")];
      tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
      tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
      tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
      int32 gr = const()[name=string("gr"), val=int32(1)];
      fp16 c044 = const()[name=string("gelu_cubic"), val=fp16(0.044715)];
      fp16 c079 = const()[name=string("gelu_scale"), val=fp16(0.79788458)];
      fp16 half = const()[name=string("half"), val=fp16(0.5)];
      fp16 one = const()[name=string("one"), val=fp16(1.0)];
  \(milConstants)
  \(milOperations)
    } -> (\(previous));
  }
  """
  var compileError = [CChar](repeating: 0, count: 1024)
  let compileStart = ProcessInfo.processInfo.systemUptime
  let weightPaths = weightBlobs.indices.map { "@model_path/weights/mlp_\($0).bin" }
  let weightSizes = weightBlobs.map(\.count)
  func withPathPointers(
    _ index: Int, _ pointers: [UnsafePointer<CChar>?],
    _ body: ([UnsafePointer<CChar>?]) -> ANEProgram?
  ) -> ANEProgram? {
    guard index < weightPaths.count else { return body(pointers) }
    return weightPaths[index].withCString { path in
      var next = pointers
      next[index] = path
      return withPathPointers(index + 1, next, body)
    }
  }
  func withWeightPointers(
    _ index: Int, _ pointers: [UnsafePointer<UInt8>?],
    _ body: ([UnsafePointer<UInt8>?]) -> ANEProgram?
  ) -> ANEProgram? {
    guard index < weightBlobs.count else { return body(pointers) }
    return weightBlobs[index].withUnsafeBytes { bytes in
      var next = pointers
      next[index] = bytes.baseAddress!.assumingMemoryBound(to: UInt8.self)
      return withWeightPointers(index + 1, next, body)
    }
  }
  let maybeProgram = mil.withCString { source in
    withPathPointers(0, [UnsafePointer<CChar>?](repeating: nil, count: weightPaths.count)) { paths in
      withWeightPointers(0, [UnsafePointer<UInt8>?](repeating: nil, count: weightBlobs.count)) { weights in
        let nonnullPaths = paths.compactMap { $0 }
        let nonnullWeights = weights.compactMap { $0 }
        return nonnullPaths.withUnsafeBufferPointer { pathBuffer in
          nonnullWeights.withUnsafeBufferPointer { weightBuffer in
            weightSizes.withUnsafeBufferPointer { sizeBuffer in
              compileError.withUnsafeMutableBufferPointer { errorBuffer in
                WDANECompileMILWithWeightBlobs(
                  source, pathBuffer.baseAddress!, weightBuffer.baseAddress!, sizeBuffer.baseAddress!,
                  weightBlobs.count, errorBuffer.baseAddress, errorBuffer.count)
              }
            }
          }
        }
      }
    }
  }
  guard let program = maybeProgram else {
    fail("fused SwinV2 MLP MIL compile/load failed", String(cString: compileError))
  }
  defer { WDANEReleaseProgram(program) }
  let compileSeconds = ProcessInfo.processInfo.systemUptime - compileStart
  let outputChannels = firstProjectionOnly ? hidden : channels
  let inputValues = MLXArray((0..<(rows * channels)).map {
    let row = $0 / channels
    return row < fullRows ? Float(sin(Double(($0 * 13 + 7) % 65521))) * 0.25 : 0
  }, [rows, channels]).asType(.float16)
  var reference = inputValues
  if firstProjectionOnly {
    let weights = blockWeights[0]
    reference = WDMetal.gemm(reference, weight: weights.0, bias: weights.1)
  } else {
    for weights in blockWeights {
      let projected = WDMetal.gemm(WDMetal.mlpGelu(reference, weight: weights.0, bias: weights.1),
                                   weight: weights.2, bias: weights.3)
      reference = includePostNorm
        ? MLXFast.layerNorm(projected, weight: weights.4, bias: weights.5, eps: 1e-5) + reference
        : projected + reference
    }
  }
  eval(reference)
  let inputRowMajor = inputValues.asArray(Float16.self)
  var inputBits = [UInt16](repeating: 0, count: rows * channels)
  for row in 0..<rows {
    for channel in 0..<channels {
      inputBits[channel * rows + row] = inputRowMajor[row * channels + channel].bitPattern
    }
  }
  let byteCount = rows * channels * MemoryLayout<UInt16>.size
  let outputByteCount = rows * outputChannels * MemoryLayout<UInt16>.size
  let input = makeSurface(elements: rows * channels, allocationBytes: byteCount)
  let output = makeSurface(elements: rows * outputChannels, allocationBytes: outputByteCount)
  defer { WDANETensorRelease(input); WDANETensorRelease(output) }
  write(inputBits, to: input)
  func evaluateANE() {
    var error = [CChar](repeating: 0, count: 1024)
    let ok = [input].withUnsafeBufferPointer { inputs in
      [output].withUnsafeBufferPointer { outputs in
        error.withUnsafeMutableBufferPointer { errorBuffer in
          WDANEEvaluate(program, inputs.baseAddress!, inputs.count,
                        outputs.baseAddress!, outputs.count,
                        errorBuffer.baseAddress, errorBuffer.count)
        }
      }
    }
    if !ok { fail("two-block fused SwinV2 MLP evaluation failed", String(cString: error)) }
  }
  evaluateANE()
  var aneBits = [UInt16](repeating: 0, count: rows * outputChannels)
  var readError = [CChar](repeating: 0, count: 1024)
  let readOK = aneBits.withUnsafeMutableBufferPointer { values in
    readError.withUnsafeMutableBufferPointer { errorBuffer in
      WDANETensorReadF16(output, values.baseAddress!, values.count,
                         errorBuffer.baseAddress, errorBuffer.count)
    }
  }
  if !readOK { fail("cannot read fused MLP output", String(cString: readError)) }
  let expectedRowMajor = reference.asArray(Float16.self)
  var expected = [Float](repeating: 0, count: fullRows * outputChannels)
  var validANE = [UInt16](repeating: 0, count: fullRows * outputChannels)
  for row in 0..<fullRows {
    for channel in 0..<outputChannels {
      expected[channel * fullRows + row] = Float(expectedRowMajor[row * outputChannels + channel])
      validANE[channel * fullRows + row] = aneBits[channel * rows + row]
    }
  }
  let differences = zip(validANE, expected).map { abs(Float(Float16(bitPattern: $0.0)) - $0.1) }
  let aneMagnitudes = validANE.map { abs(Float(Float16(bitPattern: $0))) }
  let maximumErrorIndex = differences.firstIndex(of: differences.max() ?? 0) ?? 0
  guard differences.allSatisfy(\.isFinite), (differences.max() ?? 0) <= 0.01 else {
    fail("fused MLP output exceeded the 0.01 FP16 probe tolerance: \(differences.max() ?? .infinity)")
  }
  let gpuGraph = compile { values in
    if firstProjectionOnly {
      let weights = blockWeights[0]
      return WDMetal.gemm(values, weight: weights.0, bias: weights.1)
    }
    var current = values
    for weights in blockWeights {
      let projected = WDMetal.gemm(WDMetal.mlpGelu(current, weight: weights.0, bias: weights.1),
                                   weight: weights.2, bias: weights.3)
      current = includePostNorm
        ? MLXFast.layerNorm(projected, weight: weights.4, bias: weights.5, eps: 1e-5) + current
        : projected + current
    }
    return current
  }
  for _ in 0..<3 { reference = gpuGraph(inputValues); eval(reference) }
  var aneTimes: [Double] = []
  var gpuTimes: [Double] = []
  for _ in 0..<iterations {
    var start = ProcessInfo.processInfo.systemUptime
    evaluateANE()
    aneTimes.append(ProcessInfo.processInfo.systemUptime - start)
    start = ProcessInfo.processInfo.systemUptime
    reference = gpuGraph(inputValues)
    eval(reference)
    gpuTimes.append(ProcessInfo.processInfo.systemUptime - start)
  }
  func median(_ samples: [Double]) -> Double {
    let sorted = samples.sorted()
    return sorted[sorted.count / 2]
  }
  report([
    "backend": "direct-ane-swin-mlp-fusion-probe",
    "deviceModel": machineIdentifier(),
    "os": ProcessInfo.processInfo.operatingSystemVersionString,
    "stage": stage, "batch": batch, "rows": fullRows, "rowsPerANECall": rows,
    "paddedRows": rows - fullRows, "blocks": Array(0..<blockCount),
    "note": firstProjectionOnly
      ? "First checkpoint MLP dense projection and bias only; this isolates BLOBFILE weights, 1x1 convolution layout, and output ordering."
      : "Checkpoint MLP tail(s) only. The attention and pre-MLP portions are omitted. One ANE program retains each MLP result for the next without a host/GPU intermediate.",
    "operation": firstProjectionOnly ? "fc1 dense projection plus bias" : "fc1 GELU fc2 plus residual and optional post-LayerNorm",
    "includesPostMLPLayerNorm": includePostNorm && !firstProjectionOnly,
    "milGraphHasSingleActivationInputAndSingleFinalOutput": true,
    "staticFP16WeightBlobBytes": weightBlobs.reduce(0) { $0 + $1.count },
    "compileAndLoadSeconds": compileSeconds,
    (firstProjectionOnly ? "maximumAbsoluteErrorAgainstMetalFC1Projection" : "maximumAbsoluteErrorAgainstMetalMLPChain"): differences.max() ?? 0,
    (firstProjectionOnly ? "meanAbsoluteErrorAgainstMetalFC1Projection" : "meanAbsoluteErrorAgainstMetalMLPChain"): Double(differences.reduce(0, +)) / Double(max(1, differences.count)),
    "aneMaximumAbsoluteOutput": aneMagnitudes.max() ?? 0,
    "metalMaximumAbsoluteOutput": expected.map(abs).max() ?? 0,
    "maximumErrorOutputIndex": maximumErrorIndex,
    "aneValueAtMaximumError": Float(Float16(bitPattern: validANE[maximumErrorIndex])),
    "metalValueAtMaximumError": expected[maximumErrorIndex],
    "aneMedianMilliseconds": median(aneTimes) * 1000,
    "metalMedianMilliseconds": median(gpuTimes) * 1000,
    "aneToMetalLatencyRatio": median(aneTimes) / median(gpuTimes),
    "estimatedFLOPsForMLPSubgraphs": firstProjectionOnly
      ? 2.0 * Double(rows) * Double(channels) * Double(hidden)
      : Double(blockCount) * 4.0 * Double(rows) * Double(channels) * Double(hidden),
    "timedIterations": iterations,
  ])
}

private func runInt8QKVProbe() throws {
  guard WDANEIsAvailable() else { fail("Apple's private ANE runtime classes are unavailable") }
  let stage = Int(option("--stage", default: "3")) ?? 0
  let batch = Int(option("--batch", default: "1")) ?? 0
  let iterations = Int(option("--iterations", default: "20")) ?? 0
  guard (1...4).contains(stage), [1, 2, 4, 8].contains(batch), (5...200).contains(iterations) else {
    fail("INT8 QKV probe requires --stage 1...4, --batch 1, 2, 4 or 8, and --iterations 5...200")
  }
  let channels = [128, 256, 512, 1024][stage - 1]
  let outputChannels = channels * 3
  let fullRows = [12544, 3136, 784, 196][stage - 1] * batch
  let paddingMultiple = max(0, Int(option("--pad-rows-to-multiple", default: "0")) ?? 0)
  let rows = paddingMultiple > 0
    ? ((fullRows + paddingMultiple - 1) / paddingMultiple) * paddingMultiple : fullRows
  let weightsURL = URL(fileURLWithPath: option(
    "--weights", default: "models/published/WD-SwinV2-Tagger-v3-Swift-MLX/model.safetensors"))
  let tensors = try loadArrays(url: weightsURL, stream: .cpu)
  let prefix = "layers.\(stage - 1).blocks.0.attn"
  guard let weight = tensors[prefix + ".qkv.weight"], weight.shape == [outputChannels, channels],
        let bias = tensors[prefix + "._sdpa_qkv_bias"], bias.size == outputChannels else {
    fail("checkpoint lacks the source QKV weight and bias for stage \(stage)")
  }
  let sourceWeights = weight.asArray(Float16.self)
  let maximumWeight = sourceWeights.map { abs(Float($0)) }.max() ?? 0
  guard maximumWeight > 0 else { fail("checkpoint QKV weights have zero magnitude") }
  let weightScale = maximumWeight / 127
  let quantizedWeights = sourceWeights.map { value in
    Int8(max(-127, min(127, Int((Float(value) / weightScale).rounded()))))
  }
  let biasBits = bias.asArray(Float16.self).map(\.bitPattern)
  let weightData = Data(quantizedWeights.map(UInt8.init(bitPattern:)))
  var weightHeader = [UInt8](repeating: 0, count: 128)
  weightHeader[0] = 1; weightHeader[4] = 2
  weightHeader[64] = 0xEF; weightHeader[65] = 0xBE; weightHeader[66] = 0xAD; weightHeader[67] = 0xDE
  weightHeader[68] = 1; weightHeader[74] = 0x08
  let weightByteCount = UInt32(weightData.count).littleEndian
  let weightDataOffset = UInt32(128).littleEndian
  withUnsafeBytes(of: weightByteCount) { weightHeader.replaceSubrange(72..<76, with: $0) }
  withUnsafeBytes(of: weightDataOffset) { weightHeader.replaceSubrange(80..<84, with: $0) }
  var weightBlob = Data(weightHeader)
  weightBlob.append(weightData)
  let biasBlob = makeWeightBlob(biasBits)
  let inputShape = "[1,\(channels),1,\(rows)]"
  let weightShape = "[\(outputChannels),\(channels),1,1]"
  let outputShape = "[1,\(outputChannels),1,\(rows)]"
  let generated = (0..<(rows * channels)).map {
    let row = $0 / channels
    return row < fullRows ? Float16(Float(sin(Double(($0 * 13 + 7) % 65521))) * 0.25) : 0
  }
  let activationScale = generated.map { abs(Float($0)) }.max()! / 127
  var inputBits = [UInt16](repeating: 0, count: rows * channels)
  for row in 0..<rows {
    for channel in 0..<channels {
      inputBits[channel * rows + row] = generated[row * channels + channel].bitPattern
    }
  }
  let mil = """
  program(1.3)
  [buildInfo = dict<string, string>({
    {"coremlc-component-MIL", "3510.2.1"},
    {"coremlc-version", "3505.4.1"},
    {"coremltools-component-milinternal", ""},
    {"coremltools-version", "9.0"}
  })]
  {
    func main<ios18>(tensor<fp16, \(inputShape)> x) {
      tensor<fp16, \(weightShape)> w = constexpr_affine_dequantize()[axis=int32(0), name=string("qkv_weight_int8"), quantized_data=tensor<int8, \(weightShape)>(BLOBFILE(path=string("@model_path/weights/qkv_int8.bin"), offset=uint64(64))), scale=fp16(\(weightScale)), zero_point=int8(0)];
      tensor<fp16, [1,\(outputChannels),1,1]> b = const()[name=string("qkv_bias"), val=tensor<fp16, [1,\(outputChannels),1,1]>(BLOBFILE(path=string("@model_path/weights/qkv_bias.bin"), offset=uint64(64)))];
      fp16 activation_scale = const()[name=string("activation_scale"), val=fp16(\(activationScale))];
      string int8_dtype = const()[name=string("int8_dtype"), val=string("int8")];
      tensor<int8, \(inputShape)> qx = quantize(input=x, output_dtype=int8_dtype, scale=activation_scale)[name=string("activation_quantize")];
      tensor<fp16, \(inputShape)> dx = dequantize(input=qx, scale=activation_scale)[name=string("activation_dequantize")];
      string pt = const()[name=string("pt"), val=string("valid")];
      tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
      tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
      tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
      int32 gr = const()[name=string("gr"), val=int32(1)];
      tensor<fp16, \(outputShape)> mm = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w, x=dx)[name=string("int8_weight_qkv_projection")];
      tensor<fp16, \(outputShape)> y = add(x=mm, y=b)[name=string("qkv_bias")];
    } -> (y);
  }
  """
  var compileError = [CChar](repeating: 0, count: 1024)
  let compileStart = ProcessInfo.processInfo.systemUptime
  let program = mil.withCString { source in
    "@model_path/weights/qkv_int8.bin".withCString { weightPath in
      "@model_path/weights/qkv_bias.bin".withCString { biasPath in
        weightBlob.withUnsafeBytes { weightBytes in
          biasBlob.withUnsafeBytes { biasBytes in
            compileError.withUnsafeMutableBufferPointer { errorBuffer in
              WDANECompileMILWithTwoWeightBlobs(
                source, weightPath,
                weightBytes.baseAddress!.assumingMemoryBound(to: UInt8.self), weightBlob.count,
                biasPath,
                biasBytes.baseAddress!.assumingMemoryBound(to: UInt8.self), biasBlob.count,
                errorBuffer.baseAddress, errorBuffer.count)
            }
          }
        }
      }
    }
  }
  guard let program else { fail("INT8 W8A8 QKV MIL compile/load failed", String(cString: compileError)) }
  defer { WDANEReleaseProgram(program) }
  let compileSeconds = ProcessInfo.processInfo.systemUptime - compileStart
  let inputBytes = inputBits.count * MemoryLayout<UInt16>.size
  let outputCount = rows * outputChannels
  let input = makeSurface(elements: inputBits.count, allocationBytes: inputBytes)
  let output = makeSurface(elements: outputCount, allocationBytes: outputCount * MemoryLayout<UInt16>.size)
  defer { WDANETensorRelease(input); WDANETensorRelease(output) }
  write(inputBits, to: input)
  func evaluateANE() {
    evaluate(program, inputs: [input], output: output)
  }
  let gpuInput = MLXArray(generated.map(Float.init), [rows, channels]).asType(.float16)
  let gpuGraph = compile { values in WDMetal.gemm(values, weight: weight, bias: bias) }
  var gpuOutput = gpuGraph(gpuInput)
  eval(gpuOutput)
  evaluateANE()
  var aneBits = [UInt16](repeating: 0, count: outputCount)
  var readError = [CChar](repeating: 0, count: 1024)
  let readOK = aneBits.withUnsafeMutableBufferPointer { values in
    readError.withUnsafeMutableBufferPointer { errorBuffer in
      WDANETensorReadF16(output, values.baseAddress!, values.count,
                         errorBuffer.baseAddress, errorBuffer.count)
    }
  }
  if !readOK { fail("cannot read INT8 QKV output", String(cString: readError)) }
  let gpuRowMajor = gpuOutput.asArray(Float16.self)
  var expected = [Float](repeating: 0, count: fullRows * outputChannels)
  var validANE = [UInt16](repeating: 0, count: fullRows * outputChannels)
  for row in 0..<fullRows {
    for channel in 0..<outputChannels {
      expected[channel * fullRows + row] = Float(gpuRowMajor[row * outputChannels + channel])
      validANE[channel * fullRows + row] = aneBits[channel * rows + row]
    }
  }
  let errors = zip(validANE, expected).map { abs(Float(Float16(bitPattern: $0.0)) - $0.1) }
  let finiteErrors = errors.filter(\.isFinite)
  let nonFiniteOutputCount = validANE.filter { !Float(Float16(bitPattern: $0)).isFinite }.count
  let maximumError = finiteErrors.max() ?? 0
  let meanError = finiteErrors.reduce(0, +) / Float(max(1, finiteErrors.count))
  for _ in 0..<3 { evaluateANE(); gpuOutput = gpuGraph(gpuInput); eval(gpuOutput) }
  var aneTimes: [Double] = []
  var gpuTimes: [Double] = []
  for _ in 0..<iterations {
    var start = ProcessInfo.processInfo.systemUptime
    evaluateANE()
    aneTimes.append(ProcessInfo.processInfo.systemUptime - start)
    start = ProcessInfo.processInfo.systemUptime
    gpuOutput = gpuGraph(gpuInput)
    eval(gpuOutput)
    gpuTimes.append(ProcessInfo.processInfo.systemUptime - start)
  }
  func median(_ samples: [Double]) -> Double {
    let values = samples.sorted()
    return values[values.count / 2]
  }
  report([
    "backend": "direct-ane-w8a8-qkv-probe",
    "deviceModel": machineIdentifier(),
    "os": ProcessInfo.processInfo.operatingSystemVersionString,
    "stage": stage, "batch": batch, "rows": fullRows, "rowsPerANECall": rows,
    "paddedRows": rows - fullRows,
    "operation": "static INT8 checkpoint QKV weights with constexpr affine dequantization plus calibrated INT8 activation quantize/dequantize and FP16 1x1 convolution",
    "weightTensor": prefix + ".qkv.weight",
    "weightQuantization": "symmetric per-tensor INT8; scale=max(abs(weight))/127; zero point 0",
    "activationCalibration": "synthetic sinusoidal FP16 stage input; one per-tensor scale=max(abs(input))/127; scale retained as FP16",
    "biasPrecision": "FP16",
    "activationScale": activationScale,
    "weightScale": weightScale,
    "quantizedWeightBlobBytes": weightBlob.count + biasBlob.count,
    "compileAndLoadSeconds": compileSeconds,
    "maximumAbsoluteErrorAgainstMetalFP16": maximumError,
    "meanAbsoluteErrorAgainstMetalFP16": meanError,
    "aneMedianMilliseconds": median(aneTimes) * 1000,
    "metalFP16ProjectionMedianMilliseconds": median(gpuTimes) * 1000,
    "aneToMetalFP16LatencyRatio": median(aneTimes).isFinite && median(gpuTimes) > 0
      ? median(aneTimes) / median(gpuTimes) : NSNull(),
    "nonFiniteANEOutputCount": nonFiniteOutputCount,
    "finiteOutputComparisonCount": finiteErrors.count,
    "timedIterations": iterations,
    "endToEndModelMeasurement": false,
    "measurementCaveat": "A selective QKV-only quantization feasibility test on real prepared checkpoint weights and synthetic activations. It does not establish tagging accuracy, calibration quality on image data, full-stage performance, or an end-to-end W8A8 speedup.",
  ])
}

if CommandLine.arguments.contains("--full-block") {
  do { try runFullBlockProbe(); exit(0) }
  catch { fail("complete-block probe failed", String(describing: error)) }
}
if CommandLine.arguments.contains("--attention") {
  do { try runAttentionProbe(); exit(0) }
  catch { fail("attention probe setup failed: \(error)") }
}
if CommandLine.arguments.contains("--fused-mlp") {
  do { try runFusedMLPProbe(); exit(0) }
  catch { fail("fused MLP probe setup failed: \(error)") }
}
if CommandLine.arguments.contains("--int8-qkv") {
  do { try runInt8QKVProbe(); exit(0) }
  catch { fail("INT8 QKV probe setup failed: \(error)") }
}

let stage = Int(option("--stage", default: "3")) ?? 0
let batch = Int(option("--batch", default: "1")) ?? 0
let iterations = Int(option("--iterations", default: "30")) ?? 0
let rowsPerCallOption = Int(option("--rows-per-call", default: "0")) ?? 0
let minimumSurfaceBytesOption = Int(option("--minimum-surface-bytes", default: "0")) ?? 0
let paddingMultiple = max(0, Int(option("--pad-rows-to-multiple", default: "0")) ?? 0)
let allowLargeErrorForDiagnostics = CommandLine.arguments.contains("--allow-large-error")
let bindWeightsAtCompileTime = CommandLine.arguments.contains("--bind-weights")
  || CommandLine.arguments.contains("--bind-weight-only")
let bindBiasAtCompileTime = CommandLine.arguments.contains("--bind-weights")
guard (1...4).contains(stage), (1...8).contains(batch), (5...500).contains(iterations) else {
  fail("use --stage 1...4, --batch 1...8, and --iterations 5...500")
}
guard WDANEIsAvailable() else { fail("Apple's private ANE runtime classes are unavailable") }

let channels = [128, 256, 512, 1024][stage - 1]
let rowsPerImage = [12544, 3136, 784, 196][stage - 1]
let fullRows = rowsPerImage * batch
let paddedFullRows = paddingMultiple > 0
  ? ((fullRows + paddingMultiple - 1) / paddingMultiple) * paddingMultiple : fullRows
let defaultRowsPerCall = (batch == 8 && (stage == 1 || stage == 2)) ? 12544 : paddedFullRows
let rows = rowsPerCallOption == 0 ? defaultRowsPerCall : rowsPerCallOption
guard rows > 0, rows <= paddedFullRows, paddedFullRows.isMultiple(of: rows) else {
  fail("--rows-per-call must divide the padded stage row count")
}
let rowChunks = paddedFullRows / rows
let inputChannels = channels
let outputChannels = 3 * channels
let inputElements = rows * inputChannels
let weightElements = inputChannels * outputChannels
let outputElements = rows * outputChannels
let fullOutputElements = fullRows * outputChannels
let outputBytes = outputElements * MemoryLayout<UInt16>.size
let sharedAllocationBytes = max(outputBytes, minimumSurfaceBytesOption)
let inputBytes = inputElements * MemoryLayout<UInt16>.size
let weightBytes = weightElements * MemoryLayout<UInt16>.size
let biasBytes = outputChannels * MemoryLayout<UInt16>.size
let weightsURL = URL(fileURLWithPath: option(
  "--weights", default: "models/published/WD-SwinV2-Tagger-v3-Swift-MLX/model.safetensors"))
let modelTensors = try loadArrays(url: weightsURL, stream: .cpu)
let attentionPrefix = "layers.\(stage - 1).blocks.0.attn"
guard let qkvWeightArray = modelTensors[attentionPrefix + ".qkv.weight"],
      qkvWeightArray.shape == [outputChannels, inputChannels],
      let qkvBiasArray = modelTensors[attentionPrefix + "._sdpa_qkv_bias"],
      qkvBiasArray.size == outputChannels,
      let queryScaleArray = modelTensors[attentionPrefix + "._sdpa_query_scale"],
      queryScaleArray.size == channels / 32 else {
  fail("could not find the prepared model's QKV weight and bias for Swin stage \(stage)")
}
let sourceWeights = qkvWeightArray.asArray(Float16.self)
let sourceBias = qkvBiasArray.asArray(Float16.self)
var transposedWeightBits = [UInt16](repeating: 0, count: weightElements)
for outputChannel in 0..<outputChannels {
  for inputChannel in 0..<inputChannels {
    transposedWeightBits[inputChannel * outputChannels + outputChannel] =
      sourceWeights[outputChannel * inputChannels + inputChannel].bitPattern
  }
}
let biasBits = sourceBias.map(\.bitPattern)

func makeBlobFile(_ values: [UInt16]) -> Data {
  let byteCount = UInt32(values.count * MemoryLayout<UInt16>.size).littleEndian
  let dataOffset = UInt32(128).littleEndian
  var header = [UInt8](repeating: 0, count: 128)
  header[0] = 1
  header[4] = 2
  header[64] = 0xEF; header[65] = 0xBE; header[66] = 0xAD; header[67] = 0xDE
  header[68] = 1
  withUnsafeBytes(of: byteCount) { header.replaceSubrange(72..<76, with: $0) }
  withUnsafeBytes(of: dataOffset) { header.replaceSubrange(80..<84, with: $0) }
  var blob = Data(header)
  values.withUnsafeBytes { blob.append(contentsOf: $0) }
  return blob
}
let convolutionWeightBits = sourceWeights.map(\.bitPattern)
let qkvWeightBlob = bindWeightsAtCompileTime ? makeBlobFile(convolutionWeightBits) : Data()
let qkvBiasBlob = bindBiasAtCompileTime ? makeBlobFile(biasBits) : Data()
let weightBlobPath = "@model_path/weights/qkv.bin"
let biasBlobPath = "@model_path/weights/bias.bin"
let functionInputs: String
if bindWeightsAtCompileTime {
  functionInputs = bindBiasAtCompileTime
    ? "tensor<fp16, [1,\(inputChannels),1,\(rows)]> x"
    : "tensor<fp16, [1,\(outputChannels),1,1]> b,\n                  tensor<fp16, [1,\(inputChannels),1,\(rows)]> x"
} else {
  functionInputs = "tensor<fp16, [1,1,\(inputChannels),\(outputChannels)]> a,\n                  tensor<fp16, [1,1,1,\(outputChannels)]> b,\n                  tensor<fp16, [1,1,\(rows),\(inputChannels)]> x"
}
let modelWeights = """
    \(bindWeightsAtCompileTime ? "tensor<fp16, [\(outputChannels),\(inputChannels),1,1]> qkv_weight = const()[name=string(\"qkv_weight\"), val=tensor<fp16, [\(outputChannels),\(inputChannels),1,1]>(BLOBFILE(path=string(\"\(weightBlobPath)\"), offset=uint64(64)))];" : "")
    \(bindBiasAtCompileTime ? "tensor<fp16, [1,\(outputChannels),1,1]> qkv_bias = const()[name=string(\"qkv_bias\"), val=tensor<fp16, [1,\(outputChannels),1,1]>(BLOBFILE(path=string(\"\(biasBlobPath)\"), offset=uint64(64)))];" : "")
  """
let milWeightName = bindWeightsAtCompileTime ? "qkv_weight" : "a"
let milBiasName = bindBiasAtCompileTime ? "qkv_bias" : "b"
let milOperations = bindWeightsAtCompileTime ? """
    string qkv_pt = const()[name=string("qkv_pt"), val=string("valid")];
    tensor<int32, [2]> qkv_st = const()[name=string("qkv_st"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> qkv_pd = const()[name=string("qkv_pd"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [2]> qkv_dl = const()[name=string("qkv_dl"), val=tensor<int32, [2]>([1,1])];
    int32 qkv_gr = const()[name=string("qkv_gr"), val=int32(1)];
    tensor<fp16, [1,\(outputChannels),1,\(rows)]> mm = conv(
      dilations=qkv_dl, groups=qkv_gr, pad=qkv_pd, pad_type=qkv_pt,
      strides=qkv_st, weight=qkv_weight, x=x)[name=string("conv")];
    tensor<fp16, [1,\(outputChannels),1,\(rows)]> y = add(x=mm, y=\(milBiasName))[name=string("biased")];
  """ : """
    bool qkv_tx = const()[name=string("qkv_tx"), val=bool(false)];
    bool qkv_ty = const()[name=string("qkv_ty"), val=bool(false)];
    tensor<fp16, [1,1,\(rows),\(outputChannels)]> mm = matmul(
      transpose_x=qkv_tx, transpose_y=qkv_ty, x=x, y=\(milWeightName))[name=string("matmul")];
    tensor<fp16, [1,1,\(rows),\(outputChannels)]> y = add(x=mm, y=\(milBiasName))[name=string("biased")];
  """

let mil = """
program(1.3)
[buildInfo = dict<string, string>({
  {"coremlc-component-MIL", "3510.2.1"},
  {"coremlc-version", "3505.4.1"},
  {"coremltools-component-milinternal", ""},
  {"coremltools-version", "9.0"}
})]
{
  func main<ios18>(\(functionInputs)) {
    \(modelWeights)
    \(milOperations)
  } -> (y);
}
"""

var compileError = [CChar](repeating: 0, count: 1024)
let compileStart = ProcessInfo.processInfo.systemUptime
let compiledProgram: ANEProgram? = mil.withCString { source in
  compileError.withUnsafeMutableBufferPointer { errorBuffer in
    if bindWeightsAtCompileTime {
      return weightBlobPath.withCString { weightPath in
        qkvWeightBlob.withUnsafeBytes { weightBytes in
          if bindBiasAtCompileTime {
            return biasBlobPath.withCString { biasPath in
              qkvBiasBlob.withUnsafeBytes { biasBytes in
                WDANECompileMILWithTwoWeightBlobs(
                  source, weightPath,
                  weightBytes.baseAddress!.assumingMemoryBound(to: UInt8.self), qkvWeightBlob.count,
                  biasPath, biasBytes.baseAddress!.assumingMemoryBound(to: UInt8.self), qkvBiasBlob.count,
                  errorBuffer.baseAddress, errorBuffer.count)
              }
            }
          }
          return WDANECompileMILWithOneWeightBlob(
            source, weightPath,
            weightBytes.baseAddress!.assumingMemoryBound(to: UInt8.self), qkvWeightBlob.count,
            errorBuffer.baseAddress, errorBuffer.count)
        }
      }
    }
    return WDANECompileMIL(source, errorBuffer.baseAddress, errorBuffer.count)
  }
}
guard let program = compiledProgram else {
  fail("MIL compilation or load failed", String(cString: compileError))
}
defer { WDANEReleaseProgram(program) }
let compileSeconds = ProcessInfo.processInfo.systemUptime - compileStart

let weights: ANETensor? = bindWeightsAtCompileTime
  ? nil : makeSurface(elements: weightElements, allocationBytes: sharedAllocationBytes)
let bias: ANETensor? = bindBiasAtCompileTime
  ? nil : makeSurface(elements: outputChannels, allocationBytes: sharedAllocationBytes)
let input = makeSurface(elements: inputElements, allocationBytes: sharedAllocationBytes)
let output = makeSurface(elements: outputElements, allocationBytes: sharedAllocationBytes)
defer {
  if let weights { WDANETensorRelease(weights) }
  if let bias { WDANETensorRelease(bias) }
  WDANETensorRelease(input)
  WDANETensorRelease(output)
}
let inputHalf = Float16(0.5).bitPattern
if let weights { write(transposedWeightBits, to: weights) }
if let bias { write(biasBits, to: bias) }
write([UInt16](repeating: inputHalf, count: inputElements), to: input)
let evaluationInputs: [ANETensor]
if bindWeightsAtCompileTime {
  evaluationInputs = bindBiasAtCompileTime ? [input] : [bias!, input]
} else {
  evaluationInputs = [weights!, bias!, input]
}
let usedSurfaces = [input, output] + [weights, bias].compactMap { $0 }
let surfaceAllocationBytes = WDANETensorAllocationBytes(output)
let allIOSurfacesEqualAllocation = usedSurfaces.allSatisfy {
  WDANETensorAllocationBytes($0) == surfaceAllocationBytes
}

// The MIL signature and IOSurface order stay fixed for every evaluation.
func evaluateFullProjection(capture: Bool = false, into capturedOutput: inout [UInt16]) {
  for chunk in 0..<rowChunks {
    evaluate(program, inputs: evaluationInputs, output: output)
    if capture {
      var tile = [UInt16](repeating: 0, count: outputElements)
      var readError = [CChar](repeating: 0, count: 1024)
      let readOK = tile.withUnsafeMutableBufferPointer { buffer in
        readError.withUnsafeMutableBufferPointer { errorBuffer in
          WDANETensorReadF16(output, buffer.baseAddress!, buffer.count,
                             errorBuffer.baseAddress, errorBuffer.count)
        }
      }
      if !readOK { fail("IOSurface read failed", String(cString: readError)) }
      if bindWeightsAtCompileTime {
        var rowMajor = [UInt16](repeating: 0, count: outputElements)
        for row in 0..<rows {
          for channel in 0..<outputChannels {
            rowMajor[row * outputChannels + channel] = tile[channel * rows + row]
          }
        }
        tile = rowMajor
      }
      let startRow = chunk * rows
      let validRows = min(rows, max(0, fullRows - startRow))
      let validCount = validRows * outputChannels
      let start = startRow * outputChannels
      capturedOutput.replaceSubrange(start..<(start + validCount), with: tile.prefix(validCount))
    }
  }
}
var discardedOutput: [UInt16] = []
for _ in 0..<3 { evaluateFullProjection(into: &discardedOutput) }

let gpuFullInputElements = fullRows * inputChannels
let gpuInput = MLXArray([Float](repeating: 0.5, count: gpuFullInputElements), [fullRows, inputChannels])
  .asType(.float16)
let gpuTile: WDMetal.GEMMTile = stage == 3 ? .m64n64k32 : .baseline
let gpuGraph = compile { values in
  WDMetal.gemm(values, weight: qkvWeightArray, bias: qkvBiasArray, tile: gpuTile)
}
var gpuOutput = gpuGraph(gpuInput)
eval(gpuOutput)
let gpuProductionGraph = compile { values in
  let qkv = WDMetal.projectQKV(values, weight: qkvWeightArray, bias: qkvBiasArray,
                               scale: queryScaleArray.reshaped(-1), heads: channels / 32, tile: gpuTile)
  return concatenated(qkv.map { $0.reshaped(fullRows, inputChannels) }, axis: 1)
}
var gpuProductionOutput = gpuProductionGraph(gpuInput)
eval(gpuProductionOutput)
for _ in 0..<3 {
  evaluateFullProjection(into: &discardedOutput)
  gpuOutput = gpuGraph(gpuInput)
  eval(gpuOutput)
  gpuProductionOutput = gpuProductionGraph(gpuInput)
  eval(gpuProductionOutput)
}

var outputValues = [UInt16](repeating: 0, count: fullOutputElements)
evaluateFullProjection(capture: true, into: &outputValues)
var finiteOutputs = 0
for bits in outputValues {
  let actual = Float(Float16(bitPattern: bits))
  guard actual.isFinite else { continue }
  finiteOutputs += 1
}
let gpuValues = gpuOutput.asArray(Float.self)
var gpuFiniteOutputs = 0
for value in gpuValues {
  guard value.isFinite else { continue }
  gpuFiniteOutputs += 1
}
var productionGPUFiniteOutputs = 0
for value in gpuProductionOutput.asArray(Float.self) {
  guard value.isFinite else { continue }
  productionGPUFiniteOutputs += 1
}
guard finiteOutputs == fullOutputElements,
      gpuFiniteOutputs == fullOutputElements,
      gpuValues.count == fullOutputElements,
      productionGPUFiniteOutputs == fullOutputElements else {
  fail("ANE or optimized Metal returned a non-finite or incomplete QKV projection (ANE \(finiteOutputs)/\(fullOutputElements), Metal GEMM \(gpuFiniteOutputs)/\(gpuValues.count), fused QKV \(productionGPUFiniteOutputs))")
}
var maximumAbsoluteError = Float.zero
var maximumErrorIndex = 0
var totalAbsoluteError = Double.zero
for index in 0..<outputValues.count {
  let aneValue = Float(Float16(bitPattern: outputValues[index]))
  let metalValue = gpuValues[index]
  let error = abs(aneValue - metalValue)
  if error > maximumAbsoluteError {
    maximumAbsoluteError = error
    maximumErrorIndex = index
  }
  totalAbsoluteError += Double(error)
}
guard maximumAbsoluteError <= 0.05 || allowLargeErrorForDiagnostics else {
  let aneValue = Float(Float16(bitPattern: outputValues[maximumErrorIndex]))
  let channel = maximumErrorIndex % outputChannels
  let expected = sourceWeights[(channel * inputChannels)..<((channel + 1) * inputChannels)]
    .reduce(Float(sourceBias[channel])) { $0 + Float($1) * 0.5 }
  fail("ANE and optimized Metal QKV projections differ by \(maximumAbsoluteError) at output \(maximumErrorIndex), channel \(channel) (ANE \(aneValue), Metal \(gpuValues[maximumErrorIndex]), source-weight reference \(expected), bias \(sourceBias[channel]))")
}

var aneDurations: [Double] = []
var gpuDurations: [Double] = []
var productionGpuDurations: [Double] = []
aneDurations.reserveCapacity(iterations)
gpuDurations.reserveCapacity(iterations)
productionGpuDurations.reserveCapacity(iterations)
for _ in 0..<iterations {
  let start = ProcessInfo.processInfo.systemUptime
  evaluateFullProjection(into: &discardedOutput)
  aneDurations.append(ProcessInfo.processInfo.systemUptime - start)
  let gpuStart = ProcessInfo.processInfo.systemUptime
  gpuOutput = gpuGraph(gpuInput)
  eval(gpuOutput)
  gpuDurations.append(ProcessInfo.processInfo.systemUptime - gpuStart)
  let productionStart = ProcessInfo.processInfo.systemUptime
  gpuProductionOutput = gpuProductionGraph(gpuInput)
  eval(gpuProductionOutput)
  productionGpuDurations.append(ProcessInfo.processInfo.systemUptime - productionStart)
}
func percentiles(_ values: [Double]) -> (median: Double, p95: Double) {
  let ordered = values.sorted()
  return (ordered[ordered.count / 2], ordered[min(ordered.count - 1, Int(Double(ordered.count - 1) * 0.95))])
}
let aneTiming = percentiles(aneDurations)
let gpuTiming = percentiles(gpuDurations)
let productionGpuTiming = percentiles(productionGpuDurations)
let result: [String: Any] = [
  "backend": "direct-ane-probe-with-optimized-metal-control",
  "executionAPI": "_ANEInMemoryModel.evaluateWithQoS",
  "coreMLUsed": false,
  "directANEPathUsesMetal": false,
  "gpuControlUsedMetal": true,
  "gpuControl": "WDMetal.gemm projection-only diagnostic using the existing M3 profile tile geometry; not the fused production QKV kernel",
  "gpuControlTile": gpuTile.rawValue,
  "productionGPUControl": "WDMetal.projectQKV fused projection, cosine normalization, and query scale using the existing M3 profile tile geometry",
  "productionGPUControlUsedMetal": true,
  "productionGPUDispatchMilliseconds": productionGpuDurations.map { $0 * 1000 },
  "productionGPUMedianDispatchMilliseconds": productionGpuTiming.median * 1000,
  "productionGPUP95DispatchMilliseconds": productionGpuTiming.p95 * 1000,
  "diagnosticANEToProductionGPUMedianLatencyRatioNotLikeForLike": aneTiming.median / productionGpuTiming.median,
  "comparisonNotLikeForLike": true,
  "comparisonCaveat": "Direct ANE measures only QKV FP16 dense projection plus bias (matmul or 1x1 convolution). The production Metal control also performs cosine normalization and query scaling. This ratio is diagnostic and is not a model speedup.",
  "deviceModel": machineIdentifier(),
  "os": ProcessInfo.processInfo.operatingSystemVersionString,
  "stage": stage,
  "batch": batch,
  "operation": bindWeightsAtCompileTime
    ? "stage QKV FP16 1x1 convolution plus FP16 bias, before cosine normalization"
    : "stage QKV FP16 matrix multiply plus FP16 bias, before cosine normalization",
  "inputShape": bindWeightsAtCompileTime ? [1, inputChannels, 1, rows] : [1, 1, rows, inputChannels],
  "fullInputShape": bindWeightsAtCompileTime ? [1, inputChannels, 1, fullRows] : [1, 1, fullRows, inputChannels],
  "paddedInputShape": bindWeightsAtCompileTime
    ? [1, inputChannels, 1, paddedFullRows] : [1, 1, paddedFullRows, inputChannels],
  "paddedRowsTotal": paddedFullRows,
  "paddingRows": paddedFullRows - fullRows,
  "rowChunksPerProjection": rowChunks,
  "rowsPerCall": rows,
  "weightShape": bindWeightsAtCompileTime
    ? [outputChannels, inputChannels, 1, 1] : [1, 1, inputChannels, outputChannels],
  "outputShape": bindWeightsAtCompileTime
    ? [1, outputChannels, 1, fullRows] : [1, 1, fullRows, outputChannels],
  "aneOutputShapePerCall": bindWeightsAtCompileTime
    ? [1, outputChannels, 1, rows] : [1, 1, rows, outputChannels],
  "operationFLOPs": 2 * fullRows * inputChannels * outputChannels,
  "executedOperationFLOPs": 2 * rows * rowChunks * inputChannels * outputChannels,
  "inputBytes": fullRows * inputChannels * MemoryLayout<UInt16>.size,
  "inputSurfaceBytesPerProjection": inputBytes * rowChunks,
  "weightBytes": weightBytes,
  "outputBytes": fullOutputElements * MemoryLayout<UInt16>.size,
  "outputSurfaceBytesPerProjection": outputBytes * rowChunks,
  "iosurfaceAllocationBytes": usedSurfaces.count * surfaceAllocationBytes,
  "allIOSurfacesEqualAllocation": allIOSurfacesEqualAllocation,
  "dynamicWeightInput": !bindWeightsAtCompileTime,
  "dynamicBiasInput": !bindBiasAtCompileTime,
  "compileTimeBLOBFILEWeights": bindWeightsAtCompileTime,
  "compileTimeBLOBFILEBias": bindBiasAtCompileTime,
  "weightBindingLowering": bindWeightsAtCompileTime ? "constant 1x1 convolution" : "dynamic IOSurface matmul",
  "weightBlobBytes": bindWeightsAtCompileTime ? qkvWeightBlob.count : 0,
  "biasBlobBytes": bindBiasAtCompileTime ? qkvBiasBlob.count : 0,
  "aneInputSurfaceCount": evaluationInputs.count,
  "weightsFromPreparedCheckpoint": true,
  "productionWorkerEndToEnd": false,
  "peakRSSMeasured": false,
  "weightSource": weightsURL.path,
  "weightTensor": attentionPrefix + ".qkv.weight",
  "weightOrientation": bindWeightsAtCompileTime
    ? "checkpoint [output,input] used as ANE 1x1 convolution [output,input,1,1]"
    : "checkpoint [output,input] transposed once to ANE matmul [input,output]",
  "biasTensor": attentionPrefix + "._sdpa_qkv_bias",
  "queryScaleTensor": attentionPrefix + "._sdpa_query_scale",
  "inputValues": "constant 0.5 FP16; diagnostic activations, not image data",
  "finiteOutputs": finiteOutputs,
  "aneMetalMaximumAbsoluteError": maximumAbsoluteError,
  "aneMetalMaximumErrorIndex": maximumErrorIndex,
  "aneMetalMeanAbsoluteError": totalAbsoluteError / Double(outputValues.count),
  "gpuFiniteOutputs": gpuFiniteOutputs,
  "productionGPUFiniteOutputs": productionGPUFiniteOutputs,
  "compileAndLoadSeconds": compileSeconds,
  "warmupDispatchesPerBackend": 3,
  "timedDispatches": iterations,
  "interleavedDispatchOrder": "ANE projection, Metal projection-only diagnostic, then fused production Metal QKV for each timed round",
  "aneDispatchMilliseconds": aneDurations.map { $0 * 1000 },
  "aneMedianDispatchMilliseconds": aneTiming.median * 1000,
  "aneP95DispatchMilliseconds": aneTiming.p95 * 1000,
  "aneUsefulThroughputTFLOPs": (2.0 * Double(fullRows) * Double(inputChannels) * Double(outputChannels)) / aneTiming.median / 1.0e12,
  "metalDispatchMilliseconds": gpuDurations.map { $0 * 1000 },
  "metalMedianDispatchMilliseconds": gpuTiming.median * 1000,
  "metalP95DispatchMilliseconds": gpuTiming.p95 * 1000,
  "aneProjectionToMetalGemmMedianLatencyRatio": aneTiming.median / gpuTiming.median,
  "method": "Fixed stage geometry; one direct-ANE compile/load, three warmups per backend, then interleaved synchronous full-projection direct ANE (row-chunked when requested) and existing WDMetal GEMM dispatches. IOSurface writes and final output reads are excluded from dispatch timing. BLOBFILE weights are compile-time-bound when --bind-weights is set; persistent program caching and end-to-end inference are not measured.",
]
let json = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
let reportPath = CommandLine.arguments.firstIndex(of: "--report").flatMap { $0 + 1 < CommandLine.arguments.count ? CommandLine.arguments[$0 + 1] : nil }
if let reportPath {
  try FileManager.default.createDirectory(at: URL(fileURLWithPath: reportPath).deletingLastPathComponent(),
                                          withIntermediateDirectories: true)
  try json.write(to: URL(fileURLWithPath: reportPath))
}
FileHandle.standardOutput.write(json)
FileHandle.standardOutput.write(Data("\n".utf8))
