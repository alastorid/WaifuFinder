import Darwin
import Foundation
import MLX
import WDTaggerMLX

private func footprint() -> UInt64 {
  var info = task_vm_info_data_t()
  var count = mach_msg_type_number_t(
    MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
  let status = withUnsafeMutablePointer(to: &info) { p in
    p.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
      task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
    }
  }
  return status == KERN_SUCCESS ? info.phys_footprint : 0
}
private final class PeakMemory: @unchecked Sendable {
  let lock = NSLock()
  var peak: UInt64 = 0
  let timer = DispatchSource.makeTimerSource(queue: DispatchQueue(label: "wd.native.footprint"))
  init() {
    timer.schedule(deadline: .now(), repeating: .milliseconds(10))
    timer.setEventHandler { [weak self] in
      guard let self else { return }
      let value = footprint()
      self.lock.lock()
      self.peak = max(self.peak, value)
      self.lock.unlock()
    }
    timer.resume()
  }
  deinit { timer.cancel() }
  var bytes: UInt64 {
    lock.lock()
    defer { lock.unlock() }
    return max(peak, footprint())
  }
}
private func check(_ name: String, _ actual: MLXArray, _ expected: MLXArray, tolerance: Float)
  throws -> [String: Any]
{
  guard actual.shape == expected.shape else {
    throw WDNativeError.invalid("Shape mismatch: \(name)")
  }
  let difference = abs(actual.asType(.float32) - expected.asType(.float32))
  let maximum = MLX.max(difference).item(Float.self)
  let average = mean(difference).item(Float.self)
  guard maximum.isFinite && maximum <= tolerance else {
    throw WDNativeError.invalid("\(name): max error \(maximum), limit \(tolerance)")
  }
  return [
    "name": name, "maximumAbsoluteError": maximum, "meanAbsoluteError": average, "limit": tolerance,
  ]
}
private func deterministic(_ shape: [Int], scale: Float = 1) -> MLXArray {
  let count = shape.reduce(1, *)
  return MLXArray((0..<count).map { Float(sin(Double(($0 * 13 + 7) % 65521))) * scale }, shape)
}
private func kernelChecks(tiles: [WDMetal.GEMMTile] = [.baseline]) throws -> [[String: Any]] {
  var results: [[String: Any]] = []
  let plane = 448 * 448
  let pixels = (0..<3 * plane).map { Float((($0 % plane) * 37 + ($0 / plane) * 83 + 11) % 256) }
  let image = MLXArray(pixels, [1, 3, 448, 448])
  var patches: [Float] = []
  for grid in WDNativeTagger.windowOrder(side: 112, shift: 0) {
    for dh in 0..<4 {
      for dw in 0..<4 {
        for channel in 0..<3 {
          let i = channel * plane + (grid / 112 * 4 + dh) * 448 + grid % 112 * 4 + dw
          patches.append(pixels[i] * Float(2.0 / 255.0) - Float(1))
        }
      }
    }
  }
  results.append(
    try check(
      "input-normalize-patch-window", WDMetal.inputPatches(image),
      MLXArray(patches, [112 * 112, 48]).asType(.float16), tolerance: 0.001))
  var rgba = [UInt8](repeating: 0, count: 2 * plane * 4)
  for batch in 0..<2 {
    for pixel in 0..<plane {
      for channel in 0..<3 {
        let value = UInt8(pixels[channel * plane + pixel])
        rgba[(batch * plane + pixel) * 4 + channel] = batch == 0 ? value : 255 - value
      }
      rgba[(batch * plane + pixel) * 4 + 3] = UInt8(pixel % 256)
    }
  }
  let floatBatch = concatenated([image, Float(255) - image], axis: 0)
  results.append(try check("rgba-byte-input-exact-two-batches",
    WDMetal.inputPatches(MLXArray(rgba, [2, 448, 448, 4])),
    WDMetal.inputPatches(floatBatch), tolerance: 0))
  let oldSide = 56
  let newSide = 28
  let channels = 128
  let order = WDNativeTagger.windowOrder(side: oldSide, shift: 7)
  var inverse = Array(repeating: 0, count: oldSide * oldSide)
  for (i, j) in order.enumerated() { inverse[j] = i }
  var mergeRoute: [Int32] = []
  var sourceRows: [Int32] = []
  for grid in WDNativeTagger.windowOrder(side: newSide, shift: 0) {
    for (dh, dw) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
      let source = (grid / newSide * 2 + dh) * oldSide + grid % newSide * 2 + dw
      mergeRoute.append(Int32(inverse[source]))
      sourceRows.append(Int32(source))
    }
  }
  let grid = deterministic([2, oldSide * oldSide, channels]).asType(.float16)
  let windowed = take(grid, MLXArray(order.map(Int32.init)), axis: 1).reshaped(-1, channels)
  let merged = WDMetal.patchMerge(windowed, route: MLXArray(mergeRoute), spatial: newSide * newSide)
  results.append(
    try check(
      "patch-merge-window-transition", merged,
      take(grid, MLXArray(sourceRows), axis: 1).reshaped(-1, 4 * channels), tolerance: 0))
  for channels in [128, 256, 512, 1024] {
    let x = deterministic([392, channels]).asType(.float16)
    let residual = deterministic([392, channels], scale: 0.25).asType(.float16)
    let weight = deterministic([channels], scale: 0.5).asType(.float16)
    let bias = deterministic([channels], scale: 0.1).asType(.float16)
    let indices = (0..<196).reversed().map(Int32.init)
    let route = MLXArray(indices)
    let expected = (MLXFast.layerNorm(x, weight: weight, bias: bias, eps: 1e-5) + residual)
      .reshaped(2, 196, channels)
    let actual = WDMetal.normalize(
      x, residual: residual, weight: weight, bias: bias, route: route, spatial: 196)
    results.append(
      try check(
        "norm-route-\(channels)", actual, take(expected, route, axis: 1).reshaped(392, channels),
        tolerance: 0.003))
    results.append(
      try check(
        "simd-norm-route-\(channels)",
        WDMetal.normalize(
          x, residual: residual, weight: weight, bias: bias, route: route,
          spatial: 196, simdRows: true),
        take(expected, route, axis: 1).reshaped(392, channels), tolerance: 0.003))
    let constant = MLXArray.full([392, channels], values: MLXArray(Float(1234)), dtype: .float16)
    results.append(
      try check(
        "constant-norm-\(channels)",
        WDMetal.normalize(
          constant, residual: nil, weight: weight, bias: bias, route: route, spatial: 196),
        broadcast(bias, to: [392, channels]), tolerance: 0))
    results.append(
      try check(
        "constant-simd-norm-\(channels)",
        WDMetal.normalize(
          constant, residual: nil, weight: weight, bias: bias, route: route,
          spatial: 196, simdRows: true),
        broadcast(bias, to: [392, channels]), tolerance: 0))
    let matrix = deterministic([4 * channels, channels], scale: Float(1 / sqrt(Double(channels))))
      .asType(.float16)
    let offset = deterministic([4 * channels], scale: 0.1).asType(.float16)
    let projected = addMM(offset.asType(.float32), x.asType(.float32), matrix.asType(.float32).T)
    let cubicCoefficient = MLXArray(Float(0.044715))
    let tanhScale = MLXArray(Float(0.7978845608028654))
    let half = MLXArray(Float(0.5))
    let one = MLXArray(Float(1))
    let cubicTerm = ((projected * cubicCoefficient) * projected) * projected
    let tanhInput = (projected + cubicTerm) * tanhScale
    let halfProjected = projected * half
    let onePlusTanh = one + tanh(tanhInput)
    let gelu = halfProjected * onePlusTanh
    for tile in tiles {
      results.append(
        try check(
          "half-gemm-\(channels)-\(tile.rawValue)",
          WDMetal.gemm(x, weight: matrix, bias: offset, tile: tile),
          projected.asType(.float16), tolerance: 0.008))
      results.append(
        try check(
          "fused-mlp-\(channels)-\(tile.rawValue)",
          WDMetal.mlpGelu(x, weight: matrix, bias: offset, tile: tile),
          gelu.asType(.float16), tolerance: 0.008))
    }
  }
  for (heads, windows, shifted) in [(4, 4, true), (32, 1, false)] {
    let packed = deterministic([windows * 196, 3 * heads * 32]).asType(.float16)
    let scale = MLXArray.full([heads], values: MLXArray(Float(5.656854)), dtype: .float16)
    let projectionInput = deterministic([windows * 196, heads * 32]).asType(.float16)
    let projectionWeight = deterministic(
      [3 * heads * 32, heads * 32],
      scale: Float(1 / sqrt(Double(heads * 32)))
    ).asType(.float16)
    let projectionBias = deterministic([3 * heads * 32], scale: 0.1).asType(.float16)
    let fusedProjection = WDMetal.projectQKV(
      projectionInput, weight: projectionWeight,
      bias: projectionBias, scale: scale, heads: heads)
    let separateProjection = WDMetal.qkv(
      WDMetal.gemm(projectionInput, weight: projectionWeight, bias: projectionBias),
      scale: scale, heads: heads)
    for index in 0..<3 {
      results.append(
        try check(
          "fused-qkv-project-\(heads)-\(index)",
          fusedProjection[index], separateProjection[index], tolerance: index == 2 ? 0 : 0.004))
    }
    let occupancyProjection = WDMetal.projectQKV(
      projectionInput, weight: projectionWeight,
      bias: projectionBias, scale: scale, heads: heads, tile: .occupancy)
    for index in 0..<3 {
      results.append(
        try check(
          "fused-qkv-occupancy-\(heads)-\(index)",
          occupancyProjection[index], separateProjection[index], tolerance: index == 2 ? 0 : 0.004))
    }
    let qkv = WDMetal.qkv(packed, scale: scale, heads: heads)
    let raw = packed.reshaped(windows, 196, 3, heads, 32).transposed(2, 0, 3, 1, 4)
    func normalize(_ a: MLXArray) -> MLXArray {
      let full = a.asType(.float32)
      return
        (full / maximum(sqrt(sum(full * full, axis: -1, keepDims: true)), MLXArray(Float(1e-12))))
        .asType(.float16)
    }
    let referenceQ = normalize(raw[0]) * scale.reshaped(1, heads, 1, 1)
    results.append(try check("q-pack-\(heads)", qkv[0], referenceQ, tolerance: 0.002))
    results.append(try check("k-pack-\(heads)", qkv[1], normalize(raw[1]), tolerance: 0.001))
    results.append(try check("v-pack-\(heads)", qkv[2], raw[2], tolerance: 0))
    let bias = deterministic([1, heads, 196, 196], scale: 0.2).asType(.float16)
    let regions = WDNativeTagger.regions(side: windows == 4 ? 28 : 14)
    var scores =
      matmul(qkv[0].asType(.float32), qkv[1].asType(.float32).transposed(0, 1, 3, 2))
      * Float(1 / sqrt(32.0)) + bias.asType(.float32)
    if shifted {
      let ids = regions.reshaped(windows, 196)
      let mask =
        (ids.expandedDimensions(axis: 2) .!= ids.expandedDimensions(axis: 1)).asType(.float32)
        * Float(-100)
      scores = scores + mask.expandedDimensions(axis: 1)
    }
    let expected = matmul(softmax(scores, axis: -1), qkv[2].asType(.float32)).transposed(0, 2, 1, 3)
      .reshaped(-1, heads * 32).asType(.float16)
    results.append(
      try check(
        "flash32-\(heads)-shift\(shifted)",
        WDMetal.flash(
          qkv[0], qkv[1], qkv[2], bias: bias, regions: regions, windows: windows, shifted: shifted),
        expected, tolerance: 0.006))
    let relative = deterministic([heads, 729], scale: 0.2).asType(.float16)
    var deltas: [Int32] = []
    deltas.reserveCapacity(196 * 196)
    for query in 0..<196 {
      let queryRow = query / 14
      let queryColumn = query % 14
      for key in 0..<196 {
        let rowDelta = queryRow - key / 14
        let columnDelta = queryColumn - key % 14
        let relativeBiasIndex = (rowDelta + 13) * 27 + columnDelta + 13
        deltas.append(Int32(relativeBiasIndex))
      }
    }
    let expanded = take(relative, MLXArray(deltas), axis: 1).reshaped(1, heads, 196, 196)
    results.append(
      try check(
        "compact-bias-flash32-\(heads)-shift\(shifted)",
        WDMetal.flash(
          qkv[0], qkv[1], qkv[2], bias: relative, regions: regions,
          windows: windows, shifted: shifted, compactBias: true),
        WDMetal.flash(
          qkv[0], qkv[1], qkv[2], bias: expanded, regions: regions,
          windows: windows, shifted: shifted), tolerance: 0))
    let zeros = MLXArray.zeros(qkv[0].shape, dtype: .float16)
    let zeroExpected = matmul(softmax(scores * Float(0), axis: -1), zeros.asType(.float32))
      .transposed(0, 2, 1, 3).reshaped(-1, heads * 32)
    results.append(
      try check(
        "zero-flash-\(heads)",
        WDMetal.flash(
          zeros, zeros, zeros, bias: bias, regions: regions, windows: windows, shifted: shifted),
        zeroExpected, tolerance: 0))
  }
  return results
}

/// Times real stage shapes with rotated candidate order. No trace is recorded.
private func benchmarkGEMM(batch: Int) throws -> [String: Any] {
  guard [2, 8].contains(batch) else {
    throw WDNativeError.invalid("Benchmark batch must be 2 or 8")
  }
  let thermalBefore = ProcessInfo.processInfo.thermalState.rawValue
  var records: [[String: Any]] = []
  for (stage, side) in [112, 56, 28, 14].enumerated() {
    let channels = [128, 256, 512, 1024][stage]
    let rows = batch * side * side
    for (name, k, n, activation) in [
      ("qkv", channels, 3 * channels, false),
      ("proj", channels, channels, false),
      ("fc1", channels, 4 * channels, true),
      ("fc2", 4 * channels, channels, false),
    ] {
      autoreleasepool {
        let x = deterministic([rows, k], scale: 0.5).asType(.float16)
        let weight = deterministic([n, k], scale: Float(1 / sqrt(Double(k)))).asType(.float16)
        let bias = deterministic([n], scale: 0.1).asType(.float16)
        eval(x, weight, bias)
        let tiles = WDMetal.GEMMTile.allCases
        let runners = tiles.map { tile in
          compile { (input: MLXArray, matrix: MLXArray, offset: MLXArray) -> MLXArray in
            WDMetal.gemm(input, weight: matrix, bias: offset, gelu: activation, tile: tile)
          }
        }
        var samples = Array(repeating: [Double](), count: tiles.count)
        for runner in runners {
          eval(runner(x, weight, bias))
        }
        // Three rounds, three synchronized dispatches each. Rotate order to
        // spread clock/thermal drift across variants within every shape.
        for round in 0..<3 {
          for offset in tiles.indices {
            let index = (offset + round * 2) % tiles.count
            let start = ProcessInfo.processInfo.systemUptime
            for _ in 0..<3 {
              eval(runners[index](x, weight, bias))
            }
            samples[index].append((ProcessInfo.processInfo.systemUptime - start) / 3)
          }
        }
        let candidates: [[String: Any]] = tiles.enumerated().map { index, tile in
          [
            "tile": tile.rawValue, "seconds": samples[index],
            "medianSeconds": samples[index].sorted()[1],
          ]
        }
        let record: [String: Any] = [
          "stage": stage, "operation": name, "rows": rows, "k": k, "n": n,
          "blockCount": [2, 2, 18, 2][stage], "candidates": candidates,
        ]
        records.append(record)
        print("GEMM stage \(stage) \(name) [\(rows),\(k)]×[\(n),\(k)] complete")
      }
    }
  }
  return [
    "native": true, "batch": batch, "records": records,
    "thermalBefore": thermalBefore, "thermalAfter": ProcessInfo.processInfo.thermalState.rawValue,
    "metric": "Median of three rounds, each averaging three synchronized native dispatches",
    "limitations":
      "Isolated GEMM timings include dispatch/evaluation overhead, not a whole-model speed claim.",
  ]
}

do {
  let args = CommandLine.arguments
  let fusedQKV = !args.contains("--unfused-qkv")
  func option(_ key: String) -> String? {
    guard let i = args.firstIndex(of: key), i + 1 < args.count else { return nil }
    return args[i + 1]
  }
  if args.contains("--thermal") {
    print(ProcessInfo.processInfo.thermalState.rawValue)
    exit(0)
  }
  Memory.cacheLimit = (option("--cache-mb").flatMap { Int($0) } ?? WDNativeTagger.defaultCacheMiB) * 1024 * 1024
  // Soft scheduling budget, not a process-memory cap. Explicit flags preserve
  // the old settings for comparisons with the selected RGBA worker settings.
  Memory.memoryLimit = (option("--memory-mb").flatMap { Int($0) } ?? WDNativeTagger.defaultMemoryMiB) * 1024 * 1024
  guard let gemmTile = WDMetal.GEMMTile(rawValue: option("--gemm-tile") ?? "32x64x32") else {
    throw WDNativeError.invalid("Unknown --gemm-tile")
  }
  if args.contains("--benchmark-gemm") {
    let report = try benchmarkGEMM(batch: Int(option("--batch") ?? "2") ?? 2)
    guard let output = option("--report") else { throw WDNativeError.invalid("Supply --report") }
    try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
      .write(to: URL(fileURLWithPath: output))
    exit(0)
  }
  if args.contains("--check-kernels") {
    let checks = try kernelChecks(
      tiles: args.contains("--check-gemm-tiles") ? WDMetal.GEMMTile.allCases : [gemmTile])
    if let report = option("--report") {
      try JSONSerialization.data(
        withJSONObject: ["native": true, "checks": checks, "passed": true],
        options: [.prettyPrinted, .sortedKeys]
      ).write(to: URL(fileURLWithPath: report))
    }
    print("Native Metal correctness: \(checks.count) checks passed")
    exit(0)
  }
  guard let weights = option("--weights") else { throw WDNativeError.invalid("Supply --weights") }
  if let source = option("--prepare-from") {
    try WDNativeTagger.prepare(
      source: URL(fileURLWithPath: source), destination: URL(fileURLWithPath: weights))
    print("Native FP16 weights prepared")
    exit(0)
  }
  if let coreml = option("--compare-coreml") {
    guard let inputs = option("--inputs"), let output = option("--report") else {
      throw WDNativeError.invalid("Supply --inputs and --report")
    }
    let report = try compareCoreML(
      weights: URL(fileURLWithPath: weights), modelURL: URL(fileURLWithPath: coreml),
      inputsURL: URL(fileURLWithPath: inputs), rounds: Int(option("--repeats") ?? "3") ?? 3,
      tile: gemmTile, compactBias: args.contains("--compact-bias"),
      probabilities: option("--probabilities"),
      includeOriginal: args.contains("--compare-original") || args.contains("--compare-current"),
      fusedQKV: fusedQKV, currentReference: args.contains("--compare-current")
    )
    try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
      .write(to: URL(fileURLWithPath: output))
    exit(0)
  }
  guard let input = option("--inputs"), let report = option("--report") else {
    throw WDNativeError.invalid("Supply --inputs and --report")
  }
  let batch = Int(option("--batch") ?? "8") ?? 8
  let frames = Int(option("--frames") ?? "64") ?? 64
  let repeats = args.contains("--validate-only") ? 1 : (Int(option("--repeats") ?? "3") ?? 3)
  let rgbaInput = args.contains("--rgba-input")
  let streamInput = args.contains("--stream-input")
  let row = rgbaInput ? 4 * 448 * 448 : 3 * 448 * 448
  let elementBytes = rgbaInput ? 1 : 4
  let head = 10861
  let inputFile = try FileHandle(forReadingFrom: URL(fileURLWithPath: input))
  defer { try? inputFile.close() }
  let inputSize = try inputFile.seekToEnd()
  let inputs = streamInput ? Data() : try Data(contentsOf: URL(fileURLWithPath: input), options: .mappedIfSafe)
  guard
    batch > 0 && frames > 0 && repeats > 0 && frames % batch == 0
      && inputSize >= frames * row * elementBytes
  else { throw WDNativeError.invalid("Invalid fixed input/batch count") }
  let memory = PeakMemory()
  let tagger = try WDNativeTagger(
    weights: URL(fileURLWithPath: weights), compileModel: !args.contains("--no-compile"),
    gemmTile: gemmTile, simdNorm: args.contains("--simd-norm"),
    compactBias: args.contains("--compact-bias"), fusedQKV: fusedQKV)
  Memory.clearCache()
  var times: [Double] = []
  var probabilities = Data()
  let initialThermal = ProcessInfo.processInfo.thermalState.rawValue
  for repeatIndex in -1..<repeats {
    var elapsed = 0.0
    for offset in stride(from: 0, to: frames, by: batch) {
      if repeatIndex == -1 && offset > 0 { break }
      try autoreleasepool {
        let source: Data
        let range: Range<Int>
        if streamInput {
          try inputFile.seek(toOffset: UInt64(offset * row * elementBytes))
          var bytes = Data(capacity: batch * row * elementBytes)
          while bytes.count < batch * row * elementBytes {
            guard let part = try inputFile.read(upToCount: batch * row * elementBytes - bytes.count), !part.isEmpty else {
              throw WDNativeError.invalid("Truncated streamed input")
            }
            bytes.append(part)
          }
          source = bytes
          range = 0..<bytes.count
        } else {
          source = inputs
          range = (offset * row * elementBytes)..<((offset + batch) * row * elementBytes)
        }
        let x = source.withUnsafeBytes { bytes in
          let pointer = UnsafeRawBufferPointer(rebasing: bytes[range])
          return rgbaInput ? MLXArray(pointer, [batch, 448, 448, 4], type: UInt8.self)
                           : MLXArray(pointer, [batch, 3, 448, 448], type: Float.self)
        }
        eval(x)
        let capture = repeatIndex == 0 && offset == 0 ? option("--capture") : nil
        if let capture { GPU.startCapture(url: URL(fileURLWithPath: capture)) }
        let start = ProcessInfo.processInfo.systemUptime
        let y = tagger.predict(x)
        eval(y)
        elapsed += ProcessInfo.processInfo.systemUptime - start
        if let capture { GPU.stopCapture(url: URL(fileURLWithPath: capture)) }
        guard y.shape == [batch, head] else {
          throw WDNativeError.invalid("Wrong classifier output")
        }
        if repeatIndex == 0 {
          let array = y.asArray(Float.self)
          guard array.allSatisfy({ $0.isFinite && $0 >= 0 && $0 <= 1 }) else {
            throw WDNativeError.invalid("Invalid probabilities")
          }
          array.withUnsafeBytes { probabilities.append(contentsOf: $0) }
        }
      }
    }
    if repeatIndex >= 0 { times.append(elapsed) }
  }
  if let output = option("--probabilities") {
    try probabilities.write(to: URL(fileURLWithPath: output))
  }
  let median = times.sorted()[times.count / 2]
  var record: [String: Any] = [
    "native": true, "implementation": WDNativeTagger.implementationVersion,
    "inputLayout": rgbaInput ? "rgba-uint8" : "rgb-planar-float32", "streamedInput": streamInput,
    "weightFormatVersion": WDNativeTagger.version, "mlxSwift": "0.31.2", "batch": batch,
    "gemmTile": gemmTile.rawValue,
    "normImplementation": args.contains("--simd-norm") ? "simd-registers" : "threadgroup-two-pass",
    "compactAttentionBias": args.contains("--compact-bias"),
    "fusedQKV": fusedQKV,
    "frames": frames, "repeats": repeats, "compiledWholeModel": !args.contains("--no-compile"),
    "memoryLimitBytes": Memory.memoryLimit, "cacheLimitBytes": Memory.cacheLimit,
    "cacheBytes": Memory.cacheMemory, "activeBytes": Memory.activeMemory,
    "peakMLXActiveBytes": Memory.peakMemory, "peakPhysicalFootprintBytes": memory.bytes,
    "thermalBefore": initialThermal, "thermalAfter": ProcessInfo.processInfo.thermalState.rawValue,
    "predictionSweepSeconds": times, "validationOnly": args.contains("--validate-only"),
  ]
  if !args.contains("--validate-only") {
    record["predictionFramesPerSecond"] = Double(frames) / median
  }
  if let reference = option("--reference") {
    let expected = try Data(contentsOf: URL(fileURLWithPath: reference), options: .mappedIfSafe)
    guard expected.count >= probabilities.count else {
      throw WDNativeError.invalid("Short reference")
    }
    var maximum: Float = 0
    var total = 0.0
    var thresholdChanges = 0
    var ratingChanges = 0
    probabilities.withUnsafeBytes { actualBytes in
      expected.withUnsafeBytes { expectedBytes in
        let a = actualBytes.bindMemory(to: Float.self)
        let e = expectedBytes.bindMemory(to: Float.self)
        for i in a.indices {
          let d = abs(a[i] - e[i])
          maximum = max(maximum, d)
          total += Double(d)
          if i % head >= 4 && (a[i] > 0.35) != (e[i] > 0.35) { thresholdChanges += 1 }
        }
        for frame in 0..<frames {
          let range = (frame * head)..<(frame * head + 4)
          if range.max(by: { a[$0] < a[$1] }) != range.max(by: { e[$0] < e[$1] }) {
            ratingChanges += 1
          }
        }
      }
    }
    record["maximumAbsoluteError"] = maximum
    record["meanAbsoluteError"] = total / Double(frames * head)
    record["thresholdDifferencesNonRatingHeads"] = thresholdChanges
    record["ratingDifferences"] = ratingChanges
  }
  try JSONSerialization.data(withJSONObject: record, options: [.prettyPrinted, .sortedKeys]).write(
    to: URL(fileURLWithPath: report))
  print(
    args.contains("--validate-only")
      ? "Native full-model validation completed"
      : String(
        format: "Native Metal: %.2f prediction frames/s, %.1f MB peak", Double(frames) / median,
        Double(memory.bytes) / 1e6))
} catch {
  fputs("Native WDTagger failed: \(error)\n", stderr)
  exit(1)
}
