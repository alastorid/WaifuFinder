import CryptoKit
import Foundation
import Metal
import MLX

public enum WDNativeError: Error { case invalid(String) }

/// Native Swift graph orchestration and fixed-shape Metal kernels.
public final class WDNativeTagger {
  public static let version = "swinv2-native-metal32-v1"
  public static let implementationVersion = "swinv2-native-metal32-fused-qkv-v2"
  // Soft scheduling/cache budgets selected with the direct RGBA input path.
  // Workers isolate these process-wide MLX settings from other app models.
  public static let defaultMemoryMiB = 448
  public static let defaultCacheMiB = 64
  private let p: [String: MLXArray]
  private let stages: [Stage]
  private let identityRoutePlaceholder = MLXArray([Int32(0)])
  public enum GEMMOperation: String, CaseIterable {
    case qkv, projection, fc1, fc2, reduction
  }
  private let gemmOverrides: [GEMMOperation: [WDMetal.GEMMTile]]
  private let capabilities: WDGPUCapabilities
  private let compileModel: Bool
  private let attentionRowsOverride: Int?
  private let gemmTile: WDMetal.GEMMTile?
  private let simdNorm: Bool?
  private let compactBias: Bool?
  private let fusedQKV: Bool
  private var configurations: [Int: WDResolvedKernelConfiguration] = [:]
  private var compiled: [String: (MLXArray) -> MLXArray] = [:]
  private var compactParameters: [String: MLXArray]?
  private struct Block {
    let prefix: String
    let shifted: Bool
    let transition: MLXArray
    let transitionIsIdentity: Bool
  }
  private struct Stage {
    let side: Int, heads: Int, windows: Int
    let regions: MLXArray
    let merge: MLXArray?
    let blocks: [Block]
  }

  /// One-time native conversion of the retained upstream/attention-export weights.
  /// CPU conversion keeps FP32 source tensors out of GPU workspaces.
  public static func prepare(source: URL, destination: URL) throws {
    try Stream.withNewDefaultStream(device: .cpu) {
      try prepareOnCPU(source: source, destination: destination)
    }
  }

  private static func prepareOnCPU(source: URL, destination: URL) throws {
    let original = try loadArrays(url: source, stream: .cpu)
    let hasExportedAttention = original["layers.0.blocks.0.attn._sdpa_bias"] != nil
    var half = convertedBaseWeights(original, hasExportedAttention: hasExportedAttention)
    if !hasExportedAttention {
      try appendOriginalAttentionWeights(from: original, to: &half)
    }

    let data = try Data(contentsOf: source, options: .mappedIfSafe)
    let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    let preparedSource = hasExportedAttention ? "attention-export" : "original-upstream"
    try save(
      arrays: half,
      metadata: [
        "waifu.experiment": version, "sourceSHA256": digest,
        "preparedNativelyFrom": preparedSource,
      ], url: destination, stream: .cpu)
  }

  private static func convertedBaseWeights(
    _ original: [String: MLXArray], hasExportedAttention: Bool
  ) -> [String: MLXArray] {
    var half: [String: MLXArray] = [:]
    for (name, array) in original {
      let isRelativeBias = name.contains("cpb_mlp") || name.contains("relative_")
        || name.contains("attn_mask")
      if isRelativeBias { continue }
      let isSkippedScale = [".logit_scale", ".q_bias", ".v_bias"].contains {
        name.hasSuffix($0)
      }
      if isSkippedScale { continue }

      var value = array
      if name == "patch_embed.proj.weight" {
        if !hasExportedAttention {
          let channelOrder = MLXArray([Int32(2), 1, 0])
          value = take(value, channelOrder, axis: 1, stream: .cpu)
        }
        value = value.transposed(0, 2, 3, 1, stream: .cpu)
      }
      value = value.asType(.float16, stream: .cpu)
      eval(value)
      half[name] = value
    }
    return half
  }

  private static func appendOriginalAttentionWeights(
    from original: [String: MLXArray], to half: inout [String: MLXArray]
  ) throws {
    var coordinates: [Float] = []
    coordinates.reserveCapacity(729 * 2)
    for y in -13...13 {
      for x in -13...13 {
        for coordinate in [y, x] {
          let scaled = Float(coordinate) / Float(13) * Float(8)
          coordinates.append(copysignf(log2f(abs(scaled) + Float(1)) / Float(3), scaled))
        }
      }
    }

    var indices: [Int32] = []
    indices.reserveCapacity(196 * 196)
    for query in 0..<196 {
      for key in 0..<196 {
        let rowOffset = (query / 14 - key / 14 + 13) * 27
        let columnOffset = query % 14 - key % 14 + 13
        indices.append(Int32(rowOffset + columnOffset))
      }
    }

    let tableCoordinates = MLXArray(coordinates, [729, 2])
    let positionIndices = MLXArray(indices)
    let stageDepths = [2, 2, 18, 2]
    for (stage, depth) in stageDepths.enumerated() {
      for block in 0..<depth {
        let attentionPrefix = "layers.\(stage).blocks.\(block).attn"
        try appendOriginalAttentionWeights(
          prefix: attentionPrefix, coordinates: tableCoordinates,
          indices: positionIndices, original: original, to: &half)
      }
    }
  }

  private static func appendOriginalAttentionWeights(
    prefix: String, coordinates: MLXArray, indices: MLXArray,
    original: [String: MLXArray], to half: inout [String: MLXArray]
  ) throws {
    guard let w0 = original[prefix + ".cpb_mlp.0.weight"],
      let b0 = original[prefix + ".cpb_mlp.0.bias"],
      let w2 = original[prefix + ".cpb_mlp.2.weight"],
      let q = original[prefix + ".q_bias"],
      let v = original[prefix + ".v_bias"],
      let temperature = original[prefix + ".logit_scale"]
    else { throw WDNativeError.invalid("Incomplete original attention weights: \(prefix)") }

    let hiddenProjection = addMM(b0, coordinates, w0.T)
    let hidden = maximum(hiddenProjection, MLXArray(Float(0)))
    let relativeTable = matmul(hidden, w2.T)
    let gathered = take(relativeTable, indices, axis: 0)
    let reshaped = gathered.reshaped(196, 196, w2.shape[0])
    let channelFirst = reshaped.transposed(2, 0, 1)
    let withHead = channelFirst.expandedDimensions(axis: 0)
    let bias = Float(16) * sigmoid(withHead)

    let maximumLogScale = Float(log(100.0))
    let clippedTemperature = clip(temperature, max: maximumLogScale)
    let temperatureScale = exp(clippedTemperature)
    let scale = temperatureScale * Float(sqrt(32.0))
    let emptyKeyBias = MLXArray.zeros(q.shape, dtype: .float32)
    let qkvBias = concatenated([q, emptyKeyBias, v])
    let attentionWeights: [(String, MLXArray)] = [
      ("._sdpa_bias", bias),
      ("._sdpa_query_scale", scale),
      ("._sdpa_qkv_bias", qkvBias),
    ]
    for (suffix, value) in attentionWeights {
      let converted = value.asType(.float16)
      eval(converted)
      half[prefix + suffix] = converted
    }
  }

  private static func compactBiasWeights(_ loaded: [String: MLXArray]) -> [String: MLXArray] {
    var parameters = loaded
    // Every dense bias value depends only on the 27×27 relative displacement.
    // Gather representatives from FP16 values, preserving their exact bits.
    var representatives: [Int32] = []
    for dy in -13...13 {
      for dx in -13...13 {
        let query = max(dy, 0) * 14 + max(dx, 0)
        let key = max(-dy, 0) * 14 + max(-dx, 0)
        representatives.append(Int32(query * 196 + key))
      }
    }
    let indices = MLXArray(representatives)
    for (name, value) in loaded where name.hasSuffix("._sdpa_bias") {
      let packed = take(value.reshaped(-1, 196 * 196), indices, axis: 1)
      eval(packed)
      parameters[name] = packed
    }
    return parameters
  }

  public init(
    weights: URL, compileModel: Bool = true, gemmTile: WDMetal.GEMMTile? = nil,
    simdNorm: Bool? = nil, compactBias: Bool? = nil, fusedQKV: Bool = true,
    gemmOverrides: [GEMMOperation: [WDMetal.GEMMTile]] = [:],
    attentionQueryRows: Int? = nil
  ) throws {
    let capabilities = try WDGPUCapabilities.discover()
    var plans: [Int: WDResolvedKernelConfiguration] = [:]
    for batch in 1...8 {
      plans[batch] = try WDOptimizationRegistry.resolve(gpu: capabilities, batch: batch, gemmTile: gemmTile,
        gemmOverrides: gemmOverrides, attentionQueryRows: attentionQueryRows,
        simdNorm: simdNorm, compactBias: compactBias, fusedQKV: fusedQKV)
    }
    self.capabilities = capabilities
    self.configurations = plans
    self.compileModel = compileModel
    self.gemmOverrides = gemmOverrides
    self.attentionRowsOverride = attentionQueryRows
    self.gemmTile = gemmTile
    self.simdNorm = simdNorm
    self.compactBias = compactBias
    self.fusedQKV = fusedQKV
    let (loaded, metadata) = try loadArraysAndMetadata(url: weights)
    guard metadata["waifu.experiment"] == Self.version else {
      throw WDNativeError.invalid("Prepare native weights first")
    }
    guard loaded.values.allSatisfy({ $0.dtype == .float16 }) else {
      throw WDNativeError.invalid("Expected FP16 weights")
    }
    p = compactBias == true ? Self.compactBiasWeights(loaded) : loaded
    if compactBias == true { compactParameters = p }
    var configurations: [Stage] = []
    var previousOrder: [Int] = []
    var previousSide = 0
    for (stage, side) in [112, 56, 28, 14].enumerated() {
      let depth = [2, 2, 18, 2][stage]
      let heads = [4, 8, 16, 32][stage]
      let first = Self.windowOrder(side: side, shift: 0)
      var merge: MLXArray?
      if stage > 0 {
        let inverse = Self.inverse(previousOrder)
        var route: [Int32] = []
        for grid in first {
          let h = grid / side
          let w = grid % side
          for (dh, dw) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            route.append(Int32(inverse[(h * 2 + dh) * previousSide + w * 2 + dw]))
          }
        }
        merge = MLXArray(route)
      }
      var blocks: [Block] = []
      // Alternating blocks reuse the same immutable directional route. Stage 3
      // needs two maps, rather than seventeen independently allocated copies.
      var transitionRoutes: [Int: MLXArray] = [:]
      for block in 0..<depth {
        let shift = block % 2 == 1 && side > 14 ? 7 : 0
        let order = Self.windowOrder(side: side, shift: shift)
        let nextShift = (block + 1) % 2 == 1 && side > 14 ? 7 : 0
        let next = block + 1 < depth ? Self.windowOrder(side: side, shift: nextShift) : order
        let inverse = Self.inverse(order)
        let transition: MLXArray
        if next == order {
          transition = identityRoutePlaceholder
        } else if let cached = transitionRoutes[shift] {
          transition = cached
        } else {
          transition = MLXArray(next.map { Int32(inverse[$0]) })
          transitionRoutes[shift] = transition
        }
        blocks.append(
          Block(
            prefix: "layers.\(stage).blocks.\(block)", shifted: shift != 0,
            transition: transition,
            transitionIsIdentity: next == order))
        previousOrder = order
      }
      configurations.append(
        Stage(
          side: side, heads: heads, windows: (side / 14) * (side / 14),
          regions: Self.regions(side: side),
          merge: merge, blocks: blocks))
      previousSide = side
    }
    stages = configurations
    eval(Array(p.values))
    for stage in stages {
      eval(stage.regions)
      if let merge = stage.merge { eval(merge) }
      for block in stage.blocks { eval(block.transition) }
    }
    // Plans are resolved before the first graph trace for each input signature.
  }
  private func weight(_ name: String) -> MLXArray {
    guard let array = p[name] else { preconditionFailure("Missing native tagger weight \(name)") }
    return array
  }
  /// Shares immutable weights and routing for controlled kernel comparisons.
  public func variant(
    gemmTile: WDMetal.GEMMTile, fusedQKV: Bool? = nil, compactingBias: Bool = false,
    gemmOverrides: [GEMMOperation: [WDMetal.GEMMTile]]? = nil, attentionQueryRows: Int? = nil
  ) -> WDNativeTagger {
    WDNativeTagger(
      copying: self, gemmTile: gemmTile, fusedQKV: fusedQKV ?? self.fusedQKV,
      compactingBias: compactingBias, gemmOverrides: gemmOverrides, attentionQueryRows: attentionQueryRows)
  }
  private init(
    copying source: WDNativeTagger, gemmTile: WDMetal.GEMMTile, fusedQKV: Bool,
    compactingBias: Bool, gemmOverrides: [GEMMOperation: [WDMetal.GEMMTile]]?, attentionQueryRows: Int?
  ) {
    precondition((gemmOverrides ?? [:]).values.allSatisfy { $0.count == 4 })
    precondition(attentionQueryRows.map { [16, 32].contains($0) } ?? true)
    p = compactingBias && source.compactBias != true ? Self.compactBiasWeights(source.p) : source.p
    compactParameters = compactingBias ? p : source.compactParameters
    capabilities = source.capabilities
    compileModel = source.compileModel
    stages = source.stages
    self.gemmTile = gemmTile
    self.gemmOverrides = gemmOverrides ?? source.gemmOverrides
    attentionRowsOverride = attentionQueryRows ?? source.attentionRowsOverride
    simdNorm = source.simdNorm
    compactBias = compactingBias ? true : source.compactBias
    self.fusedQKV = fusedQKV
    // Preserve the nonthrowing comparison API, but fail clearly on invalid tuning.
    for batch in 1...8 {
      do { _ = try resolvedConfiguration(batch: batch) }
      catch { preconditionFailure("Invalid native variant: \(error)") }
    }
  }
  /// Cached once per workload, outside graph tracing and block execution.
  public func resolvedConfiguration(batch: Int) throws -> WDResolvedKernelConfiguration {
    if let cached = configurations[batch] { return cached }
    let resolved = try WDOptimizationRegistry.resolve(gpu: capabilities, batch: batch,
      gemmTile: gemmTile, gemmOverrides: gemmOverrides, attentionQueryRows: attentionRowsOverride,
      simdNorm: simdNorm, compactBias: compactBias, fusedQKV: fusedQKV)
    configurations[batch] = resolved
    return resolved
  }
  public func selectedTile(_ operation: GEMMOperation, stage: Int, batch: Int) -> WDMetal.GEMMTile {
    precondition((0..<4).contains(stage))
    return try! resolvedConfiguration(batch: batch).stages[stage].tiles[operation]!
  }
  public func selectedAttentionQueryRows(batch: Int) -> Int {
    try! resolvedConfiguration(batch: batch).stages[0].attentionQueryRows
  }
  private func linear(_ x: MLXArray, _ name: String, tile: WDMetal.GEMMTile? = nil) -> MLXArray {
    let matrix = weight(name + ".weight")
    if matrix.shape[0] % 64 == 0 && matrix.shape[1] % 32 == 0 {
      return WDMetal.gemm(
        x, weight: matrix,
        bias: p[name + ".bias"] ?? MLXArray.zeros([matrix.shape[0]], dtype: .float16),
        tile: tile ?? gemmTile ?? .baseline)
    }
    if let bias = p[name + ".bias"] { return addMM(bias, x, matrix.T) }
    return matmul(x, matrix.T)
  }
  public func predict(_ x: MLXArray) -> MLXArray {
    precondition(x.shape.count == 4 && (x.dtype == .uint8
      ? Array(x.shape.dropFirst()) == [448, 448, 4]
      : x.dtype == .float32 && Array(x.shape.dropFirst()) == [3, 448, 448]))
    let configuration: WDResolvedKernelConfiguration
    do { configuration = try resolvedConfiguration(batch: x.shape[0]) }
    catch { preconditionFailure("Invalid native workload: \(error)") }
    if configuration.compactBias && compactParameters == nil {
      compactParameters = Self.compactBiasWeights(p)
    }
    let biases = configuration.compactBias ? compactParameters : nil
    guard compileModel else { return forward(x, configuration: configuration, biasWeights: biases) }
    let key = "\(x.dtype):\(x.shape)"
    if let graph = compiled[key] { return graph(x) }
    let graph = compile { [unowned self] input in
      self.forward(input, configuration: configuration, biasWeights: biases)
    }
    compiled[key] = graph
    return graph(x)
  }
  private func forward(_ image: MLXArray, configuration: WDResolvedKernelConfiguration,
                       biasWeights: [String: MLXArray]?) -> MLXArray {
    let batch = image.shape[0]
    var x = WDMetal.inputPatches(image)
    x = addMM(
      weight("patch_embed.proj.bias"), x, weight("patch_embed.proj.weight").reshaped(128, 48).T)
    x = WDMetal.normalize(
      x, residual: nil, weight: weight("patch_embed.norm.weight"),
      bias: weight("patch_embed.norm.bias"), route: identityRoutePlaceholder, spatial: 112 * 112,
      simdRows: configuration.stages[0].simdNorm, identityRoute: true)
    for (index, stage) in stages.enumerated() {
      let spatial = stage.side * stage.side
      let kernels = configuration.stages[index]
      if let merge = stage.merge {
        x = WDMetal.patchMerge(x, route: merge, spatial: spatial)
        x = linear(x, "layers.\(index).downsample.reduction", tile: kernels.tiles[.reduction]!)
        x = MLXFast.layerNorm(
          x, weight: weight("layers.\(index).downsample.norm.weight"),
          bias: weight("layers.\(index).downsample.norm.bias"), eps: 1e-5)
      }
      for block in stage.blocks {
        let attn = block.prefix + ".attn"
        let qkv: [MLXArray]
        if fusedQKV {
          qkv = WDMetal.projectQKV(
            x, weight: weight(attn + ".qkv.weight"),
            bias: weight(attn + "._sdpa_qkv_bias"),
            scale: weight(attn + "._sdpa_query_scale").reshaped(-1),
            heads: stage.heads, tile: kernels.tiles[.qkv]!)
        } else {
          let packed = WDMetal.gemm(
            x, weight: weight(attn + ".qkv.weight"), bias: weight(attn + "._sdpa_qkv_bias"),
            tile: kernels.tiles[.qkv]!)
          qkv = WDMetal.qkv(
            packed, scale: weight(attn + "._sdpa_query_scale").reshaped(-1), heads: stage.heads)
        }
        var y = WDMetal.flash(
          qkv[0], qkv[1], qkv[2], bias: biasWeights?[attn + "._sdpa_bias"] ?? weight(attn + "._sdpa_bias"), regions: stage.regions,
          windows: stage.windows, shifted: block.shifted, compactBias: configuration.compactBias,
          queryRows: kernels.attentionQueryRows)
        y = linear(y, attn + ".proj", tile: kernels.tiles[.projection]!)
        x = WDMetal.normalize(
          y, residual: x, weight: weight(block.prefix + ".norm1.weight"),
          bias: weight(block.prefix + ".norm1.bias"), route: identityRoutePlaceholder, spatial: spatial,
          simdRows: kernels.simdNorm, identityRoute: true)
        y = WDMetal.mlpGelu(
          x, weight: weight(block.prefix + ".mlp.fc1.weight"),
          bias: weight(block.prefix + ".mlp.fc1.bias"), tile: kernels.tiles[.fc1]!)
        y = linear(y, block.prefix + ".mlp.fc2", tile: kernels.tiles[.fc2]!)
        x = WDMetal.normalize(
          y, residual: x, weight: weight(block.prefix + ".norm2.weight"),
          bias: weight(block.prefix + ".norm2.bias"), route: block.transition, spatial: spatial,
          simdRows: kernels.simdNorm, identityRoute: block.transitionIsIdentity)
      }
    }
    x = MLXFast.layerNorm(x, weight: weight("norm.weight"), bias: weight("norm.bias"), eps: 1e-5)
    x = mean(x.reshaped(batch, 196, 1024), axis: 1)
    return sigmoid(linear(x, "head.fc").asType(.float32))
  }
  public static func windowOrder(side: Int, shift: Int) -> [Int] {
    var result: [Int] = []
    for wh in 0..<side / 14 {
      for ww in 0..<side / 14 {
        for h in 0..<14 {
          for w in 0..<14 {
            result.append(((wh * 14 + h + shift) % side) * side + (ww * 14 + w + shift) % side)
          }
        }
      }
    }
    return result
  }
  private static func inverse(_ order: [Int]) -> [Int] {
    var inverse = Array(repeating: 0, count: order.count)
    for (i, j) in order.enumerated() { inverse[j] = i }
    return inverse
  }
  public static func regions(side: Int) -> MLXArray {
    var result: [Int32] = []
    func region(_ v: Int) -> Int { v < side - 14 ? 0 : (v < side - 7 ? 1 : 2) }
    for wh in 0..<side / 14 {
      for ww in 0..<side / 14 {
        for h in 0..<14 {
          for w in 0..<14 {
            result.append(Int32(region(wh * 14 + h) * 3 + region(ww * 14 + w)))
          }
        }
      }
    }
    return MLXArray(result)
  }
}
