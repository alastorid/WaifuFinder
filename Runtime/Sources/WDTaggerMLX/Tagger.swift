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
  private static let isMeasuredM3 = MTLCreateSystemDefaultDevice()?.name == "Apple M3"
  private let automaticBatch8Profile: Bool
  private let attentionRowsOverride: Int?
  private let gemmTile: WDMetal.GEMMTile
  private let simdNorm: Bool
  private let compactBias: Bool
  private let fusedQKV: Bool
  private var compiled: ((MLXArray) -> MLXArray)?
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
    simdNorm: Bool = false, compactBias: Bool = false, fusedQKV: Bool = true,
    gemmOverrides: [GEMMOperation: [WDMetal.GEMMTile]] = [:],
    attentionQueryRows: Int? = nil
  ) throws {
    precondition(gemmOverrides.values.allSatisfy { $0.count == 4 })
    precondition(attentionQueryRows.map { [16, 32].contains($0) } ?? true)
    self.gemmOverrides = gemmOverrides
    self.attentionRowsOverride = attentionQueryRows
    self.automaticBatch8Profile = gemmTile == nil && Self.isMeasuredM3 && fusedQKV && !simdNorm
    self.gemmTile = gemmTile ?? .baseline
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
    p = compactBias ? Self.compactBiasWeights(loaded) : loaded
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
      for block in 0..<depth {
        let shift = block % 2 == 1 && side > 14 ? 7 : 0
        let order = Self.windowOrder(side: side, shift: shift)
        let nextShift = (block + 1) % 2 == 1 && side > 14 ? 7 : 0
        let next = block + 1 < depth ? Self.windowOrder(side: side, shift: nextShift) : order
        let inverse = Self.inverse(order)
        blocks.append(
          Block(
            prefix: "layers.\(stage).blocks.\(block)", shifted: shift != 0,
            transition: next == order ? identityRoutePlaceholder : MLXArray(next.map { Int32(inverse[$0]) }),
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
    if compileModel { compiled = compile { [unowned self] x in self.forward(x) } }
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
    p = compactingBias && !source.compactBias ? Self.compactBiasWeights(source.p) : source.p
    stages = source.stages
    self.gemmTile = gemmTile
    self.gemmOverrides = gemmOverrides ?? source.gemmOverrides
    automaticBatch8Profile = false
    attentionRowsOverride = attentionQueryRows ?? source.attentionRowsOverride
    simdNorm = source.simdNorm
    compactBias = source.compactBias || compactingBias
    self.fusedQKV = fusedQKV
    if source.compiled != nil {
      compiled = compile { [unowned self] x in self.forward(x) }
    }
  }
  /// Explicit global tiles disable automatic selection; per-operation overrides win.
  /// Only the measured base M3 and batch eight use the confirmed Stage 3 profile.
  public func selectedTile(_ operation: GEMMOperation, stage: Int, batch: Int) -> WDMetal.GEMMTile {
    precondition((0..<4).contains(stage) && batch > 0)
    if let explicit = gemmOverrides[operation] { return explicit[stage] }
    if automaticBatch8Profile && batch == 8 && stage == 2 && (operation == .qkv || operation == .fc2) {
      return .m64n64k32
    }
    return gemmTile
  }
  public func selectedAttentionQueryRows(batch: Int) -> Int {
    precondition(batch > 0)
    return attentionRowsOverride ?? (automaticBatch8Profile && batch == 8 ? 16 : 32)
  }
  private func linear(_ x: MLXArray, _ name: String, tile: WDMetal.GEMMTile? = nil) -> MLXArray {
    let matrix = weight(name + ".weight")
    if matrix.shape[0] % 64 == 0 && matrix.shape[1] % 32 == 0 {
      return WDMetal.gemm(
        x, weight: matrix,
        bias: p[name + ".bias"] ?? MLXArray.zeros([matrix.shape[0]], dtype: .float16),
        tile: tile ?? gemmTile)
    }
    if let bias = p[name + ".bias"] { return addMM(bias, x, matrix.T) }
    return matmul(x, matrix.T)
  }
  public func predict(_ x: MLXArray) -> MLXArray {
    precondition(x.shape.count == 4 && (x.dtype == .uint8
      ? Array(x.shape.dropFirst()) == [448, 448, 4]
      : x.dtype == .float32 && Array(x.shape.dropFirst()) == [3, 448, 448]))
    return compiled?(x) ?? forward(x)
  }
  private func forward(_ image: MLXArray) -> MLXArray {
    let batch = image.shape[0]
    var x = WDMetal.inputPatches(image)
    x = addMM(
      weight("patch_embed.proj.bias"), x, weight("patch_embed.proj.weight").reshaped(128, 48).T)
    x = WDMetal.normalize(
      x, residual: nil, weight: weight("patch_embed.norm.weight"),
      bias: weight("patch_embed.norm.bias"), route: identityRoutePlaceholder, spatial: 112 * 112,
      simdRows: simdNorm, identityRoute: true)
    for (index, stage) in stages.enumerated() {
      let spatial = stage.side * stage.side
      if let merge = stage.merge {
        x = WDMetal.patchMerge(x, route: merge, spatial: spatial)
        x = linear(x, "layers.\(index).downsample.reduction", tile: selectedTile(.reduction, stage: index, batch: batch))
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
            heads: stage.heads, tile: selectedTile(.qkv, stage: index, batch: batch))
        } else {
          let packed = WDMetal.gemm(
            x, weight: weight(attn + ".qkv.weight"), bias: weight(attn + "._sdpa_qkv_bias"),
            tile: selectedTile(.qkv, stage: index, batch: batch))
          qkv = WDMetal.qkv(
            packed, scale: weight(attn + "._sdpa_query_scale").reshaped(-1), heads: stage.heads)
        }
        var y = WDMetal.flash(
          qkv[0], qkv[1], qkv[2], bias: weight(attn + "._sdpa_bias"), regions: stage.regions,
          windows: stage.windows, shifted: block.shifted, compactBias: compactBias,
          queryRows: selectedAttentionQueryRows(batch: batch))
        y = linear(y, attn + ".proj", tile: selectedTile(.projection, stage: index, batch: batch))
        x = WDMetal.normalize(
          y, residual: x, weight: weight(block.prefix + ".norm1.weight"),
          bias: weight(block.prefix + ".norm1.bias"), route: identityRoutePlaceholder, spatial: spatial,
          simdRows: simdNorm, identityRoute: true)
        y = WDMetal.mlpGelu(
          x, weight: weight(block.prefix + ".mlp.fc1.weight"),
          bias: weight(block.prefix + ".mlp.fc1.bias"), tile: selectedTile(.fc1, stage: index, batch: batch))
        y = linear(y, block.prefix + ".mlp.fc2", tile: selectedTile(.fc2, stage: index, batch: batch))
        x = WDMetal.normalize(
          y, residual: x, weight: weight(block.prefix + ".norm2.weight"),
          bias: weight(block.prefix + ".norm2.bias"), route: block.transition, spatial: spatial,
          simdRows: simdNorm, identityRoute: block.transitionIsIdentity)
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
