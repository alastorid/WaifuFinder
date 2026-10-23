import Foundation
import MLX

/// Fixed-shape native Metal dispatch. No Python interpreter or Python bridge.
public enum WDMetal {
  public enum GEMMTile: String, CaseIterable {
    case baseline = "32x64x32"
    case m64n64k32 = "64x64x32"
    case m32n128k32 = "32x128x32"
    case m64n128k32 = "64x128x32"
    case m64n64k64 = "64x64x64"
    case m32n64k64 = "32x64x64"
    case library = "mlx"
    case relaxed = "relaxed32x64x32"
    case aligned = "aligned32x64x32"
    case alignedRelaxed = "aligned-relaxed32x64x32"
    case narrow = "32x32x32"
    case occupancy = "64x64x16-w4n2"

    /// QKV already selects full-tile loads by shape. These aliases have
    /// identical template arguments; library remains a genuinely unfused path.
    public var effectiveQKVTile: GEMMTile {
      switch self {
      case .aligned: return .baseline
      case .alignedRelaxed: return .relaxed
      default: return self
      }
    }

    var dimensions: (Int, Int, Int) {
      precondition(self != .library)
      if self == .occupancy { return (64, 64, 16) }
      let specialized = [Self.relaxed, .aligned, .alignedRelaxed].contains(self)
      let values = (specialized ? "32x64x32" : rawValue).split(separator: "x").map { Int($0)! }
      return (values[0], values[1], values[2])
    }
  }
  private static func make(
    _ file: String, name: String, inputs: [String], outputs: [String], header: Bool = false,
    headerFile: String? = nil
  ) -> MLXFast.MLXFastKernel {
    // Installed workers keep the source bundle beside the executable. Resolve
    // it explicitly: SwiftPM's generated accessor also embeds a build-machine
    // path, which must never be required by the distributed app.
    let installed = Bundle.main.executableURL!.deletingLastPathComponent()
      .appendingPathComponent("WaifuRuntime_WDTaggerMLX.bundle/Metal")
    let directory = FileManager.default.fileExists(atPath: installed.path)
      ? installed : Bundle.module.resourceURL!.appendingPathComponent("Metal")
    let source = try! String(
      contentsOf: directory.appendingPathComponent(file + ".metal"), encoding: .utf8)
    let include =
      header
      ? try! String(
        contentsOf: directory.appendingPathComponent((headerFile ?? file) + ".h"), encoding: .utf8)
      : ""
    return MLXFast.metalKernel(
      name: name, inputNames: inputs, outputNames: outputs, source: source, header: include)
  }
  private static let attention = make(
    "Flash32", name: "waifu_flash32", inputs: ["Q", "K", "V", "bias", "regions"], outputs: ["O"],
    header: true)
  private static let cosine = make(
    "QKVCosine", name: "waifu_qkv_cosine_pack", inputs: ["packed", "scale"],
    outputs: ["Q", "K", "V"])
  private static let projection = make(
    "QKVProject", name: "waifu_qkv_project_normalize_pack",
    inputs: ["x", "weight", "bias", "scale"],
    outputs: ["Q", "K", "V"], header: true, headerFile: "MLPGelu")
  private static let norm = make(
    "NormRoute", name: "waifu_postnorm_residual_route",
    inputs: ["x", "residual", "weight", "bias", "route"], outputs: ["out"])
  private static let normSIMD = make(
    "NormSIMD", name: "waifu_simd_postnorm_residual_route",
    inputs: ["x", "residual", "weight", "bias", "route"], outputs: ["out"])
  private static let merge = make(
    "PatchMerge", name: "waifu_patch_merge_route", inputs: ["x", "route"], outputs: ["out"])
  private static let mlp = make(
    "MLPGelu", name: "waifu_mlp_gemm_bias_gelu", inputs: ["x", "weight", "bias"], outputs: ["out"],
    header: true)
  private static let patch = make(
    "PatchInput", name: "waifu_input_normalize_patch_window", inputs: ["image"], outputs: ["out"])

  public static func inputPatches(_ image: MLXArray) -> MLXArray {
    let rgba = image.dtype == .uint8
    precondition(rgba ? Array(image.shape.dropFirst()) == [448, 448, 4]
                     : image.dtype == .float32 && Array(image.shape.dropFirst()) == [3, 448, 448])
    let rows = image.shape[0] * 112 * 112
    return patch(
      [image], template: [("RGBA", rgba)], grid: (rows * 48, 1, 1), threadGroup: (256, 1, 1), outputShapes: [[rows, 48]],
      outputDTypes: [.float16])[0]
  }

  public static func qkv(_ packed: MLXArray, scale: MLXArray, heads: Int) -> [MLXArray] {
    precondition(
      packed.dtype == .float16 && packed.ndim == 2 && packed.shape[0] % 196 == 0
        && packed.shape[1] == 3 * heads * 32)
    let windows = packed.shape[0] / 196
    return cosine(
      [packed, scale], template: [("H", heads)], grid: (windows * 196 * heads * 32, 1, 1),
      threadGroup: (128, 1, 1),
      outputShapes: Array(repeating: [windows, heads, 196, 32], count: 3),
      outputDTypes: [.float16, .float16, .float16])
  }
  /// Effective native matrix/SIMD geometry, or nil for MLX GEMM + packing.
  public static func qkvGeometry(tile: GEMMTile, heads: Int)
    -> (bm: Int, bn: Int, bk: Int, wm: Int)? {
    precondition(heads > 0 && heads % 2 == 0)
    guard tile != .library else { return nil }
    let (bm, requestedBN, bk) = tile.effectiveQKVTile.dimensions
    var bn = min(requestedBN, heads * 32)
    while heads * 32 % bn != 0 { bn /= 2 }
    return (bm, bn, bk, tile == .occupancy ? 4 : 2)
  }
  public static func projectQKV(
    _ x: MLXArray, weight: MLXArray, bias: MLXArray, scale: MLXArray, heads: Int,
    tile: GEMMTile = .baseline
  ) -> [MLXArray] {
    precondition(
      x.dtype == .float16 && x.ndim == 2 && x.shape[0] % 196 == 0 && x.shape[1] == heads * 32
        && heads > 0 && heads % 2 == 0)
    precondition(weight.shape == [3 * heads * 32, heads * 32]
      && bias.size == 3 * heads * 32 && scale.size == heads)
    guard let (bm, bn, bk, wm) = qkvGeometry(tile: tile, heads: heads) else {
      return qkv(gemm(x, weight: weight, bias: bias, tile: .library), scale: scale, heads: heads)
    }
    let tile = tile.effectiveQKVTile
    let shape = [x.shape[0] / 196, heads, 196, 32]
    let threads = wm * 2 * 32
    return projection(
      [x, weight, bias, scale], template: [
        ("H", heads), ("TBM", bm), ("TBN", bn), ("TBK", bk), ("TWM", wm),
        ("RELAXED", tile == .relaxed || tile == .alignedRelaxed),
        ("M_ALIGNED", x.shape[0] % bm == 0),
      ],
      grid: ((3 * heads * 32 / bn) * threads, (x.shape[0] + bm - 1) / bm, 1),
      threadGroup: (threads, 1, 1),
      outputShapes: [shape, shape, shape], outputDTypes: [.float16, .float16, .float16])
  }
  public static func flash(
    _ q: MLXArray, _ k: MLXArray, _ v: MLXArray, bias: MLXArray, regions: MLXArray, windows: Int,
    shifted: Bool, compactBias: Bool = false, queryRows: Int = 32
  ) -> MLXArray {
    precondition(
      q.dtype == .float16 && q.shape == k.shape && q.shape == v.shape && q.shape[2] == 196
        && q.shape[3] == 32)
    precondition([16, 32].contains(queryRows))
    let threads = queryRows * 4
    let heads = q.shape[1]
    precondition(bias.size == heads * (compactBias ? 729 : 196 * 196))
    return attention(
      [q, k, v, bias, regions],
      template: [
        ("H", heads), ("NW", windows), ("SHIFT", shifted), ("COMPACT_BIAS", compactBias), ("TBQ", queryRows),
      ],
      grid: (((196 + queryRows - 1) / queryRows) * threads, heads, q.shape[0]), threadGroup: (threads, 1, 1),
      outputShapes: [[q.shape[0] * 196, heads * 32]], outputDTypes: [.float16])[0]
  }
  public static func normalize(
    _ x: MLXArray, residual: MLXArray?, weight: MLXArray, bias: MLXArray, route: MLXArray,
    spatial: Int, simdRows: Bool = false, identityRoute: Bool = false
  ) -> MLXArray {
    let channels = x.shape[1]
    let threads = min(256, x.shape[1])
    precondition(x.dtype == .float16 && x.ndim == 2 && x.shape[0] % spatial == 0
      && (identityRoute || route.size == spatial))
    if simdRows {
      precondition(channels % 32 == 0)
      return normSIMD(
        [x, residual ?? x, weight, bias, route],
        template: [("C", channels), ("S", spatial), ("ADD_RESIDUAL", residual != nil), ("IDENTITY_ROUTE", identityRoute)],
        grid: (x.shape[0] * 32, 1, 1), threadGroup: (128, 1, 1), outputShapes: [x.shape],
        outputDTypes: [.float16])[0]
    }
    return norm(
      [x, residual ?? x, weight, bias, route],
      template: [
        ("C", channels), ("S", spatial), ("NT", threads), ("ADD_RESIDUAL", residual != nil), ("IDENTITY_ROUTE", identityRoute),
      ],
      grid: (x.shape[0] * threads, 1, 1), threadGroup: (threads, 1, 1), outputShapes: [x.shape],
      outputDTypes: [.float16])[0]
  }
  public static func patchMerge(_ x: MLXArray, route: MLXArray, spatial: Int) -> MLXArray {
    let channels = x.shape[1]
    let shape = [x.shape[0] / 4, x.shape[1] * 4]
    precondition(x.dtype == .float16 && x.shape[0] % (spatial * 4) == 0)
    return merge(
      [x, route], template: [("C", channels), ("S", spatial)], grid: (shape[0] * shape[1], 1, 1),
      threadGroup: (256, 1, 1), outputShapes: [shape], outputDTypes: [.float16])[0]
  }
  public static func mlpGelu(
    _ x: MLXArray, weight: MLXArray, bias: MLXArray, tile: GEMMTile = .baseline
  ) -> MLXArray {
    gemm(x, weight: weight, bias: bias, gelu: true, tile: tile)
  }
  public static func gemm(
    _ x: MLXArray, weight: MLXArray, bias: MLXArray, gelu: Bool = false,
    tile: GEMMTile = .baseline
  )
    -> MLXArray
  {
    if tile == .library {
      let projected = addMM(bias, x, weight.T)
      guard gelu else { return projected }
      let y = projected.asType(.float32)
      let cubicCoefficient = MLXArray(Float(0.044715))
      let tanhScale = MLXArray(Float(0.7978845608028654))
      let half = MLXArray(Float(0.5))
      let one = MLXArray(Float(1))
      let cubicTerm = ((y * cubicCoefficient) * y) * y
      let tanhInput = (y + cubicTerm) * tanhScale
      let halfY = y * half
      let onePlusTanh = one + tanh(tanhInput)
      let activated = halfY * onePlusTanh
      return activated.asType(.float16)
    }
    let rows = x.shape[0]
    let channels = x.shape[1]
    let expanded = weight.shape[0]
    let (bm, bn, bk) = tile.dimensions
    let wm = tile == .occupancy ? 4 : 2
    let wn = 2
    let threads = wm * wn * 32
    precondition(
      x.dtype == .float16 && channels % bk == 0 && expanded % bn == 0 && weight.shape[1] == channels
    )
    return mlp(
      [x, weight, bias],
      template: [
        ("K", channels), ("N", expanded), ("ACT", gelu), ("TBM", bm), ("TBN", bn), ("TBK", bk),
        ("RELAXED", tile == .relaxed || tile == .alignedRelaxed),
        ("FAST_FULL_TILES", tile == .aligned || tile == .alignedRelaxed),
        ("M_ALIGNED", rows % bm == 0),
        ("TWM", wm), ("TWN", wn),
      ],
      grid: ((expanded / bn) * threads, (rows + bm - 1) / bm, 1), threadGroup: (threads, 1, 1),
      outputShapes: [[rows, expanded]], outputDTypes: [.float16])[0]
  }
}
