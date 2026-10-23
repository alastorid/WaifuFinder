import Foundation
import Metal

/// Feature sets are compatibility contracts, not marketing generations.
public struct WDGPUCapabilities {
  public let name: String
  public let apple7: Bool
  public let apple9: Bool
  public let threadgroupBytes: Int
  public let threadgroupWidth: Int
  public let threadgroupHeight: Int
  public let threadgroupDepth: Int

  public init(name: String, apple7: Bool, apple9: Bool, threadgroupBytes: Int,
              threadgroupWidth: Int, threadgroupHeight: Int = 1, threadgroupDepth: Int = 1) {
    self.name = name; self.apple7 = apple7; self.apple9 = apple9
    self.threadgroupBytes = threadgroupBytes; self.threadgroupWidth = threadgroupWidth
    self.threadgroupHeight = threadgroupHeight; self.threadgroupDepth = threadgroupDepth
  }

  public static func discover() throws -> Self {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw WDNativeError.invalid("Native WDTagger requires a Metal device")
    }
    let size = device.maxThreadsPerThreadgroup
    return Self(name: device.name, apple7: device.supportsFamily(.apple7),
      apple9: device.supportsFamily(.apple9), threadgroupBytes: device.maxThreadgroupMemoryLength,
      threadgroupWidth: size.width, threadgroupHeight: size.height, threadgroupDepth: size.depth)
  }
}

public enum WDProfileProvenance: String {
  case measured, inherited, architectureDerived = "architecture-derived"
  case universal, userOverride = "user-override"
}

public struct WDStageKernelConfiguration {
  public var tiles: [WDNativeTagger.GEMMOperation: WDMetal.GEMMTile]
  public var attentionQueryRows: Int
  public var simdNorm: Bool

  public init(tile: WDMetal.GEMMTile = .baseline, attentionQueryRows: Int = 32, simdNorm: Bool = false) {
    tiles = Dictionary(uniqueKeysWithValues: WDNativeTagger.GEMMOperation.allCases.map { ($0, tile) })
    self.attentionQueryRows = attentionQueryRows; self.simdNorm = simdNorm
  }
}

/// Registry entries describe this fixed graph, rather than arbitrary GEMMs.
public struct WDProfileWorkload {
  public let imageSide: Int
  public let windowTokens: Int
  public let headDimension: Int
  public let channels: [Int]
  public let batches: [Int]?

  public init(imageSide: Int = 448, windowTokens: Int = 196, headDimension: Int = 32,
              channels: [Int] = [128, 256, 512, 1024], batches: [Int]? = nil) {
    self.imageSide = imageSide; self.windowTokens = windowTokens
    self.headDimension = headDimension; self.channels = channels; self.batches = batches
  }

  fileprivate func matches(batch: Int) -> Bool {
    imageSide == 448 && windowTokens == 196 && headDimension == 32
      && channels == [128, 256, 512, 1024] && (batches?.contains(batch) ?? true)
  }
}

public struct WDResolvedKernelConfiguration {
  public let profile: String
  public let provenance: WDProfileProvenance
  public let baseProvenance: WDProfileProvenance
  public let reason: String
  public let evidence: String
  public let matchingRule: String
  public let profileWorkload: WDProfileWorkload
  public let gpu: WDGPUCapabilities
  public let batch: Int
  public let stages: [WDStageKernelConfiguration]
  public let compactBias: Bool
  public let fusedQKV: Bool
  public let overrides: [String]

  /// JSON-compatible, emitted on request by engineering tools, never per inference.
  public var diagnostics: [String: Any] {
    ["gpu": gpu.name, "profile": profile, "provenance": provenance.rawValue,
     "baseProvenance": baseProvenance.rawValue, "reason": reason, "evidence": evidence,
     "matchingRule": matchingRule,
     "profileWorkload": ["imageSide": profileWorkload.imageSide,
       "windowTokens": profileWorkload.windowTokens, "headDimension": profileWorkload.headDimension,
       "channels": profileWorkload.channels, "batches": profileWorkload.batches as Any? ?? "all valid batches"],
     "kernelRevision": WDNativeTagger.implementationVersion, "batch": batch,
     "input": "448x448; FP16 matrices/FP32 accumulation; 196 tokens/head32",
     "capabilities": ["apple7": gpu.apple7, "apple9": gpu.apple9,
       "threadgroupBytes": gpu.threadgroupBytes, "maxThreads": [gpu.threadgroupWidth, gpu.threadgroupHeight, gpu.threadgroupDepth]],
     "compactBias": compactBias, "fusedQKV": fusedQKV, "overrides": overrides,
     "stages": stages.enumerated().map { index, stage -> [String: Any] in
       let c = [128, 256, 512, 1024][index], m = batch * [12544, 3136, 784, 196][index]
       return ["stage": index + 1, "rows": m, "channels": c,
         "attentionQueryRows": stage.attentionQueryRows, "attentionKeyRows": 32,
         "normalization": stage.simdNorm ? "NormSIMD" : "NormRoute",
         "gemm": Dictionary(uniqueKeysWithValues: stage.tiles.map { ($0.key.rawValue, $0.value.rawValue) }),
         "qkvEffective": fusedQKV ? stage.tiles[.qkv]!.effectiveQKVTile.rawValue : stage.tiles[.qkv]!.rawValue,
         "qkvBackend": fusedQKV && stage.tiles[.qkv]! != .library ? "native-fused"
           : stage.tiles[.qkv]! == .library ? "mlx-gemm-cosine-pack" : "native-gemm-cosine-pack",
         "operationShapes": Dictionary(uniqueKeysWithValues: stage.tiles.keys.map { operation in
           (operation.rawValue, ["M": m,
             "K": operation == .fc2 ? 4 * c : operation == .reduction ? 2 * c : c,
             "N": operation == .fc1 ? 4 * c : operation == .qkv ? 3 * c : c,
             "executed": operation != .reduction || index > 0] as [String: Any])
         }),
         "rowAligned": Dictionary(uniqueKeysWithValues: stage.tiles.map {
           ($0.key.rawValue, $0.value == .library ? false : m % $0.value.dimensions.0 == 0)
         })]
     }]
  }
}

/// Entries are data: adding a measured device/workload does not change inference.
public struct WDGPUOptimizationProfile {
  public enum Match {
    case exactDevice(String), appleFamily9, appleMatrix, universal
    fileprivate func matches(_ gpu: WDGPUCapabilities) -> Bool {
      switch self {
      case .exactDevice(let name): return gpu.name == name
      case .appleFamily9: return gpu.apple9
      case .appleMatrix: return gpu.apple7
      case .universal: return true
      }
    }
    fileprivate var priority: Int {
      switch self { case .exactDevice: return 0; case .appleFamily9: return 1; case .appleMatrix: return 2; case .universal: return 3 }
    }
    fileprivate var description: String {
      switch self {
      case .exactDevice(let name): return "exact device: \(name)"
      case .appleFamily9: return "supportsFamily(apple9)"
      case .appleMatrix: return "supportsFamily(apple7)"
      case .universal: return "universal; native compatibility required"
      }
    }
  }
  public let identifier: String
  public let provenance: WDProfileProvenance
  public let match: Match
  public let kernelRevision: String
  public let workload: WDProfileWorkload
  public let requiresStandardFused: Bool
  public let stages: [WDStageKernelConfiguration]
  public let compactBias: Bool
  public let evidence: String
  public let reason: String

  public init(identifier: String, provenance: WDProfileProvenance, match: Match,
    kernelRevision: String = WDNativeTagger.implementationVersion,
    workload: WDProfileWorkload = .init(),
    requiresStandardFused: Bool = false, stages: [WDStageKernelConfiguration],
    compactBias: Bool = false, evidence: String, reason: String) {
    self.identifier = identifier; self.provenance = provenance; self.match = match
    self.kernelRevision = kernelRevision
    self.workload = workload
    self.requiresStandardFused = requiresStandardFused; self.stages = stages
    self.compactBias = compactBias; self.evidence = evidence; self.reason = reason
  }
}

public enum WDOptimizationRegistry {
  private static var m3Stages: [WDStageKernelConfiguration] {
    var stages = Array(repeating: WDStageKernelConfiguration(attentionQueryRows: 16), count: 4)
    stages[2].tiles[.qkv] = .m64n64k32; stages[2].tiles[.fc2] = .m64n64k32
    return stages
  }
  /// Round 4 base-M4 measurement: 64×64×16 (WM=2/WN=2) won all six cooled paired
  /// rounds against the M3-derived inheritance and the universal baseline.
  private static var m4Stages: [WDStageKernelConfiguration] {
    var stages = Array(repeating: WDStageKernelConfiguration(attentionQueryRows: 16), count: 4)
    stages[2].tiles[.qkv] = .m64n64k16; stages[2].tiles[.fc2] = .m64n64k16
    return stages
  }
  public static let profiles: [WDGPUOptimizationProfile] = [
    .init(identifier: "measured-m3-b8-round2", provenance: .measured, match: .exactDevice("Apple M3"),
      workload: .init(batches: [8]), requiresStandardFused: true, stages: m3Stages,
      evidence: "docs/wd-tagger-native-kernel-review.json#round2; 1.5–1.9% prediction gain; exact on 64 synthetic inputs",
      reason: "Exact measured base M3 and batch-eight workload"),
    .init(identifier: "measured-m4-b8-round4", provenance: .measured, match: .exactDevice("Apple M4"),
      workload: .init(batches: [8]), requiresStandardFused: true, stages: m4Stages,
      evidence: "docs/wd-tagger-native-kernel-review.json#round4; 6/6 cooled paired wins, median +0.2–0.3% over inherited and universal; byte-identical probabilities",
      reason: "Exact measured base M4 (Mac mini) and batch-eight workload"),
    .init(identifier: "apple-family9-m3-derived-b8", provenance: .inherited, match: .appleFamily9,
      workload: .init(batches: [8]), requiresStandardFused: true, stages: m3Stages,
      evidence: "Inherited Round 2 configuration; unmeasured on this device",
      reason: "Reports Apple family 9 compatibility; resources validated; no claim of optimal occupancy or throughput"),
    .init(identifier: "apple-matrix-conservative", provenance: .architectureDerived, match: .appleMatrix,
      stages: Array(repeating: WDStageKernelConfiguration(), count: 4),
      evidence: "Portable native implementation; no device-specific performance claim",
      reason: "Apple SIMD matrix/reduction support; conservative native geometry for this workload"),
    .init(identifier: "universal-native", provenance: .universal, match: .universal,
      stages: Array(repeating: WDStageKernelConfiguration(), count: 4),
      evidence: "Portable native implementation; hardware compatibility still required",
      reason: "No more specific applicable profile")
  ]

  public static func resolve(gpu: WDGPUCapabilities, batch: Int,
    gemmTile: WDMetal.GEMMTile? = nil,
    gemmOverrides: [WDNativeTagger.GEMMOperation: [WDMetal.GEMMTile]] = [:],
    attentionQueryRows: Int? = nil, simdNorm: Bool? = nil, compactBias: Bool? = nil,
    fusedQKV: Bool = true, profiles: [WDGPUOptimizationProfile] = profiles
  ) throws -> WDResolvedKernelConfiguration {
    guard batch > 0 && batch <= Int(Int32.max) / (12544 * 512) else {
      throw WDNativeError.invalid("Native profile requires a positive batch within 32-bit kernel indexing limits")
    }
    guard gemmOverrides.values.allSatisfy({ $0.count == 4 }) else {
      throw WDNativeError.invalid("GEMM operation overrides require exactly four stage entries")
    }
    // All retained native attention kernels use SIMD matrix operations. Never
    // silently replace the native architecture on an unsupported device.
    guard gpu.apple7, gpu.threadgroupWidth >= 256, gpu.threadgroupHeight >= 1, gpu.threadgroupDepth >= 1 else {
      throw WDNativeError.invalid("\(gpu.name): native kernels require Apple7 SIMD matrix/reduction support and 256-wide threadgroups")
    }
    var rejected: [String] = []
    let ordered = profiles.enumerated().sorted {
      let a = $0.element.match.priority, b = $1.element.match.priority
      return a == b ? $0.offset < $1.offset : a < b
    }.map(\.element)
    for profile in ordered {
      guard profile.match.matches(gpu), profile.kernelRevision == WDNativeTagger.implementationVersion,
        profile.workload.matches(batch: batch),
        !profile.requiresStandardFused || (fusedQKV && simdNorm != true) else { continue }
      var stages = profile.stages
      guard stages.count == 4 else {
        rejected.append("\(profile.identifier): requires four stages"); continue
      }
      var overrides: [String] = []
      // Legacy explicit global GEMM configuration also pins baseline attention.
      if let gemmTile {
        stages = Array(repeating: WDStageKernelConfiguration(tile: gemmTile, simdNorm: simdNorm ?? false), count: 4)
        overrides.append("global GEMM (and baseline attention scheduling)")
      }
      for (operation, values) in gemmOverrides {
        for index in 0..<4 { stages[index].tiles[operation] = values[index] }
        overrides.append("\(operation.rawValue) stages 1–4")
      }
      for index in stages.indices {
        if let attentionQueryRows { stages[index].attentionQueryRows = attentionQueryRows }
        if let simdNorm { stages[index].simdNorm = simdNorm }
      }
      if attentionQueryRows != nil { overrides.append("attention query rows") }
      if simdNorm != nil { overrides.append("normalization") }
      if compactBias != nil { overrides.append("bias storage") }
      if !fusedQKV { overrides.append("unfused QKV") }
      do {
        try validate(stages: stages, gpu: gpu, batch: batch, fusedQKV: fusedQKV,
          explicitOperations: Set(gemmOverrides.keys), globalOverride: gemmTile != nil,
          attentionOverride: attentionQueryRows != nil)
      } catch let error as ValidationFailure {
        // Explicit invalid choices are errors; automatic incompatible entries
        // descend through the registry, with the reason retained in diagnostics.
        if error.explicit { throw WDNativeError.invalid(error.description) }
        rejected.append("\(profile.identifier): \(error)"); continue
      }
      return .init(profile: profile.identifier,
        provenance: overrides.isEmpty ? profile.provenance : .userOverride,
        baseProvenance: profile.provenance,
        reason: ([profile.reason] + rejected).joined(separator: "; "), evidence: profile.evidence,
        matchingRule: profile.match.description, profileWorkload: profile.workload,
        gpu: gpu, batch: batch, stages: stages, compactBias: compactBias ?? profile.compactBias,
        fusedQKV: fusedQKV, overrides: overrides.sorted())
    }
    throw WDNativeError.invalid("\(gpu.name): no compatible native profile. \(rejected.joined(separator: "; "))")
  }

  private struct ValidationFailure: Error, CustomStringConvertible {
    let description: String
    let explicit: Bool
  }

  private static func validate(stages: [WDStageKernelConfiguration], gpu: WDGPUCapabilities,
                               batch: Int, fusedQKV: Bool,
                               explicitOperations: Set<WDNativeTagger.GEMMOperation>,
                               globalOverride: Bool, attentionOverride: Bool) throws {
    guard stages.count == 4 else { throw ValidationFailure(description: "Profile requires four stages", explicit: false) }
    for (index, stage) in stages.enumerated() {
      guard [16, 32].contains(stage.attentionQueryRows), stage.attentionQueryRows * 4 <= gpu.threadgroupWidth,
        gpu.threadgroupBytes >= 2562 else {
          throw ValidationFailure(description: "Stage \(index + 1): unsupported Flash32 query geometry/resources", explicit: attentionOverride)
        }
      let c = [128, 256, 512, 1024][index]
      for operation in WDNativeTagger.GEMMOperation.allCases {
        guard let tile = stage.tiles[operation] else { throw ValidationFailure(description: "Missing \(operation.rawValue) stage \(index + 1)", explicit: false) }
        if operation == .reduction && index == 0 { continue } // No such operation.
        if tile == .library { continue }
        var (bm, bn, bk) = tile.dimensions
        var threads = tile == .occupancy ? 256 : 128
        let k = operation == .fc2 ? 4 * c : operation == .reduction ? 2 * c : c
        let n = operation == .fc1 ? 4 * c : operation == .qkv ? 3 * c : c
        var bytes = (bm + bn) * (bk + 8) * 2
        if operation == .qkv && fusedQKV, let geometry = WDMetal.qkvGeometry(tile: tile, heads: c / 32) {
          (bm, bn, bk) = (geometry.bm, geometry.bn, geometry.bk)
          threads = geometry.wm * 64
          bytes = (bm + bn) * (bk + 8) * 2 + bm * bn / 4
        }
        guard k % bk == 0, n % bn == 0, threads <= gpu.threadgroupWidth, bytes <= gpu.threadgroupBytes else {
          throw ValidationFailure(description: "\(gpu.name): \(operation.rawValue) stage \(index + 1), tile \(tile.rawValue), M=\(batch * [12544,3136,784,196][index]) K=\(k) N=\(n) needs \(bytes) threadgroup bytes / \(threads) threads; limits \(gpu.threadgroupBytes)/\(gpu.threadgroupWidth)", explicit: globalOverride || explicitOperations.contains(operation))
        }
      }
    }
  }
}
