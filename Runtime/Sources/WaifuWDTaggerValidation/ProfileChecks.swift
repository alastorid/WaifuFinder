import Foundation
import WDTaggerMLX

/// Pure policy tests: synthetic feature contracts, never simulated performance.
func profileChecks() throws -> [[String: Any]] {
  var checks: [[String: Any]] = []
  func expect(_ name: String, _ condition: @autoclosure () throws -> Bool) throws {
    guard try condition() else { throw WDNativeError.invalid("Profile check failed: \(name)") }
    checks.append(["name": "profile-\(name)", "passed": true])
  }
  func rejects(_ name: String, _ body: () throws -> Void) throws {
    do { try body() } catch WDNativeError.invalid { checks.append(["name": "profile-\(name)", "passed": true]); return }
    throw WDNativeError.invalid("Profile check expected diagnostic: \(name)")
  }
  func gpu(_ name: String = "Unknown future GPU", modern: Bool = true, bytes: Int = 32768,
           threads: Int = 1024, matrix: Bool = true) -> WDGPUCapabilities {
    .init(name: name, apple7: matrix, apple9: modern, threadgroupBytes: bytes, threadgroupWidth: threads)
  }
  for name in ["Apple M1", "Apple M1 Pro", "Apple M1 Max", "Apple M1 Ultra", "Apple M2", "Apple M2 Pro", "Apple M2 Max", "Apple M2 Ultra"] {
    let r = try WDOptimizationRegistry.resolve(gpu: gpu(name, modern: false), batch: 8)
    try expect("conservative-\(name)", r.provenance == .architectureDerived && r.stages[2].tiles[.qkv] == .baseline && r.fusedQKV)
  }
  let measured = try WDOptimizationRegistry.resolve(gpu: gpu("Apple M3"), batch: 8)
  try expect("measured-M3", measured.provenance == .measured && measured.stages[2].tiles[.qkv] == .m64n64k32 && measured.stages[2].tiles[.fc2] == .m64n64k32 && measured.stages.allSatisfy { $0.attentionQueryRows == 16 && !$0.simdNorm })
  try expect("dense-default", !measured.compactBias)
  let inherited = try WDOptimizationRegistry.resolve(gpu: gpu(), batch: 8)
  try expect("unknown-modern-inherits", inherited.provenance == .inherited && inherited.stages[2].tiles[.qkv] == .m64n64k32)
  let misleading = try WDOptimizationRegistry.resolve(gpu: gpu("Apple M999", modern: false), batch: 8)
  try expect("names-do-not-invent-features", misleading.provenance == .architectureDerived)
  for batch in 1...7 {
    let r = try WDOptimizationRegistry.resolve(gpu: gpu("Apple M3"), batch: batch)
    try expect("partial-batch-\(batch)", r.stages.allSatisfy { $0.attentionQueryRows == 32 && $0.tiles.values.allSatisfy { $0 == .baseline } })
  }
  let resource = try WDOptimizationRegistry.resolve(gpu: gpu(bytes: 8192), batch: 8)
  try expect("resource-fallback", resource.provenance == .architectureDerived && resource.reason.contains("needs"))
  let repaired = try WDOptimizationRegistry.resolve(gpu: gpu(bytes: 8192), batch: 8,
    gemmOverrides: [.qkv: Array(repeating: .baseline, count: 4), .fc2: Array(repeating: .baseline, count: 4)])
  try expect("valid-overrides-repair-candidate", repaired.baseProvenance == .inherited && repaired.stages[0].attentionQueryRows == 16)
  let partial = try WDOptimizationRegistry.resolve(gpu: gpu(bytes: 8192), batch: 8, compactBias: true)
  try expect("unrelated-override-allows-resource-fallback", partial.baseProvenance == .architectureDerived && partial.compactBias)
  let universal = try WDOptimizationRegistry.resolve(gpu: gpu(), batch: 8,
    profiles: WDOptimizationRegistry.profiles.filter { $0.provenance == .universal })
  try expect("universal-remains-fused", universal.provenance == .universal && universal.fusedQKV)
  let override = try WDOptimizationRegistry.resolve(gpu: gpu("Apple M3"), batch: 8, gemmTile: .library,
    gemmOverrides: [.qkv: [.baseline, .baseline, .m64n64k32, .baseline]], attentionQueryRows: 16,
    simdNorm: true, compactBias: true)
  try expect("override-precedence", override.provenance == .userOverride && override.stages[2].tiles[.qkv] == .m64n64k32 && override.stages[2].tiles[.fc2] == .library && override.stages[0].attentionQueryRows == 16 && override.compactBias && override.stages[0].simdNorm)
  let global = try WDOptimizationRegistry.resolve(gpu: gpu("Apple M3"), batch: 8, gemmTile: .baseline)
  try expect("legacy-global-pins-attention", global.stages.allSatisfy { $0.attentionQueryRows == 32 })
  let standard = try WDOptimizationRegistry.resolve(gpu: gpu("Apple M3"), batch: 8, simdNorm: false)
  try expect("explicit-standard-preserves-M3", standard.stages[2].tiles[.qkv] == .m64n64k32)
  let unfused = try WDOptimizationRegistry.resolve(gpu: gpu("Apple M3"), batch: 8, fusedQKV: false)
  try expect("comparison-baseline", unfused.stages[2].tiles[.qkv] == .baseline)
  var customStages = Array(repeating: WDStageKernelConfiguration(), count: 4)
  customStages[3].attentionQueryRows = 16
  let exact = WDGPUOptimizationProfile(identifier: "test-measured", provenance: .measured,
    match: .exactDevice("Unknown future GPU"), workload: .init(batches: [8]), stages: customStages,
    evidence: "synthetic policy test only", reason: "exact registry entry")
  let replaced = try WDOptimizationRegistry.resolve(gpu: gpu(), batch: 8,
    profiles: WDOptimizationRegistry.profiles + [exact])
  try expect("new-exact-entry-supersedes-inheritance", replaced.profile == "test-measured" && replaced.stages[3].attentionQueryRows == 16)
  let stale = WDGPUOptimizationProfile(identifier: "stale", provenance: .measured,
    match: .exactDevice("Unknown future GPU"), kernelRevision: "old", stages: customStages,
    evidence: "test", reason: "test")
  try expect("revision-scoped-evidence", WDOptimizationRegistry.resolve(gpu: gpu(), batch: 8,
    profiles: [stale] + WDOptimizationRegistry.profiles).profile != "stale")
  let otherShape = WDGPUOptimizationProfile(identifier: "other-shape", provenance: .measured,
    match: .exactDevice("Unknown future GPU"), workload: .init(imageSide: 224), stages: customStages,
    evidence: "test", reason: "test")
  try expect("shape-scoped-evidence", WDOptimizationRegistry.resolve(gpu: gpu(), batch: 8,
    profiles: [otherShape] + WDOptimizationRegistry.profiles).profile != "other-shape")
  let malformed = WDGPUOptimizationProfile(identifier: "malformed", provenance: .measured,
    match: .exactDevice("Unknown future GPU"), stages: [], evidence: "test", reason: "test")
  try expect("malformed-automatic-entry-falls-through", WDOptimizationRegistry.resolve(gpu: gpu(), batch: 8,
    profiles: [malformed] + WDOptimizationRegistry.profiles).profile != "malformed")
  try rejects("invalid-stage-count") { _ = try WDOptimizationRegistry.resolve(gpu: gpu(), batch: 8, gemmOverrides: [.fc2: [.baseline]]) }
  try rejects("invalid-attention") { _ = try WDOptimizationRegistry.resolve(gpu: gpu(), batch: 8, attentionQueryRows: 64) }
  try rejects("explicit-resource-error") { _ = try WDOptimizationRegistry.resolve(gpu: gpu(bytes: 8192), batch: 8, gemmTile: .m64n128k32) }
  try rejects("missing-matrix-support") { _ = try WDOptimizationRegistry.resolve(gpu: gpu(matrix: false), batch: 8) }
  try rejects("insufficient-threads") { _ = try WDOptimizationRegistry.resolve(gpu: gpu(threads: 128), batch: 8) }
  try rejects("insufficient-memory") { _ = try WDOptimizationRegistry.resolve(gpu: gpu(bytes: 4096), batch: 8) }
  try rejects("zero-batch") { _ = try WDOptimizationRegistry.resolve(gpu: gpu(), batch: 0) }
  try rejects("index-overflow") { _ = try WDOptimizationRegistry.resolve(gpu: gpu(), batch: Int.max) }
  let encoded = try JSONSerialization.data(withJSONObject: measured.diagnostics)
  try expect("diagnostics-serializable", !encoded.isEmpty)
  try expect("unconstrained-diagnostics-serializable", !JSONSerialization.data(withJSONObject: universal.diagnostics).isEmpty)
  return checks
}
