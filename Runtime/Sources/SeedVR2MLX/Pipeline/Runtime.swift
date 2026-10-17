import Foundation
import MLX

/// One isolated worker performs one job serially. Force graph boundaries so evaluated
/// transformer blocks and causal VAE slices release their intermediate buffers.
public enum SeedVR2Runtime {
    enum CausalState { case disabled, initializing, active }
    static var causalState: CausalState = .disabled
    public static var memoryBudget = 6 * 1024 * 1024 * 1024
    public static var progress: ((String) -> Void)?
    public static var phaseSeconds: [String: Double] = [:]
    public static var disableSharedAttentionFusion = false
    public static var disableVAENormalizationFusion = false
    public static var disableVAECausalFusion = false
    public static var disableVAEPointwiseConv = false
    public static var disableVAEResidualFusion = false
    public static func measure<T>(_ phase: String, _ operation: () throws -> T) rethrows -> T {
        let start = ProcessInfo.processInfo.systemUptime
        defer { phaseSeconds[phase, default: 0] += ProcessInfo.processInfo.systemUptime - start }
        return try operation()
    }
    public static func materialize(_ arrays: MLXArray...) {
        eval(arrays)
        // The worker caps MLX's cache at 64 MB. Keep reusable buffers between
        // operations; evaluation still releases each completed lazy graph.
        if Memory.activeMemory > memoryBudget {
            fputs("Native inference exceeded the configured working-memory budget.\n", stderr)
            exit(75)
        }
    }
}
