import Foundation
import Darwin

/// Share a native-frame working budget between raster preparation and scoring.
/// Each slot allows for source pixels, a native raster intermediate and the
/// analysis image. This estimates concurrency, not Vision or decoder allocations.
struct FrameProcessingConcurrency: Sendable {
    let preparationWorkers: Int
    let scoringWorkers: Int

    init(width: Int, height: Int, analysisWidth: Int = 512,
         processorCount: Int = ProcessInfo.processInfo.activeProcessorCount,
         physicalMemory: UInt64 = ProcessInfo.processInfo.physicalMemory,
         performanceCoreCount: Int? = nil) {
        let processors = max(1, min(processorCount, performanceCoreCount ?? Self.performanceCoreCount ?? processorCount))
        let analysisBytes = Double(analysisWidth) * Double(analysisWidth) * 4
        let bytesPerSlot = max(1, Double(width) * Double(height) * 8 + analysisBytes)
        let budget = Double(min(256 * 1024 * 1024, physicalMemory / 32))
        let slots = max(2, Int(min(8, budget / bytesPerSlot)))
        preparationWorkers = min(4, processors, max(1, slots / 2))
        scoringWorkers = min(4, processors, slots - preparationWorkers)
    }

    /// Apple exposes performance and efficiency CPU clusters as performance
    /// levels. Level zero is the fastest cluster; fall back to the active
    /// processor count on systems that do not publish the topology keys.
    private static var performanceCoreCount: Int? {
        var count: Int32 = 0
        var size = MemoryLayout<Int32>.size
        guard sysctlbyname("hw.perflevel0.physicalcpu", &count, &size, nil, 0) == 0,
              count > 0 else { return nil }
        return Int(count)
    }
}
