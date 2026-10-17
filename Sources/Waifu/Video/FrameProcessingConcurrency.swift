import Foundation

/// Share a native-frame working budget between raster preparation and scoring.
/// Each slot allows for source pixels, a native raster intermediate and the
/// analysis image. This estimates concurrency, not Vision or decoder allocations.
struct FrameProcessingConcurrency: Sendable {
    let preparationWorkers: Int
    let scoringWorkers: Int

    init(width: Int, height: Int, analysisWidth: Int = 512,
         processorCount: Int = ProcessInfo.processInfo.activeProcessorCount,
         physicalMemory: UInt64 = ProcessInfo.processInfo.physicalMemory) {
        let processors = max(1, processorCount)
        let analysisBytes = Double(analysisWidth) * Double(analysisWidth) * 4
        let bytesPerSlot = max(1, Double(width) * Double(height) * 8 + analysisBytes)
        let budget = Double(min(256 * 1024 * 1024, physicalMemory / 32))
        let slots = max(2, Int(min(6, budget / bytesPerSlot)))
        preparationWorkers = processors > 1 && slots >= 4 ? 2 : 1
        scoringWorkers = min(4, processors, slots - preparationWorkers)
    }
}
