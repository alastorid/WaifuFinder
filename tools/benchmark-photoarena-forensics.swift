// Compile with PhotoArena.swift, ArenaAnalysisTypes.swift and ArenaForensics.swift.
// Apple-native deterministic 40 MP coverage benchmark. No interactive UI or model inference.
import CoreImage
import Darwin
import Foundation
import ImageIO
import UniformTypeIdentifiers

private func memorySample() -> (UInt64, UInt64) {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let status = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
    }
    return status == KERN_SUCCESS ? (info.phys_footprint, info.compressed) : (0, 0)
}
private final class Sampler: @unchecked Sendable {
    private let lock = NSLock()
    private var footprint: UInt64 = 0
    private var compressed: UInt64 = 0
    private let timer = DispatchSource.makeTimerSource(queue: .global(qos: .utility))
    func start() {
        timer.schedule(deadline: .now(), repeating: .milliseconds(20))
        timer.setEventHandler { [weak self] in
            guard let self else { return }
            let sample = memorySample()
            self.lock.lock(); self.footprint=max(self.footprint,sample.0); self.compressed=max(self.compressed,sample.1); self.lock.unlock()
        }
        timer.resume()
    }
    func finish() -> (UInt64,UInt64) { timer.cancel(); lock.lock(); defer { lock.unlock() }; return (footprint,compressed) }
}
@main struct ArenaForensicsBenchmark {
    static func main() throws {
        let args = CommandLine.arguments
        let report = URL(fileURLWithPath: args.count > 1 ? args[1] : "docs/photoarena-forensics-40mp.json")
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("photoarena-40mp-\(UUID())")
        try FileManager.default.createDirectory(at: directory,withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let source = directory.appendingPathComponent("pattern.png")
        let width=8000, height=5000
        let context = CIContext(options: [.cacheIntermediates: false])
        let checker = CIFilter(name: "CICheckerboardGenerator",parameters: ["inputWidth": 17.0,"inputColor0": CIColor(red: 0.13,green: 0.31,blue: 0.57),"inputColor1": CIColor(red: 0.83,green: 0.67,blue: 0.29)])!.outputImage!
        let pattern = checker.cropped(to: CGRect(x: 0,y: 0,width: width,height: height))
        try context.writePNGRepresentation(of: pattern,to: source,format: .RGBA8,colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
        context.clearCaches()
        let sampler=Sampler(); sampler.start()
        let start=Date()
        let analysis=try ArenaForensics.analyzeFiles([ArenaSource(id: "native-A",url: source,seconds: 0),ArenaSource(id: "native-B",url: source,seconds: 0)])
        let elapsed=Date().timeIntervalSince(start)
        let peak=sampler.finish(), end=memorySample()
        let object: [String:Any] = ["schemaVersion": 1,"fixture": "Apple Core Image deterministic checkerboard, two identical 8000×5000 sources", "limitations": ["Coverage and throughput benchmark only; repetitive pattern does not validate registration accuracy or forensic quality.","20ms sampled whole-process physical footprint includes Apple image decode/render allocations; not a guaranteed instantaneous peak.","No VLM/MLX/Core ML inference; ANE placement and utilization are unmeasured.","macOS paging remains enabled; process swap-ins were not measured."],"referenceWidth": analysis.width,"referenceHeight": analysis.height,"megapixels": 40,"regionCount": analysis.regions.count,"acceptedSourceCount": analysis.sources.filter(\.accepted).count,"maximumDifference": analysis.regions.flatMap(\.measurements).map(\.difference).max() ?? -1,"unresolvedRegionCount": analysis.regions.filter(\.unresolved).count,"analysisSeconds": elapsed,"sampledPeakPhysicalFootprintBytes": peak.0,"sampledPeakCompressedBytes": peak.1,"finalPhysicalFootprintBytes": end.0,"finalCompressedBytes": end.1,"processSwapIns": NSNull(),"modelInference": false,"hardware": ProcessInfo.processInfo.hostName,"operatingSystem": ProcessInfo.processInfo.operatingSystemVersionString]
        try JSONSerialization.data(withJSONObject: object,options: [.prettyPrinted,.sortedKeys]).write(to: report,options: .atomic)
        print("40MP analysis: \(analysis.regions.count) regions, \(String(format: "%.3f",elapsed))s, sampled physical footprint \(peak.0) bytes")
    }
}
