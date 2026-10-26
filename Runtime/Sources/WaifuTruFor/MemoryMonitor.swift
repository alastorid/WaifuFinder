import Darwin
import Foundation
import MLX
import TruForMLX

/// MLX's allocator limit is advisory. Monitor the complete process footprint,
/// including Metal/framework allocations, and stop with headroom below 2 GB.
final class TruForMemoryMonitor: @unchecked Sendable {
    private let lock=NSLock()
    private var peak=0
    private let timer=DispatchSource.makeTimerSource(queue:DispatchQueue(label:"trufor-memory"))
    static func footprint()->Int {
        var info=task_vm_info_data_t()
        var count=mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size/MemoryLayout<integer_t>.size)
        let status=withUnsafeMutablePointer(to:&info) { pointer in
            pointer.withMemoryRebound(to:integer_t.self,capacity:Int(count)) {
                task_info(mach_task_self_,task_flavor_t(TASK_VM_INFO),$0,&count)
            }
        }
        guard status==KERN_SUCCESS else {
            fputs("TruFor stopped because its 2 GB memory monitor is unavailable.\n",stderr);_exit(75)
        }
        return Int(info.phys_footprint)
    }
    init() {
        timer.schedule(deadline:.now(),repeating:.milliseconds(5))
        timer.setEventHandler { [weak self] in self?.sample() }
        timer.resume()
    }
    deinit { timer.cancel() }
    @discardableResult func sample()->Int {
        var bytes=Self.footprint()
        lock.lock();peak=max(peak,bytes);lock.unlock()
        if bytes>=TruForMemoryPolicy.stopBytes {
            TruForMemoryPolicy.releaseUnusedCPU();bytes=Self.footprint()
        }
        if bytes>=TruForMemoryPolicy.stopBytes {
            fputs("TruFor memory: physical \(bytes), active MLX \(Memory.activeMemory), cache \(Memory.cacheMemory).\n",stderr)
            fputs("TruFor stopped before its 2 GB memory budget was exhausted. Use a smaller image.\n",stderr)
            fflush(stderr)
            // Do not tear down MLX synchronization while the inference thread runs.
            _exit(75)
        }
        return bytes
    }
    func reset() {
        lock.lock();peak=Self.footprint();lock.unlock()
        Memory.peakMemory=0
    }
    func report(_ phase:String) {
        let current=sample()
        guard ProcessInfo.processInfo.environment["WAIFU_TRUFOR_PROFILE"] != nil ||
            ProcessInfo.processInfo.environment["WAIFU_TRUFOR_MEMORY_REPORT"] == "1" else { return }
        lock.lock();let maximum=peak;lock.unlock()
        let values:[String:Any]=["phase":phase,"physicalBytes":current,"peakPhysicalBytes":maximum,
            "peakMLXBytes":Memory.peakMemory,"activeMLXBytes":Memory.activeMemory,"cachedMLXBytes":Memory.cacheMemory,
            "budgetBytes":TruForMemoryPolicy.budgetBytes,"stopBytes":TruForMemoryPolicy.stopBytes]
        if let data=try? JSONSerialization.data(withJSONObject:values,options:.sortedKeys) {
            FileHandle.standardError.write(data);FileHandle.standardError.write(Data([10]))
        }
    }
}
