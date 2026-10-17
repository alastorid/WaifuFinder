import Darwin
import Foundation

/// The in-process Core ML backend shares the native workers' residency lock.
final class RealESRGANJobLock {
    private let descriptor: Int32
    init() throws {
        let directory = FileManager.default.urls(for:.applicationSupportDirectory,in:.userDomainMask)[0].appendingPathComponent("Waifu/Uncover")
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
        let fd = open(directory.appendingPathComponent("native.lock").path,O_CREAT|O_RDWR|O_NOFOLLOW,0o600)
        guard fd >= 0 else { throw UncoverError.backend("Could not open the native restoration lock.") }
        guard flock(fd,LOCK_EX|LOCK_NB) == 0 else { close(fd);throw UncoverError.backend("Another native restoration job is running.") }
        descriptor = fd
    }
    deinit { flock(descriptor,LOCK_UN);close(descriptor) }
}

final class RealESRGANMemorySampler: @unchecked Sendable {
    private let lock = NSLock()
    private var peak: UInt64 = 0
    private let timer = DispatchSource.makeTimerSource(queue:DispatchQueue(label:"realesrgan-footprint"))
    let budget = min(UInt64(6_000_000_000),ProcessInfo.processInfo.physicalMemory/2)
    init() {
        timer.schedule(deadline:.now(),repeating:.milliseconds(20))
        timer.setEventHandler { [weak self] in
            guard let self else { return }
            let bytes = Self.footprint();self.lock.lock();self.peak=max(self.peak,bytes);self.lock.unlock()
        }
        timer.resume()
    }
    deinit { timer.cancel() }
    var peakBytes: UInt64 { lock.lock();defer { lock.unlock() };return max(peak,Self.footprint()) }
    func checkBudget() throws {
        guard peakBytes <= budget else { throw UncoverError.backend("RealESRGAN exceeded its memory budget. Reduce the selected region.") }
    }
    private static func footprint() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size/MemoryLayout<integer_t>.size)
        let status = withUnsafeMutablePointer(to:&info) { $0.withMemoryRebound(to:integer_t.self,capacity:Int(count)) { task_info(mach_task_self_,task_flavor_t(TASK_VM_INFO),$0,&count) } }
        return status == KERN_SUCCESS ? info.phys_footprint : 0
    }
}
