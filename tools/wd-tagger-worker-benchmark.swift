// Native WDT3 protocol benchmark. Timings include IPC, input upload, completed
// inference, output validation/transfer and receipt of every probability row.
import CryptoKit
import Darwin
import Foundation

private func childFootprint(_ pid: Int32) -> UInt64 {
    var info = rusage_info_v4()
    let status = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
            proc_pid_rusage(pid, RUSAGE_INFO_V4, $0)
        }
    }
    return status == 0 ? info.ri_phys_footprint : 0
}

private final class PeakChildMemory: @unchecked Sendable {
    private let lock = NSLock()
    private var peak: UInt64 = 0
    private let timer = DispatchSource.makeTimerSource(queue: DispatchQueue(label: "wd.worker.memory"))
    private let pid: Int32
    init(pid: Int32) {
        self.pid = pid
        timer.schedule(deadline: .now(), repeating: .milliseconds(10))
        timer.setEventHandler { [weak self] in
            guard let self else { return }
            let value = childFootprint(self.pid)
            self.lock.withLock { self.peak = max(self.peak, value) }
        }
        timer.resume()
    }
    deinit { timer.cancel() }
    var bytes: UInt64 { lock.withLock { max(peak, childFootprint(pid)) } }
}

@main struct WorkerBenchmark {
    static func main() throws {
        let args = CommandLine.arguments
        func option(_ name: String) throws -> String {
            guard let i = args.firstIndex(of: name), i + 1 < args.count else {
                throw NSError(domain: "Supply \(name)", code: 1)
            }
            return args[i + 1]
        }
        let frameBytes = 448 * 448 * 4, rowBytes = 10_861 * 4
        let inputBytes = frameBytes * 8
        let corpus = try Data(contentsOf: URL(fileURLWithPath: option("--inputs")), options: .mappedIfSafe)
        let expected = try Data(contentsOf: URL(fileURLWithPath: option("--reference")))
        guard corpus.count == frameBytes * 64, expected.count == rowBytes * 64 else {
            throw NSError(domain: "Expected the retained 64-frame corpus", code: 1)
        }
        let reportURL = URL(fileURLWithPath: try option("--report"))
        let scratch = FileManager.default.temporaryDirectory.appendingPathComponent("wd-worker-bench-\(UUID())")
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true,
                                                attributes: [.posixPermissions: 0o700])
        defer { try? FileManager.default.removeItem(at: scratch) }
        let inputURL = scratch.appendingPathComponent("input.bin")
        FileManager.default.createFile(atPath: inputURL.path, contents: nil)
        let file = try FileHandle(forUpdating: inputURL)
        defer { try? file.close() }
        try file.truncate(atOffset: UInt64(inputBytes))
        guard let mapping = mmap(nil, inputBytes, PROT_READ | PROT_WRITE, MAP_SHARED, file.fileDescriptor, 0),
              mapping != MAP_FAILED else { throw NSError(domain: "mmap", code: 1) }
        defer { munmap(mapping, inputBytes) }
        let child = Process(), incoming = Pipe(), outgoing = Pipe()
        let logURL = reportURL.deletingPathExtension().appendingPathExtension("worker.log")
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        let log = try FileHandle(forWritingTo: logURL)
        defer { try? log.close() }
        child.executableURL = URL(fileURLWithPath: try option("--worker"))
        child.arguments = ["--weights", try option("--weights"), "--input", inputURL.path]
            + (args.contains("--copy-input") ? ["--copy-input"] : [])
            + (args.contains("--shared-input") ? ["--shared-input"] : [])
        child.standardInput = incoming; child.standardOutput = outgoing; child.standardError = log
        try child.run()
        defer { if child.isRunning { child.terminate() } }
        try incoming.fileHandleForReading.close(); try outgoing.fileHandleForWriting.close()
        let memory = PeakChildMemory(pid: child.processIdentifier)
        func read(_ count: Int) throws -> Data {
            var data = Data(capacity: count)
            while data.count < count {
                guard let part = try outgoing.fileHandleForReading.read(upToCount: count - data.count), !part.isEmpty else {
                    throw NSError(domain: "Worker EOF: \(logURL.path)", code: 1)
                }
                data.append(part)
            }
            return data
        }
        guard try read(4) == Data("WDT3".utf8) else { throw NSError(domain: "Invalid protocol", code: 1) }
        func predict(batch: Int, count: Int = 8) throws -> (Double, Data) {
            corpus.withUnsafeBytes { source in
                mapping.copyMemory(from: source.baseAddress!.advanced(by: batch * inputBytes), byteCount: inputBytes)
            }
            let tick = ProcessInfo.processInfo.systemUptime
            try incoming.fileHandleForWriting.write(contentsOf: Data([UInt8(count)]))
            let values = try read(count * rowBytes)
            let seconds = ProcessInfo.processInfo.systemUptime - tick
            let reference = expected.subdata(in: (batch * 8 * rowBytes)..<((batch * 8 + count) * rowBytes))
            guard values == reference else { throw NSError(domain: "Changed/stale output in batch \(batch)", code: 1) }
            return (seconds, values)
        }
        _ = try predict(batch: 7) // Compile/warm up with a different batch from the first timed input.
        let thermalBefore = ProcessInfo.processInfo.thermalState.rawValue
        var sweeps: [Double] = [], batches: [[String: Any]] = []
        var output = Data()
        for round in 0..<3 {
            var total = 0.0
            for step in 0..<8 {
                let batch = round.isMultiple(of: 2) ? step : 7 - step
                let (seconds, values) = try predict(batch: batch)
                total += seconds
                batches.append(["round": round, "batch": batch, "seconds": seconds])
                if round == 0 { output.append(values) }
            }
            sweeps.append(total)
        }
        // Exercise tails after many overwrites of the same persistent GPU input.
        for count in [1, 5, 8] { _ = try predict(batch: 6, count: count) }
        let peak = memory.bytes
        try incoming.fileHandleForWriting.close()
        child.waitUntilExit()
        guard child.terminationStatus == 0 else { throw NSError(domain: "Worker failed", code: Int(child.terminationStatus)) }
        let result: [String: Any] = [
            "frames": 64, "batch": 8, "rounds": 3,
            "worker": child.executableURL!.path, "copyInput": args.contains("--copy-input"),
            "sweepSeconds": sweeps, "predictionAndIPCFPS": 64 / sweeps.sorted()[1],
            "batches": batches, "peakWorkerPhysicalFootprintBytes": peak,
            "exactAllOutputs": true, "tailCountsExact": [1, 5, 8],
            "outputSHA256": SHA256.hash(data: output).map { String(format: "%02x", $0) }.joined(),
            "thermalBefore": thermalBefore, "thermalAfter": ProcessInfo.processInfo.thermalState.rawValue,
            "method": "Warm worker, three 64-frame sweeps, forward/reverse/forward batch order. Input file preparation excluded; IPC through full response receipt included. Worker-only footprint sampled every 10 ms with proc_pid_rusage. No GPU trace.",
        ]
        try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys]).write(to: reportURL)
        print("\(64 / sweeps.sorted()[1]) fps, \(Double(peak) / 1e6) MB worker peak, exact")
    }
}
