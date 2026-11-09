import CoreGraphics
import Darwin
import Foundation
import ImageIO
import UniformTypeIdentifiers

struct ArenaJudgment: Codable, Sendable {
    let watermark: ArenaOutcome
    let authenticity: ArenaOutcome
    let detail: ArenaOutcome
    let quality: ArenaOutcome
    let consistency: ArenaOutcome
    let recommended_source: ArenaOutcome
    let uncertain: Bool
    let evidence: [String]
    func outcome(_ category: ArenaCategory) -> ArenaOutcome {
        switch category {
        case .watermark: watermark
        case .authenticity: authenticity
        case .detail: detail
        case .quality: quality
        case .consistency: consistency
        }
    }
    func validate() throws {
        guard evidence.count <= 32, evidence.allSatisfy({ $0.utf8.count <= 4096 }) else {
            throw ModelInstaller.fail("Invalid PhotoArena evidence")
        }
    }
}
struct ArenaJudgeResult: Codable, Sendable {
    let judgment: ArenaJudgment
    let tokens: Int
    let prefillSeconds: Double
    let decodeSeconds: Double
    let physicalFootprintBytes: UInt64
    let compressedBytes: UInt64
    var review: ArenaGroupAssessment? = nil
    var workerPeakPhysicalFootprintBytes: UInt64? = nil
    var workerSwapins: Int64? = nil
    var metalAllocatedBytes: UInt64? = nil
    var contextInitSeconds: Double? = nil
    func validate() throws {
        try judgment.validate()
        if let review { try review.validate(count: review.sources.count) }
        guard (0...(review == nil ? 1024 : 4096)).contains(tokens), prefillSeconds.isFinite, decodeSeconds.isFinite,
              prefillSeconds >= 0, decodeSeconds >= 0 else {
            throw ModelInstaller.fail("Invalid PhotoArena inference metrics")
        }
    }
}
enum ArenaJudgeConfiguration {
    static let revision = "minicpm-v4.7-f16-16x-llama-7071ba3-v1"
    static var configurationURL: URL { SeedVR2Configuration.configurationURL.deletingLastPathComponent().appendingPathComponent("photoarena-model.txt") }
    static var directory: URL? {
        guard let path = try? String(contentsOf: configurationURL, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines),
              !path.isEmpty else { return nil }
        let url = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: url.appendingPathComponent("model.gguf").path),
              FileManager.default.fileExists(atPath: url.appendingPathComponent("mmproj.gguf").path), executable != nil else { return nil }
        return url
    }
    static var executable: URL? {
        if let url = Bundle.main.url(forResource: "waifu-photoarena", withExtension: nil, subdirectory: "Runtime"),
           FileManager.default.isExecutableFile(atPath: url.path) { return url }
        return nil
    }
}

/// Bounded line framing without one system call per response byte.
final class ArenaPipeReader {
    private let handle: FileHandle
    private var pending = Data()
    init(_ handle: FileHandle) { self.handle = handle }
    func line() throws -> Data {
        while true {
            if let end = pending.firstIndex(of: 10) {
                guard pending.distance(from: pending.startIndex, to: end) < 128 * 1024 else {
                    throw ModelInstaller.fail("PhotoArena worker response exceeds limit")
                }
                let result = Data(pending[..<end])
                pending.removeSubrange(...end)
                return result
            }
            guard pending.count < 128 * 1024 else { throw ModelInstaller.fail("PhotoArena worker response exceeds limit") }
            // Foundation read(upToCount:) can wait to fill a pipe buffer. A
            // single POSIX read returns available bytes immediately, including
            // a short ready line while the child is waiting for its request.
            var bytes = Data(count: 4096)
            let count = bytes.withUnsafeMutableBytes { Darwin.read(handle.fileDescriptor, $0.baseAddress!, $0.count) }
            if count < 0 && errno == EINTR { continue }
            guard count > 0 else { throw ModelInstaller.fail("PhotoArena worker stopped before replying") }
            bytes.count = count
            pending.append(bytes)
        }
    }
}

/// Each request owns its worker; cancellation cannot stop another queued request.
private final class ArenaProcess: @unchecked Sendable {
    private let lock = NSLock()
    private var process: Process?
    private var cancelled = false
    func start(_ process: Process) throws {
        lock.lock(); defer { lock.unlock() }
        if cancelled { throw CancellationError() }
        self.process = process
        try process.run()
    }
    func stop() {
        lock.lock(); defer { lock.unlock() }
        cancelled = true
        if let process, process.isRunning { process.terminate() }
    }

    func run(images: [Data], directory: URL, executable: URL) throws -> ArenaJudgeResult {
        let child = Process(), input = Pipe(), output = Pipe()
        child.executableURL = executable
        child.arguments = [directory.appendingPathComponent("model.gguf").path, directory.appendingPathComponent("mmproj.gguf").path]
        child.standardInput = input; child.standardOutput = output; child.standardError = FileHandle.nullDevice
        try start(child)
        var watchdog: DispatchWorkItem?
        func armWatchdog() {
            watchdog?.cancel()
            let item = DispatchWorkItem { [weak self] in self?.stop() }
            watchdog = item
            DispatchQueue.global().asyncAfter(deadline: .now() + 300, execute: item)
        }
        armWatchdog()
        defer {
            watchdog?.cancel()
            try? input.fileHandleForWriting.close()
            if child.isRunning { child.terminate() }
            child.waitUntilExit()
            try? output.fileHandleForReading.close()
        }
        let reader = ArenaPipeReader(output.fileHandleForReading)
        let ready = try reader.line()
        guard let header = try JSONSerialization.jsonObject(with: ready) as? [String: Any], header["ready"] as? Bool == true else {
            throw ModelInstaller.fail("PhotoArena could not load its model: \(String(decoding: ready, as: UTF8.self))")
        }
        let paired = header["pairedComparison"] as? Bool == true
        func send(_ values: [Data], paired: Bool) throws {
            // Base64 has no JSON quoting characters. Stream each encoded image
            // once instead of retaining two strings plus a full JSON copy.
            try input.fileHandleForWriting.write(contentsOf: Data((paired ? "{\"compareBoth\":true,\"images\":[\"" : "{\"images\":[\"").utf8))
            for (index, image) in values.enumerated() {
                if index > 0 { try input.fileHandleForWriting.write(contentsOf: Data("\",\"".utf8)) }
                try input.fileHandleForWriting.write(contentsOf: image.base64EncodedData())
            }
            try input.fileHandleForWriting.write(contentsOf: Data("\"]}\n".utf8))
        }
        func receive() throws -> ArenaJudgeResult {
            let reply = try reader.line()
            if let json = try JSONSerialization.jsonObject(with: reply) as? [String: Any], let error = json["error"] as? String {
                throw ModelInstaller.fail(error)
            }
            let result = try JSONDecoder().decode(ArenaJudgeResult.self, from: reply)
            try result.validate()
            return result
        }
        try send(images, paired: paired)
        let first = try receive()
        armWatchdog()
        // Older bundled workers already accept repeated single requests. Reuse
        // their model safely too, without requiring a wire-protocol upgrade.
        if !paired { try send(Array(images.reversed()), paired: false) }
        let reversed = try receive()
        return ArenaJudge.reconcile(first, reversed)
    }

    /// One worker for a bounded review action. Evidence is prepared and released one job at a time.
    func review(analysis: ArenaAnalysis?, jobs: [ArenaReviewJob], previous: [ArenaReviewRecord], suppliedPayload: ArenaGroupPayload? = nil,
                directory: URL, executable: URL,
                onResult: @Sendable (ArenaReviewRecord) -> Void) throws -> [ArenaReviewRecord] {
        var child: Process?, input: Pipe?, reader: ArenaPipeReader?, watchdog: DispatchWorkItem?
        var records: [ArenaReviewRecord] = []
        defer {
            watchdog?.cancel(); try? input?.fileHandleForWriting.close()
            if let child { if child.isRunning { child.terminate() }; child.waitUntilExit() }
        }
        func arm() {
            watchdog?.cancel()
            let item = DispatchWorkItem { [weak self] in self?.stop() }
            watchdog = item; DispatchQueue.global().asyncAfter(deadline: .now()+300, execute: item)
        }
        for job in jobs {
            lock.lock(); let stopped = cancelled; lock.unlock()
            if stopped { throw CancellationError() }
            let record: ArenaReviewRecord = try autoreleasepool {
                let payload: ArenaGroupPayload
                if let suppliedPayload { payload = suppliedPayload }
                else if let analysis { payload = try ArenaGroupEvidence.prepare(analysis, job: job) }
                else { throw ModelInstaller.fail("Missing PhotoArena evidence") }
                try payload.validate()
                let key = try ArenaGroupEvidence.key(payload, job: job)
                if let saved = previous.last(where: { $0.id == job.id && $0.cacheKey == key && $0.sourceIDs == payload.sourceIDs }),
                   let assessment = saved.result.review, (try? assessment.validate(count: payload.sourceIDs.count)) != nil {
                    return saved
                }
                if child == nil {
                    let process = Process(), stdin = Pipe(), stdout = Pipe()
                    process.executableURL = executable
                    process.arguments = [directory.appendingPathComponent("model.gguf").path, directory.appendingPathComponent("mmproj.gguf").path]
                    process.standardInput = stdin; process.standardOutput = stdout; process.standardError = FileHandle.nullDevice
                    try start(process); child = process; input = stdin; reader = ArenaPipeReader(stdout.fileHandleForReading)
                    arm()
                    let data = try reader!.line()
                    let header = try JSONSerialization.jsonObject(with: data) as? [String: Any]
                    guard header?["groupReview"] as? String == ArenaGroupPayload.revision else {
                        throw ModelInstaller.fail("This PhotoArena runtime does not support multi-source review. Update the inference app.")
                    }
                }
                arm()
                let request: [String: Any] = ["reviewCount": payload.sourceIDs.count, "reviewContext": payload.context]
                var data = try JSONSerialization.data(withJSONObject: request); data.removeLast()
                data.append(Data(",\"images\":[\"".utf8))
                let writer = input!.fileHandleForWriting
                try writer.write(contentsOf: data)
                for (i,image) in payload.images.enumerated() {
                    if i > 0 { try writer.write(contentsOf:Data("\",\"".utf8)) }
                    try writer.write(contentsOf:image.base64EncodedData())
                }
                try writer.write(contentsOf:Data("\"]}\n".utf8))
                let reply = try reader!.line()
                if let json = try JSONSerialization.jsonObject(with: reply) as? [String: Any], let error = json["error"] as? String {
                    throw ModelInstaller.fail(error)
                }
                let result = try JSONDecoder().decode(ArenaJudgeResult.self, from: reply)
                try result.validate()
                guard let assessment = result.review else { throw ModelInstaller.fail("PhotoArena omitted multi-source findings") }
                try assessment.validate(count: payload.sourceIDs.count)
                return ArenaReviewRecord(job: job, sourceIDs: payload.sourceIDs, cacheKey: key, result: result)
            }
            records.append(record); onResult(record)
        }
        return records
    }
}
enum ArenaJudge {
    private static let queue = DispatchQueue(label: "waifu.photoarena.inference", qos: .userInitiated)
    static func group(_ payload: ArenaGroupPayload) async throws -> ArenaJudgeResult {
        try payload.validate()
        guard let directory = ArenaJudgeConfiguration.directory, let executable = ArenaJudgeConfiguration.executable else {
            throw ModelInstaller.fail("Install PhotoArena MiniCPM-V 4.7 on the inference server")
        }
        let process = ArenaProcess()
        let job = ArenaReviewJob(id: "group", rect: ArenaPixelRect(.zero), overview: true)
        let records: [ArenaReviewRecord] = try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                queue.async {
                    do { continuation.resume(returning: try process.review(analysis: nil, jobs: [job], previous: [], suppliedPayload: payload,
                        directory: directory, executable: executable, onResult: { _ in })) }
                    catch { continuation.resume(throwing: error) }
                }
            }
        } onCancel: { process.stop() }
        try Task.checkCancellation()
        return records[0].result
    }
    static func review(analysis: ArenaAnalysis, jobs: [ArenaReviewJob], previous: [ArenaReviewRecord] = [],
                       onResult: @escaping @Sendable (ArenaReviewRecord) -> Void) async throws -> [ArenaReviewRecord] {
        guard jobs.count <= 17, let directory = ArenaJudgeConfiguration.directory, let executable = ArenaJudgeConfiguration.executable else {
            throw ModelInstaller.fail("Install PhotoArena MiniCPM-V 4.7 in Settings → Models on the inference Mac")
        }
        return try await review(analysis: analysis, jobs: jobs, previous: previous, directory: directory, executable: executable, onResult: onResult)
    }
    static func review(analysis: ArenaAnalysis, jobs: [ArenaReviewJob], previous: [ArenaReviewRecord],
                       directory: URL, executable: URL,
                       onResult: @escaping @Sendable (ArenaReviewRecord) -> Void) async throws -> [ArenaReviewRecord] {
        let process = ArenaProcess()
        let result = try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                queue.async {
                    do { continuation.resume(returning: try process.review(analysis: analysis, jobs: jobs, previous: previous, directory: directory, executable: executable, onResult: onResult)) }
                    catch { continuation.resume(throwing: error) }
                }
            }
        } onCancel: { process.stop() }
        try Task.checkCancellation()
        return result
    }
    static func png(_ image: CGImage) throws -> Data {
        let bounded = try PhotosMedia.boundedRendition(image, maxPixelSize: 1600)
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil) else {
            throw ModelInstaller.fail("Cannot encode PhotoArena image")
        }
        CGImageDestinationAddImage(destination, bounded, nil)
        guard CGImageDestinationFinalize(destination) else { throw ModelInstaller.fail("Cannot encode PhotoArena image") }
        return data as Data
    }
    static func reconcile(_ first: ArenaJudgeResult, _ reversed: ArenaJudgeResult) -> ArenaJudgeResult {
        func agree(_ a: ArenaOutcome, _ b: ArenaOutcome) -> ArenaOutcome {
            let inverted: ArenaOutcome = b == .a ? .b : b == .b ? .a : b
            return a == inverted ? a : .unknown
        }
        let a = first.judgment, b = reversed.judgment
        let values = ArenaCategory.allCases.map { agree(a.outcome($0), b.outcome($0)) }
        return ArenaJudgeResult(judgment: ArenaJudgment(watermark: agree(a.watermark, b.watermark),
            authenticity: agree(a.authenticity, b.authenticity), detail: agree(a.detail, b.detail),
            quality: agree(a.quality, b.quality), consistency: agree(a.consistency, b.consistency),
            recommended_source: agree(a.recommended_source, b.recommended_source),
            uncertain: a.uncertain || b.uncertain || values.contains(.unknown),
            evidence: Array((["Forward pass:"] + a.evidence + ["Reverse pass (A=original B, B=original A):"] + b.evidence).prefix(32))),
            tokens: first.tokens + reversed.tokens, prefillSeconds: first.prefillSeconds + reversed.prefillSeconds,
            decodeSeconds: first.decodeSeconds + reversed.decodeSeconds,
            physicalFootprintBytes: max(first.physicalFootprintBytes, reversed.physicalFootprintBytes),
            compressedBytes: max(first.compressedBytes, reversed.compressedBytes))
    }
    static func local(_ images: [Data]) async throws -> ArenaJudgeResult {
        guard images.count == 2 else { throw ModelInstaller.fail("Two PhotoArena images required") }
        if images[0] == images[1] {
            return ArenaJudgeResult(judgment: ArenaJudgment(watermark: .tie, authenticity: .tie, detail: .tie,
                quality: .tie, consistency: .tie, recommended_source: .tie, uncertain: false,
                evidence: ["The encoded overview inputs are byte-identical; no visual preference is supported."]),
                tokens: 0, prefillSeconds: 0, decodeSeconds: 0, physicalFootprintBytes: 0, compressedBytes: 0)
        }
        return try await single(images)
    }
    private static func single(_ images: [Data]) async throws -> ArenaJudgeResult {
        guard images.count == 2, images.allSatisfy({ !$0.isEmpty && $0.count <= 16 * 1024 * 1024 }),
              let directory = ArenaJudgeConfiguration.directory, let executable = ArenaJudgeConfiguration.executable else {
            throw ModelInstaller.fail("Install PhotoArena MiniCPM-V 4.7 in Settings → Models first")
        }
        return try await compare(images: images, directory: directory, executable: executable)
    }
    /// Explicit paths permit background protocol tests and the production-path
    /// before/after harness without changing the user's model configuration.
    static func compare(images: [Data], directory: URL, executable: URL) async throws -> ArenaJudgeResult {
        guard images.count == 2, images.allSatisfy({ !$0.isEmpty && $0.count <= 16 * 1024 * 1024 }) else {
            throw ModelInstaller.fail("Invalid PhotoArena image payload")
        }
        let process = ArenaProcess()
        let result = try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<ArenaJudgeResult, Error>) in
                queue.async {
                    do { continuation.resume(returning: try process.run(images: images, directory: directory, executable: executable)) }
                    catch { continuation.resume(throwing: error) }
                }
            }
        } onCancel: { process.stop() }
        try Task.checkCancellation()
        return result
    }
}
