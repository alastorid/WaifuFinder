import CoreGraphics
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
    func validate() throws {
        try judgment.validate()
        guard (0...1024).contains(tokens), prefillSeconds.isFinite, decodeSeconds.isFinite,
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
    static func line(_ handle: FileHandle) throws -> Data {
        var data = Data()
        while data.count < 128 * 1024 {
            guard let byte = try handle.read(upToCount: 1), !byte.isEmpty else {
                throw ModelInstaller.fail("PhotoArena worker stopped before replying")
            }
            if byte[0] == 10 { return data }
            data.append(byte)
        }
        throw ModelInstaller.fail("PhotoArena worker response exceeds limit")
    }
    func run(images: [Data], directory: URL, executable: URL) throws -> ArenaJudgeResult {
        let child = Process(), input = Pipe(), output = Pipe()
        child.executableURL = executable
        child.arguments = [directory.appendingPathComponent("model.gguf").path, directory.appendingPathComponent("mmproj.gguf").path]
        child.standardInput = input; child.standardOutput = output; child.standardError = FileHandle.nullDevice
        try start(child)
        let watchdog = DispatchWorkItem { [weak self] in self?.stop() }
        DispatchQueue.global().asyncAfter(deadline: .now() + 300, execute: watchdog)
        defer {
            watchdog.cancel()
            try? input.fileHandleForWriting.close()
            if child.isRunning { child.terminate() }
            child.waitUntilExit()
            try? output.fileHandleForReading.close()
        }
        let ready = try Self.line(output.fileHandleForReading)
        guard let header = try JSONSerialization.jsonObject(with: ready) as? [String: Any], header["ready"] as? Bool == true else {
            throw ModelInstaller.fail("PhotoArena could not load its model: \(String(decoding: ready, as: UTF8.self))")
        }
        var request = try JSONSerialization.data(withJSONObject: ["images": images.map { $0.base64EncodedString() }])
        request.append(10)
        try input.fileHandleForWriting.write(contentsOf: request)
        let reply = try Self.line(output.fileHandleForReading)
        if let json = try JSONSerialization.jsonObject(with: reply) as? [String: Any], let error = json["error"] as? String {
            throw ModelInstaller.fail(error)
        }
        let result = try JSONDecoder().decode(ArenaJudgeResult.self, from: reply)
        try result.validate()
        return result
    }
}
enum ArenaJudge {
    private static let queue = DispatchQueue(label: "waifu.photoarena.inference", qos: .userInitiated)
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
        let first = try await single(images)
        try Task.checkCancellation()
        let reversed = try await single(Array(images.reversed()))
        return reconcile(first, reversed)
    }
    private static func single(_ images: [Data]) async throws -> ArenaJudgeResult {
        guard images.count == 2, images.allSatisfy({ !$0.isEmpty && $0.count <= 16 * 1024 * 1024 }),
              let directory = ArenaJudgeConfiguration.directory, let executable = ArenaJudgeConfiguration.executable else {
            throw ModelInstaller.fail("Install PhotoArena MiniCPM-V 4.7 in Settings → Models first")
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
