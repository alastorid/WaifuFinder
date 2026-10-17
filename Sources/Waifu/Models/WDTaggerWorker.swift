import Darwin
import Foundation

/// One model per tagging session. The actor serializes requests; the lock only
/// protects process lifetime against cancellation from another executor.
final class WDTaggerWorker: @unchecked Sendable {
    static let inputBytes = 8 * 448 * 448 * 4
    static var executableURL: URL? {
        Bundle.main.url(forResource: "waifu-wdtagger", withExtension: nil, subdirectory: "Runtime")
    }
    let pointer: UnsafeMutableRawPointer
    private let directory: URL
    private let inputURL: URL
    private let weights: URL
    private let executable: URL
    private let lock = NSLock()
    private var process: Process?
    private var cancelled = false
    private var request: FileHandle?
    private var response: FileHandle?
    private var log: FileHandle?

    init(weights: URL, executable: URL) throws {
        self.weights = weights
        self.executable = executable
        directory = FileManager.default.temporaryDirectory.appendingPathComponent("wd-tagger-\(UUID().uuidString)")
        inputURL = directory.appendingPathComponent("input.bin")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true,
                                                attributes: [.posixPermissions: 0o700])
        do {
            guard FileManager.default.createFile(atPath: inputURL.path, contents: nil,
                                                  attributes: [.posixPermissions: 0o600]) else {
                throw TaggerError.preprocessing
            }
            let file = try FileHandle(forUpdating: inputURL)
            defer { try? file.close() }
            try file.truncate(atOffset: UInt64(Self.inputBytes))
            guard let mapped = mmap(nil, Self.inputBytes, PROT_READ | PROT_WRITE, MAP_SHARED, file.fileDescriptor, 0),
                  mapped != MAP_FAILED else { throw TaggerError.preprocessing }
            pointer = mapped
        } catch {
            try? FileManager.default.removeItem(at: directory)
            throw error
        }
    }

    deinit {
        cancel()
        try? request?.close()
        try? response?.close()
        try? log?.close()
        munmap(pointer, Self.inputBytes)
        try? FileManager.default.removeItem(at: directory)
    }

    func cancel() {
        lock.withLock {
            cancelled = true
            if let process, process.isRunning { process.terminate() }
        }
    }

    private func start() throws {
        if process != nil { return }
        let child = Process(), incoming = Pipe(), outgoing = Pipe()
        let logURL = directory.appendingPathComponent("worker.log")
        _ = FileManager.default.createFile(atPath: logURL.path, contents: nil)
        let logging = try FileHandle(forWritingTo: logURL)
        child.executableURL = executable
        child.arguments = ["--weights", weights.path, "--input", inputURL.path]
        child.standardInput = incoming
        child.standardOutput = outgoing
        child.standardError = logging
        try lock.withLock {
            guard !cancelled else { throw CancellationError() }
            process = child
            try child.run()
        }
        request = incoming.fileHandleForWriting
        response = outgoing.fileHandleForReading
        log = logging
        // Close parent copies of the child's endpoints so crashes produce EOF.
        try incoming.fileHandleForReading.close()
        try outgoing.fileHandleForWriting.close()
        guard try readExactly(4) == Data("WDT3".utf8) else { throw failure() }
    }

    private func failure() -> Error {
        if lock.withLock({ cancelled }) { return CancellationError() }
        let url = directory.appendingPathComponent("worker.log")
        let detail = (try? String(contentsOf: url, encoding: .utf8))?.suffix(3000) ?? ""
        return TaggerError.invalidModel("Native tagger stopped. \(detail)")
    }

    private func readExactly(_ count: Int) throws -> Data {
        guard let response else { throw failure() }
        var data = Data(capacity: count)
        while data.count < count {
            guard let part = try response.read(upToCount: count - data.count), !part.isEmpty else { throw failure() }
            data.append(part)
        }
        return data
    }

    func predict(count: Int) throws -> [[Float]] {
        do {
            try start()
            guard let request else { throw failure() }
            try request.write(contentsOf: Data([UInt8(count)]))
            let bytes = try readExactly(count * TagVocabulary.headCount * 4)
            try Task.checkCancellation()
            return try Self.decodeProbabilities(bytes, count: count)
        } catch {
            cancel()
            throw error
        }
    }

    /// Both supported macOS architectures are little endian. Copy complete rows
    /// into aligned Float storage; the incoming Data need not be Float-aligned.
    static func decodeProbabilities(_ bytes: Data, count: Int) throws -> [[Float]] {
        let columns = TagVocabulary.headCount
        let rowBytes = columns * MemoryLayout<Float>.stride
        guard (1...8).contains(count), bytes.count == count * rowBytes else {
            throw TaggerError.invalidProbabilities
        }
        return bytes.withUnsafeBytes { source in
            (0..<count).map { row in
                [Float](unsafeUninitializedCapacity: columns) { storage, initialized in
                    UnsafeMutableRawPointer(storage.baseAddress!).copyMemory(
                        from: source.baseAddress!.advanced(by: row * rowBytes), byteCount: rowBytes)
                    initialized = columns
                }
            }
        }
    }
}
