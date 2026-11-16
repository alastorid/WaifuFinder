import Darwin
import Foundation

/// One model per tagging session. The actor serializes requests; the lock only
/// protects process lifetime against cancellation from another executor.
final class WDTaggerWorker: @unchecked Sendable {
    static let inputBytes = 8 * 448 * 448 * 4
    static var executableURL: URL? {
        Bundle.main.url(forResource: "waifu-wdtagger", withExtension: nil, subdirectory: "Runtime")
    }
    private let weights: URL
    private let executable: URL
    private let lock = NSLock()
    private var process: Process?
    private var cancelled = false
    private var request: FileHandle?
    private var response: FileHandle?

    init(weights: URL, executable: URL) {
        self.weights = weights
        self.executable = executable
    }

    deinit {
        cancel()
        try? request?.close()
        try? response?.close()
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
        let writer = incoming.fileHandleForWriting
        // A cancellation or a worker exit during a large write must surface
        // as a pipe error in the app.
        guard fcntl(writer.fileDescriptor, F_SETNOSIGPIPE, 1) != -1 else {
            throw TaggerError.invalidModel("Unable to configure the native tagger input pipe.")
        }
        child.executableURL = executable
        child.arguments = ["--weights", weights.path]
        child.standardInput = incoming
        child.standardOutput = outgoing
        child.standardError = FileHandle.standardError
        try lock.withLock {
            guard !cancelled else { throw CancellationError() }
            process = child
            try child.run()
        }
        request = writer
        response = outgoing.fileHandleForReading
        // Close parent copies of the child's endpoints so crashes produce EOF.
        try incoming.fileHandleForReading.close()
        try outgoing.fileHandleForWriting.close()
        guard try readExactly(4) == Data("WDT5".utf8) else {
            throw TaggerError.invalidModel("Incompatible native tagger. Rebuild WaifuFinder with its matching worker.")
        }
    }

    private func failure() -> Error {
        if lock.withLock({ cancelled }) { return CancellationError() }
        return TaggerError.invalidModel("The native tagger stopped before returning a complete response.")
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

    func predict(input: Data, count: Int) throws -> [[Float]] {
        let bytes = try requestRows(input: input, count: count, quantized: false)
        return try Self.decodeProbabilities(bytes, count: count)
    }

    func predictQuantized(input: Data, count: Int) throws -> [QuantizedTagRow] {
        let bytes = try requestRows(input: input, count: count, quantized: true)
        return try Self.decodeQuantized(bytes, count: count)
    }

    private func requestRows(input: Data, count: Int, quantized: Bool) throws -> Data {
        guard (1...8).contains(count), input.count == Self.inputBytes else { throw TaggerError.invalidBatch }
        do {
            try Task.checkCancellation()
            try start()
            guard let request else { throw failure() }
            let command = UInt8(count) | (quantized ? 0x80 : 0)
            try request.write(contentsOf: Data([command]))
            try request.write(contentsOf: input)
            let rowBytes = TagVocabulary.headCount * (quantized ? 1 : MemoryLayout<Float>.stride)
            let bytes = try readExactly(count * rowBytes)
            try Task.checkCancellation()
            return bytes
        } catch {
            let wasCancelled = Task.isCancelled || lock.withLock { cancelled }
            cancel()
            if wasCancelled { throw CancellationError() }
            throw error
        }
    }

    static func decodeQuantized(_ bytes: Data, count: Int) throws -> [QuantizedTagRow] {
        let stride = TagVocabulary.headCount
        guard (1...8).contains(count), bytes.count == count * stride else { throw TaggerError.invalidProbabilities }
        return try (0..<count).map { row in
            let offset = row * stride
            guard let scores = TagProbabilityCodec.encode(quantized: Array(bytes.dropFirst(offset).prefix(TagVocabulary.headCount)))
            else { throw TaggerError.invalidProbabilities }
            return QuantizedTagRow(scores: scores)
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
