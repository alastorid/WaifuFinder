import Darwin
import Foundation
import MLX
import WDTaggerMLX

// Protocol v4: WDT4 ready marker; count 1...8 requests raw Float32 validation
// rows; count | 0x80 requests CPU-quantized rows (10,861 UInt8 scores). Input
// is a shared fixed-size RGBA UInt8 [8,448,448,4] mapping, owned by the app.
// A response acknowledges completion before the app may overwrite the mapping.
let inputBytes = 8 * 448 * 448 * 4

/// The Metal buffer's finalizer retains this owner until its last GPU use has
/// completed, including error paths. Closing the file does not unmap its pages.
private final class SharedInputMapping {
    let pointer: UnsafeMutableRawPointer
    init(path: String) throws {
        let file = try FileHandle(forReadingFrom: URL(fileURLWithPath: path))
        defer { try? file.close() }
        guard try file.seekToEnd() == UInt64(inputBytes) else {
            throw WDNativeError.invalid("Incorrect shared input size")
        }
        guard let mapped = mmap(nil, inputBytes, PROT_READ, MAP_SHARED, file.fileDescriptor, 0),
              mapped != MAP_FAILED else {
            throw WDNativeError.invalid("Cannot map shared input")
        }
        pointer = mapped
    }
    deinit { munmap(pointer, inputBytes) }
}

func option(_ name: String) -> String? {
    let args = CommandLine.arguments
    guard let index = args.firstIndex(of: name), index + 1 < args.count else { return nil }
    return args[index + 1]
}

/// Quantize inside the worker so IPC still carries one byte per score.
func quantizeScores(_ values: [Float], count: Int) throws -> [UInt8] {
    guard (1...8).contains(count), values.count >= count * 10861 else {
        throw WDNativeError.invalid("Invalid probability row count")
    }
    return try values.prefix(count * 10861).enumerated().map { index, value in
        guard value.isFinite, (0...1).contains(value) else {
            throw WDNativeError.invalid("Invalid native probability")
        }
        if index % 10861 < 4 { return UInt8(value * 255) }
        return value < 0.35 ? 0 : value == 0.35 ? 1 : UInt8(value * 255)
    }
}

func checkQuantization() throws {
    var values = (0..<(8 * 10861)).map { Float(($0 * 7919) % 100001) / 100000 }
    for row in 0..<8 {
        values[row * 10861] = 0.1; values[row * 10861 + 1] = 0.2
        values[row * 10861 + 2] = 0.20001; values[row * 10861 + 3] = 0.15
        values[row * 10861 + 4] = Float(0.35).nextDown
        values[row * 10861 + 5] = 0.35; values[row * 10861 + 6] = Float(0.35).nextUp
    }
    let input = MLXArray(values, [8, 10861]) + Float(0)
    eval(input)
    let reference = try quantizeScores(values, count: 8)
    for row in 0..<8 {
        guard Array(reference[(row * 10861)..<(row * 10861 + 7)]) == [25, 51, 51, 38, 0, 1, 89] else {
            throw WDNativeError.invalid("Four rating scores or trait boundaries changed")
        }
    }
    for count in [1, 5, 8] {
        guard try quantizeScores(values, count: count) == Array(reference.prefix(count * 10861)) else {
            throw WDNativeError.invalid("Quantized tail changed")
        }
    }
    for invalid: Float in [.nan, .infinity, -0.01, 1.01] {
        var bad = values; bad[0] = invalid
        do {
            _ = try quantizeScores(bad, count: 1)
        } catch { continue }
        throw WDNativeError.invalid("Invalid score was accepted")
    }
    var times: [Double] = []
    for _ in 0..<31 {
        let start = Date()
        let actual = try quantizeScores(input.asArray(Float.self), count: 8)
        times.append(Date().timeIntervalSince(start) * 1000)
        guard actual == reference else { throw WDNativeError.invalid("Quantized readback changed scores") }
    }
    print("CPU UInt8 check: 86,888 scores; four ratings, cutoff boundaries, tails and invalid scores passed")
    print("Median Float32 readback + validated CPU quantization ms: \(times.sorted()[15])")
}

do {
    if CommandLine.arguments.contains("--check-quantization") {
        try checkQuantization()
        exit(0)
    }
    guard let weights = option("--weights") else {
        throw WDNativeError.invalid("Missing --weights")
    }
    Memory.memoryLimit = WDNativeTagger.defaultMemoryMiB * 1024 * 1024
    Memory.cacheLimit = WDNativeTagger.defaultCacheMiB * 1024 * 1024
    let tagger = try WDNativeTagger(weights: URL(fileURLWithPath: weights))
    if CommandLine.arguments.contains("--validate") {
        let output = tagger.predict(MLXArray.zeros([8, 448, 448, 4], dtype: .uint8))
        eval(output)
        let values = output.asArray(Float.self)
        guard output.shape == [8, 10_861], values.allSatisfy({ $0.isFinite && (0...1).contains($0) }) else {
            throw WDNativeError.invalid("Invalid native prediction")
        }
        print("Native WDTagger ready: \(WDNativeTagger.implementationVersion)")
    } else {
        guard let path = option("--input") else { throw WDNativeError.invalid("Missing --input") }
        let mapping = try SharedInputMapping(path: path)
        defer { withExtendedLifetime(mapping) {} }
        let data = Data(bytesNoCopy: mapping.pointer, count: inputBytes, deallocator: .none)
        let sharedInput: MLXArray?
        let requestsSharedInput = CommandLine.arguments.contains("--shared-input")
            || ProcessInfo.processInfo.environment["WAIFU_WD_SHARED_INPUT"] == "1"
        if !requestsSharedInput || CommandLine.arguments.contains("--copy-input") {
            // Retain the released copy path by default. External-buffer memory
            // savings did not reproduce consistently across tested GPU models.
            sharedInput = nil
        } else {
            let candidate = MLXArray(rawPointer: mapping.pointer, [8, 448, 448, 4], dtype: .uint8) {
                [mapping] in withExtendedLifetime(mapping) {}
            }
            // MLX can fall back to copying an unsupported external allocation.
            // Such a snapshot cannot be reused after the app writes a new batch.
            let aliasesMapping = candidate.asData(access: .noCopy).data.withUnsafeBytes {
                $0.baseAddress == UnsafeRawPointer(mapping.pointer)
            }
            sharedInput = aliasesMapping ? candidate : nil
        }
        FileHandle.standardError.write(Data("WDTagger input: \(sharedInput == nil ? "copy" : "shared")\n".utf8))
        try FileHandle.standardOutput.write(contentsOf: Data("WDT4".utf8))
        while let command = try FileHandle.standardInput.read(upToCount: 1), !command.isEmpty {
            try autoreleasepool {
                let quantized = command[0] & 0x80 != 0
                let count = Int(command[0] & 0x7f)
                guard (1...8).contains(count) else { throw WDNativeError.invalid("Invalid batch size") }
                // The app only overwrites the mapping after receiving a complete
                // response. eval/asArray finish GPU reads before that response.
                let x = sharedInput ?? MLXArray(data, [8, 448, 448, 4], type: UInt8.self)
                let output = tagger.predict(x)
                guard output.shape == [8, 10_861] else { throw WDNativeError.invalid("Invalid output shape") }
                eval(output)
                let values = output.asArray(Float.self)
                if quantized {
                    let scores = try quantizeScores(values, count: count)
                    try FileHandle.standardOutput.write(contentsOf: Data(scores))
                    return
                }
                guard values.allSatisfy({ $0.isFinite && (0...1).contains($0) }) else {
                    throw WDNativeError.invalid("Invalid native probabilities")
                }
                try values.withUnsafeBytes { bytes in
                    try FileHandle.standardOutput.write(contentsOf: Data(bytes.prefix(count * 10_861 * 4)))
                }
            }
        }
    }
} catch {
    FileHandle.standardError.write(Data("WDTagger: \(error)\n".utf8))
    exit(1)
}
