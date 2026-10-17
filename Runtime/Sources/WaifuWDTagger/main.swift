import Darwin
import Foundation
import MLX
import WDTaggerMLX

// Protocol v3: WDT3 ready marker; one byte (real frame count 1...8) per
// request; that many contiguous Float32 probability rows per response. Input
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

do {
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
        try FileHandle.standardOutput.write(contentsOf: Data("WDT3".utf8))
        while let command = try FileHandle.standardInput.read(upToCount: 1), !command.isEmpty {
            try autoreleasepool {
                let count = Int(command[0])
                guard (1...8).contains(count) else { throw WDNativeError.invalid("Invalid batch size") }
                // The app only overwrites the mapping after receiving a complete
                // response. eval/asArray finish GPU reads before that response.
                let x = sharedInput ?? MLXArray(data, [8, 448, 448, 4], type: UInt8.self)
                let output = tagger.predict(x)
                eval(output)
                guard output.shape == [8, 10_861] else { throw WDNativeError.invalid("Invalid output shape") }
                let values = output.asArray(Float.self)
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
