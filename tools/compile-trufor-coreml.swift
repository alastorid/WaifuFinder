import CoreML
import Foundation

@main
private struct CompileTruForCoreML {
    static func main() async throws {
        let args = CommandLine.arguments
        guard args.count == 3 else {
            fputs("Usage: compile-trufor-coreml MODEL.mlpackage OUTPUT.mlmodelc\n", stderr)
            exit(64)
        }
        let source = URL(fileURLWithPath: args[1])
        let destination = URL(fileURLWithPath: args[2])
        guard !FileManager.default.fileExists(atPath: destination.path) else {
            throw NSError(domain: "TruForCoreMLCompile", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Output already exists"])
        }
        let compiled = try await MLModel.compileModel(at: source)
        defer { try? FileManager.default.removeItem(at: compiled) }
        try FileManager.default.copyItem(at: compiled, to: destination)
        _ = try MLModel(contentsOf: destination)
        print("Compiled and loaded Core ML model at \(destination.path)")
    }
}
