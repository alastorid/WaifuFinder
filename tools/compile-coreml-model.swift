import CoreML
import Foundation

guard CommandLine.arguments.count == 3 else {
    fputs("Usage: compile-coreml-model.swift INPUT.mlpackage OUTPUT.mlmodelc\n", stderr)
    exit(2)
}

let fileManager = FileManager.default
let source = URL(fileURLWithPath: CommandLine.arguments[1]).standardizedFileURL
let destination = URL(fileURLWithPath: CommandLine.arguments[2]).standardizedFileURL
guard source.pathExtension == "mlpackage", fileManager.fileExists(atPath: source.path) else {
    fputs("A Core ML .mlpackage source is required.\n", stderr)
    exit(2)
}

do {
    let compiled = try MLModel.compileModel(at: source)
    try fileManager.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
    if fileManager.fileExists(atPath: destination.path) { try fileManager.removeItem(at: destination) }
    try fileManager.moveItem(at: compiled, to: destination)
    print("Compiled Core ML model: \(destination.path)")
} catch {
    fputs("Core ML model compilation failed: \(error.localizedDescription)\n", stderr)
    exit(1)
}
