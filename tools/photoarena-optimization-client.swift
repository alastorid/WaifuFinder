// Compiled against each actual ArenaJudge.swift. Only model/worker location is
// supplied externally; judging, subprocess IPC and reconciliation stay unchanged.
import Foundation
import CoreGraphics

enum ModelInstaller {
    static func fail(_ text: String) -> NSError { NSError(domain: "WaifuModels", code: 1, userInfo: [NSLocalizedDescriptionKey: text]) }
}
enum PhotosMedia {
    // This benchmark starts with the same pre-encoded overview/crop inputs used
    // by the app. PNG encoding is outside the measured judging action.
    static func boundedRendition(_ image: CGImage, maxPixelSize: Int) throws -> CGImage { image }
}
enum ArenaBenchmarkTelemetry {
    static func save(_ data: Data) {
        guard let path = ProcessInfo.processInfo.environment["ARENA_REPLIES"] else { return }
        if !FileManager.default.fileExists(atPath: path) { FileManager.default.createFile(atPath: path, contents: nil) }
        guard let handle = FileHandle(forWritingAtPath: path) else { return }
        defer { try? handle.close() }
        let reply = try? JSONSerialization.jsonObject(with: data)
        let record: [String: Any] = ["recordedAt": Date().timeIntervalSince1970, "reply": reply ?? NSNull()]
        guard let encoded = try? JSONSerialization.data(withJSONObject: record) else { return }
        _ = try? handle.seekToEnd(); try? handle.write(contentsOf: encoded); try? handle.write(contentsOf: Data([10]))
    }
}
@main struct PhotoArenaOptimizationClient {
    static func main() async throws {
        let start = Date()
        let images = try CommandLine.arguments.dropFirst().map { try Data(contentsOf: URL(fileURLWithPath: $0)) }
        let result = try await ArenaJudge.local(images)
        let data = try JSONEncoder().encode(result)
        let object = try JSONSerialization.jsonObject(with: data)
        let json = try JSONSerialization.data(withJSONObject: ["result": object, "wallSeconds": Date().timeIntervalSince(start)])
        print(String(decoding: json, as: UTF8.self))
    }
}
