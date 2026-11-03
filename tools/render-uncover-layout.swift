import AppKit
import SwiftUI

/// Offscreen rendering only. No windows, event injection, app activation or UI interaction.
private struct LayoutBackend: TemporalVideoUpscaler {
    func capabilities() async -> UpscalerCapabilities {
        UpscalerCapabilities(available: true, message: "Layout preview", scales: UncoverScale.allCases,
                             minimumFrames: 1, requiresUnsafeMemoryOptIn: false)
    }
    func reconstruct(_ prepared: UncoverPrepared, allowUnsafeMemory: Bool,
                     progress: @escaping @Sendable (String) -> Void) async throws -> UncoverResult {
        throw UncoverError.backend("Rendering does not run inference")
    }
}

@main
struct RenderUncoverLayout {
    @MainActor
    static func main() async throws {
        let output = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let video = VideoInfo(url: output.appendingPathComponent("layout-placeholder.mov"), durationSeconds: 10,
                              width: 1920, height: 1080, frameRate: 30, estimatedFrameCount: 300)
        let session = UncoverSession(video: video, seconds: 0, backend: LayoutBackend())
        defer { session.dispose() }
        try await Task.sleep(for: .milliseconds(100))
        for width in [480, 640, 800, 1280] {
            let size = NSSize(width: width, height: 820)
            let host = NSHostingView(rootView: UncoverView(session: session, close: {}).environment(\.colorScheme, .light))
            host.frame = NSRect(origin: .zero, size: size)
            host.layoutSubtreeIfNeeded()
            guard let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds) else {
                throw UncoverError.backend("Unable to allocate offscreen bitmap")
            }
            host.cacheDisplay(in: host.bounds, to: bitmap)
            guard let png = bitmap.representation(using: .png, properties: [:]) else {
                throw UncoverError.backend("Unable to encode offscreen bitmap")
            }
            let path = output.appendingPathComponent("uncover-\(width).png")
            try png.write(to: path)
            print(path.path)
        }
    }
}
