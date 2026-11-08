import AppKit
import Combine
import Foundation

@MainActor
final class ClipExportController: ObservableObject {
    @Published private(set) var isExporting = false
    @Published private(set) var status = ""
    @Published private(set) var exportedURLs: [URL] = []
    @Published var error: String?

    func begin(video: VideoInfo, frames: [AnalyzedFrame]) {
        guard !isExporting else { return }
        let ranges = VideoClipExporter.ranges(video: video, frames: frames)
        guard !ranges.isEmpty else { return }
        if let directory = DefaultExportFolder.shared.url {
            start(ranges, directory: directory)
            return
        }
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = true
        panel.title = "Choose Clip Export Folder"
        panel.prompt = "Export"
        panel.message = "Save \(ranges.count) clip\(ranges.count == 1 ? "" : "s") covering the selected ±3-second ranges."
        panel.begin { [weak self] response in
            guard response == .OK, let directory = panel.url else { return }
            self?.start(ranges, directory: directory)
        }
    }

    private func start(_ ranges: [VideoClipRange], directory: URL) {
        isExporting = true
        exportedURLs = []
        error = nil
        status = "Exporting clips · 0/\(ranges.count)"
        Task {
            let access = directory.startAccessingSecurityScopedResource()
            defer {
                if access { directory.stopAccessingSecurityScopedResource() }
                isExporting = false
            }
            do {
                for range in ranges {
                    exportedURLs.append(try await VideoClipExporter.export(range, to: directory))
                    status = "Exporting clips · \(exportedURLs.count)/\(ranges.count)"
                }
                status = "Exported \(exportedURLs.count) clip\(exportedURLs.count == 1 ? "" : "s")"
            } catch {
                status = "Exported \(exportedURLs.count)/\(ranges.count) clips"
                self.error = error.localizedDescription
            }
        }
    }
}
