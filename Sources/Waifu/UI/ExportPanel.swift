import SwiftUI
import UniformTypeIdentifiers
import ImageIO

/// Handles export directory selection and results for the toolbar Export action.
///
/// Presents a folder importer when the toolbar requests export, then calls
/// `session.exportSelection(to:)` and shows a results sheet with the list of
/// exported files and any errors. The export runs asynchronously so the UI
/// stays responsive.
struct ExportPanel: View {
    @ObservedObject var session: WaifuSession
    @Binding var isPresented: Bool

    @State private var isExporting = false
    @State private var exportResults: [ExportResult] = []
    @State private var exportError: String?
    @State private var showResults = false
    @State private var selectedDirectory: URL?

    var body: some View {
        Color.clear
            .onAppear {
                if isPresented { presentFolderPicker() }
            }
            .onChange(of: isPresented) { _, presented in
                if presented { presentFolderPicker() }
            }
            .sheet(isPresented: $showResults) {
                ExportResultsSheet(
                    results: exportResults,
                    error: exportError,
                    directory: selectedDirectory
                )
            }
    }

    /// Presents a folder picker whose primary action reflects the export operation.
    private func presentFolderPicker() {
        selectedDirectory = nil

        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Save"
        panel.title = "Choose Export Folder"
        panel.message = "Choose where to save the exported frames."
        panel.begin { response in
            isPresented = false
            guard response == .OK, let directory = panel.url else { return }
            selectedDirectory = directory
            startExport(to: directory)
        }
    }

    /// Starts the export asynchronously.
    private func startExport(to directory: URL) {
        isExporting = true
        exportResults = []
        exportError = nil

        Task {
            let scopedAccess = directory.startAccessingSecurityScopedResource()
            defer { if scopedAccess { directory.stopAccessingSecurityScopedResource() } }
            do {
                let results = try await session.exportSelection(to: directory)
                await MainActor.run {
                    exportResults = results
                    isExporting = false
                    showResults = true
                }
            } catch {
                await MainActor.run {
                    exportError = error.localizedDescription
                    isExporting = false
                    showResults = true
                }
            }
        }
    }
}

/// Preview exports use the already-rendered model output rather than the source frame.
struct PreviewExportPanel: View {
    @ObservedObject var session: WaifuSession
    @Binding var isPresented: Bool
    let frame: AnalyzedFrame
    let photoURL: URL?
    let evidenceInput: PreviewEvidenceInput?
    @ObservedObject var realESRGAN: RealESRGANPreviewModel
    @ObservedObject var trufor: TruForPreviewModel
    let truforEnabled: Bool
    @ObservedObject var safire: SafirePreviewModel
    let safireEnabled: Bool

    @State private var exportResults: [ExportResult] = []
    @State private var exportError: String?
    @State private var showResults = false
    @State private var selectedDirectory: URL?

    var body: some View {
        Color.clear
            .onAppear { if isPresented { presentFolderPicker() } }
            .onChange(of: isPresented) { _, presented in
                if presented { presentFolderPicker() }
            }
            .sheet(isPresented: $showResults) {
                ExportResultsSheet(results: exportResults, error: exportError, directory: selectedDirectory)
            }
    }

    private var hasProcessedSelection: Bool {
        photoURL != nil && (realESRGAN.selection != nil || truforEnabled || safireEnabled)
    }

    private func presentFolderPicker() {
        selectedDirectory = nil
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Save"
        panel.title = "Choose Export Folder"
        panel.message = "Choose where to save the preview images."
        panel.begin { response in
            isPresented = false
            guard response == .OK, let directory = panel.url else { return }
            selectedDirectory = directory
            startExport(to: directory)
        }
    }

    private func startExport(to directory: URL) {
        exportResults = []
        exportError = nil
        Task {
            let scopedAccess = directory.startAccessingSecurityScopedResource()
            defer { if scopedAccess { directory.stopAccessingSecurityScopedResource() } }
            do {
                let results = hasProcessedSelection
                    ? try await exportProcessedImages(to: directory)
                    : try await exportOriginalSelection(to: directory)
                exportResults = results
            } catch {
                exportError = error.localizedDescription
            }
            showResults = true
        }
    }

    private func exportOriginalSelection(to directory: URL) async throws -> [ExportResult] {
        if session.selectedFrameIDs.isEmpty {
            return try await session.exportFrames([frame.id], to: directory)
        }
        return try await session.exportSelection(to: directory)
    }

    private func exportProcessedImages(to directory: URL) async throws -> [ExportResult] {
        guard let photoURL, let input = evidenceInput, input.photoURL == photoURL else {
            throw PreviewImageExportFailure.photoUnavailable
        }
        let sourceStem = photoURL.deletingPathExtension().lastPathComponent
        let frameID = frame.id
        let quality = session.settings.exportJPEGQuality
        let truforSnapshot = truforEnabled ? try trufor.exportSnapshot(inputID: input.id) : nil
        var safireImages: [(image: CGImage, suffix: String)] = []
        if safireEnabled {
            guard safire.photoURL == photoURL, safire.inputID == input.id,
                  let overlay = safire.overlayPixels, let mask = safire.maskPixels else {
                throw PreviewImageExportFailure.modelStillProcessing("SAFIRE")
            }
            safireImages = [(overlay, "safire-overlay"), (mask, "safire-mask")]
        }
        var realESRGANImage: (image: CGImage, suffix: String, format: ExportFormat)?
        if let variant = realESRGAN.selection {
            guard realESRGAN.outputInputID == input.id,
                  let rendered = realESRGAN.image(for: input),
                  let image = Self.cgImage(from: rendered) else {
                throw PreviewImageExportFailure.modelStillProcessing("RealESRGAN")
            }
            realESRGANImage = (image, "realesrgan-\(variant.label)", session.settings.exportFormat)
        }
        var results: [ExportResult] = []

        if let snapshot = truforSnapshot {
            for mode in TruForDisplay.allCases {
                try Task.checkCancellation()
                let image = try await TruForEvidenceRenderer.shared.render(
                    original: snapshot.original, evidence: snapshot.evidence, mode: mode)
                let result = try await Task.detached(priority: .utility) {
                    try ProcessedPreviewImageExporter.write(image, sourceStem: sourceStem,
                        frameID: frameID, suffix: mode.exportSuffix, directory: directory,
                        format: .png, jpegQuality: quality)
                }.value
                results.append(result)
            }
        }

        if !safireImages.isEmpty {
            for (image, suffix) in safireImages {
                try Task.checkCancellation()
                let result = try await Task.detached(priority: .utility) {
                    try ProcessedPreviewImageExporter.write(image, sourceStem: sourceStem,
                        frameID: frameID, suffix: suffix, directory: directory,
                        format: .png, jpegQuality: quality)
                }.value
                results.append(result)
            }
        }

        if let realESRGANImage {
            let result = try await Task.detached(priority: .utility) {
                try ProcessedPreviewImageExporter.write(realESRGANImage.image, sourceStem: sourceStem,
                    frameID: frameID, suffix: realESRGANImage.suffix, directory: directory,
                    format: realESRGANImage.format, jpegQuality: quality)
            }.value
            results.append(result)
        }

        guard !results.isEmpty else { throw PreviewImageExportFailure.noProcessedOutput }
        return results
    }

    private static func cgImage(from image: NSImage) -> CGImage? {
        var bounds = CGRect(origin: .zero, size: image.size)
        return image.cgImage(forProposedRect: &bounds, context: nil, hints: nil)
    }
}

private enum PreviewImageExportFailure: LocalizedError {
    case photoUnavailable
    case modelStillProcessing(String)
    case noProcessedOutput

    var errorDescription: String? {
        switch self {
        case .photoUnavailable:
            "The preview photo is not ready to export. Try again after it finishes loading."
        case .modelStillProcessing(let model):
            "\(model) is still processing this photo. Wait for it to finish, then export again."
        case .noProcessedOutput:
            "No processed preview images are ready to export."
        }
    }
}

/// A sheet showing the results of an export operation.
private struct ExportResultsSheet: View {
    let results: [ExportResult]
    let error: String?
    let directory: URL?

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 16) {
            // Header
            HStack {
                Image(systemName: error == nil ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                    .font(.system(size: 28))
                    .foregroundStyle(error == nil ? .green : .orange)

                VStack(alignment: .leading, spacing: 2) {
                    Text(error == nil ? "Export Complete" : "Export Completed with Errors")
                        .font(.title2.weight(.semibold))
                    if let directory {
                        Text(directory.path)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                }
                Spacer()
            }

            // Summary
            HStack(spacing: 24) {
                StatBox(label: "Exported", value: "\(results.count)", color: .green)
                if error != nil {
                    StatBox(label: "Error", value: "1", color: .orange)
                }
            }

            // Results list
            if !results.isEmpty {
                ScrollView {
                    VStack(alignment: .leading, spacing: 8) {
                        ForEach(results, id: \.url) { result in
                            ExportResultRow(result: result)
                        }
                    }
                }
                .frame(maxHeight: 300)
            }

            // Error message
            if let error {
                Text(error)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
                    .background(Color.orange.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
            }

            // Actions
            HStack {
                Spacer()
                if let directory {
                    Button("Open in Finder") {
                        NSWorkspace.shared.activateFileViewerSelecting([directory])
                    }
                    .keyboardShortcut(.defaultAction)
                }
                Button("Done") { dismiss() }
                    .keyboardShortcut(.cancelAction)
            }
        }
        .padding(24)
        .frame(width: 500)
    }
}

/// A single row in the export results list.
private struct ExportResultRow: View {
    let result: ExportResult

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "doc.fill")
                .font(.system(size: 14))
                .foregroundStyle(.secondary)
                .frame(width: 20)

            VStack(alignment: .leading, spacing: 1) {
                Text(result.url.lastPathComponent)
                    .font(.system(size: 12, design: .monospaced))
                    .lineLimit(1)
                    .truncationMode(.middle)

                Text("\(ByteCountFormatter.string(fromByteCount: Int64(result.byteCount), countStyle: .file)) • Frame \(result.frameID.raw)")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }

            Spacer()

            Button {
                NSWorkspace.shared.activateFileViewerSelecting([result.url])
            } label: {
                Image(systemName: "arrow.up.forward.square")
                    .font(.system(size: 12))
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 6))
    }
}

/// A small statistic box for the summary.
private struct StatBox: View {
    let label: String
    let value: String
    let color: Color

    var body: some View {
        VStack(spacing: 2) {
            Text(value)
                .font(.system(size: 24, weight: .bold, design: .rounded))
                .foregroundStyle(color)
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(minWidth: 80)
        .padding(.vertical, 8)
        .background(color.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
    }
}
