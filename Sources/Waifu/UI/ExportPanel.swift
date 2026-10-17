import SwiftUI
import UniformTypeIdentifiers

/// Handles export directory selection and results for the toolbar Export action.
///
/// Presents an `NSOpenPanel` as soon as the toolbar requests export, then calls
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
            .onChange(of: isPresented) { _, shouldPresent in
                guard shouldPresent else { return }
                isPresented = false
                selectDirectoryAndExport()
            }
            .sheet(isPresented: $showResults) {
                ExportResultsSheet(
                    results: exportResults,
                    error: exportError,
                    directory: selectedDirectory
                )
            }
    }

    /// Opens a directory picker and starts the export.
    private func selectDirectoryAndExport() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.title = "Choose Export Directory"
        panel.message = "Selected frames will be exported as \(session.settings.exportFormat.fileExtension.uppercased()) files"
        panel.prompt = "Export Here"

        if panel.runModal() == .OK, let url = panel.url {
            selectedDirectory = url
            startExport(to: url)
        }
    }

    /// Starts the export asynchronously.
    private func startExport(to directory: URL) {
        isExporting = true
        exportResults = []
        exportError = nil

        Task {
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
                        ForEach(results, id: \.frameID) { result in
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
            .help("Show in Finder")
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
