import AppKit
import SwiftUI

struct WaifuSettingsView: View {
    @AppStorage("settingsTab") private var selection = "general"
    var body: some View {
        TabView(selection: $selection) {
            GeneralSettingsView().tag("general").tabItem { Label("General", systemImage: "gearshape") }
            ModelSettingsView().tag("models").tabItem { Label("Models", systemImage: "internaldrive") }
            FirefoxSettingsView().tag("firefox").tabItem { Label("Firefox", systemImage: "network") }
        }
        .padding(16)
        .frame(width: 760, height: 620)
    }
}

private struct GeneralSettingsView: View {
    @ObservedObject private var tagMemory = TagResultMemory.shared
    @ObservedObject private var analysisMemory = AnalysisResultMemory.shared
    @State private var cacheToDelete: CacheKind?

    private enum CacheKind: String {
        case tagging
        case analysis

        var title: String {
            switch self {
            case .tagging: "tagging results"
            case .analysis: "analysis results"
            }
        }
    }

    var body: some View {
        Form {
            Section("WaifuFinder") {
                Text("Find and restore the best frames in your videos.")
                Text("Analysis, grouping, and export controls are available in the main window. Manage restoration models in the Models tab.")
                    .foregroundStyle(.secondary)
            }

            Section("Tagging memory") {
                Toggle("Remember tagging results", isOn: $tagMemory.rememberTaggingResults)
                    .help("Reuse completed tagger results the next time the same video is analyzed.")

                Stepper("Delete results older than \(tagMemory.retentionDays) days",
                        value: $tagMemory.retentionDays,
                        in: 1...36_500)
                    .help("Expired results are removed when the cache is checked. The minimum is one day.")

                Text("Scores are saved in compressed binary form using vocabulary indices. Tag names are reconstructed when the results are loaded.")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                GroupBox {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("File location")
                            .font(.headline)
                        Text(tagMemory.fileURL.path)
                            .font(.caption.monospaced())
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                            .fixedSize(horizontal: false, vertical: true)
                        HStack {
                            Text("Current size")
                            Spacer()
                            Text(ByteCountFormatter.string(fromByteCount: tagMemory.fileSize, countStyle: .file))
                                .monospacedDigit()
                        }
                        HStack {
                            Text("Entries")
                            Spacer()
                            Text(tagMemory.entryCount.formatted())
                                .monospacedDigit()
                        }
                        Button("Delete cached results…", role: .destructive) {
                            cacheToDelete = .tagging
                        }
                        if let error = tagMemory.errorMessage {
                            Text("Cache status: \(error)")
                                .font(.caption)
                                .foregroundStyle(.red)
                                .textSelection(.enabled)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(4)
                }
            }

            Section("Analysis memory") {
                Toggle("Remember analysis results", isOn: $analysisMemory.rememberAnalysisResults)
                    .help("Reuse frame scores and image features when the same video is analyzed again.")

                Stepper("Delete results older than \(analysisMemory.retentionDays) days",
                        value: $analysisMemory.retentionDays,
                        in: 1...36_500)
                    .help("Expired results are removed when the cache is checked. The minimum is one day.")

                Text("Saves aesthetics and face scores plus compressed image features. Frames are still decoded for thumbnails, and grouping and ranking are recalculated with current settings.")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                GroupBox {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("File location")
                            .font(.headline)
                        Text(analysisMemory.fileURL.path)
                            .font(.caption.monospaced())
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                            .fixedSize(horizontal: false, vertical: true)
                        HStack {
                            Text("Current size")
                            Spacer()
                            Text(ByteCountFormatter.string(fromByteCount: analysisMemory.fileSize, countStyle: .file))
                                .monospacedDigit()
                        }
                        HStack {
                            Text("Entries")
                            Spacer()
                            Text(analysisMemory.entryCount.formatted())
                                .monospacedDigit()
                        }
                        Button("Delete cached results…", role: .destructive) {
                            cacheToDelete = .analysis
                        }
                        if let error = analysisMemory.errorMessage {
                            Text("Cache status: \(error)")
                                .font(.caption)
                                .foregroundStyle(.red)
                                .textSelection(.enabled)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(4)
                }
            }
        }.formStyle(.grouped)
            .confirmationDialog(
                "Delete remembered \(cacheToDelete?.title ?? "cache")?",
                isPresented: Binding(
                    get: { cacheToDelete != nil },
                    set: { if !$0 { cacheToDelete = nil } }
                ),
                titleVisibility: .visible
            ) {
                Button("Delete", role: .destructive) {
                    guard let cache = cacheToDelete else { return }
                    cacheToDelete = nil
                    switch cache {
                    case .tagging:
                        Task { await tagMemory.deleteAll() }
                    case .analysis:
                        Task { await analysisMemory.deleteAll() }
                    }
                }
                Button("Cancel", role: .cancel) { cacheToDelete = nil }
            } message: {
                Text("This removes the saved cache file. Results can be created again as videos are processed.")
            }
    }
}

struct ModelSettingsView: View {
    @ObservedObject private var library = ModelLibrary.shared
    @AppStorage("settingsModelFocus") private var settingsModelFocus = ""
    @State private var highlightedModel: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Model Library").font(.title2.bold())
                    Text("\(library.entries.filter(\.ready).count) ready · \(ByteCountFormatter.string(fromByteCount: library.totalBytes, countStyle: .file)) on disk")
                        .foregroundStyle(.secondary).font(.callout)
                }
                Spacer()
                Button { library.refresh() } label: { Image(systemName: "arrow.clockwise") }.help("Refresh model status")
            }
            GroupBox {
                HStack(alignment: .top) {
                    Image(systemName: "folder").font(.title3).foregroundStyle(.secondary)
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Download location").font(.headline)
                        Text(library.storage.path).font(.caption).foregroundStyle(.secondary).textSelection(.enabled).lineLimit(2)
                        Text("Changing this folder applies to future downloads. Existing models stay where they are.").font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button("Change…") { library.chooseStorage() }.disabled(library.active != nil)
                }.padding(5)
            }
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(spacing: 10) {
                        ForEach(library.entries) { entry in modelRow(entry).id(entry.model.rawValue) }
                        GroupBox {
                            HStack(alignment: .top) {
                                Image(systemName: "apple.logo").frame(width: 28)
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("Apple Vision").font(.headline)
                                    Text("Aesthetics, face quality, face detection, and image similarity").font(.caption).foregroundStyle(.secondary)
                                    Text("Included with macOS · managed by the system · no separate download").font(.caption).foregroundStyle(.secondary)
                                }
                                Spacer()
                            }
                            .padding(6)
                        }.padding(6)
                    }
                }
                .onAppear {
                    revealRequestedModel(using: proxy)
                }
                .onChange(of: settingsModelFocus) { _, _ in
                    revealRequestedModel(using: proxy)
                }
            }
            if let error = library.error {
                HStack(alignment: .top) {
                    Image(systemName: "exclamationmark.triangle").foregroundStyle(.orange)
                    Text(error).font(.caption).textSelection(.enabled)
                    Spacer()
                    Button("Dismiss") { library.error = nil }
                }
            }
        }
        .onAppear { library.refresh() }
    }

    private func revealRequestedModel(using proxy: ScrollViewProxy) {
        guard !settingsModelFocus.isEmpty else { return }
        let target = settingsModelFocus
        highlightedModel = target
        withAnimation(.easeInOut(duration: 0.35)) {
            proxy.scrollTo(target, anchor: .center)
        }
        settingsModelFocus = ""
    }

    private func modelRow(_ entry: ModelLibraryEntry) -> some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 8) {
                HStack(alignment: .top) {
                    Image(systemName: entry.ready ? "checkmark.circle.fill" : "arrow.down.circle")
                        .foregroundStyle(entry.ready ? Color.green : Color.secondary).font(.title3).frame(width: 28)
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text(entry.model.name).font(.headline)
                            Spacer()
                            Text(entry.ready ? "Ready" : entry.location == nil ? (entry.pending == nil ? "Not installed" : "Download incomplete") : "Needs attention")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        Text(entry.model.detail).font(.caption).foregroundStyle(.secondary)
                        if entry.model.isNonCommercialRestricted {
                            Text("Non-commercial use only").font(.caption).foregroundStyle(.secondary)
                        }
                        if let location = entry.location {
                            Text("\(ByteCountFormatter.string(fromByteCount: entry.bytes, countStyle: .file)) · \(library.isManaged(entry) ? "App managed" : "External folder")")
                                .font(.caption).foregroundStyle(.secondary)
                            Text(location.path).font(.caption).foregroundStyle(.secondary).textSelection(.enabled).lineLimit(2)
                        }
                    }
                }
                if let pending = entry.pending, library.active != entry.model {
                    Text("Incomplete download · \(pending.path)").font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                }
                if (library.inUse[entry.model] ?? 0) > 0 {
                    Text("In use").font(.caption).foregroundStyle(.secondary)
                }
                if library.active == entry.model {
                    HStack {
                        if let fraction = library.fraction { ProgressView(value: fraction).frame(maxWidth: 220) }
                        else { ProgressView().controlSize(.small) }
                        Text(library.status).font(.caption).lineLimit(1)
                        Spacer()
                        Button("Cancel") { library.cancel() }
                    }
                } else {
                    HStack {
                        Spacer()
                        if let location = entry.location {
                            Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([location]) }
                            Button(library.isManaged(entry) ? "Remove…" : "Disconnect…") { library.remove(entry) }
                        }
                        if entry.pending != nil {
                            Button("Discard Download…") { library.discardDownload(entry) }
                        }
                        if !entry.ready {
                            Button(entry.pending == nil ? "Download" : "Retry Download") { library.download(entry.model) }.buttonStyle(.borderedProminent)
                        }
                        Button("Choose Existing…") { library.chooseExisting(entry.model) }
                    }.disabled(library.active != nil || (library.inUse[entry.model] ?? 0) > 0)
                }
            }.padding(6)
        }
        .overlay {
            RoundedRectangle(cornerRadius: 8)
                .stroke(highlightedModel == entry.model.rawValue ? Color.accentColor : .clear, lineWidth: 2)
        }
    }
}
