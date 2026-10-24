import SwiftUI

/// Photos keeps its album navigation beside the existing ranking workspace.
/// Choosing another album replaces only the source; the scoring controls and
/// result grid continue to use the same session as ordinary video imports.
struct PhotosWorkspaceView<RankedContent: View>: View {
    @ObservedObject var session: WaifuSession
    private let rankedContent: () -> RankedContent
    private let onBack: () -> Void
    @State private var sidebarExpanded = true

    init(session: WaifuSession, onBack: @escaping () -> Void,
         @ViewBuilder rankedContent: @escaping () -> RankedContent) {
        self.session = session
        self.onBack = onBack
        self.rankedContent = rankedContent
    }

    @State private var albums: [PhotosAlbum] = []
    @State private var selectedAlbum: String?
    @State private var loading = true
    @State private var opening = false
    @State private var error: String?
    @State private var selectionTask: Task<Void, Never>?
    @State private var selectionGeneration = UUID()
    private var album: PhotosMedia.Album? {
        guard let url = session.photosAlbumURL else { return nil }
        return PhotosMedia.shared.album(url)
    }

    var body: some View {
        HStack(spacing: 0) {
            if sidebarExpanded {
                PhotosAlbumSidebar(albums: albums, selectedAlbum: $selectedAlbum,
                                   selectionID: selectedAlbum, loading: loading, error: error)
                    .equatable()
                    .frame(width: 230)
                Divider()
            }
            workspaceContent
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .background(Color(nsColor: .windowBackgroundColor))
        // Strip stray animations from the workspace, but let the grid's zoom
        // reflow commit ride its own marked transaction through.
        .transaction { if !$0[GridZoomTransactionKey.self] { $0.animation = nil } }
        .toolbar {
            ToolbarItemGroup(placement: .navigation) {
                // Back pops a workspace level (orb, then the ranked workspace the
                // album was opened from, then welcome), not always the main menu.
                Button(action: onBack) {
                    Image(systemName: "chevron.left")
                }
                .accessibilityLabel("Back")

                Button { sidebarExpanded.toggle() } label: {
                    Image(systemName: "sidebar.left")
                }
                .accessibilityLabel(sidebarExpanded ? "Collapse albums" : "Expand albums")
                Label("Photos", systemImage: "photo.on.rectangle.angled")
                    .font(.headline)
                    .labelStyle(.titleAndIcon)
            }
            ToolbarItem(placement: .principal) { albumTitle }
        }
        .task { await loadAlbums() }
        .onChange(of: selectedAlbum) { _, selection in openAlbum(selection) }
        .onDisappear { selectionTask?.cancel(); selectionGeneration = UUID() }
    }

    // All title-bar slots remain mounted while the source and analysis change.
    private var albumTitle: some View {
        HStack(spacing: 8) {
            ProgressView().controlSize(.small)
                .opacity(loading || opening || session.progress.isRunning || session.isSyncingPhotosOrder ? 1 : 0)
                .frame(width: 16, height: 16)
            VStack(alignment: .leading, spacing: 2) {
                Text(album?.title ?? "Photos")
                    .font(.headline)
                    .frame(maxWidth: .infinity, alignment: .leading)
                Text(album.map { "\($0.items.count) photos · \(session.frames.count) ranked" } ?? "Choose an album")
                    .font(.caption).foregroundStyle(.secondary)
                    .monospacedDigit()
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .lineLimit(1)
        }
        .frame(width: 230, height: 36)
    }

    @ViewBuilder
    private var workspaceContent: some View {
        if let album {
            rankedContent()
                .id(album.url)
                .transaction { if !$0[GridZoomTransactionKey.self] { $0.animation = nil } }
        } else {
            VStack(spacing: 12) {
                Image(systemName: "photo.stack").font(.system(size: 42)).foregroundStyle(.secondary)
                Text(loading ? "Reading your Photos albums…" : "Choose an album in the sidebar")
                    .font(.title3)
                Text("Photos appear here as they are analyzed and ranked.")
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    /// Read sidebar metadata once when opening the workspace. Album analysis is
    /// refreshed only by the existing Analyze album button.
    private func loadAlbums() async {
        loading = true
        do {
            albums = try await PhotosMedia.shared.listAlbums()
            try Task.checkCancellation()
            error = albums.allSatisfy({ $0.count == 0 }) ? "No accessible albums contain photos." : nil
        } catch is CancellationError { return }
        catch { self.error = error.localizedDescription }
        loading = false
    }

    private func openAlbum(_ id: String?) {
        guard let id, let selection = albums.lazy.compactMap({ $0.find(id) }).first,
              !selection.isFolder, selection.count > 0 else { return }
        selectionTask?.cancel()
        let generation = UUID(); selectionGeneration = generation
        opening = true; error = nil
        selectionTask = Task {
            do {
                let url = try await Task.detached(priority: .userInitiated) {
                    try PhotosMedia.shared.openAlbum(selection)
                }.value
                guard !Task.isCancelled, selectionGeneration == generation else {
                    PhotosMedia.shared.release(url); return
                }
                session.loadVideo(at: url)
            } catch {
                if !Task.isCancelled && selectionGeneration == generation { self.error = error.localizedDescription }
            }
            if selectionGeneration == generation { opening = false }
        }
    }
}

/// Album navigation depends only on its own inputs, never on analysis ticks.
private struct PhotosAlbumSidebar: View, Equatable {
    let albums: [PhotosAlbum]
    @Binding var selectedAlbum: String?
    let selectionID: String?
    let loading: Bool
    let error: String?

    nonisolated static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.albums == rhs.albums && lhs.selectionID == rhs.selectionID
            && lhs.loading == rhs.loading && lhs.error == rhs.error
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            if loading && albums.isEmpty {
                VStack(spacing: 10) {
                    ProgressView()
                        .controlSize(.small)
                    Text("Loading albums…")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .accessibilityElement(children: .combine)
            } else {
                if loading {
                    HStack(spacing: 8) {
                        ProgressView()
                            .controlSize(.small)
                        Text("Refreshing albums…")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                }
                List(selection: $selectedAlbum) {
                    if albums.contains(where: { !$0.isSmartAlbum }) {
                        Section("My Albums") {
                            OutlineGroup(albums.filter { !$0.isSmartAlbum }, children: \.children) { album in
                                albumRow(album)
                                    .selectionDisabled(album.isFolder || album.count == 0)
                            }
                        }
                    }
                    if albums.contains(where: \.isSmartAlbum) {
                        Section("Smart Albums") {
                            ForEach(albums.filter(\.isSmartAlbum)) { album in albumRow(album) }
                        }
                    }
                }
                .listStyle(.sidebar)
            }
            if let error {
                Text(error).font(.caption).foregroundStyle(.secondary).textSelection(.enabled).padding(12)
            }
        }
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))
    }

    private func albumRow(_ album: PhotosAlbum) -> some View {
        HStack(spacing: 9) {
            Image(systemName: album.isFolder ? "folder" : album.isSmartAlbum ? "sparkles" : "rectangle.stack")
                .foregroundStyle(.secondary)
            Text(album.title).lineLimit(1)
            Spacer(minLength: 4)
            if !album.isFolder {
                Text("\(album.count)").font(.caption).foregroundStyle(.secondary)
            }
        }
        .tag(album.id)
    }

}
