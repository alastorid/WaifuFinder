import SwiftUI
import Combine

/// Photos keeps its album navigation beside the existing ranking workspace.
/// Choosing another album replaces only the source; the scoring controls and
/// result grid continue to use the same session as ordinary video imports.
struct PhotosWorkspaceView<RankedContent: View, TrailingControls: View>: View {
    @ObservedObject var session: WaifuSession
    private let rankedContent: () -> RankedContent
    private let trailingControls: () -> TrailingControls
    private let onBackToWelcome: () -> Void
    /// Shared with the Sort and Filter fraction; see `ChiaroscuroLink`.
    private let link: ChiaroscuroLink
    @AppStorage("photosDownloadFromICloud") private var downloadFromICloud = false
    @AppStorage("photosSquareThumbnails") private var squareThumbnails = false
    @State private var sidebarExpanded = true
    @Environment(\.scenePhase) private var scenePhase

    init(session: WaifuSession, link: ChiaroscuroLink, onBackToWelcome: @escaping () -> Void,
         @ViewBuilder trailingControls: @escaping () -> TrailingControls,
         @ViewBuilder rankedContent: @escaping () -> RankedContent) {
        self.session = session
        self.link = link
        self.onBackToWelcome = onBackToWelcome
        self.rankedContent = rankedContent
        self.trailingControls = trailingControls
    }

    @State private var libraryRevision = 0
    @State private var keywordRefreshPending = false
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
        .transaction { $0.animation = nil }
        .toolbar {
            ToolbarItemGroup(placement: .navigation) {
                Button(action: onBackToWelcome) {
                    Image(systemName: "chevron.left")
                }
                .help("Back to welcome screen")
                .accessibilityLabel("Back to welcome screen")

                Button { sidebarExpanded.toggle() } label: {
                    Image(systemName: "sidebar.left")
                }
                .help(sidebarExpanded ? "Collapse albums" : "Expand albums")
                .accessibilityLabel(sidebarExpanded ? "Collapse albums" : "Expand albums")
                Label("Photos", systemImage: "photo.on.rectangle.angled")
                    .font(.headline)
                    .labelStyle(.titleAndIcon)
            }
            ToolbarItem(placement: .principal) { albumTitle }
            ToolbarItem(placement: .primaryAction) {
                HStack(spacing: 8) {
                    PhotosKeywordJobView(status: session.photosKeywordJob, cancel: session.cancelPhotosKeywordJob)
                    albumControls
                    trailingControls()
                }
            }
        }
        .task(id: libraryRevision) { await refreshLibrary() }
        .onReceive(NotificationCenter.default.publisher(for: PhotosLibraryChanges.notification)
            .debounce(for: .milliseconds(350), scheduler: DispatchQueue.main)) { _ in
                if session.isWritingPhotosTags {
                    keywordRefreshPending = true
                } else {
                    libraryRevision += 1
                }
        }
        .onChange(of: session.isWritingPhotosTags) { _, running in
            if !running && keywordRefreshPending {
                keywordRefreshPending = false
                libraryRevision += 1
            }
        }
        .onChange(of: scenePhase) { _, phase in
            if phase == .active {
                if session.isWritingPhotosTags { keywordRefreshPending = true }
                else { libraryRevision += 1 }
            }
        }
        .onChange(of: downloadFromICloud) { _, enabled in
            session.photosDownloadPolicyChanged(allowNetwork: enabled)
        }
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
        .help(opening ? "Opening album…" : session.isSyncingPhotosOrder ? "Syncing order to Photos…" : session.photosOrderSyncMessage ?? statusText)
    }

    @ViewBuilder
    private var albumControls: some View {
        Button { squareThumbnails.toggle() } label: {
            Label("Thumbnail shape", systemImage: squareThumbnails ? "square.grid.2x2" : "rectangle.ratio.16.to.9")
        }
        .labelStyle(.iconOnly)
        .frame(width: 30)
        .accessibilityValue(squareThumbnails ? "Square" : "Full aspect ratio")
        .help(squareThumbnails ? "Show full aspect ratio" : "Crop thumbnails to squares")

        Button { session.photosOrderFilter = session.photosOrderFilter.next } label: {
            PhotosOrderIcon(filter: session.photosOrderFilter)
                .foregroundStyle(session.photosOrderFilter == .all ? Color.secondary : Color.accentColor)
        }
        .frame(width: 30)
        .disabled(album == nil)
        .accessibilityLabel("Order filter")
        .accessibilityValue(session.photosOrderFilter.title)
        .help("Order filter: \(session.photosOrderFilter.title). Click for \(session.photosOrderFilter.next.title.lowercased()) photos.")

        Button { downloadFromICloud.toggle() } label: {
            Label("iCloud downloads", systemImage: downloadFromICloud ? "icloud.and.arrow.down.fill" : "icloud.slash")
        }
        .labelStyle(.iconOnly)
        .frame(width: 30)
        // The light's destination, but only while the cloud is switched off: that
        // is the icon the fraction in the Sort and Filter panel is linked to.
        .chiaroscuroDestination(link, isEligible: !downloadFromICloud)
        .accessibilityValue(downloadFromICloud ? "On" : "Off")
        .help(downloadFromICloud ? "Disable iCloud downloads" : "Enable iCloud downloads; currently using only media on this Mac")

        Button {
            if session.progress.isRunning { session.cancel() } else { session.analyze() }
        } label: {
            Label(session.progress.isRunning ? "Cancel analysis" : "Analyze album",
                  systemImage: session.progress.isRunning ? "stop.circle" : "arrow.clockwise")
        }
        .labelStyle(.iconOnly)
        .frame(width: 30)
        .disabled(opening || session.video == nil)
        .help(session.progress.isRunning ? "Cancel analysis" : "Analyze album")
    }

    private var statusText: String {
        switch session.progress.phase {
        case .cancelled: "Analysis cancelled. Your album and completed results remain available."
        case .failed: session.progress.message
        default: session.progress.message == "Complete" ? "Rank, filter, tag, and export with the controls beside your photos." : session.progress.message
        }
    }

    @ViewBuilder
    private var workspaceContent: some View {
        if let album {
            rankedContent()
                .id(album.url)
                .transaction { $0.animation = nil }
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

    /// Refresh the current source as well as the sidebar. A fresh source URL
    /// gives previews and thumbnails a new identity after edits or deletions.
    private func refreshLibrary() async {
        let sourceURL = session.photosAlbumURL
        let selectionToken = selectionGeneration
        loading = true
        do {
            let refreshedAlbums = try await PhotosMedia.shared.listAlbums()
            try Task.checkCancellation()
            albums = refreshedAlbums
            error = albums.allSatisfy({ $0.count == 0 }) ? "No accessible albums contain photos." : nil
            if let sourceURL, let previous = PhotosMedia.shared.album(sourceURL) {
                let selection = albums.lazy.compactMap { $0.find(previous.collectionIdentifier) }.first
                guard let selection else {
                    if session.photosAlbumURL == sourceURL, selectionGeneration == selectionToken {
                        session.clearPhotosAlbum()
                        selectedAlbum = nil
                    }
                    loading = false
                    return
                }
                let refreshedURL = try await Task.detached(priority: .utility) {
                    try PhotosMedia.shared.openAlbum(selection, allowEmpty: true)
                }.value
                guard !Task.isCancelled, session.photosAlbumURL == sourceURL,
                      selectionGeneration == selectionToken else {
                    PhotosMedia.shared.release(refreshedURL)
                    return
                }
                if let refreshed = PhotosMedia.shared.album(refreshedURL),
                   refreshed.fingerprint != previous.fingerprint || refreshed.title != previous.title {
                    session.refreshPhotosAlbum(at: refreshedURL)
                } else {
                    PhotosMedia.shared.release(refreshedURL)
                }
            }
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
        .help(album.isFolder ? "\(album.title) · \(album.children?.count ?? 0) albums and folders"
              : "\(album.title) · \(album.count) photos")
    }

}
