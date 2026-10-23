import SwiftUI
import UniformTypeIdentifiers

/// The main window content.
///
/// Two panes and one question — *which frame do I want?* The grid holds the
/// candidates; the right column decides what is in it (the order, the per-group
/// cap, and what the cap left out) and shows the focused frame in motion. There is
/// no timeline: the preview's own transport scrubs, and the grid is a finer way to
/// land on a moment than a 10-pixel-per-minute track was.
///
/// The old left sidebar is gone. Its two lists answered "which frames are alike?"
/// and "what are the top ones?"; the first is legible from the group badges in the
/// grid, and the second is what the order panel is for.
struct RootView: View {
    @StateObject private var session = WaifuSession()
    @StateObject private var orb = WaifuCrystalOrbModel()
    // Only the player panel observes its clock; playback must not refresh the grid.
    @State private var preview = PreviewController()
    // One per window, shared by the Sort and Filter fraction and the cloud in the
    // title bar. See ChiaroscuroLink.
    @StateObject private var link = ChiaroscuroLink()
    @State private var showPhotosWorkspace = false
    @State private var showWelcomeScreen = false
    @AppStorage("rightPanelExpanded") private var rightPanelExpanded = true
    // Keep the existing preference key so current Photos users retain their choice.
    @AppStorage("photosSquareThumbnails") private var squareThumbnails = false

    private var photosAlbum: PhotosMedia.Album? {
        guard let url = session.photosAlbumURL else { return nil }
        return PhotosMedia.shared.album(url)
    }

    var body: some View {
        ZStack {
            if showWelcomeScreen {
                DropZone(session: session, isPrimary: true, openPhotos: openPhotosWorkspace, imageDrop: { CrystalOrbImport.accept($0, model: orb, session: session) })
            } else if showPhotosWorkspace {
                PhotosWorkspaceView(session: session, link: link, onBackToWelcome: returnToWelcome,
                                    orbControl: { orbButton }, trailingControls: { workspaceControls }) {
                    mainContent
                }
                .opacity(session.previewFrame == nil && !orb.isOpen ? 1 : 0)
                .allowsHitTesting(session.previewFrame == nil && session.uncover == nil && !orb.isOpen)
                .accessibilityHidden(session.previewFrame != nil || session.uncover != nil || orb.isOpen)
            } else if session.progress.isRunning {
                AnalysisProgressView(progress: session.progress, video: session.video, cancel: session.cancel)
            } else if session.video != nil {
                // Keep the grid and AVPlayer view mounted while frame preview is
                // open. Tearing down this subtree forces the thumbnail grid to be
                // rebuilt and makes AVPlayer reattach/seek when the user closes it.
                //
                // Keep keyboard focus intact as well. Pointer and accessibility
                // interaction are gated below; monitors own preview keyboard input.
                mainContent
                    .opacity(session.previewFrame == nil && !orb.isOpen ? 1 : 0)
                    .allowsHitTesting(session.previewFrame == nil && session.uncover == nil && !orb.isOpen)
                    .accessibilityHidden(session.previewFrame != nil || session.uncover != nil || orb.isOpen)
            } else {
                // Full-window drop zone when no video is loaded
                DropZone(session: session, isPrimary: true, openPhotos: openPhotosWorkspace, imageDrop: { CrystalOrbImport.accept($0, model: orb, session: session) })
            }
            // Keep dismissal immediate; the grid restores scrolling and focus only
            // after the session mutation has committed.
            if let frame = session.previewFrame {
                FramePreviewView(frame: frame, session: session)
                // No `.id(frame.id)` here on purpose. A changing identity tears the view
                // down and rebuilds it on every step — blank, then thumbnail, then the
                // full-resolution frame — which is exactly the flicker it caused. The
                // view keeps one identity and reloads through `.task(id: frame.id)`.
                .transition(.identity)
            }
            if let uncover = session.uncover {
                UncoverView(session: uncover, close: { session.closeUncover() })
            }
            if orb.isOpen {
                WaifuCrystalOrbView(model: orb, session: session).transition(.identity)
            }
            // The east egg's light. Mounted for the life of the window rather than
            // while a beam exists: it costs one empty view until something is drawn,
            // and it keeps a light that moves sixty times a second out of the
            // hierarchy that draws the grid. See ChiaroscuroLink.
            ChiaroscuroWindowLayer(link: link)
                .frame(width: 0, height: 0)
        }
        .onDrop(of: [.fileURL, .url, .image], isTargeted: nil) { providers in
            CrystalOrbImport.accept(providers, model: orb, session: session)
        }
        .onDisappear { if orb.isOpen { orb.close() } }
        .onChange(of: session.thumbnailGeneration) { _, _ in orb.sourceChanged(session: session) }
        .onChange(of: orb.isOpen) { _, active in if active { preview.pause() } }
        .onOpenURL { url in
            if CrystalOrbImageSource.isImage(url) {
                orb.importImage(url: url, session: session)
            } else if let source = FirefoxIntegration.incomingURL(url) {
                showWelcomeScreen = false
                showPhotosWorkspace = false
                session.loadVideo(at: source)
            }
            else { session.errorMessage = "Unsupported media URL. Use an HTTP or HTTPS link." }
        }
        .background {
            PreviewToggleKeyboard(session: session, isPhotosMode: showPhotosWorkspace, orb: orb)
                .frame(width: 0, height: 0)
        }
        .background {
            ExportPanel(session: session, isPresented: $session.showExportSheet)
                .frame(width: 0, height: 0)
        }
        .onChange(of: session.photosAlbumURL) { old, new in
            if old != nil && new == nil { showPhotosWorkspace = false }
        }
        .onChange(of: session.previewFrame?.id) { _, frameID in
            Diag.trace(.preview, message: "root preview rendered frame=\(frameID?.description ?? "nil")")
        }
        .onChange(of: session.uncover != nil) { _, active in if active { preview.pause() } }
        .onChange(of: session.previewFrame != nil) { _, active in if active { preview.pause() } }
        .onChange(of: session.progress.isRunning) { _, running in
            if !running, orb.isOpen, orb.query != nil { orb.search() }
            if running {
                showWelcomeScreen = false
                preview.pause()
            }
        }
        .toolbar {
            if showWelcomeScreen && !showPhotosWorkspace && session.isWritingPhotosTags {
                ToolbarItem(placement: .primaryAction) {
                    PhotosKeywordJobView(status: session.photosKeywordJob, cancel: session.cancelPhotosKeywordJob)
                }
            }
            if !showWelcomeScreen && !showPhotosWorkspace && (session.video != nil || session.progress.isRunning) {
                ToolbarItem(placement: .navigation) {
                    Button(action: returnToWelcome) {
                        Label("Back to welcome screen", systemImage: "chevron.left")
                    }
                    .labelStyle(.iconOnly)
                    .help("Back to welcome screen")
                    .accessibilityLabel("Back to welcome screen")
                }
            }
            if !showPhotosWorkspace && session.video != nil {
                ToolbarItem(placement: .primaryAction) {
                    HStack(spacing: 8) {
                        PhotosKeywordJobView(status: session.photosKeywordJob, cancel: session.cancelPhotosKeywordJob)
                        orbButton
                        if session.uncover == nil {
                            if photosAlbum != nil {
                                Button { showPhotosWorkspace = true } label: {
                                    Label("Open Photos", systemImage: "photo.on.rectangle.angled")
                                }
                                .labelStyle(.iconOnly)
                                .frame(width: 30)
                                .help("Open Photos")
                            } else {
                                Button {
                                    if isAnalyzing { session.cancel() } else { session.analyze() }
                                } label: {
                                    Label(isAnalyzing ? "Cancel analysis" : "Analyze",
                                          systemImage: isAnalyzing ? "stop.circle" : "arrow.clockwise")
                                }
                                .labelStyle(.iconOnly)
                                .frame(width: 30)
                                .help(isAnalyzing ? "Cancel analysis" : "Re-run analysis")
                            }
                        }
                        Button { squareThumbnails.toggle() } label: {
                            Label("Thumbnail shape", systemImage: squareThumbnails ? "square.grid.2x2" : "rectangle.ratio.16.to.9")
                        }
                        .labelStyle(.iconOnly)
                        .frame(width: 30)
                        .accessibilityValue(squareThumbnails ? "Square" : "Full aspect ratio")
                        .help(squareThumbnails ? "Show full aspect ratio" : "Crop thumbnails to squares")
                        workspaceControls
                    }
                }
            }
        }
        .alert("Error", isPresented: .constant(session.errorMessage != nil)) {
            Button("OK") { session.errorMessage = nil }
        } message: {
            Text(session.errorMessage ?? "")
        }
    }

    private var orbButton: some View {
        CrystalOrbToolbarButton(model: orb) {
            if orb.isOpen { orb.close() } else { orb.open(session: session) }
        }
    }

    private func openPhotosWorkspace() {
        showWelcomeScreen = false
        showPhotosWorkspace = true
    }

    private func returnToWelcome() {
        if orb.isOpen { orb.close() }
        preview.pause()
        if session.progress.isRunning { session.cancel() }
        session.closeUncover()
        session.previewFrame = nil
        showPhotosWorkspace = false
        showWelcomeScreen = true
    }

    // Photos embeds these after its own actions in one toolbar item, so
    // SwiftUI cannot merge Refresh into a slot to the right of Export.
    private var workspaceControls: some View {
        HStack(spacing: 8) {
            Button {
                if rightPanelExpanded { preview.pause() }
                rightPanelExpanded.toggle()
            } label: {
                Label("Preview and controls", systemImage: "sidebar.right")
            }
            .labelStyle(.iconOnly)
            .help(rightPanelExpanded ? "Collapse preview and controls" : "Expand preview and controls")
            .accessibilityLabel(rightPanelExpanded ? "Collapse preview and controls" : "Expand preview and controls")

            Button { session.showExportSheet = true } label: {
                Label("Export selected frames", systemImage: "square.and.arrow.up")
            }
            .labelStyle(.iconOnly)
            .frame(width: 30)
            .disabled(session.selectedFrameIDs.isEmpty || session.uncover != nil)
            .keyboardShortcut("e", modifiers: .command)
            .help("Export \(session.selectedFrameIDs.count) selected frames")
        }
    }

    @ViewBuilder
    private var mainContent: some View {
        HStack(spacing: 0) {
            // Candidates.
            FrameGridView(
                session: session,
                gridRevision: session.gridRevision,
                squareThumbnails: squareThumbnails,
                onFrameClick: { frame, kind in
                    session.focusFromPointer(frame.id)
                    switch kind {
                    case .single: session.selectFrame(frame.id)
                    case .toggle: session.selectFrame(frame.id, command: true)
                    case .range: session.selectFrame(frame.id, shift: true)
                    case .additiveRange: session.selectFrame(frame.id, command: true, shift: true)
                    case .double: break
                    }
                }
            )
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            if rightPanelExpanded {
                Divider()
                VStack(spacing: 0) {
                    VideoPreviewPanel(session: session, preview: preview)
                    Divider()
                    FrameOrderPanel(session: session, link: link)
                }
                .frame(width: 340)
                .frame(maxHeight: .infinity, alignment: .top)
                .transaction { $0.animation = nil }
            }
        }
    }

    /// Whether an analysis pass is in flight, which the toolbar's Analyze/Cancel pair
    /// keys off.
    private var isAnalyzing: Bool {
        session.progress.isRunning
    }
}
