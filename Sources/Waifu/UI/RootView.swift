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
    /// Whether Photos was opened from the ranked workspace (Back returns there)
    /// or straight from the welcome screen (Back returns to welcome).
    @State private var photosOpenedFromRanked = false
    /// Whether a drag currently hovers the window. The single drop registration
    /// below owns routing; the DropZone highlight binds to this.
    @State private var isDropTargeted = false
    @AppStorage("rightPanelExpanded") private var rightPanelExpanded = true
    // Keep the existing preference key so current Photos users retain their choice.
    @AppStorage("photosSquareThumbnails") private var squareThumbnails = false
    /// The grid's zoom rung as a cell width; Photos.app-style zoom walks the ladder.
    @AppStorage("gridThumbnailZoomWidth") private var gridZoomWidth = Double(GridZoom.defaultValue)
    @AppStorage("photosDownloadFromICloud") private var downloadFromICloud = false

    private var gridZoomTarget: Binding<CGFloat> {
        Binding(
            get: { GridZoom.snapped(CGFloat(gridZoomWidth)) },
            set: { gridZoomWidth = Double($0) }
        )
    }

    private var photosAlbum: PhotosMedia.Album? {
        guard let url = session.photosAlbumURL else { return nil }
        return PhotosMedia.shared.album(url)
    }

    private var workspaceLayers: some View {
        ZStack {
            if showWelcomeScreen {
                DropZone(session: session, isPrimary: true, isTargeted: isDropTargeted, openPhotos: openPhotosWorkspace)
            } else if showPhotosWorkspace {
                PhotosWorkspaceView(session: session, onBack: navigateBack) {
                    mainContent
                }
                .opacity(session.previewFrame == nil && !orb.isOpen ? 1 : 0)
                .allowsHitTesting(session.previewFrame == nil && session.uncover == nil && !orb.isOpen)
                .accessibilityHidden(session.previewFrame != nil || session.uncover != nil || orb.isOpen)
            } else if session.progress.isRunning && session.photosAlbumURL == nil {
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
                DropZone(session: session, isPrimary: true, isTargeted: isDropTargeted, openPhotos: openPhotosWorkspace)
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
    }

    private var routedWorkspace: some View {
        workspaceLayers
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .contentShape(Rectangle())
        // The one drop registration for the whole window. A second nested
        // registration over the same pixels races two loads on the same drag
        // providers; one drop must be handled exactly once.
        .onDrop(of: [.fileURL, .url, .image], isTargeted: $isDropTargeted) { providers in
            CrystalOrbImport.accept(providers, model: orb, session: session)
        }
        .onDisappear { if orb.isOpen { orb.close() } }
        .onChange(of: session.thumbnailGeneration) { _, _ in
            orb.sourceChanged(session: session)
        }
        .onChange(of: orb.isOpen) { _, active in if active { preview.pause() } }
        .onOpenURL { url in
            if CrystalOrbImageSource.isImage(url) {
                orb.importImage(url: url, session: session)
            } else if let source = FirefoxIntegration.incomingURL(url) {
                showWelcomeScreen = false
                showPhotosWorkspace = false
                photosOpenedFromRanked = false
                session.loadVideo(at: source)
            }
            else { session.errorMessage = "Unsupported media URL. Use an HTTP or HTTPS link." }
        }
    }

    var body: some View {
        routedWorkspace
        .background {
            PreviewToggleKeyboard(session: session, isPhotosMode: showPhotosWorkspace, orb: orb)
                .frame(width: 0, height: 0)
        }
        .background {
            thumbnailZoomShortcuts
        }
        .background {
            ExportPanel(session: session, isPresented: $session.showExportSheet)
                .frame(width: 0, height: 0)
        }
        .onChange(of: session.photosAlbumURL) { old, new in
            if old != nil && new == nil { showPhotosWorkspace = false }
        }
        .onChange(of: downloadFromICloud) { _, enabled in
            session.photosDownloadPolicyChanged(allowNetwork: enabled)
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
            // The title bar carries the window-level controls and the workspace's
            // icon actions, top right; status text rides in the same right edge,
            // just before those icons, inside its tag-like capsule.
            if session.isWritingPhotosTags {
                ToolbarItem(placement: .primaryAction) {
                    PhotosKeywordJobView(status: session.photosKeywordJob, cancel: session.cancelPhotosKeywordJob)
                }
            }
            // Back pops one workspace level: the orb, then Photos, then ranked, then welcome.
            if !showPhotosWorkspace && (orb.isOpen || (!showWelcomeScreen && (session.video != nil || session.progress.isRunning))) {
                ToolbarItem(placement: .navigation) {
                    Button(action: navigateBack) {
                        Label("Back", systemImage: "chevron.left")
                    }
                    .labelStyle(.iconOnly)
                    .accessibilityLabel(backAccessibilityLabel)
                }
            }
            if showPhotosWorkspace || orb.isOpen || (!showWelcomeScreen && session.video != nil) {
                ToolbarItem(placement: .primaryAction) { workspaceToolbarControls }
            }
        }
        .alert("Error", isPresented: .constant(session.errorMessage != nil)) {
            Button("OK") { session.errorMessage = nil }
        } message: {
            Text(ErrorAlertText.sanitized(session.errorMessage ?? ""))
        }
    }

    private var orbButton: some View {
        CrystalOrbToolbarButton(model: orb) {
            if orb.isOpen { orb.close() } else { orb.open(session: session) }
        }
    }

    private func openPhotosWorkspace() {
        // Reached from the welcome screen: Back from Photos returns to welcome.
        photosOpenedFromRanked = false
        showWelcomeScreen = false
        showPhotosWorkspace = true
    }

    private func openPhotosFromRanked() {
        photosOpenedFromRanked = true
        showPhotosWorkspace = true
    }

    private var backAccessibilityLabel: String {
        if orb.isOpen { return "Close Crystal Orb and go back" }
        return "Back to welcome screen"
    }

    /// Back pops one workspace level: the orb closes onto the workspace beneath
    /// it, Photos returns to the ranked workspace it was opened from, and the
    /// ranked workspace returns to welcome.
    private func navigateBack() {
        if orb.isOpen {
            orb.close()
            return
        }
        if showPhotosWorkspace {
            if photosOpenedFromRanked, session.video != nil {
                showPhotosWorkspace = false
            } else {
                returnToWelcome()
            }
            return
        }
        returnToWelcome()
    }

    private func returnToWelcome() {
        if orb.isOpen { orb.close() }
        preview.pause()
        if session.progress.isRunning { session.cancel() }
        session.closeUncover()
        session.previewFrame = nil
        showPhotosWorkspace = false
        photosOpenedFromRanked = false
        showWelcomeScreen = true
    }

    /// The panel toggle stays in the title bar so a collapsed panel can be
    /// brought back; it also toggles the Crystal Orb's inspector.
    private var panelToggleButton: some View {
        Button {
            if rightPanelExpanded { preview.pause() }
            rightPanelExpanded.toggle()
        } label: {
            Label(orb.isOpen ? "Image information" : "Preview and controls", systemImage: "sidebar.right")
        }
        .labelStyle(.iconOnly)
        .accessibilityLabel(rightPanelExpanded ? "Hide right panel" : "Show right panel")
    }

    /// The workspace's icon actions, top right of the title bar, with status
    /// capsules to their left. The back button owns leaving a workspace, so the
    /// orb button only appears while the orb is closed. Photos refresh and iCloud
    /// controls remain available over the Crystal Orb.
    private var workspaceToolbarControls: some View {
        HStack(spacing: 8) {
            if !orb.isOpen {
                orbButton
            }
            if showPhotosWorkspace { photosPanelActions }
            else if !orb.isOpen && session.video != nil { videoPanelActions }
            panelToggleButton
            exportButton
        }
    }

    @ViewBuilder
    private var videoPanelActions: some View {
        if session.uncover == nil {
            if photosAlbum != nil {
                Button { openPhotosFromRanked() } label: {
                    Label("Open Photos", systemImage: "photo.on.rectangle.angled")
                }
                .labelStyle(.iconOnly)
                .frame(width: 30)
            } else {
                Button {
                    if isAnalyzing { session.cancel() } else { session.analyze() }
                } label: {
                    Label(isAnalyzing ? "Cancel analysis" : "Analyze",
                          systemImage: isAnalyzing ? "stop.circle" : "arrow.clockwise")
                }
                .labelStyle(.iconOnly)
                .frame(width: 30)
            }
        }
        thumbnailShapeButton
    }

    private var photosPanelActions: some View {
        HStack(spacing: 8) {
            if !orb.isOpen { thumbnailShapeButton }

            Button { downloadFromICloud.toggle() } label: {
                Label("iCloud downloads", systemImage: downloadFromICloud ? "icloud.and.arrow.down.fill" : "icloud.slash")
            }
            .labelStyle(.iconOnly)
            .frame(width: 30)
            // The light's destination, but only while the cloud is switched off: that
            // is the icon the fraction in the Sort and Filter panel is linked to.
            .chiaroscuroDestination(link, isEligible: !downloadFromICloud)
            .accessibilityValue(downloadFromICloud ? "On" : "Off")

            Button {
                if session.progress.isRunning { session.cancel() } else { session.analyze() }
            } label: {
                Label(session.progress.isRunning ? "Cancel analysis" : "Refresh album",
                      systemImage: session.progress.isRunning ? "stop.circle" : "arrow.clockwise")
            }
            .labelStyle(.iconOnly)
            .frame(width: 30)
            .disabled(session.video == nil)
        }
    }

    private var thumbnailShapeButton: some View {
        Button { squareThumbnails.toggle() } label: {
            Label("Thumbnail shape", systemImage: squareThumbnails ? "square.grid.2x2" : "rectangle.ratio.16.to.9")
        }
        .labelStyle(.iconOnly)
        .frame(width: 30)
        .accessibilityValue(squareThumbnails ? "Square" : "Full aspect ratio")
    }

    /// Title-bar buttons go away with the panel they live in, so the shortcuts
    /// ride buttons that stay mounted for the window's lifetime.
    private var thumbnailZoomShortcuts: some View {
        Group {
            Button("Zoom In", action: zoomThumbnailsIn)
                .keyboardShortcut("=", modifiers: .command)
            Button("Zoom In", action: zoomThumbnailsIn)
                .keyboardShortcut("+", modifiers: .command)
            Button("Zoom Out", action: zoomThumbnailsOut)
                .keyboardShortcut("-", modifiers: .command)
            Button("Default Zoom", action: resetThumbnailsZoom)
                .keyboardShortcut("0", modifiers: .command)
        }
        .frame(width: 0, height: 0)
        .opacity(0)
        .accessibilityHidden(true)
    }

    private func zoomThumbnailsIn() {
        gridZoomWidth = Double(GridZoom.zoomIn(from: CGFloat(gridZoomWidth)))
    }

    private func zoomThumbnailsOut() {
        gridZoomWidth = Double(GridZoom.zoomOut(from: CGFloat(gridZoomWidth)))
    }

    private func resetThumbnailsZoom() {
        gridZoomWidth = Double(GridZoom.defaultValue)
    }

    private var exportButton: some View {
        Button { session.requestExport() } label: {
            Label(session.previewFrame == nil ? "Export selected frames" : "Export preview",
                  systemImage: "square.and.arrow.up")
        }
        .labelStyle(.iconOnly)
        .frame(width: 30)
        .disabled((session.previewFrame == nil && session.selectedFrameIDs.isEmpty) || session.uncover != nil)
        .keyboardShortcut("e", modifiers: .command)
    }

    @ViewBuilder
    private var mainContent: some View {
        HStack(spacing: 0) {
            // Candidates.
            FrameGridView(
                session: session,
                gridRevision: session.gridRevision,
                squareThumbnails: squareThumbnails,
                zoomTarget: gridZoomTarget,
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

            // Photos and ranked media share the preview and sorting controls.
            if rightPanelExpanded {
                Divider()
                VStack(spacing: 0) {
                    // The right panel starts directly with its panes: the keyword
                    // job status shows in the title bar, right of center and just
                    // before the icon actions. Icon actions are in the title bar;
                    // grid zoom is gesture and keyboard shortcut only.
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
