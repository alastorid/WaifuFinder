import AppKit
import AVFoundation
import SwiftUI

/// The zoomable look at a single frame — what Space opens from the grid, and what
/// Space, Escape or Close leaves.
///
/// **An overlay in the same window, not a sheet.** A sheet slides in and out, and this
/// is a look-at-this gesture that should appear and disappear with the key press; an
/// overlay is inserted by an `if`, so there is nothing to animate. It also leaves the
/// grid suspended while it is open, so navigation cannot redraw hidden thumbnails.
/// Its key monitor owns navigation; closing restores the grid at the focused frame.
///
/// ## Navigating without flicker
///
/// This view's identity is **stable across the frames it shows** (the parent does not
/// re-`.id` it), and it reloads through `.task(id: frame.id)`. Both matter:
///
///   * A changing identity would tear the view down and build a new one per step —
///     a blank frame followed by the thumbnail.
///   * The old image stays on screen until the new one has been decoded, rather than
///     being cleared to a placeholder first. The header says "reading…" meanwhile, so
///     the picture lagging a step behind the readout is stated rather than surprising.
///   * Fresh key presses replace a read; held keys wait for each decode to finish.
///
/// Reads the exact source timestamp at native resolution. Grid thumbnails are
/// analysis-sized JPEGs and cannot supply the pixels needed for inspection.
struct FramePreviewView: View {
    let frame: AnalyzedFrame
    @ObservedObject var session: WaifuSession

    @ObservedObject private var inferencePool=RemoteInferencePool.shared
    @StateObject private var trufor=TruForPreviewModel()
    @State private var truforEnabled=false
    @State private var truforDisplay=TruForDisplay.combined
    @StateObject private var safire=SafirePreviewModel()
    @State private var safireEnabled=false
    @State private var showSafireMask=false
    @State private var showOriginalEvidence=false
    @State private var evidenceInput: PreviewEvidenceInput?
    @State private var evidenceRetry = UUID()
    @AppStorage(GoogleLensSearch.enabledKey) private var googleLensEnabled = false
    @State private var image: NSImage?
    @State private var animation: PhotosAnimatedImage?
    /// Keep the previous original visible until the next source frame is ready.
    @State private var displayedSourceURL: URL?
    @State private var displayedIsOriginal = false
    @State private var displayedFrameID: FrameID?
    @State private var note: String?
    @State private var reading = false
    @State private var showReadingStatus = false
    /// A failed read is finished too; an unreadable frame must not trap held arrows.
    @State private var completedReadFrameID: FrameID?
    /// Zoom as a multiple of the frame's fit-to-window size (1 = fit).
    @State private var zoom: CGFloat = 1
    @State private var displayedImageScale: CGFloat = 1
    /// The point in the image kept at the viewport center while frames change.
    @State private var zoomCenter = CGPoint(x: 0.5, y: 0.5)
    /// The local key monitor installed while the preview is up.
    @State private var keyMonitor: Any?
    @State private var arrows = PreviewArrowNavigation()
    @State private var arrowRepeater = PreviewArrowRepeater()
    @State private var loadID = UUID()
    @State private var photoMetadata: PhotoMetadata?
    @State private var metadataLoading = false
    @State private var metadataError: String?
    @State private var metadataFrameID: FrameID?
    @State private var metadataTags: [String] = []
    @State private var metadataRating: String?
    /// The metadata sidebar answers to the title bar's panel toggle. It shares
    /// the workspace's key so one expand/collapse icon rules every right panel,
    /// including the Crystal Orb's inspector.
    @AppStorage("rightPanelExpanded") private var inspectorExpanded = true
    @AppStorage("photosDownloadFromICloud") private var downloadFromICloud = false

    init(frame: AnalyzedFrame, session: WaifuSession) {
        self.frame = frame; self.session = session
        let identity = session.thumbnailGeneration.uuidString + ":" + frame.thumbnailKey
        let cached = GridThumbnailLoader.shared.cachedImage(identity: identity)
        let initial = cached.map { NSImage(cgImage: $0, size: NSSize(width: $0.width, height: $0.height)) }
            ?? session.cachedThumbnailData(forKey: frame.thumbnailKey).flatMap(NSImage.init(data:))
        _image = State(initialValue: initial)
        _displayedFrameID = State(initialValue: initial == nil ? nil : frame.id)
        _displayedSourceURL = State(initialValue: initial == nil ? nil : session.video?.url)
    }

    var body: some View {
        ZStack {
            // The *window* background, not black: the preview belongs to the same app in
            // the same appearance, and a light app flashing a black sheet is exactly the
            // kind of thing that makes one panel feel bolted on.
            Color(nsColor: .windowBackgroundColor)
                .opacity(0.97)
                .ignoresSafeArea()

            VStack(spacing: 0) {
                header
                Divider()
                HStack(spacing: 0) {
                    content
                        .frame(minWidth: 0, maxWidth: .infinity, minHeight: 0, maxHeight: .infinity)
                        .layoutPriority(1)
                        .contextMenu {
                            if let photoInfoURL {
                                Button("Open in Photos.app") { PhotosAppIntegration.reveal(photoInfoURL) }
                                Divider()
                            }
                            if googleLensEnabled {
                                Button("Search with Google Lens") {
                                    let pixels = displayedFrameID == frame.id
                                        ? image?.cgImage(forProposedRect: nil, context: nil, hints: nil) : nil
                                    Task { await GoogleLensSearch.search(frame: frame, preferredImage: pixels, in: session) }
                                }
                            }
                        }

                    if inspectorExpanded, photoInfoURL != nil {
                        Divider()
                            .frame(maxHeight: .infinity)
                        ScrollView(.vertical) {
                            VStack(spacing: 0) {
                                PhotoMetadataInspectorView(metadata: photoMetadata,
                                    isLoading: metadataLoading,
                                    errorMessage: metadataError,
                                    tags: metadataFrameID == frame.id ? orderedTopTags : metadataTags,
                                    rating: metadataFrameID == frame.id ? frame.tagRating?.rawValue.capitalized : metadataRating,
                                    scrollsDetails: false)
                                if animation == nil {
                                    Divider()
                                    evidenceControls
                                }
                            }
                            .frame(width: 350, alignment: .topLeading)
                        }
                        .frame(width: 350)
                        .frame(minHeight: 0, maxHeight: .infinity)
                        .clipped()
                        .background(Color(nsColor: .controlBackgroundColor).opacity(0.45))
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            .padding(24)
        }
        .task(id: "\(metadataTaskID)|\(session.thumbnailGeneration)|\(downloadFromICloud)") {
            await load(frame)
            await preloadNeighbours(of: frame)
        }
        .task(id: metadataTaskID) { await loadPhotoMetadata(for: frame) }
        .task(id:"\(metadataTaskID)|\(evidenceInput?.id.uuidString ?? "")|\(evidenceRetry)|\(safireEnabled)|\(animation == nil)") {
            await safire.load(url:photoInfoURL,enabled:safireEnabled && animation==nil,input:evidenceInput)
        }
        .task(id:"\(metadataTaskID)|\(evidenceInput?.id.uuidString ?? "")|\(evidenceRetry)|\(truforEnabled)|\(animation == nil)") {
            await trufor.load(url:photoInfoURL,enabled:truforEnabled && animation==nil,mode:truforDisplay,input:evidenceInput)
        }
        .task(id:truforDisplay) { await trufor.display(truforDisplay) }
        .onChange(of:truforEnabled) { _, enabled in if enabled { safireEnabled=false } }
        .onChange(of:safireEnabled) { _, enabled in if enabled { truforEnabled=false } }
        .task(id: "\(reading):\(loadID)") {
            showReadingStatus = false
            guard reading else { return }
            do { try await Task.sleep(for: .milliseconds(330)) }
            catch { return }
            guard !Task.isCancelled, reading else { return }
            showReadingStatus = true
        }
        // While this overlay is up it *owns* the navigation keys, rather than hoping they
        // reach whoever holds the keyboard. Using a zoom button moves focus into this
        // window, and a focused SwiftUI `Button` does not pass the arrow keys up to an
        // ancestor's `onKeyPress` — so setting a zoom and then stepping through frames,
        // which is the whole point, silently did nothing. A local monitor is the one
        // mechanism that does not depend on focus; it is removed when the overlay goes.
        .onAppear {
            Diag.trace(.preview, .focus, message: "overlay onAppear frame=\(frame.id) | \(Diag.responderDescription())")
            installKeyMonitor()
        }
        .onDisappear {
            Diag.trace(.preview, .focus, message: "overlay onDisappear | \(Diag.responderDescription())")
            removeKeyMonitor()
        }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didResignActiveNotification)) { _ in stopArrows() }
        .onReceive(session.previewFrameUpdates) { frame in
            if frame == nil, session.previewFrame == nil { stopArrows() }
        }
        // The overlay never takes keyboard focus. Navigation uses its monitor;
        // Space/Escape use the window handler that survives overlay teardown.
    }

    // MARK: - Keys

    /// Intercepts the navigation keys while the preview is up.
    ///
    /// Returns `nil` for the keys it handles, which consumes them before the responder
    /// chain sees them — so the grid's own handlers cannot act on the same press.
    private func installKeyMonitor() {
        Diag.trace(.preview, message: "installKeyMonitor alreadyInstalled=\(keyMonitor != nil)")
        guard keyMonitor == nil else { return }
        keyMonitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown, .keyUp],
                handler: Diag.tracedKeyHandler("preview.monitor") { event in
                // `keyCode` is taken out here, before the main-actor hop: an `NSEvent` is
                // not Sendable, so carrying it into the isolated closure does not compile
                // under Swift 6 — and the number is all the closure needs.
                let code = event.keyCode
                let isUp = event.type == .keyUp
                let isRepeat = event.isARepeat
                return MainActor.assumeIsolated {
                    // Space/Escape belong to the persistent window handler, including
                    // their releases after this overlay has disappeared.
                    guard code != 49, code != 53 else { return false }
                    if isUp {
                        guard code == 123 || code == 124 else { return false }
                        arrowUp(code == 123 ? -1 : 1)
                        return session.previewFrame != nil
                    }
                    guard case .consume(let key) = PreviewKeyRouter.outcome(
                        forCode: code,
                        previewFrameIsUp: session.previewFrame != nil,
                        hasTextFocus: event.window?.firstResponder is NSTextView,
                        modifiers: event.modifierFlags) else { return false }

                    switch key {
                    case .up: session.focusGroupEdge(forward: false)
                    case .down: session.focusGroupEdge(forward: true)
                    case .left: arrowDown(-1, isRepeat: isRepeat)
                    case .right: arrowDown(1, isRepeat: isRepeat)
                    case .space, .escape: return false
                    }
                    return true
                }
            })
    }

    private func arrowDown(_ direction: Int, isRepeat: Bool) {
        let loading = reading || completedReadFrameID != session.previewFrame?.id
        if let step = arrows.down(direction, isRepeat: isRepeat, loading: loading) {
            session.focusNeighbour(step)
        }
        if !isRepeat {
            arrowRepeater.start(direction) { arrowDown($0, isRepeat: true) }
        }
    }

    private func arrowUp(_ direction: Int) {
        arrows.up(direction)
        if arrows.held == nil { arrowRepeater.stop() }
    }

    private func stopArrows() {
        arrows.reset()
        arrowRepeater.stop()
    }

    private func removeKeyMonitor() {
        stopArrows()
        loadID = UUID()
        if let keyMonitor { NSEvent.removeMonitor(keyMonitor) }
        keyMonitor = nil
    }

    // MARK: - Chrome

    /// The saved label list is normally already score-sorted. Re-sort it from the
    /// stored probabilities as well, so older results and restored sessions show
    /// the strongest traits first.
    private var orderedTopTags: [String] {
        let tags = frame.topTags ?? []
        guard let vocabulary = session.tagVocabulary,
              frame.tagVocabularyFingerprint == vocabulary.cacheFingerprint,
              let probabilities = frame.tagProbabilities.flatMap(TagProbabilityCodec.decode) else {
            return tags
        }

        func score(for tag: String) -> Float? {
            guard let index = vocabulary.index(for: tag), probabilities.indices.contains(index) else {
                return nil
            }
            return probabilities[index]
        }

        return tags.enumerated().sorted { left, right in
            let leftScore = score(for: left.element)
            let rightScore = score(for: right.element)
            if let leftScore, let rightScore, leftScore != rightScore {
                return leftScore > rightScore
            }
            if leftScore != nil, rightScore == nil { return true }
            if leftScore == nil, rightScore != nil { return false }
            return left.offset < right.offset
        }.map(\.element)
    }

    private var header: some View {
        HStack(spacing: 10) {
            Text(timecode(frame.timeSeconds))
                .font(.system(size: 12, design: .monospaced))
            Text(scoreText)
                .font(.system(size: 12, design: .monospaced))
                .foregroundStyle(.secondary)
            if let note {
                Text(note)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if frame.tagRating != nil || !orderedTopTags.isEmpty {
                traitLine
            }
            Spacer(minLength: 6)
            Button { session.openUncover(at: frame.timeSeconds) } label: {
                // Marks the *workspace*, not a specific algorithm: it says the
                // server can do something here. `Reconstruct` marks per
                // algorithm, so the two together say which is which.
                Text("Uncover Waifu")
                    .remoteMark(RemoteClient.shared.offersAnyRemoteUncover)
            }
                .disabled(displayedFrameID != frame.id || reading)
            Text(showReadingStatus ? "preview · loading original" : (note == nil ? "original frame" : "preview"))
                .font(.caption)
                .foregroundStyle(.tertiary)
            Divider().frame(height: 16)
            Button { zoom = FrameZoomGeometry.clampedRatio(zoom / 1.5, fitScale: currentFitScale) } label: { Image(systemName: "minus.magnifyingglass") }
            Text(String(format: "%.0f%%", displayedImageScale * 100))
                .font(.system(size: 11, design: .monospaced))
                .frame(width: 46)
            Button { zoom = FrameZoomGeometry.clampedRatio(zoom * 1.5, fitScale: currentFitScale) } label: { Image(systemName: "plus.magnifyingglass") }
            Button("Close") { session.previewFrame = nil }
        }
        .buttonStyle(.borderless)
        .foregroundStyle(.primary)
        .padding(.vertical, 8)
    }

    private var currentFitScale: CGFloat {
        displayedImageScale / max(zoom, .leastNonzeroMagnitude)
    }

    private var currentTruForEvidence: Bool {
        trufor.photoURL == photoInfoURL && trufor.inputID == evidenceInput?.id && evidenceInput != nil
    }

    private var currentSafireEvidence: Bool {
        safire.photoURL == photoInfoURL && safire.inputID == evidenceInput?.id && evidenceInput != nil
    }

    private var evidenceControls: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 0) {
                modelButton("Off", selected: !truforEnabled && !safireEnabled) {
                    truforEnabled = false; safireEnabled = false
                }
                if WaifuDistribution.isNonCommercialOnly {
                    modelButton("TruFor", selected: truforEnabled, remote: !inferencePool.truforClients.isEmpty) {
                        safireEnabled = false; truforEnabled = true; showOriginalEvidence = false
                    }
                }
                modelButton("SAFIRE", selected: safireEnabled, remote: !inferencePool.safireClients.isEmpty) {
                    truforEnabled = false; safireEnabled = true; showOriginalEvidence = false
                }
            }
            .accessibilityLabel("Photo analysis")
            if truforEnabled || safireEnabled {
                evidenceStatus(truforEnabled ? (currentTruForEvidence ? trufor.status : "Loading photo…") :
                    (currentSafireEvidence ? safire.status : "Loading photo…"),
                    model: truforEnabled ? "TruFor" : "SAFIRE",
                    loading: truforEnabled ? trufor.loading : safire.loading)
                LazyVGrid(columns: [GridItem(.fixed(155)), GridItem(.fixed(155))], spacing: 10) {
                    EvidenceThumbnail(image: evidenceInput?.photoURL == photoInfoURL ? evidenceInput?.thumbnail : nil,
                        label: "Original", detail: "Show the source image without analysis", selected: showOriginalEvidence) {
                        showOriginalEvidence = true
                    }
                    if truforEnabled {
                        ForEach(TruForDisplay.allCases) { mode in
                            EvidenceThumbnail(image: currentTruForEvidence ? trufor.thumbnails[mode] : nil,
                                label: mode.previewLabel, detail: mode.previewDetail,
                                selected: !showOriginalEvidence && truforDisplay == mode) {
                                truforDisplay = mode; showOriginalEvidence = false
                            }
                        }
                    } else {
                        EvidenceThumbnail(image: currentSafireEvidence ? safire.overlayThumbnail : nil,
                            label: "Overlay", detail: "SAFIRE regions highlighted over the photo",
                            selected: !showOriginalEvidence && !showSafireMask) {
                            showSafireMask = false; showOriginalEvidence = false
                        }
                        EvidenceThumbnail(image: currentSafireEvidence ? safire.maskThumbnail : nil,
                            label: "Mask", detail: "SAFIRE region mask",
                            selected: !showOriginalEvidence && showSafireMask) {
                            showSafireMask = true; showOriginalEvidence = false
                        }
                    }
                }
            }
        }
        .padding(14)
        .frame(width: 350, alignment: .leading)
        .clipped()
    }

    private func modelButton(_ title: String, selected: Bool, remote: Bool = false,
                             action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title).remoteMark(remote)
                .font(.caption).lineLimit(1).truncationMode(.tail)
                .frame(maxWidth: .infinity, minHeight: 26)
                .background(selected ? Color.accentColor.opacity(0.2) : Color.secondary.opacity(0.06))
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(selected ? .isSelected : [])
    }

    private func evidenceStatus(_ status: String, model: String, loading: Bool) -> some View {
        let label = status.replacingOccurrences(of: model + " · ", with: "")
            .replacingOccurrences(of: model + " ", with: "")
            .replacingOccurrences(of: "detection score ", with: "")
            .components(separatedBy: " · ").first ?? ""
        let preview = evidenceInput.map { !$0.isOriginal } == true
        let detail = status + (preview ? " · Using the available preview; results may differ from the original." : "")
        return HStack(spacing: 5) {
            if loading || (evidenceInput == nil && reading) { ProgressView().controlSize(.mini) }
            if !label.isEmpty {
                Text("- " + label + (preview ? " · preview" : ""))
                    .font(.caption).foregroundStyle(.secondary).lineLimit(1).truncationMode(.tail)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            if !loading, status.contains("unavailable") {
                Button("Retry") { evidenceRetry = UUID() }.buttonStyle(.borderless)
                    .lineLimit(1).truncationMode(.tail).frame(width: 38)
            }
        }
        .frame(width: 322, height: 20, alignment: .leading)
        .clipped()
        .help(detail)
    }

    private var traitLine: some View {
        HStack(spacing: 8) {
            Text("Traits")
                .font(.system(size: 14))
                .foregroundStyle(.secondary)
            if let rating = frame.tagRating {
                Text(rating.rawValue.capitalized)
                    .font(.system(size: 14))
                    .foregroundStyle(rating == .explicit ? Color.red : rating == .questionable ? Color.orange : Color.secondary)
            }
            ScrollView(.horizontal) {
                HStack(spacing: 5) {
                    ForEach(Array(orderedTopTags.prefix(8)), id: \.self) { tag in
                        Text(tag)
                            .font(.system(size: 14))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color.secondary.opacity(0.15))
                            .clipShape(Capsule())
                    }
                    if orderedTopTags.count > 8 {
                        Text("+\(orderedTopTags.count - 8)")
                            .font(.system(size: 14))
                            .foregroundStyle(.tertiary)
                    }
                }
            }
            .scrollIndicators(.hidden)
            .layoutPriority(-1)
        }
    }

    @ViewBuilder
    private var content: some View {
        ZStack(alignment: .topLeading) {
            if let animation {
                AnimatedPhotoPlayer(animation: animation, isActive: !reading, maximumPixelSize: 4096) { frameImage, index in
                    ZoomableFrameImage(image: frameImage, frameID: FrameID(sampleIndex: index),
                                       zoomRatio: $zoom, center: $zoomCenter, displayedScale: $displayedImageScale)
                }
                .id(animation.identity)
            } else if let image {
                ZoomableFrameImage(image: previewImage(image),
                    frameID: displayedFrameID ?? frame.id,
                                   zoomRatio: $zoom, center: $zoomCenter, displayedScale: $displayedImageScale)
            } else {
                VStack(spacing: 8) {
                    if showReadingStatus { ProgressView() }
                    Text(note ?? (showReadingStatus ? "Loading the original frame…" : ""))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            if showReadingStatus {
                Text("reading…")
                    .font(.caption)
                    .foregroundStyle(.white)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 5)
                    .background(Color.black.opacity(0.6), in: Capsule())
                    .padding(10)
                    .allowsHitTesting(false)
            }
        }
    }

    private func previewImage(_ original:NSImage)->NSImage {
        guard !showOriginalEvidence, photoInfoURL != nil,displayedFrameID==frame.id else { return original }
        if truforEnabled,currentTruForEvidence { return trufor.rendered ?? original }
        if safireEnabled,currentSafireEvidence { return (showSafireMask ? safire.mask:safire.overlay) ?? original }
        return original
    }

    // MARK: - Loading and zoom

    /// Keep full source pixels for zooming, with one cancellable read per target.
    /// The previous image stays visible until this frame's original is available.
    private func load(_ target: AnalyzedFrame) async {
        guard !Task.isCancelled, session.previewFrame?.id == target.id,
              let source = session.video?.url else { return }
        let request = UUID()
        loadID = request
        note = nil
        let keepingCurrent = displayedFrameID == target.id && displayedSourceURL == source
        evidenceInput = nil
        reading = true
        showReadingStatus = false
        animation = nil
        defer {
            if loadID == request {
                reading = false
                showReadingStatus = false
                if !Task.isCancelled { completedReadFrameID = target.id }
            }
        }

        // A load can restart for the same photo when thumbnail caches refresh.
        // Only replace its displayed pixels with a sharper rendition. A new
        // target must still get its own thumbnail, regardless of resolution.
        let identity = session.thumbnailGeneration.uuidString + ":" + target.thumbnailKey
        let current = keepingCurrent ? image.flatMap(PhotosMedia.renditionPixels) : nil
        if let pixels = GridThumbnailLoader.shared.cachedImage(identity: identity) {
            if PreviewPhotoSource.isSharper(pixels, than: current) {
                image = NSImage(cgImage: pixels, size: NSSize(width: pixels.width, height: pixels.height))
                displayedFrameID = target.id
                displayedSourceURL = source
                displayedIsOriginal = false
            }
        } else if let data = await session.thumbnailData(for: target),
                  let thumbnail = NSImage(data: data), let pixels = PhotosMedia.renditionPixels(thumbnail) {
            guard !Task.isCancelled, loadID == request, session.previewFrame?.id == target.id else { return }
            if PreviewPhotoSource.isSharper(pixels, than: current) {
                image = NSImage(cgImage: pixels, size: NSSize(width: pixels.width, height: pixels.height))
                displayedFrameID = target.id
                displayedSourceURL = source
                displayedIsOriginal = false
            }
        }
        do {
            if let photoURL = PhotosMedia.shared.photoItemURL(for: source, seconds: target.timeSeconds) {
                let cached = displayedFrameID == target.id && displayedSourceURL == source
                    ? image.flatMap(PhotosMedia.renditionPixels) : nil
                let preview = try await PreviewPhotoSource.load(
                    original: { try await PhotosMedia.shared.previewContent(photoURL) },
                    rendition: { try await PhotosMedia.shared.inspectionPreview(photoURL, onPreview: { _ in }) },
                    cached: cached, cachedIsOriginal: displayedIsOriginal && displayedFrameID == target.id && displayedSourceURL == source)
                guard !Task.isCancelled, loadID == request,
                      session.previewFrame?.id == target.id, session.video?.url == source else { return }
                switch preview.content {
                case .still(let pixels):
                    let native = NSImage(cgImage: pixels, size: NSSize(width: pixels.width, height: pixels.height))
                    native.cacheMode = .never
                    image = native
                    displayedIsOriginal = preview.isOriginal
                    animation = nil
                    evidenceInput = PreviewEvidenceInput(photoURL: photoURL, image: pixels, isOriginal: preview.isOriginal)
                    if !preview.isOriginal { note = "Preview image" }
                case .animated(let sequence):
                    animation = sequence
                }
                displayedFrameID = target.id
                displayedSourceURL = source
                if let step = arrows.loaded() { session.focusNeighbour(step) }
                return
            }
            let original = try await PhotosMedia.shared.image(source, seconds: target.timeSeconds)
            guard !Task.isCancelled, loadID == request,
                  session.previewFrame?.id == target.id, session.video?.url == source else { return }
            let native = NSImage(cgImage: original, size: NSSize(width: original.width, height: original.height))
            native.cacheMode = .never
            image = native
            displayedIsOriginal = true
            displayedFrameID = target.id
            displayedSourceURL = source
            if let step = arrows.loaded() { session.focusNeighbour(step) }
        } catch {
            guard !Task.isCancelled, loadID == request,
                  session.previewFrame?.id == target.id, session.video?.url == source else { return }
            note = "Original image unavailable."
        }
    }

    private func preloadNeighbours(of target: AnalyzedFrame) async {
        guard !Task.isCancelled, downloadFromICloud, session.previewFrame?.id == target.id,
              let source = session.video?.url, PhotosMedia.isPhotos(source) else { return }
        let neighbours = session.previewPreloadFrames(after: target.id)
        for neighbour in neighbours {
            guard !Task.isCancelled, downloadFromICloud, session.previewFrame?.id == target.id,
                  session.video?.url == source else { return }
            guard let url = PhotosMedia.shared.photoItemURL(for: source, seconds: neighbour.timeSeconds) else { continue }
            do { try await PhotosMedia.shared.preloadPreview(url) }
            catch is CancellationError { return }
            catch { /* Unavailable neighbours do not stop the idle preload window. */ }
        }
    }

    private var photoInfoURL: URL? {
        guard let source = session.video?.url else { return nil }
        return PhotosMedia.shared.photoItemURL(for: source, seconds: frame.timeSeconds)
    }

    private var metadataTaskID: String {
        "\(frame.id.raw):\(session.video?.url.absoluteString ?? ""):\(photoInfoURL?.absoluteString ?? "")"
    }

    /// Metadata fills the visible sidebar independently of original-image decoding.
    private func loadPhotoMetadata(for target: AnalyzedFrame) async {
        guard let source = session.video?.url,
              let photoURL = PhotosMedia.shared.photoItemURL(for: source, seconds: target.timeSeconds) else { return }
        metadataLoading = false
        metadataError = nil
        let loadingIndicator = Task { @MainActor in
            do { try await Task.sleep(for: .milliseconds(330)) }
            catch { return }
            guard !Task.isCancelled, session.previewFrame?.id == target.id,
                  session.video?.url == source else { return }
            metadataLoading = true
        }
        defer {
            loadingIndicator.cancel()
            if !Task.isCancelled, session.previewFrame?.id == target.id {
                metadataLoading = false
            }
        }
        do {
            let result = try await PhotoMetadataService.shared.load(for: photoURL)
            guard !Task.isCancelled, session.previewFrame?.id == target.id,
                  session.video?.url == source,
                  PhotosMedia.shared.photoItemURL(for: source, seconds: target.timeSeconds) == photoURL else { return }
            photoMetadata = result
            metadataFrameID = target.id
            metadataTags = orderedTopTags
            metadataRating = frame.tagRating?.rawValue.capitalized
        } catch is CancellationError {
            return
        } catch {
            guard !Task.isCancelled, session.previewFrame?.id == target.id else { return }
            photoMetadata = nil
            metadataFrameID = nil
            metadataTags = []
            metadataRating = nil
            metadataError = "Photo details unavailable."
        }
    }

    /// `art` or `art/face`, two decimals — the same reading the grid's overlay gives.
    private var scoreText: String {
        if let definition = frame.faceDefinitionScore {
            return "Face definition " + String(format: "%.2f", definition)
        }
        let art = String(format: "%.2f", frame.artScore)
        guard let face = frame.faceScore, frame.hasSignificantFace else { return art }
        return art + "/" + String(format: "%.2f", face)
    }
}
