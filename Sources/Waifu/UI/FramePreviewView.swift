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

    @State private var image: NSImage?
    @State private var animation: PhotosAnimatedImage?
    /// Keep the previous original visible until the next source frame is ready.
    @State private var displayedFrameID: FrameID?
    @State private var note: String?
    @State private var reading = false
    @State private var showReadingStatus = false
    /// A failed read is finished too; an unreadable frame must not trap held arrows.
    @State private var completedReadFrameID: FrameID?
    /// Zoom as a multiple of the frame's fit-to-window size (1 = fit).
    @State private var zoom: CGFloat = 1
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

    init(frame: AnalyzedFrame, session: WaifuSession) {
        self.frame = frame; self.session = session
        let identity = session.thumbnailGeneration.uuidString + ":" + frame.thumbnailKey
        let cached = GridThumbnailLoader.shared.cachedImage(identity: identity)
        let initial = cached.map { NSImage(cgImage: $0, size: NSSize(width: $0.width, height: $0.height)) }
            ?? session.cachedThumbnailData(forKey: frame.thumbnailKey).flatMap(NSImage.init(data:))
        _image = State(initialValue: initial)
        _displayedFrameID = State(initialValue: initial == nil ? nil : frame.id)
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
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .layoutPriority(1)

                    if photoInfoURL != nil {
                        Divider()
                            .frame(maxHeight: .infinity)
                        PhotoMetadataInspectorView(metadata: photoMetadata,
                                                   isLoading: metadataLoading,
                                                   errorMessage: metadataError,
                                                   tags: metadataFrameID == frame.id ? orderedTopTags : metadataTags,
                                                   rating: metadataFrameID == frame.id ? frame.tagRating?.rawValue.capitalized : metadataRating)
                            .frame(maxHeight: .infinity)
                            .background(Color(nsColor: .controlBackgroundColor).opacity(0.45))
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            .padding(24)
        }
        .task(id: frame.id) { await load(frame) }
        .task(id: metadataTaskID) { await loadPhotoMetadata(for: frame) }
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
                    .remoteMark(RemoteClient.shared.offersAnyRemoteUncover,
                                serverName: RemoteClient.shared.capabilities?.serverName)
            }
                .disabled(displayedFrameID != frame.id || reading)
                .help(RemoteClient.shared.offersAnyRemoteUncover
                      ? "Open Uncover — reconstruction can run on \(RemoteClient.shared.capabilities?.serverName ?? "the server")"
                      : "Open Uncover")
            Text(showReadingStatus ? "preview · loading original" : (note == nil ? "original frame" : "preview"))
                .font(.caption)
                .foregroundStyle(.tertiary)
            Divider().frame(height: 16)
            Button { zoom = max(0.25, zoom / 1.5) } label: { Image(systemName: "minus.magnifyingglass") }
                .help("Zoom out")
            Text(String(format: "%.0f%%", zoom * 100))
                .font(.system(size: 11, design: .monospaced))
                .frame(width: 46)
            Button { zoom = min(16, zoom * 1.5) } label: { Image(systemName: "plus.magnifyingglass") }
                .help("Zoom in")
            Button("Fit") { zoom = 1; zoomCenter = CGPoint(x: 0.5, y: 0.5) }
            Button("Close") { session.previewFrame = nil }
                .help("Close the preview — Space or Escape also close it")
        }
        .buttonStyle(.borderless)
        .foregroundStyle(.primary)
        .padding(.vertical, 8)
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
                                       zoomRatio: $zoom, center: $zoomCenter)
                }
                .id(animation.identity)
            } else if let image {
                ZoomableFrameImage(image: image, frameID: displayedFrameID ?? frame.id,
                                   zoomRatio: $zoom, center: $zoomCenter)
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

    // MARK: - Loading and zoom

    /// Keep full source pixels for zooming, with one cancellable read per target.
    /// The previous image stays visible until this frame's original is available.
    private func load(_ target: AnalyzedFrame) async {
        guard !Task.isCancelled, session.previewFrame?.id == target.id,
              let source = session.video?.url else { return }
        let request = UUID()
        loadID = request
        note = nil
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

        // Paint the target thumbnail before awaiting its original, including
        // navigation to a different photo while the overlay stays mounted.
        let identity = session.thumbnailGeneration.uuidString + ":" + target.thumbnailKey
        if let pixels = GridThumbnailLoader.shared.cachedImage(identity: identity) {
            image = NSImage(cgImage: pixels, size: NSSize(width: pixels.width, height: pixels.height))
            displayedFrameID = target.id
        } else if let data = await session.thumbnailData(for: target), let thumbnail = NSImage(data: data) {
            guard !Task.isCancelled, loadID == request, session.previewFrame?.id == target.id else { return }
            image = thumbnail
            displayedFrameID = target.id
        }
        do {
            if let photoURL = PhotosMedia.shared.photoItemURL(for: source, seconds: target.timeSeconds) {
                let preview = try await PhotosMedia.shared.previewContent(photoURL)
                guard !Task.isCancelled, loadID == request,
                      session.previewFrame?.id == target.id, session.video?.url == source else { return }
                switch preview {
                case .still(let pixels):
                    let native = NSImage(cgImage: pixels, size: NSSize(width: pixels.width, height: pixels.height))
                    native.cacheMode = .never
                    image = native
                    animation = nil
                case .animated(let sequence):
                    animation = sequence
                }
                displayedFrameID = target.id
                if let step = arrows.loaded() { session.focusNeighbour(step) }
                return
            }
            let original = try await PhotosMedia.shared.image(source, seconds: target.timeSeconds)
            guard !Task.isCancelled, loadID == request,
                  session.previewFrame?.id == target.id, session.video?.url == source else { return }
            let native = NSImage(cgImage: original, size: NSSize(width: original.width, height: original.height))
            native.cacheMode = .never
            image = native
            displayedFrameID = target.id
            if let step = arrows.loaded() { session.focusNeighbour(step) }
        } catch {
            guard !Task.isCancelled, loadID == request,
                  session.previewFrame?.id == target.id, session.video?.url == source else { return }
            note = error.localizedDescription
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
            metadataError = error.localizedDescription
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
