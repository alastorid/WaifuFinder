import AVKit
import CoreGraphics
import SwiftUI

/// The preview: the video itself, and the controls that put it on a frame.
///
/// Waifu is about choosing *moments*, so the preview's job is to show the frame
/// the grid is pointing at — not to be a general-purpose player. Selecting a frame
/// seeks here; the play button continues from that frame. That is the whole
/// interaction, and it is why the surface has no scrubber of its own: the window's
/// timeline is the scrubber, and it drives the same playhead.
struct VideoPreviewPanel: View {
    @ObservedObject var session: WaifuSession
    @ObservedObject var preview: PreviewController
    var videoShortcutEnabled: Bool
    @StateObject private var keyboard = VideoPreviewKeyboardState()
    @State private var selectedThumbnail: NSImage?
    @State private var photosPlayerSource: URL?
    @State private var videoPlayerOffset: Double?

    /// The frame the preview is currently parked on, if any.
    private var focusedFrame: AnalyzedFrame? {
        guard let id = session.focusedFrameID else { return nil }
        return session.frame(id: id)
    }

    private var isPhotosMode: Bool {
        session.video.map { PhotosMedia.isPhotos($0.url) } == true
    }

    private var focusedPhotoURL: URL? {
        guard let source = session.video?.url, let frame = focusedFrame else { return nil }
        return PhotosMedia.shared.photoItemURL(for: source, seconds: frame.timeSeconds)
    }

    /// Still photos have no playback transport. Resolve through the album at the
    /// same timestamp used by `syncPlayer`, so mixed photo/video albums keep the
    /// controls only while a video item is focused.
    private var isPhotosStill: Bool {
        guard let source = session.video?.url, PhotosMedia.isPhotos(source) else { return false }
        return PhotosMedia.shared.photoItemURL(for: source, seconds: focusedFrame?.timeSeconds ?? 0) != nil
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            surface
            if !isPhotosStill {
                Divider()
                controls
            }
        }
        .background(Color(nsColor: .controlBackgroundColor))
        .background {
            VideoPreviewKeyboard(state: keyboard, session: session, preview: preview,
                                 enabled: videoShortcutEnabled && !isPhotosMode)
        }
        .task(id: focusedPhotoURL) {
            guard let frame = focusedFrame, let source = session.video?.url,
                  PhotosMedia.isPhotos(source) else { return }
            let urls = session.previewPreloadFrames(after: frame.id).prefix(2).compactMap {
                PhotosMedia.shared.photoItemURL(for: source, seconds: $0.timeSeconds)
            }
            for url in urls {
                guard !Task.isCancelled else { return }
                await PhotoMetadataService.shared.prefetchFilename(for: url)
            }
        }
        .task(id: focusedFrame?.id) {
            guard let frame = focusedFrame,
                  let data = await session.thumbnailData(for: frame) else {
                selectedThumbnail = nil
                return
            }
            let identity = session.thumbnailGeneration.uuidString + ":" + frame.thumbnailKey
            let image: CGImage?
            if let cached = GridThumbnailLoader.shared.cachedImage(identity: identity) {
                image = cached
            } else {
                selectedThumbnail = nil
                image = await GridThumbnailLoader.shared.image(data: data, identity: identity)
            }
            guard !Task.isCancelled, let image else { return }
            selectedThumbnail = NSImage(cgImage: image, size: NSSize(width: image.width, height: image.height))
        }
        .onAppear {
            syncPlayer()
            if let frame = focusedFrame { preview.seek(to: frame.timeSeconds) }
        }
        // A new video replaces the item; re-running the analysis does not (the URL
        // is unchanged), so the playhead is not thrown away by "Analyze".
        .onChange(of: session.video?.url) { _, _ in
            photosPlayerSource = nil; videoPlayerOffset = nil
            syncPlayer()
        }
        // The one rule that makes this a *frame* preview: whatever the grid focuses,
        // the video shows.
        .onChange(of: session.focusedFrameID) { _, id in
            guard session.previewFrame == nil else { return }
            guard let id, let frame = session.frame(id: id) else { return }
            if session.video.map({ PhotosMedia.isPhotos($0.url) || VideoCollection.get($0.url) != nil }) == true { syncPlayer() }
            else { preview.seek(to: frame.timeSeconds) }
        }
        .onChange(of: session.previewFrame == nil) { _, closed in
            if closed, focusedFrame != nil { syncPlayer(); if let frame = focusedFrame { preview.seek(to: frame.timeSeconds) } }
            if !closed { preview.pause() }
        }
    }

    // MARK: - Sections

    private var header: some View {
        HStack(spacing: 6) {
            Image(systemName: "film")
                .foregroundStyle(.secondary)
            Text("Preview")
                .font(.headline)
            if let url = focusedPhotoURL,
               let photoFilename = PhotoMetadataService.shared.filename(for: url) {
                Text(photoFilename).font(.caption).foregroundStyle(.secondary)
                    .lineLimit(1).truncationMode(.middle).help(photoFilename)
                    .contentTransition(.identity)
            }
            Spacer()
            if !isPhotosMode {
                if let frame = focusedFrame {
                    Text(timecode(frame.timeSeconds))
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(.secondary)
                } else {
                    Text("no frame selected")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
    }

    @ViewBuilder
    private var surface: some View {
        if session.video.map({ PhotosMedia.isPhotos($0.url) }) == true, preview.player == nil, let selectedThumbnail {
            Image(nsImage: selectedThumbnail).resizable().aspectRatio(contentMode: .fit)
                .frame(maxWidth: .infinity).background(Color.black)
        } else if preview.player != nil {
            // `.clipped()` is not cosmetic. The player is an `NSView`-backed
            // `AVPlayerView`, and AppKit does not clip a view's subviews to its
            // bounds by default — so anything the player's own controls draw beyond
            // the 16:9 box (a slider track, an edge of the control bar) can appear
            // anywhere in the window as a stray line. The preview is a fixed box;
            // nothing it contains may draw outside it.
            Group {
                if isPhotosMode {
                    VideoPlayer(player: preview.player)
                } else {
                    NativeVideoPreview(player: preview.player, keyboard: keyboard)
                }
            }
                .aspectRatio(16.0 / 9.0, contentMode: .fit)
                .frame(maxWidth: .infinity)
                .background(Color.black)
                .clipped()
                .overlay {
                    if preview.isSeeking, let selectedThumbnail {
                        Image(nsImage: selectedThumbnail)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                            .background(Color.black)
                            .allowsHitTesting(false)
                    }
                }
        } else {
            Rectangle()
                .fill(Color.black)
                .aspectRatio(16.0 / 9.0, contentMode: .fit)
                .frame(maxWidth: .infinity)
                .overlay {
                    VStack(spacing: 6) {
                        Image(systemName: "film.stack")
                            .font(.system(size: 22))
                            .foregroundStyle(.white.opacity(0.5))
                        Text("No video loaded")
                            .font(.caption)
                            .foregroundStyle(.white.opacity(0.6))
                    }
                }
        }
    }

    private var controls: some View {
        HStack(spacing: 10) {
            Button {
                preview.togglePlayPause()
            } label: {
                Image(systemName: preview.isPlaying ? "pause.fill" : "play.fill")
                    .frame(width: 16)
            }
            .buttonStyle(.borderless)
            .disabled(preview.player == nil)

            if !isPhotosMode {
                Text("\(timecode(preview.currentTime)) / \(timecode(preview.duration))")
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }

            Spacer()

            Button {
                if let frame = focusedFrame { preview.seek(to: frame.timeSeconds) }
            } label: {
                Image(systemName: "scope")
            }
            .buttonStyle(.borderless)
            .disabled(focusedFrame == nil)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
    }

    // MARK: - Helpers

    private func syncPlayer() {
        if let url = session.video?.url {
            if let collection = VideoCollection.get(url) {
                let seconds = focusedFrame?.timeSeconds ?? 0
                let member = collection.member(at: seconds)
                if photosPlayerSource != member.video.url || videoPlayerOffset != member.offset {
                    photosPlayerSource = member.video.url
                    videoPlayerOffset = member.offset
                    preview.load(asset: VideoAssetFactory.asset(for: member.video.url), offset: member.offset,
                                 mediaDuration: member.video.durationSeconds, timelineDuration: collection.info.durationSeconds)
                }
                preview.seek(to: seconds)
                return
            }
            if PhotosMedia.isPhotos(url) {
                let (source, time) = PhotosMedia.shared.resolve(url, seconds: focusedFrame?.timeSeconds ?? 0)
                if PhotosMedia.shared.isStill(source) {
                    photosPlayerSource = source
                    preview.loadStill(time: focusedFrame?.timeSeconds ?? 0, duration: session.video?.durationSeconds ?? 1)
                    return
                }
                guard photosPlayerSource != source else { preview.seek(to: focusedFrame?.timeSeconds ?? time); return }
                photosPlayerSource = source
                preview.unload()
                Task {
                    do {
                        let asset = try await PhotosMedia.shared.videoAsset(source)
                        guard photosPlayerSource == source else { return }
                        let item = PhotosMedia.shared.item(source)
                        preview.load(asset: asset.value, offset: item?.offset ?? 0, mediaDuration: item?.duration,
                                     timelineDuration: session.video?.durationSeconds)
                        preview.seek(to: focusedFrame?.timeSeconds ?? time)
                    } catch { session.errorMessage = "Video preview unavailable." }
                }
                return
            }
            if let asset = preview.player?.currentItem?.asset as? AVURLAsset, asset.url == url { return }
            preview.load(url: url)
        } else {
            preview.unload()
        }
    }
}

/// Shares the native player view with the panel's keyboard monitor.
@MainActor
private final class VideoPreviewKeyboardState: ObservableObject {
    weak var playerView: AVPlayerView?
}

private struct NativeVideoPreview: NSViewRepresentable {
    let player: AVPlayer?
    let keyboard: VideoPreviewKeyboardState

    func makeNSView(context: Context) -> AVPlayerView {
        let view = AVPlayerView()
        view.player = player
        keyboard.playerView = view
        return view
    }

    func updateNSView(_ view: AVPlayerView, context: Context) {
        view.player = player
        keyboard.playerView = view
    }
}

/// A click anywhere in the preview panel selects it for the unmodified F key.
private struct VideoPreviewKeyboard: NSViewRepresentable {
    let state: VideoPreviewKeyboardState
    let session: WaifuSession
    let preview: PreviewController
    let enabled: Bool

    func makeNSView(context: Context) -> MonitorView {
        let view = MonitorView()
        updateNSView(view, context: context)
        return view
    }

    func updateNSView(_ view: MonitorView, context: Context) {
        view.state = state
        view.session = session
        view.preview = preview
        view.enabled = enabled
    }

    static func dismantleNSView(_ view: MonitorView, coordinator: ()) {
        view.removeMonitor()
    }

    final class MonitorView: NSView {
        weak var state: VideoPreviewKeyboardState?
        weak var session: WaifuSession?
        weak var preview: PreviewController?
        var enabled = false
        private var selected = false
        private var monitor: Any?

        override var acceptsFirstResponder: Bool { true }

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            removeMonitor()
            guard window != nil else { return }
            monitor = NSEvent.addLocalMonitorForEvents(matching: [.leftMouseDown, .rightMouseDown, .keyDown]) { [weak self] event in
                let consumed = MainActor.assumeIsolated {
                    guard let self, let panelWindow = self.window else { return false }
                    if event.type != .keyDown {
                        self.selected = event.window === panelWindow && self.bounds.contains(self.convert(event.locationInWindow, from: nil))
                        if self.selected, event.type == .leftMouseDown,
                           panelWindow.firstResponder is NSTextView {
                            panelWindow.makeFirstResponder(self)
                        }
                        return false
                    }
                    guard let playerView = self.state?.playerView else { return false }
                    let modifiers = event.modifierFlags.intersection([.command, .control, .option, .shift])
                    if playerView.isInFullScreenMode, event.window === playerView.window,
                       event.keyCode == 53, modifiers.isEmpty {
                        playerView.exitFullScreenMode(options: nil)
                        return true
                    }
                    guard self.selected, self.enabled, panelWindow.isKeyWindow, event.window === panelWindow,
                          !event.isARepeat, modifiers.isEmpty,
                          event.charactersIgnoringModifiers?.lowercased() == "f",
                          panelWindow.attachedSheet == nil,
                          !(panelWindow.firstResponder is NSTextView),
                          (panelWindow.firstResponder as? NSTextField)?.isEditable != true,
                          let session = self.session, !session.isPreviewInteractionBlocked,
                          session.previewFrame == nil,
                          let preview = self.preview, preview.player != nil,
                          let screen = panelWindow.screen,
                          !playerView.isInFullScreenMode else { return false }
                    if playerView.enterFullScreenMode(screen, withOptions: nil) {
                        preview.play()
                        return true
                    }
                    return false
                }
                return consumed ? nil : event
            }
        }

        func removeMonitor() {
            if let monitor { NSEvent.removeMonitor(monitor) }
            monitor = nil
            selected = false
        }
    }
}
