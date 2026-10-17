import SwiftUI
import UniformTypeIdentifiers

/// Accepts local files and web links. Explicit buttons keep file browsing and URL
/// entry independent of the drop target. Finder uses public.file-url providers;
/// browsers use public.url, so both representations are requested.
struct DropZone: View {
    /// The session to receive the dropped file.
    @ObservedObject var session: WaifuSession

    /// Whether the drop zone should be visually prominent (no video loaded).
    let isPrimary: Bool
    var openPhotos: (() -> Void)? = nil

    @State private var isTargeted = false
    @State private var showURL = false

    var body: some View {
        ZStack {
            if isPrimary {
                primaryDropZone
            } else {
                secondaryDropZone
            }
        }
        .sheet(isPresented: $showURL) { MediaURLSheet(session: session) }
        .onDrop(of: [.fileURL, .url], isTargeted: $isTargeted) { providers in
            acceptDrop(providers)
        }
        .animation(.easeInOut(duration: 0.15), value: isTargeted)
    }

    /// Full-window drop zone shown when no video is loaded.
    private var primaryDropZone: some View {
        VStack(spacing: 16) {
            Image(systemName: "arrow.down.doc.fill")
                .font(.system(size: 48))
                .foregroundStyle(isTargeted ? Color.accentColor : .secondary)
                .symbolEffect(.bounce, value: isTargeted)

            Text("Drop a video here")
                .font(.title2)
                .foregroundStyle(.primary)

            Button("Browse files…") { selectFileViaPanel() }
                .buttonStyle(.link)

            Button("Or load from URL") { showURL = true }
                .buttonStyle(.link)
                .font(.callout)

            Picker("Analysis", selection: $session.settings.analysisMode) {
                ForEach(AnalysisMode.allCases) { mode in
                    Text(mode.label)
                        .accessibilityLabel(mode.label)
                        .tag(mode)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .controlSize(.small)
            .frame(width: 190)
            .help("Keyframe analyses only the frames the encoder can decode on its own, so its progress loops. Quick analyses one frame a second. Full analyses every frame.")

            Button("Or open Photos") { openPhotos?() }
                .buttonStyle(.link)
                .font(.callout)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .strokeBorder(
                    isTargeted ? Color.accentColor : Color.secondary.opacity(0.3),
                    style: StrokeStyle(lineWidth: 2, dash: [8, 4])
                )
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(isTargeted ? Color.accentColor.opacity(0.1) : Color.clear)
                )
        )
        .padding(32)
        .contentShape(Rectangle())
    }

    /// Compact drop target in the toolbar when a video is already loaded.
    private var secondaryDropZone: some View {
        HStack(spacing: 8) {
            Image(systemName: "plus.rectangle.on.rectangle")
                .font(.system(size: 14, weight: .medium))
            Button("Replace Video") { selectFileViaPanel() }
                .buttonStyle(.plain)
                .font(.callout.weight(.medium))
            Button("URL") { showURL = true }.buttonStyle(.link)
            Button("Photos") { openPhotos?() }.buttonStyle(.link)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(
            Capsule()
                .strokeBorder(
                    isTargeted ? Color.accentColor : Color.secondary.opacity(0.4),
                    lineWidth: isTargeted ? 2 : 1
                )
                .background(
                    Capsule()
                        .fill(isTargeted ? Color.accentColor.opacity(0.15) : Color.clear)
                )
        )
        .foregroundStyle(isTargeted ? Color.accentColor : .secondary)
        .scaleEffect(isTargeted ? 1.05 : 1.0)
        // The label says "Replace Video", so it has to be clickable: a target that
        // only responds to a drag promises an action it does not have. Same panel
        // as the empty-state drop zone, which is the only way to open a file when
        // the window is already full of one.
        .contentShape(Capsule())

    }

    /// Routes a drop to the session.
    ///
    /// Returns `true` when an item provider *can* supply a file URL, which is what
    /// tells AppKit to animate acceptance; the URL itself arrives asynchronously,
    /// because a file dragged from a remote volume may still be being copied. The
    /// decision is therefore returned immediately and the load happens when the URL
    /// lands, if `VideoDrop` accepts it.
    private func acceptDrop(_ providers: [NSItemProvider]) -> Bool {
        let session = session
        for provider in providers
        where provider.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) || provider.hasItemConformingToTypeIdentifier(UTType.url.identifier) {
            let type = provider.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) ? UTType.fileURL.identifier : UTType.url.identifier
            provider.loadItem(forTypeIdentifier: type, options: nil) { item, _ in
                guard let url = DropZone.url(from: item), VideoDrop.accepts(url) else { return }
                Task { @MainActor in session.loadVideo(at: url) }
            }
            return true
        }
        return false
    }

    /// Reads a file URL out of whatever `NSItemProvider` handed back. A local drop
    /// delivers a `URL`; a cross-process one delivers the URL's serialised `Data`.
    ///
    /// `nonisolated` because it is called from `loadItem`'s completion handler,
    /// which runs off the main actor, and it touches no view state.
    nonisolated private static func url(from item: NSSecureCoding?) -> URL? {
        if let url = item as? URL { return url }
        if let text = item as? String { return URL(string: text) }
        if let data = item as? Data { return URL(dataRepresentation: data, relativeTo: nil) }
        return nil
    }

    /// Opens a file panel for selecting a video when the user clicks the drop zone.
    private func selectFileViaPanel() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.data]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.title = "Select a video to analyze"
        panel.message = "Choose a video file"
        if panel.runModal() == .OK, let url = panel.url {
            session.loadVideo(at: url)
        }
    }
}
