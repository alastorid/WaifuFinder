import SwiftUI

/// The drop zone's visuals and buttons: prompt, highlight and explicit openers.
/// The drop itself is handled by the one window-level registration in `RootView`,
/// which routes to `CrystalOrbImport` — this view only reflects whether a drag is
/// currently over the window.
struct DropZone: View {
    /// The session to receive the dropped file.
    @ObservedObject var session: WaifuSession

    /// Whether the drop zone should be visually prominent (no video loaded).
    let isPrimary: Bool

    /// Whether a drag currently hovers the window; drives the highlight only.
    var isTargeted: Bool = false
    var openPhotos: (() -> Void)? = nil

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
        .animation(.easeInOut(duration: 0.15), value: isTargeted)
    }

    /// Full-window drop zone shown when no video is loaded.
    private var primaryDropZone: some View {
        VStack(spacing: 16) {
            Image(systemName: "arrow.down.doc.fill")
                .font(.system(size: 48))
                .foregroundStyle(isTargeted ? Color.accentColor : .secondary)
                .symbolEffect(.bounce, value: isTargeted)

            Text("Drop videos here")
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

    /// Opens a file panel for selecting a video when the user clicks the drop zone.
    private func selectFileViaPanel() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.data]
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.title = "Select videos to analyze"
        panel.message = "Choose videos to analyze together"
        if panel.runModal() == .OK {
            session.loadVideos(at: panel.urls)
        }
    }
}
