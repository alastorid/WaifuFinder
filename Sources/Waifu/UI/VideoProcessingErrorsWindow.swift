import AppKit
import SwiftUI

/// A modeless log: closing it never cancels or pauses the analysis.
@MainActor
final class VideoProcessingErrorsWindow: ObservableObject {
    private var window: NSWindow?

    func show(session: WaifuSession) {
        if window == nil {
            let panel = NSPanel(contentRect: NSRect(x: 0, y: 0, width: 720, height: 430),
                                styleMask: [.titled, .closable, .resizable, .utilityWindow],
                                backing: .buffered, defer: false)
            panel.title = "Processing Errors"
            panel.isReleasedWhenClosed = false
            panel.hidesOnDeactivate = false
            panel.minSize = NSSize(width: 500, height: 300)
            panel.contentView = NSHostingView(rootView: VideoProcessingErrorsView(session: session, close: { [weak panel] in
                panel?.orderOut(nil)
            }))
            panel.center()
            window = panel
        }
        window?.orderFront(nil)
    }

    func hide() { window?.orderOut(nil) }
}

private struct VideoProcessingErrorsView: View {
    @ObservedObject var session: WaifuSession
    let close: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Label("\(session.videoProcessingIssues.count) video errors", systemImage: "exclamationmark.triangle.fill")
                .font(.headline)
                .foregroundStyle(.orange)
            Text(session.progress.isRunning
                 ? "Processing continues with the remaining readable videos. Completed frames are kept."
                 : "Completed frames are kept. The files below could not be fully processed.")
                .foregroundStyle(.secondary)
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 16) {
                    ForEach(session.videoProcessingIssues) { issue in
                        VStack(alignment: .leading, spacing: 4) {
                            Text(issue.source.lastPathComponent).font(.headline)
                            Text(issue.sourceDescription).font(.caption).foregroundStyle(.secondary)
                            Text("\(issue.stage): \(issue.reason)")
                            Text(issue.outcome).font(.caption).foregroundStyle(.secondary)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        Divider()
                    }
                }
                .textSelection(.enabled)
                .padding(10)
            }
            .background(.background.secondary, in: RoundedRectangle(cornerRadius: 6))
            HStack {
                Button("Copy Log") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(session.videoProcessingIssues.map(\.logText).joined(separator: "\n\n"), forType: .string)
                }
                Spacer()
                Button("Close", action: close).keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(minWidth: 460, minHeight: 260)
    }
}
