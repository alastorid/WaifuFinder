import SwiftUI

struct TaggingPanelStatus: View {
    @ObservedObject var session: WaifuSession
    let panel: TaggingPanel
    @Environment(\.openSettings) private var openSettings
    @AppStorage("settingsTab") private var settingsTab = "general"
    @AppStorage("settingsModelFocus") private var settingsModelFocus = ""
    /// Observed so the ᴿ mark appears and disappears as the connection changes.
    @ObservedObject private var remote = RemoteClient.shared

    var body: some View {
        Group {
            if session.taggingProgress.isRunning {
                if session.taggingProgress.origin == panel {
                    TaggingProgressCard(status: session.taggingStatus, cancel: { session.cancelTagging() })
                } else {
                    VStack(spacing: 7) {
                        ProgressView().controlSize(.small)
                        Text("Tagging in progress")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            } else if session.restoringTagMemory {
                VStack(spacing: 7) {
                    ProgressView().controlSize(.small)
                    Text("Checking saved tags")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            } else if session.tagVocabulary == nil {
                Button("Set up WD Tagger in Settings…", action: openTaggerSettings)
                    .font(.caption)
                    .buttonStyle(.link)
                    .multilineTextAlignment(.center)
            } else if session.matchedFrameCountForTagging == 0 {
                Text("No frames match the current filters.")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .multilineTextAlignment(.center)
            } else if session.untaggedFrameCountForTagging == 0 {
                Text(session.photosAlbumURL == nil ? "All matching frames are tagged." : "All photos are tagged.")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .multilineTextAlignment(.center)
            } else {
                VStack(spacing: 6) {
                    Button {
                        Task { await session.startTagging(from: panel) }
                    } label: {
                        Text("Start tagging")
                            .remoteMark(remote.offersRemoteTagging, serverName: remote.capabilities?.serverName)
                    }
                    .buttonStyle(.borderedProminent)
                    Text(session.photosAlbumURL == nil
                         ? "Tag \(session.untaggedFrameCountForTagging) matching frames"
                         : "Tag all \(session.untaggedFrameCountForTagging) untagged photos using thumbnails")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .multilineTextAlignment(.center)
                }
            }
        }
        .frame(maxWidth: .infinity, minHeight: 76, alignment: .center)
        .padding(.vertical, 4)
    }

    private func openTaggerSettings() {
        settingsModelFocus = ManagedModel.tagger.rawValue
        settingsTab = "models"
        openSettings()
    }

}

private struct TaggingProgressCard: View {
    @ObservedObject var status: TaggingStatus
    let cancel: () -> Void

    private var progress: TaggingProgress { status.progress }

    var body: some View {
        VStack(spacing: 6) {
            HStack(spacing: 6) {
                ProgressView().controlSize(.small)
                Text("Tagging frames")
                    .font(.caption.weight(.medium))
            }
            ProgressView(value: progress.fraction)
            Text("\(progress.processed.formatted()) of \(progress.total.formatted()) frames")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
            HStack(spacing: 10) {
                Text(timecode(progress.elapsed))
                Text("\(progress.framesPerSecond, specifier: "%.1f") frames/s")
            }
            .font(.caption2.monospacedDigit())
            .foregroundStyle(.tertiary)
            Button("Cancel", action: cancel)
                .controlSize(.small)
        }
        .frame(maxWidth: .infinity)
        .padding(8)
    }
}

struct TaggingStartButton: View {
    @ObservedObject var session: WaifuSession
    let panel: TaggingPanel
    @Environment(\.openSettings) private var openSettings
    @AppStorage("settingsTab") private var settingsTab = "general"
    @AppStorage("settingsModelFocus") private var settingsModelFocus = ""
    @ObservedObject private var remote = RemoteClient.shared

    var body: some View {
        Group {
            if session.untaggedFrameCountForTagging > 0 && !session.restoringTagMemory {
                HStack {
                    Spacer(minLength: 0)
                    Button {
                        if session.tagVocabulary == nil && remote.remoteVocabulary == nil {
                            settingsModelFocus = ManagedModel.tagger.rawValue
                            settingsTab = "models"
                            openSettings()
                        } else {
                            Task { await session.startTagging(from: panel) }
                        }
                    } label: {
                        Text("Start tagging")
                            .remoteMark(remote.offersRemoteTagging, serverName: remote.capabilities?.serverName)
                    }
                    .buttonStyle(.borderless)
                    .help(session.tagVocabulary == nil && remote.remoteVocabulary == nil
                          ? "Open Settings to download and set up WD Tagger, or connect to a server that has it."
                          : session.photosAlbumURL == nil
                            ? "Tag the additional frames admitted by the current filters."
                            : "Tag every untagged photo in this album using thumbnails.")
                    Spacer(minLength: 0)
                }
                .padding(.top, 4)
            }
        }
    }
}
