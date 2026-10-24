import SwiftUI

/// Uses the same native GroupBox, signal row and job presentation as Rating.
struct ForensicPanelSection<Signal:View>:View {
    @ObservedObject var session:WaifuSession
    @ObservedObject var job:TruForJob
    @ViewBuilder let signal:()->Signal
    @ObservedObject private var remote=RemoteInferencePool.shared
    @ObservedObject private var models=ModelLibrary.shared
    @Environment(\.openSettings) private var openSettings
    @AppStorage("settingsTab") private var settingsTab="general"
    @AppStorage("settingsModelFocus") private var settingsModelFocus=""
    @State private var optionHeld=false
    var body:some View {
        GroupBox {
            VStack(alignment:.leading,spacing:10) {
                if session.isLoadingPhotosAlbum {
                    Text("Loading...").font(.caption).foregroundStyle(.secondary)
                        .frame(maxWidth:.infinity,alignment:.leading)
                } else if job.progress.isRunning || job.scored==0 {
                    status.frame(maxWidth:.infinity,minHeight:76,alignment:.center).padding(.vertical,4)
                } else {
                    signal()
                    if job.remaining>0 {
                        HStack { Spacer(minLength:0);startButton.buttonStyle(.borderless);Spacer(minLength:0) }.padding(.top,4)
                        Text("\(job.scored.formatted()) of \(job.total.formatted()) photos analyzed; unscored photos remain visible.")
                            .font(.caption2).foregroundStyle(.tertiary)
                    } else {
                        Text("All photos are analyzed.").font(.caption).foregroundStyle(.tertiary)
                    }
                    if let message=job.message { Text(message).font(.caption).foregroundStyle(.secondary) }
                }
            }.frame(maxWidth:.infinity,alignment:.leading)
        } label: { Text("Forensic").font(.headline) }
        .task(id:"\(session.photosAlbumURL?.absoluteString ?? "")|\(session.thumbnailGeneration)|\(session.frames.count)") {
            do { try await Task.sleep(for:.milliseconds(250));try Task.checkCancellation();await job.refresh(session:session) } catch {}
        }
        .onAppear { optionHeld=NSEvent.modifierFlags.contains(.option) }
        .onModifierKeysChanged(mask:.option) { _,flags in optionHeld=flags.contains(.option) }
    }
    @ViewBuilder private var status:some View {
        if job.progress.isRunning {
            VStack(spacing:6) {
                HStack(spacing:6) { ProgressView().controlSize(.small);Text("Analyzing photos").font(.caption.weight(.medium)) }
                ProgressView(value:job.progress.fraction)
                Text("\(job.progress.processed.formatted()) of \(job.progress.total.formatted()) photos").font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                HStack(spacing:10) { Text(timecode(job.progress.elapsed));Text("\(job.progress.framesPerSecond,specifier:"%.1f") photos/s") }.font(.caption2.monospacedDigit()).foregroundStyle(.tertiary)
                if let message=job.message { Text(message).font(.caption2).foregroundStyle(.secondary) }
                Button("Cancel") { job.cancel() }.controlSize(.small)
            }.frame(maxWidth:.infinity).padding(8)
        } else if job.checking {
            VStack(spacing:7) { ProgressView().controlSize(.small);Text("Checking saved forensic scores").font(.caption).foregroundStyle(.secondary) }
        } else if !job.available {
            Button("Set up TruFor in Settings…") { settingsModelFocus=ManagedModel.trufor.rawValue;settingsTab="models";openSettings() }
                .font(.caption).buttonStyle(.link).multilineTextAlignment(.center)
        } else if job.total==0 {
            Text("No eligible photos.").font(.caption).foregroundStyle(.tertiary)
        } else {
            VStack(spacing:6) {
                startButton.buttonStyle(.borderedProminent)
                Text("Analyze \(job.remaining.formatted()) unscored photos at native resolution").font(.caption2).foregroundStyle(.tertiary).multilineTextAlignment(.center)
                if let message=job.message { Text(message).font(.caption2).foregroundStyle(.secondary) }
            }
        }
    }
    private var startButton:some View {
        Button { let local=NSEvent.modifierFlags.contains(.option);Task { await job.start(session:session,includeLocal:local) } } label: {
            Text("Start Forensic").remoteMark(job.remotely,includeLocal:optionHeld && job.localAvailable)
        }.disabled(!job.available)
    }
}
