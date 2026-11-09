import SwiftUI

/// The selected-photo entry point shares the same N-source workflow as File → PhotoArena.
@MainActor
final class ArenaWorkspace: ObservableObject, Identifiable {
    let id = UUID()
    let sources: [ArenaSource]
    init(sources: [ArenaSource]) { self.sources = sources }
}
struct PhotoArenaView: View {
    @ObservedObject var workspace: ArenaWorkspace
    @Environment(\.dismiss) private var dismiss
    var body: some View {
        VStack(spacing: 0) {
            HStack { Spacer(); Button("Done") { dismiss() } }.padding([.top,.trailing], 10)
            ArenaStudioView(initialSources: workspace.sources)
        }
    }
}
