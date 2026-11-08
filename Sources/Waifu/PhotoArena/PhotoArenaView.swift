import AppKit
import CryptoKit
import SwiftUI
import UniformTypeIdentifiers

@MainActor
final class ArenaWorkspace: ObservableObject, Identifiable {
    let id = UUID()
    @Published var document: ArenaDocument
    @Published var pairIndex = 0
    @Published var category = ArenaCategory.authenticity
    @Published var images: [CGImage] = []
    @Published var status = ""
    @Published var evidence = ""
    @Published var judging = false
    private var judgeTask: Task<Void, Never>?
    private let storage: URL
    init(sources: [ArenaSource]) {
        document = ArenaDocument(sources: sources)
        let identity = sources.map { "\($0.id)|\($0.url.absoluteString)|\($0.seconds)" }.joined(separator: "\n")
        let key = SHA256.hash(data: Data(identity.utf8)).map { String(format: "%02x", $0) }.joined()
        storage = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Waifu/PhotoArena/\(key).json")
        if let data = try? Data(contentsOf: storage), let saved = try? JSONDecoder().decode(ArenaDocument.self, from: data),
           saved.sources == sources, (try? saved.validate()) != nil { document = saved }
    }
    var pair: (ArenaSource, ArenaSource) { document.pairs[pairIndex] }
    func load() async {
        images = []; status = "Loading original pixels…"
        do {
            let a = try await PhotosMedia.shared.image(pair.0.url, seconds: pair.0.seconds)
            try Task.checkCancellation()
            let b = try await PhotosMedia.shared.image(pair.1.url, seconds: pair.1.seconds)
            try Task.checkCancellation()
            images = [a, b]; status = "Human judge • Whole image • Alignment not validated"
        } catch { if !Task.isCancelled { status = error.localizedDescription } }
    }
    func vote(_ outcome: ArenaOutcome) {
        let vote = ArenaVote(id: UUID(), date: Date(), candidateA: pair.0.id, candidateB: pair.1.id,
                             category: category, outcome: outcome, author: "human", region: "whole-image",
                             evidence: evidence)
        var updated = document
        updated.votes.append(vote)
        do {
            try updated.validate()
            try FileManager.default.createDirectory(at: storage.deletingLastPathComponent(), withIntermediateDirectories: true)
            try JSONEncoder().encode(updated).write(to: storage, options: .atomic)
            document = updated; evidence = ""; status = "Vote saved • \(document.votes.count) decisions"
        } catch { status = "Could not save vote: \(error.localizedDescription)" }
    }
    func cancelJudge() { judgeTask?.cancel(); judgeTask = nil }
    func judge() {
        guard images.count == 2, !judging else { return }
        let capturedPair = pair
        let capturedImages = images
        let backend = ArenaInferenceBackend.preferred(clients: RemoteInferencePool.shared.entries.map(\.client))
        judging = true
        status = "MiniCPM is judging · \(backend.label)…"
        judgeTask = Task {
            if !backend.isRemote { ModelLibrary.shared.inUseArena(true) }
            defer { judging = false; if !backend.isRemote { ModelLibrary.shared.inUseArena(false) } }
            do {
                let data = try await Task.detached { try capturedImages.map(ArenaJudge.png) }.value
                let result = try await backend.infer(data)
                try Task.checkCancellation()
                var updated = document
                for category in ArenaCategory.allCases {
                    updated.votes.append(ArenaVote(id: UUID(), date: Date(), candidateA: capturedPair.0.id,
                        candidateB: capturedPair.1.id, category: category, outcome: result.judgment.outcome(category),
                        author: "AI:" + ArenaJudgeConfiguration.revision, region: "whole-image-overview-1600px",
                        evidence: result.judgment.evidence.joined(separator: " • ") + (result.judgment.uncertain ? " • Uncertain" : "")))
                }
                try updated.validate()
                try FileManager.default.createDirectory(at: storage.deletingLastPathComponent(), withIntermediateDirectories: true)
                try JSONEncoder().encode(updated).write(to: storage, options: .atomic)
                document = updated
                evidence = result.judgment.evidence.joined(separator: " • ")
                status = "MiniCPM: \(result.judgment.recommended_source.rawValue) · \(result.tokens) tokens · \(String(format: "%.1f", result.prefillSeconds + result.decodeSeconds))s · \(result.judgment.uncertain ? "uncertain" : "uncalibrated judgment")"
            } catch { status = Task.isCancelled ? "Judging cancelled" : error.localizedDescription }
        }
    }
    func export() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.json]; panel.nameFieldStringValue = "PhotoArena-decisions.json"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do { try JSONEncoder().encode(document).write(to: url, options: .atomic) }
        catch { status = error.localizedDescription }
    }
}

struct PhotoArenaView: View {
    @ObservedObject var workspace: ArenaWorkspace
    @Environment(\.dismiss) private var dismiss
    var body: some View {
        VStack(spacing: 12) {
            HStack {
                Text("PhotoArena").font(.title2.bold())
                Text("\(workspace.document.sources.count) sources · Match \(workspace.pairIndex + 1)/\(workspace.document.pairs.count)")
                Spacer()
                Button(workspace.judging ? "Cancel AI" : "Ask MiniCPM") {
                    if workspace.judging { workspace.cancelJudge() } else { workspace.judge() }
                }.disabled(workspace.images.count != 2)
                Button("Export Decisions", action: workspace.export)
                Button("Done") { dismiss() }
            }
            HStack {
                ForEach(ArenaCategory.allCases, id: \.self) { category in
                    Button(category.rawValue) { workspace.category = category }
                        .buttonStyle(.bordered).tint(workspace.category == category ? .accentColor : .secondary)
                }
            }
            HStack(spacing: 12) {
                candidate(0, label: "A", source: workspace.pair.0)
                candidate(1, label: "B", source: workspace.pair.1)
            }.frame(maxWidth: .infinity, maxHeight: .infinity)
            Text(workspace.status).font(.caption).foregroundStyle(.secondary)
            TextField("Observable evidence (optional)", text: $workspace.evidence)
            HStack {
                Button("Previous") { workspace.pairIndex -= 1 }.disabled(workspace.pairIndex == 0 || workspace.judging)
                Spacer()
                Button("A wins") { workspace.vote(.a) }.keyboardShortcut("a", modifiers: [])
                Button("B wins") { workspace.vote(.b) }.keyboardShortcut("b", modifiers: [])
                Button("Tie") { workspace.vote(.tie) }.keyboardShortcut("t", modifiers: [])
                Button("Unknown") { workspace.vote(.unknown) }.keyboardShortcut("u", modifiers: [])
                Spacer()
                Button("Next") { workspace.pairIndex += 1 }.disabled(workspace.pairIndex + 1 == workspace.document.pairs.count || workspace.judging)
            }.disabled(workspace.images.count != 2)
            Text("AI judgments use a 1600px overview and are uncalibrated evidence. Alignment and reconstruction are not available yet.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding(20).frame(minWidth: 950, minHeight: 650)
        .task(id: workspace.pairIndex) { await workspace.load() }
        .onDisappear { workspace.cancelJudge() }
    }
    private func candidate(_ index: Int, label: String, source: ArenaSource) -> some View {
        VStack {
            Text("\(label) · \(source.id)").lineLimit(1)
            if workspace.images.indices.contains(index) {
                Image(decorative: workspace.images[index], scale: 1).resizable().scaledToFit()
            } else { ProgressView().frame(maxWidth: .infinity, maxHeight: .infinity) }
        }.frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
