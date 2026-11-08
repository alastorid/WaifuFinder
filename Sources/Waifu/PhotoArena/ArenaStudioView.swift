import AppKit
import CoreImage
import CryptoKit
import SwiftUI
import UniformTypeIdentifiers

/// Native PhotoArena workspace; analysis/export never interact with the user's UI.
@MainActor
final class ArenaStudio: ObservableObject {
    @Published var sources: [ArenaSource] = []
    @Published var analysis: ArenaAnalysis?
    @Published var selections: [ArenaSourceSelection] = []
    @Published var previews: [String: CGImage] = [:]
    @Published var nativeCrops: [String: CGImage] = [:]
    @Published var inspectNative = true
    @Published var pixelAccurate = false
    @Published var rankingAuthenticity = true
    @Published var composite: CGImage?
    @Published var overlay: CGImage?
    @Published var regionID: String?
    @Published var candidateA = ""
    @Published var candidateB = ""
    @Published var mode = "Arena"
    @Published var atlasMode = "Source"
    @Published var status = "Choose two or more versions of the same photograph."
    @Published var busy = false
    @Published var votes: [ArenaVote] = []
    @Published var category = ArenaCategory.authenticity
    @Published var showDifferences = false
    @Published var zoom: CGFloat = 1
    @Published var pan = CGSize.zero
    private var generation = UUID()
    private var previewGeneration = UUID()
    private var storage: URL?
    private var overrides: [String: String] = [:]
    private var history: [[String: String]] = []
    private var operation: Task<Void, Never>?
    private var comparisons: [ArenaRegionalComparison] = []
    var region: ArenaRegion? { analysis?.regions.first { $0.id == regionID } }

    func choose() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]; panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        guard panel.runModal() == .OK, panel.urls.count >= 2 else { return }
        operation?.cancel(); generation = UUID(); storage = nil
        sources = panel.urls.map { url in
            let id = SHA256.hash(data: Data(url.absoluteString.utf8)).prefix(12).map { String(format: "%02x", $0) }.joined()
            return ArenaSource(id: id, url: url, seconds: 0)
        }
        analysis = nil; previews = [:]; composite = nil; overlay = nil; votes = []; comparisons = []
        overrides = [:]; history = []; selections = []; regionID = nil
        candidateA = sources[0].id; candidateB = sources[1].id
        analyze()
    }
    func analyze() {
        let captured = sources, token = generation
        busy = true; status = "Registering sources and measuring native-resolution regions…"
        operation = Task {
            defer { busy = false }
            do {
                let result = try await ArenaForensics.analyze(captured)
                try Task.checkCancellation()
                let thumbnails = try await Task.detached { () -> [String: CGImage] in
                    let context = CIContext(options: [.cacheIntermediates: false])
                    var images: [String: CGImage] = [:]
                    for source in result.sources {
                        let image = try ArenaForensics.alignedImage(source)
                        let scale = min(1, 1600 / Double(max(result.width, result.height)))
                        let bounded = image.cropped(to: CGRect(x: 0, y: 0, width: result.width, height: result.height))
                            .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
                        images[source.id] = context.createCGImage(bounded, from: bounded.extent)
                    }
                    return images
                }.value
                try Task.checkCancellation()
                guard token == generation else { return }
                analysis = result; previews = thumbnails; regionID = result.regions.first?.id
                let cacheURL = try await Task.detached { try ArenaStudioPersistence.location(captured) }.value
                guard token == generation else { return }
                storage = cacheURL
                if let data = try? Data(contentsOf: cacheURL), let saved = try? JSONDecoder().decode(ArenaStudioSnapshot.self, from: data) {
                    votes = saved.votes; comparisons = saved.comparisons; overrides = saved.overrides
                }
                updateSelections(); persist()
                status = "\(result.width) × \(result.height) · \(result.regions.count) regions · \(result.sources.filter(\.accepted).count)/\(result.sources.count) aligned · authenticity unresolved"
                refreshPreview(); loadRegion()
            } catch { status = error.localizedDescription }
        }
    }
    func loadRegion() {
        guard let analysis, let region else { return }
        let regionID = region.id, token = generation
        Task {
            do {
                let crops = try await Task.detached { () -> [String: CGImage] in
                    var result: [String: CGImage] = [:]
                    for source in analysis.sources where source.accepted {
                        result[source.id] = try ArenaForensics.crop(source, rect: region.rect.cgRect)
                    }
                    return result
                }.value
                if token == generation && self.regionID == regionID { nativeCrops = crops }
            } catch { status = error.localizedDescription }
        }
    }
    func cancel() { operation?.cancel(); status = "Cancellation requested…" }
    func refreshPreview() {
        guard let analysis else { return }
        let choices = selections, sourceID = candidateB
        previewGeneration = UUID()
        let token = previewGeneration, groupToken = generation
        Task {
            do {
                let result = try await Task.detached {
                    (try ArenaReconstruction.preview(analysis, selections: choices, maxPixelSize: 1600),
                     try? ArenaForensics.differenceOverlay(analysis, sourceID: sourceID, maxPixelSize: 1600))
                }.value
                guard token == previewGeneration && groupToken == generation else { return }
                composite = result.0; overlay = result.1
            } catch { status = error.localizedDescription }
        }
    }
    func vote(_ outcome: ArenaOutcome) {
        guard candidateA != candidateB, let region else { return }
        votes.append(ArenaVote(id: UUID(), date: Date(), candidateA: candidateA, candidateB: candidateB,
            category: category, outcome: outcome, author: "human", region: region.id,
            evidence: "Human regional comparison; authenticity is an assessment, not proof."))
        updateSelections(); persist(); refreshPreview()
        status = "Regional vote recorded; export retains human and model evidence separately."
    }
    func judge() {
        guard let analysis else { return }
        let backend = ArenaInferenceBackend.preferred(clients: RemoteInferencePool.shared.entries.map(\.client))
        busy = true; status = "Judging native regions · \(backend.label)…"
        operation = Task {
            if !backend.isRemote { ModelLibrary.shared.inUseArena(true) }
            defer { busy = false; if !backend.isRemote { ModelLibrary.shared.inUseArena(false) } }
            do {
                let records = try await ArenaRegionalJudge.shared.judge(analysis: analysis, maximumComparisons: 4, infer: backend.infer)
                try Task.checkCancellation()
                comparisons += records; votes += ArenaRegionalJudge.votes(records); persist()
                status = "\(backend.label) · \(records.count) regional comparisons saved; model confidence is uncalibrated."
            } catch { status = error.localizedDescription }
        }
    }
    func override(_ id: String) {
        guard analysis != nil, let regionID else { return }
        history.append(overrides); overrides[regionID] = id
        updateSelections(); persist(); refreshPreview()
    }
    func undo() {
        guard let previous = history.popLast(), analysis != nil else { return }
        overrides = previous; updateSelections(); persist(); refreshPreview()
    }
    private func updateSelections() {
        guard let analysis else { return }
        let choices = ArenaRanking.selectionOverrides(analysis: analysis, votes: votes).merging(overrides) { _, manual in manual }
        selections = ArenaReconstruction.select(analysis, overrides: choices)
    }
    private func persist() {
        guard let storage else { return }
        do {
            try FileManager.default.createDirectory(at: storage.deletingLastPathComponent(), withIntermediateDirectories: true)
            try JSONEncoder().encode(ArenaStudioSnapshot(votes: votes, comparisons: comparisons, overrides: overrides)).write(to: storage, options: .atomic)
        } catch { status = "Could not save session: " + error.localizedDescription }
    }
    func export() {
        guard let analysis else { return }
        let panel = NSOpenPanel(); panel.canChooseDirectories = true; panel.canChooseFiles = false
        panel.canCreateDirectories = true; panel.prompt = "Export Here"
        guard panel.runModal() == .OK, let parent = panel.url else { return }
        let directory = parent.appendingPathComponent("PhotoArena-" + UUID().uuidString.prefix(8))
        let choices = selections, records = comparisons, savedVotes = votes
        busy = true; status = "Exporting original-resolution pixels and provenance…"
        operation = Task {
            defer { busy = false }
            do {
                try await Task.detached {
                    _ = try ArenaReconstruction.export(analysis, selections: choices, to: directory)
                    let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                    try encoder.encode(records).write(to: directory.appendingPathComponent("judgments.json"), options: .atomic)
                    try encoder.encode(savedVotes).write(to: directory.appendingPathComponent("votes.json"), options: .atomic)
                    let ranking = ArenaCategory.allCases.map { category in
                        ArenaCategoryRanking(category: category, ranks: ArenaRanking.pairwise(sourceIDs: analysis.sources.map(\.id), votes: savedVotes, category: category))
                    }
                    try encoder.encode(ranking).write(to: directory.appendingPathComponent("ranking.json"), options: .atomic)
                    try encoder.encode(ArenaRanking.technical(analysis)).write(to: directory.appendingPathComponent("technical-ranking.json"), options: .atomic)
                }.value
                status = "Exported \(analysis.width) × \(analysis.height) TIFF, source map and evidence to \(directory.path)"
            } catch { status = error.localizedDescription }
        }
    }
}

struct ArenaStudioView: View {
    @Environment(\.displayScale) private var displayScale
    @StateObject private var studio = ArenaStudio()
    private let modes = ["Arena", "Ranking", "Region Atlas", "Reconstruction Studio"]
    var body: some View {
        VStack(spacing: 12) {
            HStack {
                Text("PhotoArena").font(.title2.bold())
                Button("Choose Photographs…", action: studio.choose).disabled(studio.busy)
                Spacer()
                if studio.busy { ProgressView().controlSize(.small); Button("Cancel", action: studio.cancel) }
                Button("Judge Regions", action: studio.judge).disabled(studio.analysis == nil || studio.busy)
                Button("Export Reconstruction…", action: studio.export).disabled(studio.analysis == nil || studio.busy)
            }
            HStack {
                ForEach(modes, id: \.self) { mode in
                    Button(mode) { studio.mode = mode }.tint(studio.mode == mode ? .accentColor : .secondary)
                }
                Spacer()
                Button("Fit") { studio.zoom = 1; studio.pan = .zero; studio.pixelAccurate = false }
                Button("100%") {
                    studio.inspectNative = true; studio.pixelAccurate = true; studio.pan = .zero; studio.loadRegion()
                }.disabled(studio.analysis == nil)
            }.buttonStyle(.bordered)
            if let analysis = studio.analysis {
                if studio.mode == "Ranking" { ranking(analysis) }
                else if studio.mode == "Region Atlas" { atlas(analysis) }
                else if studio.mode == "Reconstruction Studio" { reconstruction(analysis) }
                else { arena(analysis) }
            } else {
                ContentUnavailableView("Compare photographic evidence", systemImage: "photo.on.rectangle.angled",
                    description: Text("Import matching photographs to inspect differences and reconstruct from existing source pixels."))
            }
            Text(studio.status).font(.caption).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
            Text("Authenticity cannot be proven from appearance. Automatic selections remain unresolved; no generative reconstruction is used.")
                .font(.caption).foregroundStyle(.secondary)
        }.padding(18).frame(minWidth: 1000, minHeight: 700)
        .onChange(of: studio.regionID) { _, _ in studio.loadRegion() }
    }
    private func choices(_ binding: Binding<String>, analysis: ArenaAnalysis) -> some View {
        HStack {
            ForEach(analysis.sources) { source in
                Button(source.url.lastPathComponent) { binding.wrappedValue = source.id; studio.refreshPreview() }
                    .tint(binding.wrappedValue == source.id ? .accentColor : .secondary)
                    .disabled(!source.accepted)
            }
        }.buttonStyle(.bordered)
    }
    private func arena(_ analysis: ArenaAnalysis) -> some View {
        VStack {
            choices($studio.candidateA, analysis: analysis)
            choices($studio.candidateB, analysis: analysis)
            HStack { Toggle("Native region crops", isOn: $studio.inspectNative); Toggle("Difference heatmap on B (overview)", isOn: $studio.showDifferences) }
            HStack {
                image(studio.inspectNative && !studio.showDifferences ? studio.nativeCrops[studio.candidateA] : studio.previews[studio.candidateA])
                image(studio.showDifferences ? studio.overlay : studio.inspectNative ? studio.nativeCrops[studio.candidateB] : studio.previews[studio.candidateB])
            }
            regionControls(analysis)
            HStack {
                ForEach(ArenaCategory.allCases, id: \.self) { category in
                    Button(category.rawValue) { studio.category = category }
                        .tint(studio.category == category ? .accentColor : .secondary)
                }
            }
            HStack {
                Button("A wins") { studio.vote(.a) }; Button("B wins") { studio.vote(.b) }
                Button("Tie") { studio.vote(.tie) }; Button("Unknown") { studio.vote(.unknown) }
            }.disabled(studio.candidateA == studio.candidateB)
        }
    }
    private func regionControls(_ analysis: ArenaAnalysis) -> some View {
        HStack {
            Button("Previous Region") { moveRegion(-1, analysis) }
            Text(studio.regionID ?? "No region").monospaced()
            Button("Next Region") { moveRegion(1, analysis) }
            if let region = studio.region {
                Text("x \(Int(region.rect.x)), y \(Int(region.rect.y)) · \(Int(region.rect.width)) × \(Int(region.rect.height)) native pixels").font(.caption)
            }
        }
    }
    private func moveRegion(_ delta: Int, _ analysis: ArenaAnalysis) {
        let index = analysis.regions.firstIndex { $0.id == studio.regionID } ?? 0
        let next = max(0, min(analysis.regions.count - 1, index + delta))
        if analysis.regions.indices.contains(next) { studio.regionID = analysis.regions[next].id }
    }
    private func image(_ cg: CGImage?) -> some View {
        GeometryReader { geometry in
            ZStack {
                Color.black.opacity(0.9)
                if let cg {
                    Image(decorative: cg, scale: 1).resizable().interpolation(.none).scaledToFit()
                        .frame(width: studio.pixelAccurate ? CGFloat(cg.width) / displayScale : geometry.size.width * studio.zoom, height: studio.pixelAccurate ? CGFloat(cg.height) / displayScale : geometry.size.height * studio.zoom)
                        .offset(studio.pan)
                }
            }.frame(width: geometry.size.width, height: geometry.size.height).clipped()
                .gesture(DragGesture().onChanged { value in studio.pan = value.translation })
                .simultaneousGesture(MagnifyGesture().onChanged { value in studio.zoom = max(1, min(32, value.magnification)) })
        }
    }
    private func ranking(_ analysis: ArenaAnalysis) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 15) {
                HStack {
                    Button("Authenticity evidence first") { studio.rankingAuthenticity = true }
                    Button("Technical quality first") { studio.rankingAuthenticity = false }
                }.buttonStyle(.bordered)
                Text(studio.rankingAuthenticity ? "Authenticity preferences · insufficient evidence remains tied" : "Technical quality order · authenticity unknown").font(.headline)
                Text("Sharpness is a technical measurement and can reward artificial sharpening. Stable equal scores are ties, not evidence of superiority.").foregroundStyle(.secondary)
                ForEach(qualityOrder(analysis), id: \.0.id) { source, score in
                    HStack {
                        if let cg = studio.previews[source.id] { Image(decorative: cg, scale: 1).resizable().scaledToFit().frame(width: 100, height: 80) }
                        VStack(alignment: .leading) {
                            Text(source.url.lastPathComponent).bold()
                            Text("Relative metric \(score, specifier: "%.4f") · alignment \(source.alignmentConfidence, specifier: "%.2f") · authenticity unknown")
                            ForEach(source.issues, id: \.self) { Text($0).font(.caption) }
                        }
                    }
                }
                Text("\(studio.votes.count) category-specific votes retained for review/export.")
                ForEach(Array(studio.votes.suffix(20).enumerated()), id: \.offset) { _, vote in
                    Text("\(vote.region) · \(vote.category.rawValue) · \(vote.outcome.rawValue) · \(vote.author): \(vote.evidence)").font(.caption)
                }
            }.frame(maxWidth: .infinity, alignment: .leading)
        }
    }
    private func qualityOrder(_ analysis: ArenaAnalysis) -> [(ArenaAnalyzedSource, Double)] {
        let ranks = studio.rankingAuthenticity
            ? ArenaRanking.pairwise(sourceIDs: analysis.sources.map(\.id), votes: studio.votes, category: .authenticity)
            : ArenaRanking.technical(analysis)
        return ranks.compactMap { rank in
            analysis.sources.first(where: { $0.id == rank.sourceID }).map { ($0, rank.score) }
        }
    }
    private func atlas(_ analysis: ArenaAnalysis) -> some View {
        VStack {
            HStack {
                ForEach(["Source", "Differences", "Text candidates", "Alignment confidence", "Unresolved"], id: \.self) { mode in
                    Button(mode) { studio.atlasMode = mode }.tint(studio.atlasMode == mode ? .accentColor : .secondary)
                }
            }.buttonStyle(.bordered)
            GeometryReader { geometry in
                let scale = min(geometry.size.width / CGFloat(analysis.width), geometry.size.height / CGFloat(analysis.height))
                ZStack(alignment: .topLeading) {
                    if let cg = studio.composite { Image(decorative: cg, scale: 1).resizable().frame(width: CGFloat(analysis.width) * scale, height: CGFloat(analysis.height) * scale) }
                    ForEach(analysis.regions) { region in
                        let rect = region.rect.cgRect
                        Rectangle().fill(region.id == studio.regionID ? Color.accentColor.opacity(0.7) : atlasColor(region, analysis))
                            .overlay(Rectangle().stroke(Color.white.opacity(0.5), lineWidth: 0.5))
                            .frame(width: rect.width * scale, height: rect.height * scale)
                            .offset(x: rect.minX * scale, y: (CGFloat(analysis.height) - rect.maxY) * scale)
                            .onTapGesture { studio.regionID = region.id }
                    }
                }
            }
            regionControls(analysis)
            if let region = studio.region {
                Text((region.evidence + region.measurements.flatMap(\.evidence)).joined(separator: " • ")).font(.caption).lineLimit(5)
            }
        }
    }
    private func atlasColor(_ region: ArenaRegion, _ analysis: ArenaAnalysis) -> Color {
        let sourceID = studio.selections.first { $0.regionID == region.id }?.sourceID
        switch studio.atlasMode {
        case "Differences":
            return Color.red.opacity(min(0.85, (region.measurements.map(\.difference).max() ?? 0) * 6))
        case "Text candidates":
            return Color.yellow.opacity(min(0.85, (region.measurements.map(\.textCandidateFraction).max() ?? 0) * 4))
        case "Alignment confidence":
            let confidence = analysis.sources.first { $0.id == sourceID }?.alignmentConfidence ?? 0
            return Color.green.opacity(confidence * 0.6)
        case "Unresolved": return Color.orange.opacity(region.unresolved ? 0.4 : 0)
        default:
            let index = analysis.sources.firstIndex { $0.id == sourceID } ?? 0
            return Color(hue: Double(index) / Double(max(1, analysis.sources.count)), saturation: 0.8, brightness: 0.9).opacity(0.4)
        }
    }
    private func reconstruction(_ analysis: ArenaAnalysis) -> some View {
        VStack {
            HStack { image(studio.previews[analysis.referenceID]); image(studio.composite) }
            regionControls(analysis)
            Text("Override selected region with source pixels:").font(.caption)
            HStack {
                ForEach(analysis.sources.filter(\.accepted)) { source in
                    Button(source.url.lastPathComponent) { studio.override(source.id) }
                }
                Button("Undo Override", action: studio.undo)
            }
            if let selection = studio.selections.first(where: { $0.regionID == studio.regionID }) {
                Text("Source: \(selection.sourceID) · \(selection.unresolved ? "Unresolved authenticity" : "Supported selection") · \(selection.reasons.joined(separator: " • "))").font(.caption)
            }
            Text("Hard source boundaries preserve samples; seams and low-resolution resampling may remain. Export includes exact source selections and transforms.").font(.caption).foregroundStyle(.secondary)
        }
    }
}

private struct ArenaCategoryRanking: Codable {
    let category: ArenaCategory
    let ranks: [ArenaSourceRank]
}
private struct ArenaStudioSnapshot: Codable {
    let votes: [ArenaVote]
    let comparisons: [ArenaRegionalComparison]
    let overrides: [String: String]
}
private enum ArenaStudioPersistence {
    static func location(_ sources: [ArenaSource]) throws -> URL {
        var key = SHA256()
        for source in sources {
            key.update(data: Data(source.id.utf8))
            let handle = try FileHandle(forReadingFrom: source.url)
            defer { try? handle.close() }
            var file = SHA256()
            while let bytes = try handle.read(upToCount: 1024 * 1024), !bytes.isEmpty { file.update(data: bytes) }
            key.update(data: Data(file.finalize()))
        }
        let name = key.finalize().map { String(format: "%02x", $0) }.joined()
        return FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Waifu/PhotoArena/studio-" + name + ".json")
    }
}
