import AppKit
import CoreImage
import CryptoKit
import Foundation
import ImageIO
import UniformTypeIdentifiers

@MainActor
final class ArenaStudio: ObservableObject {
    @Published var analysis: ArenaAnalysis?
    @Published var plan: ArenaRecoveryPlan?
    @Published var phase: ArenaWorkflowPhase = .empty
    @Published var status = "Choose several versions of one photograph to find the best source."
    @Published var previews: [String: CGImage] = [:]
    @Published var nativeCrops: [String: CGImage] = [:]
    @Published var resultCrop: CGImage?
    @Published var resultImage: CGImage?
    @Published var redOverlay: CGImage?
    @Published var difference: ArenaOutputDifference?
    @Published var records: [ArenaReviewRecord] = []
    @Published var sourceID = ""
    @Published var regionID: String?
    @Published var inspectionMode = "Result"
    @Published var differenceScope = "Detected changes"
    @Published var tolerance = 12.0
    @Published var opacity = 0.55
    @Published var zoom = 1.0
    @Published var pan = CGSize.zero
    @Published var showRed = true
    @Published var nativeDetail = false
    private var operation: Task<Void, Never>?
    private var displayTask: Task<Void, Never>?
    private var cropTask: Task<Void, Never>?
    private var generation = UUID()
    private var displayGeneration = UUID()
    private var overrides: [String: String] = [:]
    private var preferredSource: String?
    private var history: [([String: String], String?)] = []
    private var storage: URL?
    private var ingestionDirectory: URL?
    var busy: Bool { [.ingesting, .aligning, .reviewing, .planning, .verifying].contains(phase) }
    var region: ArenaRegion? { analysis?.regions.first { $0.id == regionID } }
    var selectedReview: ArenaReviewRecord? { records.last { $0.id == (regionID ?? "overview") } }
    var changeRegions: [ArenaRegion] { analysis?.regions.filter(ArenaRecoveryPlanner.needsReview) ?? [] }

    func choose() {
        let panel = NSOpenPanel(); panel.allowedContentTypes = [.image]; panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        guard panel.runModal() == .OK else { return }
        open(panel.urls.map { ArenaSource(id: $0.absoluteString, url: $0, seconds: 0) })
    }
    func open(_ candidates: [ArenaSource]) {
        guard !busy else { return }
        guard (2...16).contains(candidates.count) else { status = "Choose 2–16 candidates from the same photograph."; return }
        let inputs = candidates.map { source in
            // Remote evidence uses opaque candidate identities, without sending local paths.
            let id = SHA256.hash(data:Data(source.id.utf8)).prefix(16).map { String(format:"%02x",$0) }.joined()
            return ArenaSource(id:id,url:source.url,seconds:source.seconds)
        }
        cancelTasks(); generation = UUID(); let token = generation
        analysis = nil; plan = nil; records = []; previews = [:]; nativeCrops = [:]; resultImage = nil; redOverlay = nil; resultCrop = nil
        overrides = [:]; preferredSource = nil; history = []; regionID = nil; storage = nil
        if let ingestionDirectory { try? FileManager.default.removeItem(at: ingestionDirectory) }
        ingestionDirectory = nil
        phase = .ingesting; status = "Reading candidates…"
        operation = Task {
            do {
                var files: [ArenaSource] = []
                for input in inputs {
                    try Task.checkCancellation()
                    if input.seconds == 0, input.url.isFileURL,
                       CGImageSourceCreateWithURL(input.url as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary) != nil {
                        files.append(input); continue
                    }
                    // Photos-library assets/video frames are materialized as explicit derived inputs.
                    // Inference still uses anonymous pipes; no activations or model scratch are written.
                    if ingestionDirectory == nil {
                        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("PhotoArena-inputs-"+UUID().uuidString)
                        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true); ingestionDirectory = directory
                    }
                    let image = try await PhotosMedia.shared.image(input.url, seconds: input.seconds)
                    try Task.checkCancellation()
                    let url = ingestionDirectory!.appendingPathComponent("candidate-\(files.count+1).tiff")
                    guard let target = CGImageDestinationCreateWithURL(url as CFURL, UTType.tiff.identifier as CFString, 1, nil) else { throw ModelInstaller.fail("Cannot prepare photographic candidate") }
                    CGImageDestinationAddImage(target, image, [kCGImagePropertyTIFFDictionary: [kCGImagePropertyTIFFImageDescription: "Derived from \(input.url.absoluteString) at \(input.seconds)s"]] as CFDictionary)
                    guard CGImageDestinationFinalize(target) else { throw ModelInstaller.fail("Cannot finish photographic candidate") }
                    files.append(ArenaSource(id: input.id, url: url, seconds: 0))
                }
                guard token == generation else { return }
                phase = .aligning; status = "Aligning all candidates and finding native-pixel differences…"
                let result = try await ArenaForensics.analyze(files, tileSize: 256)
                try Task.checkCancellation(); guard token == generation else { return }
                analysis = result; sourceID = result.referenceID
                regionID = result.regions.first(where: ArenaRecoveryPlanner.needsReview)?.id
                let location = ArenaStudioPersistence.location(result)
                storage = location
                let saved = (try? Data(contentsOf: location)).flatMap { try? JSONDecoder().decode(ArenaStudioSnapshot.self, from: $0) }
                if let snapshot = saved, snapshot.revision == ArenaGroupPayload.revision {
                    // Persisted records are revalidated against freshly prepared evidence before reuse.
                    records = []; overrides = snapshot.overrides; preferredSource = snapshot.preferredSource
                }
                let previous = saved?.records ?? []
                previews = try await background {
                    var images: [String: CGImage] = [:]
                    let context = CIContext(options: [.cacheIntermediates: false])
                    for source in result.sources {
                        try Task.checkCancellation()
                        let image = try source.accepted ? ArenaForensics.alignedImage(source) : ArenaForensics.image(source)
                        let bounds = source.accepted ? CGRect(x: 0, y: 0, width: result.width, height: result.height) : image.extent
                        let scale = min(1, 1600/max(bounds.width,bounds.height))
                        let reduced = image.cropped(to: bounds).transformed(by: .init(scaleX: scale, y: scale))
                        images[source.id] = context.createCGImage(reduced, from: reduced.extent)
                        context.clearCaches()
                    }
                    return images
                }
                recomputePlan(); refreshDisplay(); loadRegion()
                await runReview(previous: previous, token: token)
            } catch { fail(error, token: token) }
        }
    }
    func reviewMore() {
        guard !busy, analysis != nil else { return }
        let token = generation
        operation = Task { await runReview(previous: records, token: token) }
    }
    private func runReview(previous: [ArenaReviewRecord], token: UUID) async {
        guard let analysis else { return }
        let backend = ArenaReviewBackend.preferred(clients: RemoteInferencePool.shared.entries.map(\.client))
        let completed = Set(records.map(\.id))
        let jobs = Array(ArenaRecoveryPlanner.jobs(analysis).filter { !completed.contains($0.id) }.prefix(13))
        guard !jobs.isEmpty else { phase = .ready; return }
        phase = .reviewing; status = "Reviewing all \(analysis.sources.count) candidates · \(backend.label)…"
        if !backend.isRemote { ModelLibrary.shared.inUseArena(true) }
        defer { if !backend.isRemote { ModelLibrary.shared.inUseArena(false) } }
        do {
            let results = try await backend.review(analysis, jobs, previous, { [weak self] record in
                Task { @MainActor in
                    guard let self, self.generation == token, self.phase == .reviewing else { return }
                    self.accept(record); self.recomputePlan(); self.persist()
                    self.status = "Reviewed \(self.records.count)/\(ArenaRecoveryPlanner.jobs(analysis).count) evidence packages across \(analysis.sources.count) candidates."
                }
            })
            try Task.checkCancellation(); guard token == generation else { return }
            for record in results { accept(record) }
            phase = .planning; recomputePlan(); persist()
            phase = .verifying; status = "Checking recovery boundaries and source coverage…"
            displayTask?.cancel()
            await updateDisplay(token: token)
            try Task.checkCancellation(); guard token == generation else { return }
            phase = plan?.pendingRegionIDs.isEmpty == true ? .ready : .partial
            status = "\(records.count) packages reviewed · \(plan?.pendingRegionIDs.count ?? 0) pending · \(plan?.unresolvedRegionIDs.count ?? 0) regions retain uncertainty."
        } catch { fail(error, token: token); recomputePlan(); refreshDisplay() }
    }
    private func accept(_ record: ArenaReviewRecord) {
        records.removeAll { $0.id == record.id }; records.append(record)
    }
    private func fail(_ error: Error, token: UUID) {
        guard token == generation else { return }
        phase = error is CancellationError || Task.isCancelled ? .cancelled : .failed
        status = phase == .cancelled ? "Review cancelled. Completed findings remain available; pending regions are unreviewed." : error.localizedDescription
    }
    private func recomputePlan() {
        guard let analysis else { return }
        plan = ArenaRecoveryPlanner.plan(analysis, reviews: records, overrides: overrides, preferredSource: preferredSource)
    }
    func pickSource(_ id: String) {
        guard analysis?.sources.contains(where: { $0.id == id && $0.accepted }) == true, !busy else { return }
        history.append((overrides,preferredSource)); preferredSource = id
        recomputePlan(); persist(); refreshDisplay()
    }
    func overrideRegion(_ id: String) {
        guard let regionID, !busy else { return }
        history.append((overrides,preferredSource)); overrides[regionID] = id
        recomputePlan(); persist(); refreshDisplay()
    }
    func undo() {
        guard !busy, let previous = history.popLast() else { return }
        overrides = previous.0; preferredSource = previous.1; recomputePlan(); persist(); refreshDisplay()
    }
    func cancel() { operation?.cancel(); displayTask?.cancel(); status = "Cancellation requested; completed evidence is retained." }
    func close() {
        cancelTasks()
        if let ingestionDirectory { try? FileManager.default.removeItem(at: ingestionDirectory) }
        ingestionDirectory = nil
    }
    private func cancelTasks() { operation?.cancel(); displayTask?.cancel(); cropTask?.cancel() }
    func refreshDisplay() {
        displayTask?.cancel(); let token = generation
        displayTask = Task { await updateDisplay(token: token) }
    }
    private func updateDisplay(token: UUID) async {
        guard let analysis, let plan else { return }
        displayGeneration = UUID(); let displayToken = displayGeneration
        let selected = sourceID, threshold = Int(tolerance), alpha = opacity, scope = differenceScope
        do {
            let output = try await background {
                let checked = try plan.verification == nil ? ArenaMergeVerifier.verify(analysis,plan:plan) : plan
                let image = try ArenaReconstruction.preview(analysis, selections: checked.selections)
                if scope == "Detected changes" {
                    let overlay = try ArenaOutputInspector.detectedOverlay(analysis,tolerance:threshold,opacity:alpha)
                    return (checked,image,Optional<ArenaOutputDifference>.none,Optional(overlay))
                }
                let diff = try? ArenaOutputInspector.compare(analysis, selections: checked.selections, sourceID: selected,
                    tolerance: threshold, opacity: alpha, heatmap: scope == "Heatmap")
                return (checked,image,diff?.0,diff?.1)
            }
            try Task.checkCancellation(); guard token == generation, displayToken == displayGeneration else { return }
            if phase != .reviewing { self.plan = output.0 }
            resultImage = output.1; difference = output.2; redOverlay = output.3
            if nativeDetail { loadRegion() }
        } catch { if !Task.isCancelled && token == generation { status = error.localizedDescription } }
    }
    func loadRegion() {
        cropTask?.cancel(); nativeCrops = [:]; resultCrop = nil
        guard let analysis, let region else { return }
        let token = generation, selections = plan?.selections ?? []
        cropTask = Task {
            do {
                let crops = try await background {
                    var images: [String: CGImage] = [:]
                    for source in analysis.sources where source.accepted {
                        try Task.checkCancellation(); images[source.id] = try ArenaForensics.crop(source, rect: region.rect.cgRect)
                    }
                    let result = selections.isEmpty ? nil : try ArenaOutputInspector.tile(analysis,selections:selections,rect:region.rect.cgRect)
                    return (images,result)
                }
                try Task.checkCancellation(); if token == generation && region.id == regionID { nativeCrops = crops.0; resultCrop = crops.1 }
            } catch { }
        }
    }
    private func persist() {
        guard let storage else { return }
        do {
            try FileManager.default.createDirectory(at: storage.deletingLastPathComponent(), withIntermediateDirectories: true)
            try JSONEncoder().encode(ArenaStudioSnapshot(revision: ArenaGroupPayload.revision, records: records, overrides: overrides, preferredSource: preferredSource))
                .write(to: storage, options: .atomic)
        } catch { status = "Could not save review: " + error.localizedDescription }
    }
    func export() {
        guard !busy, let analysis, let plan else { return }
        let panel = NSOpenPanel(); panel.canChooseDirectories = true; panel.canChooseFiles = false; panel.canCreateDirectories = true; panel.prompt = "Export Here"
        guard panel.runModal() == .OK, let parent = panel.url else { return }
        let directory = parent.appendingPathComponent("PhotoArena-" + UUID().uuidString.prefix(8))
        let records = records, token = generation
        phase = .verifying; status = "Exporting source pixels, review evidence and provenance…"
        operation = Task {
            do {
                let manifest = try await background {
                    let checked = try plan.verification == nil ? ArenaMergeVerifier.verify(analysis,plan:plan) : plan
                    let manifest = try ArenaReconstruction.export(analysis, selections: checked.selections, to: directory, plan: checked)
                    do {
                        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted,.sortedKeys]
                        try encoder.encode(records).write(to: directory.appendingPathComponent("review.json"), options: .atomic)
                        var differences: [ArenaOutputDifference] = []
                        var archived = analysis
                        archived.sources = analysis.sources.map { source in
                            ArenaAnalyzedSource(id:source.id,url:directory.appendingPathComponent(manifest.sourceFiles[source.id]!),
                                width:source.width,height:source.height,transform:source.transform,alignmentConfidence:source.alignmentConfidence,
                                accepted:source.accepted,issues:source.issues,fileSHA256:source.fileSHA256)
                        }
                        let outputURL = directory.appendingPathComponent(manifest.imageFile)
                        let output: CIImage
                        if checked.kind == .composite {
                            guard let written = CIImage(contentsOf:outputURL) else { throw ModelInstaller.fail("Cannot verify exported composite") }
                            output = written
                        } else {
                            let source = archived.sources.first { $0.id == checked.backboneID }!
                            let copied = ArenaAnalyzedSource(id:source.id,url:outputURL,width:source.width,height:source.height,
                                transform:source.transform,alignmentConfidence:source.alignmentConfidence,accepted:source.accepted,issues:source.issues)
                            output = try ArenaForensics.alignedImage(copied)
                        }
                        for source in archived.sources where source.accepted {
                            try Task.checkCancellation()
                            let report = try ArenaOutputInspector.compare(archived, selections: checked.selections,
                                sourceID: source.id, maxPixelSize: 256, renderedOutput:output).0
                            guard report.changedPixels == report.selectedFromOtherSourcePixels else {
                                throw ModelInstaller.fail("Export verification found pixels inconsistent with the recorded source. No result was saved.")
                            }
                            differences.append(report)
                        }
                        try encoder.encode(differences).write(to: directory.appendingPathComponent("output-differences.json"), options: .atomic)
                        return manifest
                    } catch { try? FileManager.default.removeItem(at: directory); throw error }
                }
                guard token == generation else { return }
                phase = plan.pendingRegionIDs.isEmpty ? .ready : .partial
                status = "Exported \(manifest.width) × \(manifest.height) result and evidence to \(directory.path)"
            } catch { fail(error, token: token) }
        }
    }
}

private struct ArenaStudioSnapshot: Codable {
    let revision: String
    let records: [ArenaReviewRecord]
    let overrides: [String: String]
    let preferredSource: String?
}
private enum ArenaStudioPersistence {
    static func location(_ analysis: ArenaAnalysis) -> URL {
        var key = SHA256(); key.update(data: Data(ArenaGroupPayload.revision.utf8))
        for source in analysis.sources {
            key.update(data: Data((source.id+"\u{0}"+(source.fileSHA256 ?? "")+"\u{0}").utf8))
        }
        let name = key.finalize().map { String(format: "%02x", $0) }.joined()
        return FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Waifu/PhotoArena/review-" + name + ".json")
    }
}
/// Cancellation propagates into detached image work, including tiled loops.
private func background<T: Sendable>(_ work: @escaping @Sendable () throws -> T) async throws -> T {
    let task = Task.detached(priority: .userInitiated, operation: work)
    return try await withTaskCancellationHandler { try await task.value } onCancel: { task.cancel() }
}
