import AppKit
import Combine
import Foundation

enum UncoverComparison: String, CaseIterable { case sideBySide = "Side by side", wipe = "Wipe", toggle = "Toggle" }

@MainActor
final class UncoverSession: ObservableObject {
    let source: URL
    let sourceDuration: Double
    @Published var roi: UncoverROI = .initial { didSet { inputsChanged() } }
    @Published var scale: UncoverScale = .two { didSet { inputsChanged() } }
    @Published var algorithm: UncoverAlgorithm = .seedVR2 { didSet { if oldValue != algorithm { algorithmChanged() } } }
    @Published var start: Double { didSet { inputsChanged() } }
    @Published var duration: Double = 1 { didSet { inputsChanged() } }
    @Published var feather: Double = 8 { didSet { refresh() } }
    @Published var strength: Double = 1 { didSet { refresh() } }
    @Published var comparison: UncoverComparison = .wipe
    @Published var showOriginal = false
    @Published var showB = true
    @Published var editing = true
    @Published var zoom: Double = 1
    @Published var viewScaleMode: UncoverViewScaleMode = .fitWidth
    @Published private(set) var fitRequest = 0
    @Published var loop = true
    @Published private(set) var playing = false
    @Published private(set) var busy = false
    @Published private(set) var status = "Select a region"
    @Published private(set) var error: String?
    @Published private(set) var original: NSImage?
    @Published private(set) var imageA: NSImage?
    @Published private(set) var imageB: NSImage?
    @Published private(set) var displayedIndex = 0
    @Published private(set) var sourceTimeline: SourceFrameTimeline?
    @Published private(set) var sourceRequestedIndex = 0
    @Published private(set) var sourceDisplayedIndex = 0
    @Published private(set) var readingSource = false
    @Published private(set) var readingResult = false
    @Published private(set) var sourceTime: Double
    @Published private(set) var result: UncoverResult?
    @Published private(set) var capabilities: UpscalerCapabilities?
    @Published private(set) var allowUnsafeMemory: Bool

    private let backendFactory: @Sendable (UncoverAlgorithm) -> any TemporalVideoUpscaler
    private var backend: any TemporalVideoUpscaler { backendFactory(algorithm) }
    private let renderer = UncoverRenderer()
    private var job: Task<Void, Never>?
    private var rendering: Task<Void, Never>?
    private var playback: Task<Void, Never>?
    private var generation = UUID()
    private var renderGeneration = UUID()
    private var resultKey: String?
    private var pendingDirectory: URL?
    private var requestedIndex = 0
    private var capabilityTask: Task<Void, Never>?
    private var sourceTimelineTask: Task<Void, Never>?
    private var sourceReadingTask: Task<Void, Never>?
    private var sourceGeneration = UUID()
    private var preparations: [String: UncoverPrepared] = [:]
    private struct CachedResult { let preparationKey: String; let result: UncoverResult }
    private var results: [String: CachedResult] = [:]
    private var resultOrder: [String] = []
    private var disposed = false

    init(video: VideoInfo, seconds: Double, backend: (any TemporalVideoUpscaler)? = nil,
         backendFactory: (@Sendable (UncoverAlgorithm) -> any TemporalVideoUpscaler)? = nil) {
        source = video.url; sourceDuration = video.durationSeconds
        sourceTime = max(0, min(seconds, video.durationSeconds))
        start = max(0, min(seconds - 0.5, video.durationSeconds - 1))
        self.backendFactory = backendFactory ?? { algorithm in backend ?? algorithm.makeBackend() }
        allowUnsafeMemory = SeedVR2Configuration.load()?.allowUnsafeMemory ?? false
        if PhotosMedia.shared.isStill(source) { algorithm = .realESRGAN; start = 0 }
        updateCapabilities()
        readSource(seconds: sourceTime, initializeRegion: true)
        let source = source
        sourceTimelineTask = Task {
            do {
                let timeline = try await SourceFrameTimeline.load(source: source)
                try Task.checkCancellation()
                guard !disposed else { return }
                sourceTimeline = timeline
                sourceRequestedIndex = timeline.index(at: sourceTime)
                sourceDisplayedIndex = sourceRequestedIndex
            } catch is CancellationError { } catch {
                if !disposed { self.error = error.localizedDescription }
            }
        }
    }

    var isStillPhoto: Bool { PhotosMedia.shared.isStill(source) }
    var availableAlgorithms: [UncoverAlgorithm] {
        UncoverAlgorithm.offered.filter { !isStillPhoto || $0.supportsStillPhotos }
    }
    var sourceFrameCount: Int { sourceTimeline?.timestamps.count ?? 0 }
    var navigatingFrame: Bool { readingSource || readingResult }

    func toggleEditing() {
        pause()
        sourceReadingTask?.cancel(); sourceGeneration = UUID(); readingSource = false
        rendering?.cancel(); renderGeneration = UUID(); readingResult = false
        if editing, hasCurrentResult {
            editing = false
            refresh()
        } else {
            editing = true
            if let timeline = sourceTimeline, !timeline.timestamps.isEmpty {
                let index = timeline.index(at: displayedTime)
                sourceRequestedIndex = index
                readSource(seconds: timeline.timestamps[index], initializeRegion: false)
            }
        }
    }

    func seekSource(index: Int) {
        guard !busy, !disposed, let timeline = sourceTimeline, !timeline.timestamps.isEmpty else { return }
        pause()
        let index = max(0, min(index, timeline.timestamps.count - 1))
        let time = timeline.timestamps[index]
        guard index != sourceRequestedIndex || index != sourceDisplayedIndex || original == nil || start != time else { return }
        sourceRequestedIndex = index
        start = time
        editing = true
        readSource(seconds: time, initializeRegion: original == nil)
    }

    private func readSource(seconds: Double, initializeRegion: Bool) {
        sourceReadingTask?.cancel()
        let token = UUID(); sourceGeneration = token
        readingSource = true
        error = nil
        sourceReadingTask = Task {
            do {
                let image = try await renderer.preview(source: source, seconds: seconds)
                try Task.checkCancellation()
                guard sourceGeneration == token, !disposed else { return }
                original = Self.image(image)
                sourceTime = seconds
                sourceDisplayedIndex = sourceTimeline?.index(at: seconds) ?? 0
                if initializeRegion {
                    let width = min(256, image.width), height = min(256, image.height)
                    roi = .from(pixels: CGRect(x: (image.width - width) / 2, y: (image.height - height) / 2,
                                              width: width, height: height), in: CGSize(width: image.width, height: image.height))
                }
            } catch is CancellationError { } catch {
                if sourceGeneration == token, !disposed { self.error = error.localizedDescription }
            }
            if sourceGeneration == token { readingSource = false }
        }
    }

    var videoSize: CGSize { original?.size ?? .zero }
    var displaySize: CGSize { editing ? videoSize : imageB?.size ?? videoSize }
    var end: Double { min(sourceDuration, start + duration) }
    var hasCurrentResult: Bool { result != nil && resultKey == inputKey }
    var frameCount: Int { result?.prepared.manifest.frames.count ?? 0 }
    var displayedTime: Double {
        guard let frames = result?.prepared.manifest.frames, frames.indices.contains(displayedIndex) else { return start }
        return frames[displayedIndex].timestamp.seconds
    }
    var geometry: String {
        guard let rect = try? roi.pixels(in: videoSize) else { return "Draw a region" }
        let input = scale.degradedSize(rect.size)
        let outputScale = algorithm == .flashVSR11 ? FlashVSRBackend.outputScale : 1
        return "\(Int(rect.width))×\(Int(rect.height)) → \(Int(input.width))×\(Int(input.height)) → \(Int(rect.width) * outputScale)×\(Int(rect.height) * outputScale)"
    }
    var canReconstruct: Bool {
        availableAlgorithms.contains(algorithm) && capabilities?.available == true && capabilities?.scales.contains(scale) == true
            && original != nil && !readingSource && !busy && (allowUnsafeMemory || capabilities?.requiresUnsafeMemoryOptIn == false) && validRegion && withinOutputGeometry && withinMemoryGeometry
            && start.isFinite && duration.isFinite && start >= 0 && start < sourceDuration && duration > 0 && duration <= 5
            && hasEnoughSourceFrames
    }
    private var hasEnoughSourceFrames: Bool {
        guard let sourceTimeline else { return true }
        return sourceTimeline.frameCount(from: start, until: end) >= max(1, capabilities?.minimumFrames ?? 5)
    }
    private var validRegion: Bool {
        (try? roi.pixels(in: videoSize)) != nil
    }
    private var withinOutputGeometry: Bool {
        let outputScale = algorithm == .flashVSR11 ? FlashVSRBackend.outputScale : 1
        return UncoverOutputLimits.accepts(width: Int(videoSize.width), height: Int(videoSize.height), scale: outputScale)
    }
    private var withinMemoryGeometry: Bool {
        guard let rect = try? roi.pixels(in: videoSize) else { return false }
        if let maximum = capabilities?.maximumOutputPixels {
            let width = Int(rect.width), height = Int(rect.height)
            guard height > 0, width <= maximum / height, width * height <= maximum else { return false }
        }
        if let maximum = capabilities?.maximumPaddedArea {
            let outputScale = algorithm == .flashVSR11 ? FlashVSRBackend.outputScale : scale.rawValue
            let width = Int(ceil(rect.width * Double(outputScale) / 128)) * 128
            let height = Int(ceil(rect.height * Double(outputScale) / 128)) * 128
            guard height > 0, width <= maximum / height, width * height < maximum else { return false }
        }
        return true
    }
    var limitation: String? {
        if capabilities?.available != true { return capabilities?.message }
        if !validRegion { return "Select a valid region inside the video." }
        if capabilities?.scales.contains(scale) != true { return "\(algorithm.label) currently supports \((capabilities?.scales ?? []).map { "\($0.rawValue)×" }.joined(separator: ", ")) downsampling." }
        if !withinOutputGeometry { return "The upscaled video would exceed the supported output dimensions. Choose a smaller source video." }
        if !hasEnoughSourceFrames { return "Select an interval with at least \(capabilities?.minimumFrames ?? 5) source frames." }
        if !withinMemoryGeometry {
            if let maximum = capabilities?.maximumOutputPixels {
                let limit = maximum.isMultiple(of: 1_000_000)
                    ? "\(maximum / 1_000_000)-megapixel" : "\(maximum)-pixel"
                return "The selected region exceeds the \(limit) output limit. Select a smaller region."
            }
            return "This region exceeds \(algorithm.label)'s memory budget. Select a smaller region."
        }
        if !allowUnsafeMemory && capabilities?.requiresUnsafeMemoryOptIn == true { return "Enable enlarged-video processing in the backend setup." }
        return nil
    }
    private var preparationKey: String {
        let attributes = try? FileManager.default.attributesOfItem(atPath: source.path)
        return "\(source.path):\(attributes?[.size] ?? 0):\(attributes?[.modificationDate] ?? Date.distantPast):\(roi.x):\(roi.y):\(roi.width):\(roi.height):\(scale.rawValue):\(start):\(end)"
    }
    private var inputKey: String { "\(algorithm.rawValue):\(backend.cacheIdentity):\(preparationKey)" }
    private func updateCapabilities() {
        capabilityTask?.cancel(); capabilities = nil
        let selected = algorithm, engine = backend
        capabilityTask = Task {
            let value = await engine.capabilities()
            guard !Task.isCancelled, selected == algorithm else { return }
            capabilities = value
        }
    }
    func refreshBackend() { updateCapabilities(); inputsChanged() }
    private func algorithmChanged() {
        error = nil
        updateCapabilities(); inputsChanged()
        if !busy { restoreCachedResult() }
        if !busy && !hasCurrentResult { status = "Select a region · \(algorithm.label)" }
    }
    private func restoreCachedResult() {
        guard let cached = results[inputKey] else { return }
        if resultKey == inputKey, result?.directory == cached.result.directory { return }
        result = cached.result; resultKey = inputKey; editing = false; showOriginal = false
        imageA = nil; imageB = nil
        requestedIndex = min(requestedIndex, frameCount - 1)
        status = "Cached · \(algorithm.label) · \(frameCount) frames"
        refresh()
    }

    func reconstruct() {
        guard canReconstruct, ModelLibrary.shared.active?.algorithm != algorithm else { return }
        pause(); error = nil; busy = true
        rendering?.cancel(); renderGeneration = UUID()
        job?.cancel()
        let token = UUID(); generation = token
        let key = inputKey, preparationKey = preparationKey, engine = backend
        let roi = roi, scale = scale, start = start, end = end
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("waifu-uncover-\(UUID().uuidString)")
        pendingDirectory = directory
        let usedAlgorithm = algorithm
        ModelLibrary.shared.acquire(usedAlgorithm)
        job = Task {
            defer {
                ModelLibrary.shared.release(usedAlgorithm)
                if disposed { releaseCaches() }
            }
            do {
                let progress: @Sendable (String) -> Void = { [weak self] message in
                    Task { @MainActor in if self?.generation == token && self?.busy == true { self?.status = message } }
                }
                let reusable: UncoverPrepared
                if let cached = preparations[preparationKey] { reusable = cached }
                else {
                    let preparedDirectory = FileManager.default.temporaryDirectory.appendingPathComponent("waifu-uncover-inputs-\(UUID().uuidString)")
                    do {
                        reusable = try await TemporalFrameExtractor().prepare(source: source, roi: roi, scale: scale,
                            start: start, end: end, directory: preparedDirectory, minimumFrames: capabilities?.minimumFrames ?? 5, progress: progress)
                        try Task.checkCancellation()
                        preparations[preparationKey] = reusable
                    } catch { try? FileManager.default.removeItem(at: preparedDirectory); throw error }
                }
                try Task.checkCancellation()
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                for folder in ["original", "degraded", "baseline"] {
                    try FileManager.default.createSymbolicLink(at: directory.appendingPathComponent(folder),
                        withDestinationURL: reusable.directory.appendingPathComponent(folder))
                }
                try FileManager.default.copyItem(at: reusable.directory.appendingPathComponent("manifest.json"),
                                                to: directory.appendingPathComponent("manifest.json"))
                try FileManager.default.createDirectory(at: directory.appendingPathComponent("reconstructed"), withIntermediateDirectories: true)
                let prepared = UncoverPrepared(directory: directory, manifest: reusable.manifest)
                let reconstructed = try await engine.reconstruct(prepared, allowUnsafeMemory: allowUnsafeMemory, progress: progress)
                try Task.checkCancellation()
                guard generation == token, inputKey == key else { throw CancellationError() }
                status = "Compositing"
                let firstFrame = try await renderer.render(reconstructed, index: 0, feather: feather, strength: strength)
                try Task.checkCancellation()
                guard generation == token, inputKey == key else { throw CancellationError() }
                if let old = results[key]?.result.directory, old != directory { try? FileManager.default.removeItem(at: old) }
                result = reconstructed; resultKey = key; pendingDirectory = nil
                results[key] = CachedResult(preparationKey: preparationKey, result: reconstructed)
                resultOrder.removeAll { $0 == key }; resultOrder.append(key)
                display(firstFrame)
                status = "Ready · \(algorithm.label) · \(prepared.manifest.frames.count) frames · \(String(format: "%.1fs", reconstructed.seconds))"
                busy = false; editing = false; showOriginal = false; requestedIndex = 0
                trimCaches()
            } catch {
                try? FileManager.default.removeItem(at: directory)
                guard generation == token else { return }
                pendingDirectory = nil; busy = false
                if error is CancellationError { status = "Cancelled" }
                else { self.error = error.localizedDescription; status = "Failed" }
                restoreCachedResult(); trimCaches()
            }
        }
    }

    func cancel() { job?.cancel(); pause() }
    func dispose() {
        disposed = true
        generation = UUID(); renderGeneration = UUID()
        job?.cancel(); capabilityTask?.cancel(); rendering?.cancel(); pause()
        sourceTimelineTask?.cancel(); sourceReadingTask?.cancel(); sourceGeneration = UUID()
        // A worker may still read symlinked prepared inputs. Its cancelled job owns
        // cleanup after exit; idle sessions can release those inputs immediately.
        if !busy { releaseCaches() }
        // A pending inference directory is removed by its cancelled job after the
        // subprocess has exited, so it cannot race cleanup with a writer.
    }
    private func inputsChanged() {
        pause()
        rendering?.cancel(); renderGeneration = UUID(); readingResult = false
        if busy { cancel() }
        editing = !hasCurrentResult
        if !busy { restoreCachedResult() }
    }
    private func removePreparations() {
        for prepared in preparations.values { try? FileManager.default.removeItem(at: prepared.directory) }
        preparations.removeAll()
    }
    private func releaseCaches() {
        for cached in results.values { try? FileManager.default.removeItem(at: cached.result.directory) }
        results.removeAll(); resultOrder.removeAll(); removePreparations()
    }
    private func trimCaches() {
        while resultOrder.count > 4 {
            let oldest = resultOrder.removeFirst()
            if let cached = results.removeValue(forKey: oldest) { try? FileManager.default.removeItem(at: cached.result.directory) }
        }
        let needed = Set(results.values.map(\.preparationKey)).union([preparationKey])
        for key in Array(preparations.keys) where !needed.contains(key) {
            if let old = preparations.removeValue(forKey: key) { try? FileManager.default.removeItem(at: old.directory) }
        }
    }
    func seek(index: Int) {
        guard hasCurrentResult, frameCount > 0, !busy else { return }
        let target = max(0, min(index, frameCount - 1))
        guard target != requestedIndex || target != displayedIndex || imageA == nil else { return }
        pause(); requestedIndex = target; refresh()
    }
    func step(_ delta: Int) {
        if editing || !hasCurrentResult { seekSource(index: sourceRequestedIndex + delta) }
        else { seek(index: requestedIndex + delta) }
    }

    private func refresh() {
        guard !playing, !busy, !editing else { return }
        guard hasCurrentResult, let result else { return }
        rendering?.cancel()
        let token = UUID(); renderGeneration = token
        readingResult = true
        let index = requestedIndex, feather = feather, strength = strength
        rendering = Task {
            do {
                let rendered = try await renderer.render(result, index: index, feather: feather, strength: strength)
                try Task.checkCancellation()
                guard renderGeneration == token else { return }
                // Publish all three views and their readout in one main-actor turn.
                display(rendered)
            } catch is CancellationError { } catch {
                if renderGeneration == token { self.error = error.localizedDescription }
            }
            if renderGeneration == token { readingResult = false }
        }
    }

    func togglePlayback() {
        if playing { pause(); return }
        guard !busy, !editing, hasCurrentResult, let result else { return }
        rendering?.cancel(); renderGeneration = UUID()
        readingResult = false
        playing = true
        let frames = result.prepared.manifest.frames
        let first = frames[0].timestamp.seconds
        let last = frames[frames.count - 1]
        let span = last.timestamp.seconds + last.duration.seconds - first
        let origin = ContinuousClock.now
        let initial = frames[displayedIndex].timestamp.seconds - first
        playback = Task {
            while !Task.isCancelled {
                let elapsed = origin.duration(to: .now).components
                var offset = initial + Double(elapsed.seconds) + Double(elapsed.attoseconds) / 1e18
                if offset >= span {
                    if loop { offset.formTruncatingRemainder(dividingBy: span) }
                    else { pause(); break }
                }
                let index = frames.lastIndex(where: { $0.timestamp.seconds - first <= offset }) ?? 0
                requestedIndex = index
                do {
                    let rendered = try await renderer.render(result, index: index, feather: feather, strength: strength)
                    try Task.checkCancellation()
                    display(rendered)
                } catch is CancellationError { break }
                catch { self.error = error.localizedDescription; pause(); break }
                try? await Task.sleep(nanoseconds: 16_000_000)
            }
        }
    }
    func pause() { playback?.cancel(); playback = nil; playing = false }
    func setViewScaleMode(_ mode: UncoverViewScaleMode) {
        guard mode != viewScaleMode else { return }
        viewScaleMode = mode
        zoom = 1
        fitRequest += 1
    }

    func fit() { setViewScaleMode(.fitWidth) }

    private func display(_ rendered: UncoverRenderedFrame) {
        original = Self.image(rendered.original); imageA = original
        imageB = Self.image(rendered.reconstructed); displayedIndex = rendered.index
    }

    func export(to destination: URL) {
        guard hasCurrentResult, let result, !busy else { return }
        pause(); busy = true; error = nil
        let token = UUID(); generation = token
        let feather = feather, strength = strength
        let usedAlgorithm = algorithm
        ModelLibrary.shared.acquire(usedAlgorithm)
        job = Task {
            defer {
                ModelLibrary.shared.release(usedAlgorithm)
                if disposed { releaseCaches() }
            }
            do {
                if isStillPhoto {
                    guard !FileManager.default.fileExists(atPath: destination.path) else { throw UncoverError.invalid("Choose a new output file.") }
                    let frame = try await renderer.render(result, index: 0, feather: feather, strength: strength)
                    try Task.checkCancellation()
                    try RegionImages.write(frame.reconstructed, to: destination)
                } else {
                    try await UncoverVideoExporter().export(result, feather: feather, strength: strength, to: destination) { [weak self] message in
                        Task { @MainActor in if self?.generation == token { self?.status = message } }
                    }
                }
                guard generation == token else { return }
                status = "Exported · \(destination.lastPathComponent)"; busy = false
            } catch {
                guard generation == token else { return }
                busy = false
                if error is CancellationError { status = "Export cancelled" }
                else { self.error = error.localizedDescription; status = "Export failed" }
            }
        }
    }

    private static func image(_ image: CGImage) -> NSImage {
        NSImage(cgImage: image, size: CGSize(width: image.width, height: image.height))
    }
}
