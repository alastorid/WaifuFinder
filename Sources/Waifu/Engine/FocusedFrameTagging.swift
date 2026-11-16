import CoreGraphics
import Foundation

/// Hands off pixels from scoring to WD Tagger with one batch in flight and one
/// batch filling. A slow tagger pauses scoring before decoded frames pile up.
actor FocusedFrameTagging {
    static let preferenceKey = "autoTagFocusedFrames"
    static let defaultEnabled = false

    struct Completion: Sendable {
        let frames: [FrameID: AnalyzedFrame]
        let warning: String?
    }

    private struct Input: Sendable {
        let frame: AnalyzedFrame
        let image: CGImage
    }

    private let backend: any TaggerBackend
    private let vocabulary: TagVocabulary
    private let threshold: Float
    private let isCollection: Bool
    private let identities: [Int: TagResultCacheIdentity]
    private let cached: [TagResultCacheIdentity: [Int: Data]]
    private var seen = Set<FrameID>()
    private var pending: [Input] = []
    private var batchTask: Task<Void, Never>?
    private var tagged: [FrameID: AnalyzedFrame] = [:]
    private var warning: String?
    private var cancelled = false

    init(backend: any TaggerBackend, vocabulary: TagVocabulary, threshold: Float,
         videoURL: URL, analysisWidth: Int) async {
        self.backend = backend
        self.vocabulary = vocabulary
        self.threshold = threshold
        let collection = VideoCollection.get(videoURL)
        isCollection = collection != nil
        let sources = collection?.members.map { $0.video.url } ?? [videoURL]
        let identities = Dictionary(uniqueKeysWithValues: sources.enumerated().compactMap { index, url in
            TagResultCacheIdentity(videoURL: url, vocabulary: vocabulary, analysisWidth: analysisWidth)
                .map { (index, $0) }
        })
        self.identities = identities
        cached = await TagResultMemory.shared.cachedResults(for: Array(Set(identities.values)))
    }

    func collect(_ frame: AnalyzedFrame, image: CGImage) async {
        guard !Task.isCancelled, !cancelled, seen.insert(frame.id).inserted else { return }
        let index = frame.id.sampleIndex
        let sourceIndex = isCollection ? VideoCollection.sourceIndex(index) : 0
        let recordIndex = isCollection ? VideoCollection.localIndex(index) : index
        if let identity = identities[sourceIndex], let data = cached[identity]?[recordIndex],
           let restored = TagResultMemory.restoring(frame, from: data, vocabulary: vocabulary, threshold: threshold) {
            tagged[frame.id] = restored
            return
        }

        while pending.count == WDTaggerBackend.batchSize {
            if let batchTask { await batchTask.value }
            else { startBatch() }
            guard !Task.isCancelled, !cancelled, warning == nil else { return }
        }
        guard warning == nil else { return }
        pending.append(Input(frame: frame, image: image))
        if pending.count == WDTaggerBackend.batchSize, batchTask == nil { startBatch() }
    }

    func finish() async -> Completion {
        await waitForBatch()
        if !Task.isCancelled, !cancelled, warning == nil, !pending.isEmpty {
            startBatch()
            await waitForBatch()
        }
        return Completion(frames: tagged, warning: warning)
    }

    func cancel() async {
        cancelled = true
        pending.removeAll()
        batchTask?.cancel()
        await batchTask?.value
    }

    private func waitForBatch() async {
        guard let batchTask else { return }
        await withTaskCancellationHandler {
            await batchTask.value
        } onCancel: {
            batchTask.cancel()
        }
    }

    private func startBatch() {
        let batch = pending
        pending.removeAll(keepingCapacity: true)
        batchTask = Task {
            defer { batchTask = nil }
            do {
                let results = try await backend.tag(batch: batch.map(\.image))
                try Task.checkCancellation()
                guard results.count == batch.count else {
                    throw ModelInstaller.fail("The tagger returned an unexpected frame count.")
                }
                for (input, result) in zip(batch, results) {
                    tagged[input.frame.id] = try result.applying(to: input.frame, fingerprint: vocabulary.cacheFingerprint)
                }
            } catch {
                if !Task.isCancelled { warning = error.localizedDescription }
                pending.removeAll()
            }
        }
    }
}
