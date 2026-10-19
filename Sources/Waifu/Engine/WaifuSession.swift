import AVFoundation
import AppKit
import Combine
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// Selection query for the pre-sort, pre-cap grid candidates. The key contains
/// every setting that can change membership, so a cached result is never reused
/// across an incompatible filter query.
private struct DisplayFilterCacheKey: Hashable {
    let requiredTags: Set<String>
    let excludedTags: Set<String>
    let thresholdBits: UInt32
    let faceFilter: FrameFaceFilter
    let limits: FrameLimits
    let minimumFaceAreaBits: UInt32
    let vocabularyFingerprint: UInt64?
}

/// The tagger intentionally uses only these four numeric ranges; rating and
/// tag-score ranges depend on tagger output and do not control which frames it tags.
private struct TaggingCandidateCacheKey: Hashable {
    let faceFilter: FrameFaceFilter
    let time: FrameLimit
    let face: FrameLimit
    let art: FrameLimit
    let definition: FrameLimit
    let minimumFaceAreaBits: UInt32
}

private struct FrameLimitBoundsCacheKey: Equatable {
    let faceFilter: FrameFaceFilter
    let minimumFaceAreaBits: UInt32
}

private struct FrameIndexCacheEntry {
    let indices: [Int]
    let untaggedCount: Int?
}

private struct TaggingCandidateSelection {
    let key: TaggingCandidateCacheKey
    let frames: [AnalyzedFrame]
    let untaggedCount: Int
}

/// Small LRU of compact sample-index lists. Compatible entries can supply a
/// superset which is narrowed without revisiting the whole video.
private struct FrameIndexResultCache<Key: Hashable> {
    private(set) var values: [Key: FrameIndexCacheEntry] = [:]
    private var mostRecent: [Key] = []
    private let capacity = 12

    mutating func exact(_ key: Key) -> FrameIndexCacheEntry? {
        guard let result = values[key] else { return nil }
        touch(key)
        return result
    }

    mutating func compatible(where predicate: (Key) -> Bool,
                             prefer: (Key, Key) -> Bool) -> (key: Key, entry: FrameIndexCacheEntry)? {
        var bestKey: Key?
        for key in Array(mostRecent.reversed()) where predicate(key) {
            guard values[key] != nil else { continue }
            if let incumbent = bestKey {
                if prefer(key, incumbent) { bestKey = key }
            } else {
                bestKey = key
            }
        }
        guard let bestKey, let entry = values[bestKey] else { return nil }
        touch(bestKey)
        return (bestKey, entry)
    }

    mutating func store(_ indices: [Int], untaggedCount: Int? = nil, for key: Key) {
        values[key] = FrameIndexCacheEntry(indices: indices, untaggedCount: untaggedCount)
        touch(key)
        while mostRecent.count > capacity {
            let evicted = mostRecent.removeFirst()
            values.removeValue(forKey: evicted)
        }
    }

    mutating func removeAll() {
        values.removeAll(keepingCapacity: true)
        mostRecent.removeAll(keepingCapacity: true)
    }

    mutating func invalidateMetadata() {
        for key in Array(values.keys) {
            guard let entry = values[key] else { continue }
            values[key] = FrameIndexCacheEntry(indices: entry.indices, untaggedCount: nil)
        }
    }

    private mutating func touch(_ key: Key) {
        mostRecent.removeAll { $0 == key }
        mostRecent.append(key)
    }
}

private func rangeIsSuperset(_ broader: FrameLimit, of narrower: FrameLimit) -> Bool {
    let broaderLow = broader.low ?? -Double.infinity
    let narrowerLow = narrower.low ?? -Double.infinity
    let broaderHigh = broader.high ?? Double.infinity
    let narrowerHigh = narrower.high ?? Double.infinity
    return broaderLow <= narrowerLow && broaderHigh >= narrowerHigh
}

private func faceFilterIsSuperset(_ broader: FrameFaceFilter, of narrower: FrameFaceFilter) -> Bool {
    broader == .all || broader == narrower
}

private func frameLimitsAreSuperset(_ broader: FrameLimits, of narrower: FrameLimits) -> Bool {
    FrameSortKey.allCases.allSatisfy { key in
        rangeIsSuperset(broader.limit(for: key), of: narrower.limit(for: key))
    }
}

private struct VisualOrderInput: Equatable, Sendable {
    let id: FrameID
    let featurePrint: Data?
}

private struct TagSummaryInput: Equatable, Sendable {
    let ids: Set<FrameID>
    let revision: UInt64
    let threshold: Float
}

/// Frame IDs are the identity used by both the grid and its caches. Keep the
/// first occurrence if an upstream refresh ever repeats one, so a duplicate
/// cannot be counted twice, rendered twice, or trap a dictionary initializer.
private func uniqueFramesByID(_ frames: [AnalyzedFrame]) -> [AnalyzedFrame] {
    var seen = Set<FrameID>()
    seen.reserveCapacity(frames.count)
    return frames.filter { seen.insert($0.id).inserted }
}

/// The single source of truth for the UI. Every published property is read from a
/// `View.body`, so the session is `@MainActor` isolated — the long-running work
/// happens off the main actor and only publishes results back here.
@MainActor
final class WaifuSession: ObservableObject, ThumbnailStore {

    // MARK: Configuration

    @Published var settings: WaifuSettings = .default {
        didSet {
            persistSettings()
            // `perGroupFrameCount` sizes the per-group cap and the grouping knobs
            // decide which faces count, so a settings change can change what the
            // grid shows.
            recomputeDisplayedFrames()
        }
    }

    // MARK: Video state

    @Published private(set) var video: VideoInfo?
    @Published private(set) var progress: AnalysisProgress = .init()
    @Published private(set) var isRegrouping = false
    let regroupProgress = RegroupProgress()

    let taggingStatus = TaggingStatus()
    private(set) var taggingProgress = TaggingProgress() {
        didSet {
            taggingStatus.update(taggingProgress)
            // Start/stop changes which controls are shown. Per-batch counts do
            // not change the grid, group list, filters, or selection.
            if oldValue.isRunning != taggingProgress.isRunning || oldValue.origin != taggingProgress.origin {
                objectWillChange.send()
            }
        }
    }
    @Published private(set) var matchedFrameCountForTagging = 0
    @Published private(set) var untaggedFrameCountForTagging = 0
    @Published private(set) var taggedFrameCount = 0
    @Published private(set) var restoringTagMemory = false

    // MARK: Analysis results

    @Published private(set) var frames: [AnalyzedFrame] = [] {
        didSet {
            visualOrderBaseFrames = nil
            invalidateTagSummaries()
            displayFilterResults.removeAll()
            let isTagUpdate = incrementallyTaggedFrames != nil
            if isTagUpdate {
                // Tagging changes metadata, not membership or score filters.
                taggingCandidateResults.invalidateMetadata()
            } else {
                // Live Photos scoring grows the frame set (often from just one
                // photo). Cached indices and the demand snapshot must see it.
                taggingCandidateResults.removeAll()
                lastTaggingCandidateKey = nil
            }
            cachedLimitBoundsKey = nil
            cachedLimitBounds = nil
            if let incrementallyTaggedFrames {
                for frame in incrementallyTaggedFrames { frameByID[frame.id] = frame }
                self.incrementallyTaggedFrames = nil
            } else {
                frameByID = Dictionary(uniqueKeysWithValues: frames.map { ($0.id, $0) })
            }
            taggedFrameCount = frames.reduce(into: 0) { count, frame in
                if frame.hasTaggingResult { count += 1 }
            }
            if !taggingProgress.isRunning || !isTagUpdate { recomputeDisplayedFrames() }
        }
    }
    private var frameByID: [FrameID: AnalyzedFrame] = [:]
    private var displayFilterResults = FrameIndexResultCache<DisplayFilterCacheKey>()
    private var taggingCandidateResults = FrameIndexResultCache<TaggingCandidateCacheKey>()
    private var cachedLimitBoundsKey: FrameLimitBoundsCacheKey?
    private var cachedLimitBounds: FrameLimitBounds?
    private let tagSummaryComputer = TagSummaryComputer()
    private var tagSummaryRevision: UInt64 = 0
    private var tagSummaryGeneration = UUID()
    private var tagSummaryTask: Task<TagSummary.Labels?, Never>?
    private var pendingTagSummaryInput: TagSummaryInput?
    private var completedTagSummaryInput: TagSummaryInput?
    private var completedTagSummaryLabels: TagSummary.Labels?
    private var incrementallyTaggedFrames: [AnalyzedFrame]?
    private var presentedGridNavigation: FrameNavigationOrder?
    private var previewNavigation: FrameNavigationOrder?

    private var navigationOrder: FrameNavigationOrder {
        previewNavigation ?? presentedGridNavigation
            ?? FrameNavigationOrder(frames: displayOrderFrames, rows: displayedRows, groupRows: usesGroupRows)
    }

    func commitGridNavigation(_ order: FrameNavigationOrder, generation: UUID) {
        guard generation == thumbnailGeneration else { return }
        presentedGridNavigation = order
    }
    private var visualOrderTask: Task<[FrameID]?, Never>?
    private var visualOrderGeneration = UUID()
    private var pendingVisualOrderInput: [VisualOrderInput]?
    private var completedVisualOrderInput: [VisualOrderInput]?
    private var visualOrderIDs: [FrameID] = []
    /// Score-ranked, filtered and capped photos before visual sequencing. A
    /// visual toggle/reversal changes only their order, so it must not decode
    /// every tagging vector or rebuild the tag summaries on the UI actor.
    private var visualOrderBaseFrames: [AnalyzedFrame]?
    @Published private(set) var isSortingVisualDistance = false
    private var rowPosition: [FrameID: (row: Int, position: Int)] = [:]
    private var rowHeightRatios: [String: Double] = [:]
    /// Changes only when the grid's arrangement or layout inputs change. Selection,
    /// focus, and preview updates must not force a large LazyStack to diff every row.
    private(set) var gridRevision: UInt64 = 0
    private let gridLayoutSubject = CurrentValueSubject<UInt64, Never>(0)
    /// Layout consumers must read after every stored layout field and revision
    /// have committed, including the final asynchronous visual-sort result.
    var gridLayoutUpdates: AnyPublisher<UInt64, Never> {
        // A grid can mount (or renew its subscription) after an album update.
        // Replay the current revision rather than waiting for another mutation.
        gridLayoutSubject.removeDuplicates().receive(on: DispatchQueue.main).eraseToAnyPublisher()
    }
    private var defersVisualOrder = false

    private func invalidateGridLayout() {
        gridRevision &+= 1
        Diag.trace(.grid, message: "layout revision=\(gridRevision) analyzed=\(frames.count) displayed=\(displayOrderFrames.count) rows=\(displayedRows.count) loading=\(progress.isRunning)")
        gridLayoutSubject.send(gridRevision)
    }
    private(set) var thumbnailGeneration = UUID()
    @Published private(set) var groups: [FrameGroup] = []
    @Published private(set) var selectedFrameIDs: Set<FrameID> = []
    private var selectionAnchor: FrameID?
    @Published private(set) var isSyncingPhotosOrder = false
    @Published private(set) var photosOrderSyncMessage: String?

    // MARK: UI state

    @Published var errorMessage: String?
    @Published var focusedFrameID: FrameID?
    private let gridScrollSubject = PassthroughSubject<FrameID, Never>()
    private(set) var gridNavigationFrameID: FrameID?

    /// An explicit navigation event, with no initial/replayed value when a lazy
    /// row appears. Pointer selection must never generate a scrolling command.
    var gridScrollRequests: AnyPublisher<FrameID, Never> {
        gridScrollSubject.receive(on: DispatchQueue.main).eraseToAnyPublisher()
    }

    func focusFromPointer(_ id: FrameID) {
        gridNavigationFrameID = nil
        focusedFrameID = id
    }

    private func requestGridScroll(to id: FrameID) {
        gridNavigationFrameID = id
        gridScrollSubject.send(id)
    }
    @Published var showExportSheet: Bool = false
    @Published private(set) var uncover: UncoverSession?

    func openUncover(at seconds: Double) {
        guard let video else { return }
        uncover?.dispose()
        previewFrame = nil
        if PhotosMedia.isPhotos(video.url) {
            let (source, time) = PhotosMedia.shared.resolve(video.url, seconds: seconds)
            guard let item = PhotosMedia.shared.item(source) else { return }
            let info = VideoInfo(url: source, durationSeconds: item.duration, width: item.asset.pixelWidth,
                                 height: item.asset.pixelHeight, frameRate: item.isStill ? 1 : 30,
                                 estimatedFrameCount: item.isStill ? 1 : Int(item.duration * 30))
            uncover = UncoverSession(video: info, seconds: time, backendFactory: remoteBackendFactory)
        } else {
            uncover = UncoverSession(video: video, seconds: seconds, backendFactory: remoteBackendFactory)
        }
    }

    /// Returns a remote backend for algorithms the connected server offers, and
    /// the local one for everything else.
    ///
    /// A per-algorithm choice rather than an all-or-nothing one, so switching to
    /// an algorithm the server does not have simply runs locally instead of
    /// failing. That is also why the Reconstruct button's ᴿ mark can appear and
    /// disappear as the user changes the picker.
    private var remoteBackendFactory: (@Sendable (UncoverAlgorithm) -> any TemporalVideoUpscaler)? {
        let client = RemoteClient.shared
        guard client.status.isConnected else { return nil }
        // Resolved here, on the main actor, and captured as an immutable set:
        // the factory itself runs off-actor and must not reach back into
        // main-actor state to ask what the server offers.
        let available = Set((client.capabilities?.uncover ?? [])
            .filter(\.available).map(\.algorithm))
        return { algorithm in
            available.contains(algorithm.rawValue)
                ? RemoteUncoverBackend(algorithm: algorithm, client: client)
                : algorithm.makeBackend()
        }
    }

    func closeUncover() {
        uncover?.dispose()
        uncover = nil
    }

    /// Cached video stem for synchronous thumbnail access from non-isolated contexts.
    /// Updated on the main actor when a video is loaded; read from `thumbnailData`.
    nonisolated(unsafe) private var currentVideoStem: String?

    // MARK: Grid arrangement

    /// Visual continuity, followed by independent score signals for representatives and ties.
    @Published var frameOrder: FrameOrder = .defaultOrder {
        didSet {
            guard frameOrder != oldValue else { return }
            var previous = oldValue
            previous.minimalVisualDistance = frameOrder.minimalVisualDistance
            if previous == frameOrder {
                recomputeVisualOrder()
            } else {
                recomputeDisplayedFrames()
            }
        }
    }

    /// Filters Photos items by the app's private "Sync order" marker.
    @Published var photosOrderFilter: PhotosOrderFilter = .all {
        didSet { recomputeDisplayedFrames() }
    }

    /// All frames, frames with significant faces, or frames without them.
    @Published var faceFilter: FrameFaceFilter = .all {
        didSet { recomputeDisplayedFrames() }
    }

    @Published var tagFilter = TagFilter() {
        didSet { recomputeDisplayedFrames() }
    }

    @Published private(set) var tagVocabulary: TagVocabulary?

    /// Low/high limits per signal, dragged in the panel.
    @Published var frameLimits: FrameLimits = .none {
        didSet { recomputeDisplayedFrames() }
    }

    /// Whether the grid draws **one group per line** or one grid of everything.
    ///
    /// Two ways of looking at the same ordered list, so it is view state: nothing
    /// about the arrangement changes, only how the same frames are laid out.
    @Published var groupRows: Bool = true {
        didSet { invalidateGridLayout() }
    }

    /// Whether the analysis' groups participate in the grid arrangement. The group
    /// assignments stay cached while off, so turning grouping back on is immediate.
    @Published var groupingEnabled: Bool = false {
        didSet { recomputeDisplayedFrames() }
    }

    /// Write to Photos first; private markers must never claim an unsuccessful
    /// library edit was synced. Keep the user's ranking/grouping controls intact.
    func syncPhotosOrder(forPhotoIdentifiers identifiers: [String]) {
        guard !isSyncingPhotosOrder, !identifiers.isEmpty, let albumURL = photosAlbumURL,
              let albumIdentifier = PhotosMedia.shared.collectionIdentifier(for: albumURL) else { return }
        isSyncingPhotosOrder = true
        photosOrderSyncMessage = nil
        Task {
            defer { isSyncingPhotosOrder = false }
            do {
                try await Task.detached(priority: .userInitiated) {
                    try await PhotosMedia.shared.syncOrder(collectionIdentifier: albumIdentifier,
                                                           assetIdentifiers: identifiers)
                }.value
                PhotosOrderState.shared.markOrdered(albumIdentifier: albumIdentifier,
                                                     assetIdentifiers: identifiers)
                if photosAlbumURL == albumURL {
                    photosOrderSyncMessage = "Order synced to Photos · \(identifiers.count) photos"
                    recomputeDisplayedFrames()
                }
            } catch {
                errorMessage = "Couldn’t sync order to Photos: \(error.localizedDescription)"
            }
        }
    }

    func syncPhotosOrder(for frame: AnalyzedFrame) {
        guard let albumURL = photosAlbumURL else { return }
        let targetFrames: [AnalyzedFrame]
        if selectedFrameIDs.count > 1 && selectedFrameIDs.contains(frame.id) {
            targetFrames = displayOrderFrames.filter { selectedFrameIDs.contains($0.id) }
        } else {
            targetFrames = displayOrderFrames
        }
        let identifiers = targetFrames.compactMap {
            PhotosMedia.shared.assetIdentifier(for: albumURL, seconds: $0.timeSeconds)
        }
        syncPhotosOrder(forPhotoIdentifiers: identifiers)
    }

    func resetPhotosOrderState() {
        guard let albumURL = photosAlbumURL,
              let albumIdentifier = PhotosMedia.shared.collectionIdentifier(for: albumURL) else { return }
        PhotosOrderState.shared.reset(albumIdentifier: albumIdentifier)
        recomputeDisplayedFrames()
    }

    /// A global visual path must stay continuous across group boundaries.
    var usesGroupRows: Bool { groupingEnabled && groupRows && !frameOrder.minimalVisualDistance.enabled }

    /// The target of the zoomable frame preview. Set by Space in the grid.
    @Published var previewFrame: AnalyzedFrame? {
        didSet {
            if previewFrame == nil { previewNavigation = nil }
            if oldValue != nil, previewFrame == nil, let id = focusedFrameID {
                requestGridScroll(to: id)
            }
        }
    }

    /// UI effects must see the stored state that matches the publication.
    /// @Published emits in willSet. ScrollViewProxy can synchronously render the
    /// window, so delivering directly lets it consume invalidation with old state.
    var previewFrameUpdates: AnyPublisher<AnalyzedFrame?, Never> {
        $previewFrame.receive(on: DispatchQueue.main).eraseToAnyPublisher()
    }

    /// Groups whose per-group cap the user has lifted, so the grid shows every
    /// frame of theirs rather than the best `settings.perGroupFrameCount`.
    @Published var revealedGroups: Set<FrameGroupID> = [] {
        didSet { recomputeDisplayedFrames() }
    }

    /// The frames the grid renders: `frames` filtered, ordered, and capped per
    /// group by the current arrangement.
    ///
    /// **Stored, not computed.** It is read from `View.body`, and an order can rank
    /// every analysed frame — thousands of them — so recomputing on each body
    /// evaluation would put that work in the middle of a scroll. It is refreshed
    /// wherever its inputs change instead: `frames`, `settings`, and the arrangement
    /// fields (`frameOrder`, `faceFilter`, `revealedGroups`).
    @Published private(set) var displayedFrames: [AnalyzedFrame] = []

    /// Traits present on the frames currently shown in the grid, sorted for the
    /// wrapping tag filter. The selected tags are kept separately by `TagFilter`.
    @Published private(set) var tagFacetTags: [String] = []
    /// Tags present on every frame currently shown in the grid.
    @Published private(set) var tagCommonTags: [String] = []
    @Published private(set) var isSummarizingTags = false

    /// The same frames split into grid lines — one per group, plus a final line for
    /// the frames in no group. Used when `groupRows` is on.
    @Published private(set) var displayedRows: [FrameRow] = []

    /// The global visual path when enabled; otherwise groups stay together and
    /// `groupRows` decides whether they occupy separate lines.
    @Published private(set) var displayOrderFrames: [AnalyzedFrame] = []

    /// The span each signal's slider covers, measured from the frames the face
    /// filter admits. Recomputed with the arrangement, never while a thumb moves.
    @Published private(set) var limitBounds = FrameLimitBounds(time: 0...1, art: -1...1, face: nil)

    /// Frames past the per-group cap, per group.
    ///
    /// **Nothing was discarded to produce this.** Every analysed frame is still in
    /// memory; the grid is simply not drawing the ones beyond the cap until asked.
    @Published private(set) var beyondCapByGroup: [FrameGroupID: Int] = [:]
    private(set) var cappedGroupSummaries: [CappedGroupSummary] = []

    /// How many frames the per-group cap is *currently* keeping out of the grid —
    /// the counts for groups the user has not revealed.
    var hiddenFrameCount: Int {
        beyondCapByGroup.reduce(0) { total, entry in
            total + (revealedGroups.contains(entry.key) ? 0 : entry.value)
        }
    }

    /// Every frame the per-group cap puts past the limit, revealed or not.
    var beyondCapFrameCount: Int { beyondCapByGroup.values.reduce(0, +) }

    // MARK: Thumbnail access

    /// Fast, synchronous lookup for callers that can use an already-cached image.
    nonisolated func cachedThumbnailData(forKey key: String) -> Data? {
        guard let stem = currentVideoStem else { return nil }
        return thumbnailCache.thumbnailDataSync(forKey: key, videoStem: stem)
    }

    /// Returns an in-memory thumbnail, regenerating it from the source video after
    /// LRU eviction. Grid-cell tasks request only materialized cells, so recent
    /// accesses track the viewport and SwiftUI's nearby lazy cells.
    func thumbnailData(for frame: AnalyzedFrame, retainRegeneratedThumbnail: Bool = true) async -> Data? {
        guard !Task.isCancelled, let video else { return nil }
        let sourceURL = video.url
        let stem = sourceURL.deletingPathExtension().lastPathComponent
        let generation = thumbnailGeneration
        // This nonisolated async cache read runs on the generic executor. Even
        // cache bookkeeping and lock waits must not consume the scroll frame.
        let cached = await thumbnailCache.thumbnailData(
            forKey: frame.thumbnailKey, videoStem: stem, touchLRU: retainRegeneratedThumbnail,
            allowPrefixMatch: false
        )
        guard !Task.isCancelled, thumbnailGeneration == generation,
              self.video?.url == sourceURL else { return nil }
        if let cached { return cached }

        let requestKey = generation.uuidString + ":" + frame.thumbnailKey
        let request: ThumbnailRequest
        if let existing = thumbnailRequests[requestKey] {
            var promoted = existing
            promoted.retainAfterCompletion = promoted.retainAfterCompletion || retainRegeneratedThumbnail
            thumbnailRequests[requestKey] = promoted
            request = promoted
        } else {
            let requestID = UUID()
            // Photos grid/analysis thumbnails use 512px renditions, regardless
            // of the video analysis-width setting. Still-photo tagging requests
            // its own model-sized rendition through TaggingImageLoader.
            let width = PhotosMedia.isPhotos(sourceURL)
                ? PhotosMedia.analysisPixelSize
                : settings.analysisWidth
            let task = Task { [weak self] () -> Data? in
                let data = try? await ThumbnailRegenerator.shared.jpegData(
                    for: sourceURL, timeSeconds: frame.timeSeconds, analysisWidth: width
                )
                guard let self else { return nil }
                defer {
                    if self.thumbnailRequests[requestKey]?.id == requestID {
                        self.thumbnailRequests.removeValue(forKey: requestKey)
                    }
                }
                guard self.thumbnailGeneration == generation,
                      self.video?.url == sourceURL,
                      self.frameByID[frame.id]?.thumbnailKey == frame.thumbnailKey else { return nil }
                if let data, self.thumbnailRequests[requestKey]?.retainAfterCompletion == true {
                    try? self.thumbnailCache.store(data, forKey: frame.thumbnailKey, videoStem: stem)
                }
                return data
            }
            request = ThumbnailRequest(id: requestID, task: task,
                                       retainAfterCompletion: retainRegeneratedThumbnail)
            thumbnailRequests[requestKey] = request
        }

        let data = await request.task.value
        return Task.isCancelled ? nil : data
    }

    /// ThumbnailStore serves both full frame keys and the stable export key. If
    /// an export key missed the cache, resolve it to the current analyzed frame
    /// and rebuild the thumbnail in memory.
    func thumbnailData(forKey key: String) async -> Data? {
        if let cached = cachedThumbnailData(forKey: key) { return cached }
        guard key.hasPrefix("thumb-"),
              let rawID = key.dropFirst("thumb-".count).split(separator: "-").first,
              let id = FrameID(raw: String(rawID)),
              let frame = frameByID[id] else { return nil }
        return await thumbnailData(for: frame)
    }

    // MARK: Private

    private var remoteVocabularyChanges: AnyCancellable?
    private var modelChanges: AnyCancellable?
    private var tagMemorySettingChanges: AnyCancellable?
    private var analysisMemorySettingChanges: AnyCancellable?
    private var terminationObserver: AnyCancellable?
    private var analysisTask: Task<Void, Never>?
    private var analysisGeneration = UUID()
    private var taggingTask: Task<Void, Never>?
    private var taggingGeneration = UUID()
    private var taggingDemandActive = false
    private var taggingDemandGeneration = UUID()
    private var lastTaggingOrigin: TaggingPanel = .tags
    private var lastTaggingCandidateIDs: Set<FrameID>?
    private var taggingQueue: [FrameID] = []
    private var taggingQueueIndex = 0
    private var taggingQueuedIDs = Set<FrameID>()
    private var pendingTaggingRestoreFrames: [FrameID: AnalyzedFrame] = [:]
    private var taggingCandidateRestoreTask: Task<Void, Never>?
    private var taggingCandidateRestoreToken = UUID()
    private var tagMemoryRestoreOperations = Set<UUID>()
    private var pendingTaggedFrames: [FrameID: AnalyzedFrame] = [:]
    private var tagMemoryCancellationFlushPending = false
    private var taggingCacheKeys = TagResultCacheKeys.none
    private var taggingVocabularyFingerprint: UInt64?
    private var lastTaggingCandidateKey: TaggingCandidateCacheKey?
    private var tagMemoryRestoreToken = UUID()
    private var analysisCacheIdentity: AnalysisResultCacheIdentity?
    nonisolated private let thumbnailCache = ThumbnailCache()
    private struct ThumbnailRequest {
        let id: UUID
        let task: Task<Data?, Never>
        var retainAfterCompletion: Bool
    }
    private var thumbnailRequests: [String: ThumbnailRequest] = [:]
    private let settingsURL: URL
    /// A supplied settings file keeps session-level regression fixtures isolated.
    init(settingsURL: URL? = nil, vocabulary: TagVocabulary? = nil) {
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        let file = settingsURL ?? support.appendingPathComponent("Waifu", isDirectory: true)
            .appendingPathComponent("settings.json")
        let dir = file.deletingLastPathComponent()
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        self.settingsURL = file
        loadSettings()
        tagVocabulary = vocabulary ?? RemoteInferencePool.shared.remoteVocabulary ?? TagVocabulary.loadAvailable()
        remoteVocabularyChanges = RemoteInferencePool.shared.$remoteVocabulary
            .dropFirst()
            .receive(on: DispatchQueue.main)
            .sink { [weak self] vocabulary in
                MainActor.assumeIsolated {
                    guard let self, !self.taggingProgress.isRunning else { return }
                    guard let updated = vocabulary ?? TagVocabulary.loadAvailable() else { return }
                    self.tagVocabulary = updated
                    self.clearTagResultsNotMatching(updated.cacheFingerprint)
                    self.invalidateTagSummaries()
                    self.recomputeDisplayedFrames()
                    self.restoreRememberedTagResults()
                }
            }
        modelChanges = NotificationCenter.default.publisher(for: .waifuModelsChanged)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                MainActor.assumeIsolated {
                    let previousFingerprint = self?.tagVocabulary?.cacheFingerprint
                    let updatedVocabulary = RemoteInferencePool.shared.remoteVocabulary ?? TagVocabulary.loadAvailable()
                    self?.tagVocabulary = updatedVocabulary
                    if let updatedVocabulary,
                       previousFingerprint != updatedVocabulary.cacheFingerprint {
                        self?.clearTagResultsNotMatching(updatedVocabulary.cacheFingerprint)
                    }
                    self?.invalidateTagSummaries()
                    self?.recomputeDisplayedFrames()
                    self?.restoreRememberedTagResults()
                }
            }
        tagMemorySettingChanges = TagResultMemory.shared.$rememberTaggingResults
            .dropFirst()
            .receive(on: DispatchQueue.main)
            .sink { [weak self] enabled in
                MainActor.assumeIsolated {
                    guard enabled else { return }
                    self?.restoreRememberedTagResults()
                }
            }
        analysisMemorySettingChanges = AnalysisResultMemory.shared.$rememberAnalysisResults
            .dropFirst()
            .receive(on: DispatchQueue.main)
            .sink { [weak self] enabled in
                MainActor.assumeIsolated {
                    guard enabled else { return }
                    self?.persistAnalysisResultsForCurrentVideo()
                }
            }
        terminationObserver = NotificationCenter.default.publisher(for: NSApplication.willTerminateNotification)
            .sink { [weak self] _ in
                MainActor.assumeIsolated {
                    self?.persistTagResultsAtExit()
                    self?.persistAnalysisResultsAtExit()
                }
            }
        // The live knob starts where the persisted rule left off.
        groupingThreshold = settings.similarityThreshold
    }

    // MARK: Public API

    /// Loads a video file, clears prior state, and starts the analysis pipeline.
    @Published private(set) var photosAlbumURL: URL?
    private var importedMedia: PreparedMedia?

    func loadVideo(at url: URL) {
        loadVideos(at: [url])
    }

    /// Imports a drop as one continuous video, preserving the supplied order.
    func loadVideos(at urls: [URL]) {
        guard let url = urls.first else { return }
        Diag.trace(.analysis, message: "load source photos=\(PhotosMedia.shared.album(url)?.items.count ?? 0)")
        closeUncover()
        cancelTagging(keepingCompletedResults: false)
        cancel()
        clearResults()
        errorMessage = nil

        video = nil
        photosAlbumURL = PhotosMedia.shared.album(url) == nil ? nil : url
        photosOrderSyncMessage = nil
        let preparedSource = photosAlbumURL.map { PreparedMedia(url: $0) }
        if let preparedSource { importedMedia = preparedSource }

        let stem = url.deletingPathExtension().lastPathComponent
        currentVideoStem = urls.count > 1 ? "\(stem) and \(urls.count - 1) more" : stem

        progress = AnalysisProgress(phase: .probing, processed: 0, total: 1, elapsed: 0, message: urls.count > 1 ? "Preparing \(urls.count) videos…" : PhotosMedia.isPhotos(url) ? "Reading Photos album…" : (url.isFileURL ? "Probing video…" : "Downloading and preparing media…"))

        let token = analysisGeneration, config = settings.validated()
        analysisTask = Task.detached { [weak self] in
            await self?.runAnalysisPipeline(for: url, existing: nil, settings: config, token: token, preparedSource: preparedSource, sourceURLs: urls)
        }
    }

    func refreshPhotosAlbum(at url: URL) {
        let limits = frameLimits
        loadVideo(at: url)
        frameLimits = limits
    }

    /// An album removed in Photos must not leave exportable stale results.
    func clearPhotosAlbum() {
        guard photosAlbumURL != nil else { return }
        closeUncover()
        cancelTagging(keepingCompletedResults: false)
        cancel()
        clearResults()
        video = nil
        photosAlbumURL = nil
        importedMedia = nil
        photosOrderSyncMessage = nil
        errorMessage = nil
        progress = AnalysisProgress()
    }

    /// Re-runs the analysis under current settings without re-probing.
    func analyze() {
        guard video != nil else { return }
        cancelTagging(keepingCompletedResults: false)
        cancel()
        clearResults()
        errorMessage = nil

        progress = AnalysisProgress(phase: .sampling, processed: 0, total: 1, elapsed: 0, message: "Decoding and scoring every frame…")

        let token = analysisGeneration, config = settings.validated(), existing = video
        analysisTask = Task.detached { [weak self] in
            await self?.runAnalysisPipeline(for: nil, existing: existing, settings: config, token: token)
        }
    }

    /// Cancels any running analysis.
    func cancel() {
        analysisTask?.cancel()
        analysisGeneration = UUID()
        analysisTask = nil
        isRegrouping = false
        if progress.phase != .complete && progress.phase != .idle {
            progress.phase = .cancelled
            progress.message = "Cancelled"
        }
        if defersVisualOrder {
            defersVisualOrder = false
            recomputeVisualOrder()
        }
    }

    /// Starts tagging the frames admitted by the active non-tagging filters.
    /// The caller records which of the two tagging panels owns the progress card.
    private var taggingIncludesLocal = false

    func startTagging(from panel: TaggingPanel, includeLocal: Bool = false) async {
        guard !taggingProgress.isRunning, !restoringTagMemory,
              !tagMemoryCancellationFlushPending, let video else { return }
        // A connected server supplies both the weights and the labels, so neither
        // a local installation nor a local vocabulary is required. This gate used
        // to be the only way in, and it would otherwise make a remote tagger
        // unreachable from a machine that has no model of its own.
        let remote = RemoteInferencePool.shared
        let configuration = WDTaggerConfiguration.load()
        let localVocabulary = configuration.flatMap { try? TagVocabulary.load(from: $0.labelsURL) }
        let remoteVocabulary = await remote.loadVocabularyIfNeeded()
        guard configuration != nil || remoteVocabulary != nil else {
            errorMessage = "Install WD Tagger in Settings, or connect to a server that has it."
            return
        }
        let vocabulary: TagVocabulary
        if remote.offersRemoteTagging, let remoteVocabulary {
            vocabulary = remoteVocabulary
        } else if let localVocabulary {
            vocabulary = localVocabulary
        } else {
            errorMessage = "No tag vocabulary is available. Reconnect to the server."
            return
        }
        if includeLocal && remote.offersRemoteTagging {
            guard let localVocabulary, configuration != nil else {
                errorMessage = "Install WD Tagger on this Mac to use remote and local inference together."
                return
            }
            guard localVocabulary.cacheFingerprint == vocabulary.cacheFingerprint else {
                errorMessage = "Local and remote tag labels must match to tag together."
                return
            }
        }
        taggingIncludesLocal = includeLocal
        tagVocabulary = vocabulary
        clearTagResultsNotMatching(vocabulary.cacheFingerprint)
        invalidateTagSummaries()
        recomputeDisplayedFrames()

        let cacheKeys = TagResultCacheKeys.make(videoURL: video.url, frames: frames, vocabulary: vocabulary,
                                                analysisWidth: settings.analysisWidth,
                                                analysisMode: settings.analysisMode)
        taggingCacheKeys = cacheKeys
        taggingVocabularyFingerprint = vocabulary.cacheFingerprint
        if !cacheKeys.isEmpty {
            let restoreToken = UUID()
            tagMemoryRestoreToken = restoreToken
            beginTagMemoryRestore(restoreToken)
            let token = analysisGeneration
            let snapshot = frames
            let restored = await TagResultMemory.shared.restore(
                snapshot,
                keys: cacheKeys,
                vocabulary: vocabulary,
                threshold: settings.taggerThreshold
            )
            guard tagMemoryRestoreToken == restoreToken else { return }
            endTagMemoryRestore(restoreToken)
            guard token == analysisGeneration, self.video?.url == video.url,
                  !taggingProgress.isRunning else { return }
            if restored.restoredCount > 0 {
                frames = restored.frames
                invalidateTagSummaries()
                recomputeDisplayedFrames()
                await TagResultMemory.shared.remember(frames: frames, keys: cacheKeys)
            }
        }

        let candidates = taggingCandidates().frames
        let newCandidates = candidates.filter { !$0.hasTaggingResult }
        guard !newCandidates.isEmpty else { return }

        taggingDemandGeneration = UUID()
        taggingDemandActive = true
        lastTaggingOrigin = panel
        lastTaggingCandidateIDs = Set(candidates.map(\.id))
        taggingQueue = newCandidates.map(\.id)
        taggingQueueIndex = 0
        taggingQueuedIDs = Set(taggingQueue)
        taggingGeneration = UUID()
        taggingProgress = TaggingProgress(isRunning: true, origin: panel,
                                          processed: 0, total: taggingQueue.count, elapsed: 0)
        launchTaggingTask(config: configuration, remote: remoteVocabulary != nil ? remote : nil,
                          token: taggingGeneration, cacheKeys: cacheKeys,
                          vocabularyFingerprint: vocabulary.cacheFingerprint)
    }

    /// Stops the active run. Completed batches stay tagged and can be reused.
    func cancelTagging(keepingCompletedResults: Bool = true) {
        let keys = taggingCacheKeys
        let snapshot = frames + Array(pendingTaggedFrames.values)
        taggingDemandActive = false
        taggingDemandGeneration = UUID()
        if keepingCompletedResults { publishPendingTaggedFrames() }
        else { pendingTaggedFrames.removeAll() }
        taggingGeneration = UUID()
        let cancelledTask = taggingTask
        cancelledTask?.cancel()
        taggingTask = nil
        taggingQueue = []
        taggingQueueIndex = 0
        taggingQueuedIDs.removeAll()
        if taggingProgress.isRunning {
            taggingProgress.isRunning = false
            if keepingCompletedResults { recomputeDisplayedFrames() }
        }
        if let cancelledTask {
            tagMemoryCancellationFlushPending = true
            Task { [weak self] in
                await cancelledTask.value
                guard let self else { return }
                let framesToRemember = self.taggingCacheKeys == keys
                    ? self.frames + Array(self.pendingTaggedFrames.values)
                    : snapshot
                self.tagMemoryCancellationFlushPending = false
                await TagResultMemory.shared.remember(frames: framesToRemember, keys: keys)
            }
        } else {
            scheduleTagMemorySync(keys: keys, frames: snapshot)
        }
    }

    /// Chooses the backend for a tagging run.
    ///
    /// Snapshot compatible servers for this run. Option-start also reserves a local
    /// model, and follow-up batches preserve that choice.
    private func launchTaggingTask(config: WDTaggerConfiguration?, remote: RemoteInferencePool?, token: UUID,
                                   cacheKeys: TagResultCacheKeys,
                                   vocabularyFingerprint: UInt64?) {
        let threshold = settings.taggerThreshold
        let clients = remote?.taggingClients(matching: vocabularyFingerprint) ?? []
        let localConfig = clients.isEmpty || taggingIncludesLocal ? config : nil
        taggingTask = Task.detached(priority: .userInitiated) { [weak self] in
            guard let self else { return }
            await self.runTaggingPipeline(config: localConfig, clients: clients,
                                          threshold: threshold, token: token,
                                          cacheKeys: cacheKeys,
                                          vocabularyFingerprint: vocabularyFingerprint)
        }
    }

    private nonisolated func runTaggingPipeline(config: WDTaggerConfiguration?, clients: [RemoteClient],
                                                threshold: Float, token: UUID,
                                                cacheKeys: TagResultCacheKeys,
                                                vocabularyFingerprint: UInt64?) async {
        let started = Date()
        var acquired = false
        do {
            var lanes = clients.map {
                BalancedTaggerBackend.Lane(backend: RemoteTaggerBackend(client: $0, threshold: threshold,
                    vocabularyFingerprint: vocabularyFingerprint), capacity: RemoteLimits.taggingBatchWindow)
            }
            if let config {
                let localVocabulary = try TagVocabulary.load(from: config.labelsURL)
                guard localVocabulary.cacheFingerprint == vocabularyFingerprint else {
                    throw ModelInstaller.fail("Local and remote tag labels must match to tag together.")
                }
                await ModelLibrary.shared.acquireTagger()
                acquired = true
                lanes.append(.init(backend: try WDTaggerBackend(directory: config.directory,
                    generalThreshold: threshold), capacity: 1))
            }
            guard !lanes.isEmpty else { throw ModelInstaller.fail("No tagging server or local model is available.") }
            let tagger = BalancedTaggerBackend(lanes: lanes)
            let next: @Sendable () async throws -> [PreparedTagFrame]? = {
                guard let batch = try await self.nextTaggingBatch(token: token) else { return nil }
                let images = try await TaggingImageLoader.load(batch.map(\.source))
                return zip(batch, images).map { PreparedTagFrame(frame: $0.frame, image: $1) }
            }
            let whenEmpty: @Sendable () async -> Bool = {
                await self.finishTaggingIfDrained(token: token, elapsed: Date().timeIntervalSince(started))
            }
            let process: @Sendable ([PreparedTagFrame]) async throws -> [AnalyzedFrame] = { batch in
                try Task.checkCancellation()
                let results = try await tagger.tag(batch: batch.map(\.image))
                guard results.count == batch.count else {
                    throw ModelInstaller.fail("The tagger returned an unexpected frame count.")
                }

                var updates: [AnalyzedFrame] = []
                updates.reserveCapacity(batch.count)
                for (item, result) in zip(batch, results) {
                    guard let encoded = TagProbabilityCodec.encode(result.probabilities) else {
                        throw ModelInstaller.fail("The tagger returned invalid probabilities.")
                    }
                    var frame = item.frame
                    frame.tagProbabilities = encoded
                    frame.tagVocabularyFingerprint = vocabularyFingerprint
                    frame.tagRating = result.rating
                    frame.topTags = result.topTags
                    frame.tagGeneralScore = result.generalScore
                    updates.append(frame)
                }
                return updates
            }
            let commit: @Sendable ([AnalyzedFrame]) async -> Void = { updates in
                await self.acceptTaggedFrames(updates, token: token, elapsed: Date().timeIntervalSince(started),
                                              cacheKeys: cacheKeys)
            }
            if !clients.isEmpty {
                try await TaggingBatchPipeline.runPipelined(maximumInFlight: tagger.maximumInFlight,
                    commitInOrder: false, next: next, whenEmpty: whenEmpty, process: process, commit: commit)
            } else {
                try await TaggingBatchPipeline.run(next: next, whenEmpty: whenEmpty, consume: { batch in
                    await commit(try await process(batch))
                })
            }
        } catch {
            let elapsed = Date().timeIntervalSince(started)
            await failTagging(error, token: token, elapsed: elapsed)
        }
        if acquired { await ModelLibrary.shared.releaseTagger() }
    }

    private func nextTaggingBatch(token: UUID) async throws -> [PendingTagFrame]? {
        guard token == taggingGeneration, taggingProgress.isRunning, let video else { return nil }
        var batch: [PendingTagFrame] = []
        while taggingQueueIndex < taggingQueue.count && batch.count < WDTaggerBackend.batchSize {
            let id = taggingQueue[taggingQueueIndex]
            taggingQueueIndex += 1
            guard let frame = frameByID[id], !frame.hasTaggingResult else { continue }
            if let photoURL = PhotosMedia.shared.photoItemURL(for: video.url, seconds: frame.timeSeconds) {
                // Ask PhotoKit for model-sized pixels; do not encode and decode
                // the scoring/grid JPEG or shrink its 512px rendition again.
                batch.append(PendingTagFrame(frame: frame, source: .photo(photoURL)))
                continue
            }
            guard let data = await thumbnailData(for: frame, retainRegeneratedThumbnail: false) else {
                try Task.checkCancellation()
                guard token == taggingGeneration, taggingProgress.isRunning else { return nil }
                throw ModelInstaller.fail("A cached frame could not be found for tagging.")
            }
            try Task.checkCancellation()
            guard token == taggingGeneration, taggingProgress.isRunning else { return nil }
            batch.append(PendingTagFrame(frame: frame, source: .thumbnail(data)))
        }
        return batch
    }

    /// Returns true when candidates were added while the worker drained its queue.
    private func finishTaggingIfDrained(token: UUID, elapsed: TimeInterval) -> Bool {
        guard token == taggingGeneration, taggingProgress.isRunning else { return false }
        if taggingQueueIndex < taggingQueue.count { return true }
        publishPendingTaggedFrames()
        taggingProgress.elapsed = elapsed
        taggingProgress.isRunning = false
        taggingTask = nil
        taggingQueue = []
        taggingQueueIndex = 0
        taggingQueuedIDs.removeAll()
        recomputeDisplayedFrames()
        scheduleTagMemorySync(keys: taggingCacheKeys, frames: frames)
        return false
    }

    private func acceptTaggedFrames(_ updates: [AnalyzedFrame], token: UUID, elapsed: TimeInterval,
                                    cacheKeys: TagResultCacheKeys) async {
        guard token == taggingGeneration, taggingProgress.isRunning else {
            if taggingCacheKeys == cacheKeys {
                for frame in updates { pendingTaggedFrames[frame.id] = frame }
                publishPendingTaggedFrames()
                recomputeDisplayedFrames()
            }
            return
        }
        for frame in updates { pendingTaggedFrames[frame.id] = frame }
        taggingProgress.processed += updates.count
        taggingProgress.elapsed = elapsed
        if pendingTaggedFrames.count >= 1_024 { publishPendingTaggedFrames() }
    }

    private func publishPendingTaggedFrames() {
        guard !pendingTaggedFrames.isEmpty else { return }
        let updates = pendingTaggedFrames.values.sorted { $0.id.sampleIndex < $1.id.sampleIndex }
        pendingTaggedFrames.removeAll(keepingCapacity: true)
        var replacements = Dictionary(uniqueKeysWithValues: updates.map { ($0.id, $0) })
        var updatedFrames = frames
        for index in updatedFrames.indices {
            if let replacement = replacements.removeValue(forKey: updatedFrames[index].id) {
                updatedFrames[index] = replacement
            }
        }
        incrementallyTaggedFrames = updates
        frames = updatedFrames
    }

    private func failTagging(_ error: Error, token: UUID, elapsed: TimeInterval) {
        guard token == taggingGeneration else { return }
        taggingDemandActive = false
        taggingDemandGeneration = UUID()
        publishPendingTaggedFrames()
        taggingProgress.isRunning = false
        taggingProgress.elapsed = elapsed
        taggingTask = nil
        taggingQueue = []
        taggingQueueIndex = 0
        taggingQueuedIDs.removeAll()
        pendingTaggedFrames.removeAll()
        recomputeDisplayedFrames()
        scheduleTagMemorySync(keys: taggingCacheKeys, frames: frames)
        if !(error is CancellationError) { errorMessage = error.localizedDescription }
    }

    /// Album tagging covers every photo, independently of grid filters and caps.
    /// Video tagging follows non-tagging filters; rating and traits never gate it.
    private func taggingCandidates(_ grouping: GroupingSettings? = nil) -> TaggingCandidateSelection {
        let grouping = grouping ?? settings.grouping.validated()
        let candidateLimits = photosAlbumURL == nil ? frameLimits : .none
        let candidateFaceFilter = photosAlbumURL == nil ? faceFilter : .all
        let key = TaggingCandidateCacheKey(faceFilter: candidateFaceFilter,
                                           time: candidateLimits.time,
                                           face: candidateLimits.face,
                                           art: candidateLimits.art,
                                           definition: candidateLimits.definition,
                                           minimumFaceAreaBits: grouping.minimumFaceAreaFraction.bitPattern)
        // Frame replacement briefly passes through an empty set; never cache it
        // under the active filter key for the next analysis to reuse.
        guard !frames.isEmpty else {
            return TaggingCandidateSelection(key: key, frames: [], untaggedCount: 0)
        }
        if let exact = taggingCandidateResults.exact(key) {
            let candidates = frames(forCachedIndices: exact.indices)
            let untaggedCount = exact.untaggedCount ?? candidates.reduce(into: 0) { count, frame in
                if !frame.hasTaggingResult { count += 1 }
            }
            taggingCandidateResults.store(exact.indices, untaggedCount: untaggedCount, for: key)
            return TaggingCandidateSelection(key: key, frames: candidates, untaggedCount: untaggedCount)
        }

        let reusable = taggingCandidateResults.compatible { cached in
            cached.minimumFaceAreaBits == key.minimumFaceAreaBits
                && faceFilterIsSuperset(cached.faceFilter, of: key.faceFilter)
                && rangeIsSuperset(cached.time, of: key.time)
                && rangeIsSuperset(cached.face, of: key.face)
                && rangeIsSuperset(cached.art, of: key.art)
                && rangeIsSuperset(cached.definition, of: key.definition)
        } prefer: { candidate, incumbent in
            if candidate.faceFilter == key.faceFilter, incumbent.faceFilter != key.faceFilter { return true }
            if incumbent.faceFilter == key.faceFilter, candidate.faceFilter != key.faceFilter { return false }
            let candidateIsNarrower = rangeIsSuperset(incumbent.time, of: candidate.time)
                && rangeIsSuperset(incumbent.face, of: candidate.face)
                && rangeIsSuperset(incumbent.art, of: candidate.art)
                && rangeIsSuperset(incumbent.definition, of: candidate.definition)
            let incumbentIsNarrower = rangeIsSuperset(candidate.time, of: incumbent.time)
                && rangeIsSuperset(candidate.face, of: incumbent.face)
                && rangeIsSuperset(candidate.art, of: incumbent.art)
                && rangeIsSuperset(candidate.definition, of: incumbent.definition)
            return candidateIsNarrower && !incumbentIsNarrower
        }
        let source = reusable.map { frames(forCachedIndices: $0.entry.indices) } ?? frames
        let candidates = source.filter { frame in
            key.faceFilter.admits(frame, grouping: grouping)
                && candidateLimits.admitsForTagging(frame, grouping: grouping)
        }
        let untaggedCount = candidates.reduce(into: 0) { count, frame in
            if !frame.hasTaggingResult { count += 1 }
        }
        taggingCandidateResults.store(candidates.map(\.id.sampleIndex), untaggedCount: untaggedCount, for: key)
        return TaggingCandidateSelection(key: key, frames: candidates, untaggedCount: untaggedCount)
    }

    private func filteredDisplayCandidates(grouping: GroupingSettings) -> [AnalyzedFrame] {
        let key = DisplayFilterCacheKey(
            requiredTags: Set(tagFilter.requiredTags.map(TagVocabulary.normalized)),
            excludedTags: Set(tagFilter.excludedTags.map(TagVocabulary.normalized)),
            thresholdBits: settings.taggerThreshold.bitPattern,
            faceFilter: faceFilter,
            limits: frameLimits,
            minimumFaceAreaBits: grouping.minimumFaceAreaFraction.bitPattern,
            vocabularyFingerprint: tagVocabulary?.cacheFingerprint
        )
        if let exact = displayFilterResults.exact(key) {
            return frames(forCachedIndices: exact.indices)
        }

        let reusable = displayFilterResults.compatible { cached in
            cached.thresholdBits == key.thresholdBits
                && cached.minimumFaceAreaBits == key.minimumFaceAreaBits
                && cached.vocabularyFingerprint == key.vocabularyFingerprint
                && cached.requiredTags.isSubset(of: key.requiredTags)
                && cached.excludedTags.isSubset(of: key.excludedTags)
                && faceFilterIsSuperset(cached.faceFilter, of: key.faceFilter)
                && frameLimitsAreSuperset(cached.limits, of: key.limits)
        } prefer: { candidate, incumbent in
            if candidate.requiredTags.count != incumbent.requiredTags.count {
                return candidate.requiredTags.count > incumbent.requiredTags.count
            }
            if candidate.faceFilter == key.faceFilter, incumbent.faceFilter != key.faceFilter { return true }
            if incumbent.faceFilter == key.faceFilter, candidate.faceFilter != key.faceFilter { return false }
            let candidateIsNarrower = frameLimitsAreSuperset(incumbent.limits, of: candidate.limits)
            let incumbentIsNarrower = frameLimitsAreSuperset(candidate.limits, of: incumbent.limits)
            return candidateIsNarrower && !incumbentIsNarrower
        }
        let source = reusable.map { frames(forCachedIndices: $0.entry.indices) } ?? frames
        let filter = TagFilter(requiredTags: key.requiredTags, excludedTags: key.excludedTags,
                               threshold: Float(bitPattern: key.thresholdBits))
        let candidates = source.filter { frame in
            key.faceFilter.admits(frame, grouping: grouping)
                && key.limits.admits(frame, grouping: grouping)
                && filter.admits(frame, vocabulary: tagVocabulary)
        }
        displayFilterResults.store(candidates.map(\.id.sampleIndex), for: key)
        return candidates
    }

    private func frames(forCachedIndices indices: [Int]) -> [AnalyzedFrame] {
        indices.compactMap { frameByID[FrameID(sampleIndex: $0)] }
    }

    private func appendTaggingCandidates(_ additions: [AnalyzedFrame]) {
        guard taggingDemandActive, let sourceURL = video?.url else { return }
        let newFrames = additions.filter { frame in
            !frame.hasTaggingResult && taggingQueuedIDs.insert(frame.id).inserted
        }
        guard !newFrames.isEmpty else { return }

        for frame in newFrames { pendingTaggingRestoreFrames[frame.id] = frame }
        guard taggingCandidateRestoreTask == nil else { return }

        let restoreToken = UUID()
        taggingCandidateRestoreToken = restoreToken
        let analysisToken = analysisGeneration
        let demandToken = taggingDemandGeneration
        beginTagMemoryRestore(restoreToken)
        taggingCandidateRestoreTask = Task { [weak self] in
            await self?.restorePendingTaggingCandidates(token: restoreToken,
                                                        analysisToken: analysisToken,
                                                        demandToken: demandToken,
                                                        sourceURL: sourceURL)
        }
    }

    private func restorePendingTaggingCandidates(token: UUID, analysisToken: UUID,
                                                 demandToken: UUID, sourceURL: URL) async {
        defer {
            if taggingCandidateRestoreToken == token {
                for frame in pendingTaggingRestoreFrames.values { taggingQueuedIDs.remove(frame.id) }
                pendingTaggingRestoreFrames.removeAll(keepingCapacity: true)
                taggingCandidateRestoreTask = nil
                endTagMemoryRestore(token)
            }
        }

        while !Task.isCancelled, taggingCandidateRestoreToken == token,
              !pendingTaggingRestoreFrames.isEmpty {
            let candidates = pendingTaggingRestoreFrames.values.sorted { $0.id.sampleIndex < $1.id.sampleIndex }
            pendingTaggingRestoreFrames.removeAll(keepingCapacity: true)
            guard let video, video.url == sourceURL, let vocabulary = tagVocabulary else {
                for frame in candidates { taggingQueuedIDs.remove(frame.id) }
                continue
            }

            let keys = TagResultCacheKeys.make(videoURL: video.url, frames: frames,
                                               vocabulary: vocabulary,
                                               analysisWidth: settings.analysisWidth,
                                               analysisMode: settings.analysisMode)
            taggingCacheKeys = keys
            taggingVocabularyFingerprint = vocabulary.cacheFingerprint
            let restored = await TagResultMemory.shared.restore(
                candidates,
                keys: keys,
                vocabulary: vocabulary,
                threshold: settings.taggerThreshold
            )
            guard !Task.isCancelled, taggingCandidateRestoreToken == token,
                  analysisGeneration == analysisToken, self.video?.url == sourceURL,
                  tagVocabulary?.cacheFingerprint == vocabulary.cacheFingerprint else {
                for frame in candidates { taggingQueuedIDs.remove(frame.id) }
                return
            }

            var updatedFrames = frames
            let frameIndices = Dictionary(uniqueKeysWithValues: updatedFrames.enumerated().map { ($0.element.id, $0.offset) })
            var didRestore = false
            for restoredFrame in restored.frames where restoredFrame.hasTaggingResult {
                guard let index = frameIndices[restoredFrame.id],
                      !updatedFrames[index].hasTaggingResult else { continue }
                updatedFrames[index] = restoredFrame
                didRestore = true
            }
            if didRestore { frames = updatedFrames }

            let canQueue = taggingDemandActive && taggingDemandGeneration == demandToken
            var untaggedFrames: [AnalyzedFrame] = []
            for candidate in candidates {
                guard let current = frameByID[candidate.id], !current.hasTaggingResult else {
                    taggingQueuedIDs.remove(candidate.id)
                    continue
                }
                if canQueue {
                    taggingQueuedIDs.insert(candidate.id)
                    untaggedFrames.append(current)
                } else {
                    taggingQueuedIDs.remove(candidate.id)
                }
            }
            enqueueTaggingFrames(untaggedFrames)
        }

    }

    private func enqueueTaggingFrames(_ newFrames: [AnalyzedFrame]) {
        guard !newFrames.isEmpty, taggingDemandActive else {
            for frame in newFrames { taggingQueuedIDs.remove(frame.id) }
            return
        }

        if taggingProgress.isRunning {
            taggingQueue.append(contentsOf: newFrames.map(\.id))
            taggingProgress.total += newFrames.count
        } else {
            // Follow-up batches for frames that arrived mid-run. The backend is
            // chosen the same way the first run chose it, including the remote
            // case, so a run does not silently switch machines halfway through.
            let config = WDTaggerConfiguration.load()
            let remote = RemoteInferencePool.shared.offersRemoteTagging ? RemoteInferencePool.shared : nil
            guard config != nil || remote != nil else {
                for frame in newFrames { taggingQueuedIDs.remove(frame.id) }
                return
            }
            taggingQueue.append(contentsOf: newFrames.map(\.id))
            taggingProgress = TaggingProgress(isRunning: true, origin: lastTaggingOrigin,
                                              processed: 0, total: newFrames.count, elapsed: 0)
            taggingGeneration = UUID()
            launchTaggingTask(config: config, remote: remote, token: taggingGeneration,
                              cacheKeys: taggingCacheKeys,
                              vocabularyFingerprint: taggingVocabularyFingerprint)
        }
        scheduleTagMemorySync(keys: taggingCacheKeys, frames: frames)
    }

    /// IDs in the same order the user sees, with each group contiguous and
    /// ungrouped frames after the groups.
    private var selectionOrder: [FrameID] {
        navigationOrder.frames.map(\.id)
    }

    func selectFrame(_ id: FrameID, command: Bool = false, shift: Bool = false) {
        // Only a range selection needs the full visual order.
        if !shift {
            if command { select(id) }
            else { selectedFrameIDs = [id]; selectionAnchor = id }
            return
        }
        var selection = FrameSelection(ids: selectedFrameIDs, anchor: selectionAnchor)
        selection.click(id, order: selectionOrder, command: command, shift: shift)
        selectedFrameIDs = selection.ids
        selectionAnchor = selection.anchor
    }

    func selectAllDisplayedFrames() {
        guard uncover == nil, previewFrame == nil else { return }
        var selection = FrameSelection(ids: selectedFrameIDs, anchor: selectionAnchor)
        selection.selectAll(order: selectionOrder, focused: focusedFrameID)
        selectedFrameIDs = selection.ids
        selectionAnchor = selection.anchor
    }

    /// Toggles a frame's selection from its context menu.
    func select(_ frame: FrameID) {
        selectionAnchor = frame
        if selectedFrameIDs.contains(frame) {
            selectedFrameIDs.remove(frame)
        } else {
            selectedFrameIDs.insert(frame)
        }
    }

    /// Selects all frames in a group.
    func selectAll(in group: FrameGroup) {
        for id in group.memberIDs {
            if let fid = FrameID(raw: id) { selectedFrameIDs.insert(fid) }
        }
    }

    /// Clears the selection.
    func clearSelection() {
        selectedFrameIDs.removeAll()
        selectionAnchor = nil
    }

    /// Looks up a frame by identifier.
    func frame(id: FrameID) -> AnalyzedFrame? {
        frameByID[id]
    }

    func rowID(for id: FrameID) -> String? {
        guard let location = rowPosition[id] else { return nil }
        return displayedRows[location.row].id
    }

    func rowHeightRatio(for id: String) -> Double { rowHeightRatios[id] ?? 9.0 / 16 }

    /// Returns the frames belonging to a group, in time order.
    func members(of group: FrameGroup) -> [AnalyzedFrame] {
        group.memberIDs.compactMap { raw in
            FrameID(raw: raw).flatMap { frameByID[$0] }
        }
    }

    /// Returns the cached thumbnail for a frame, or nil if not yet available.
    func thumbnail(for frame: AnalyzedFrame) -> NSImage? {
        guard let data = cachedThumbnailData(forKey: frame.thumbnailKey) else { return nil }
        return NSImage(data: data)
    }

    /// Exports the currently selected frames.
    func exportSelection(to directory: URL) async throws -> [ExportResult] {
        let ids = selectedFrameIDs.sorted { $0.sampleIndex < $1.sampleIndex }
        return try await exportFrames(ids, to: directory)
    }

    /// Exports specific frames by identifier.
    func exportFrames(_ ids: [FrameID], to directory: URL) async throws -> [ExportResult] {
        guard let video = video else { throw ExportError.sourceUnreadable(directory) }
        let timestamps = Dictionary(uniqueKeysWithValues: frames.map { ($0.id, $0.timeSeconds) })
        let request = ExportRequest(frameIDs: ids, destinationDirectory: directory, format: settings.exportFormat,
                                    jpegQuality: settings.exportJPEGQuality, frameTimes: timestamps)
        let exporter = FrameExporter(thumbnailStore: self)
        var results: [ExportResult] = []
        for try await result in exporter.export(request, sourceURL: video.url) {
            results.append(result)
        }
        return results
    }

    // MARK: Pipeline

    /// Decode, score, accumulate and group away from the UI actor. Only small
    /// progress values cross actors during the pass; the grid receives one result.
    private nonisolated func runAnalysisPipeline(for url: URL?, existing: VideoInfo?,
                                                settings: WaifuSettings, token: UUID, preparedSource: PreparedMedia? = nil, sourceURLs: [URL]? = nil) async {
        let started = Date()
        var progress = AnalysisProgress(phase: .probing, total: 1, message: "Reading video…")
        var analyzed: [AnalyzedFrame] = []
        var cacheIdentity: AnalysisResultCacheIdentity?
        do {
            let videoInfo: VideoInfo
            if let url {
                let prepared = if let preparedSource { preparedSource } else { try await MediaImport.prepare(sourceURLs ?? [url]) }
                videoInfo = try await VideoProbe.probe(prepared.url)
                try Task.checkCancellation()
                guard await acceptVideo(videoInfo, token: token, prepared: prepared) else { return }
            } else if let existing { videoInfo = existing }
            else { throw AnalysisError.noVideo }

            let album = PhotosMedia.shared.album(videoInfo.url)
            let analysisWidth = album == nil ? settings.analysisWidth : PhotosMedia.analysisPixelSize
            cacheIdentity = AnalysisResultCacheIdentity(videoURL: videoInfo.url,
                                                        analysisWidth: analysisWidth,
                                                        minimumFaceAreaFraction: settings.minimumFaceAreaFraction, analysisMode: settings.analysisMode)
            await acceptAnalysisCacheIdentity(cacheIdentity, token: token)
            let cachedAnalysisResults: [Int: Data] = if let cacheIdentity {
                await AnalysisResultMemory.shared.cachedResults(for: cacheIdentity)
            } else {
                [:]
            }

            let mode = settings.analysisMode
            progress.phase = .scoring
            // The keyframe pass cannot know how many sync samples the file
            // holds, so it reports progress as a loop rather than as a
            // fraction of a count it never had. Full and Quick both compute
            // their totals exactly from the probe and stay determinate.
            progress.isIndeterminate = album == nil && mode == .keyframe
            progress.total = album?.items.count ?? max(1, mode.analysedFrameCount(frameCount: videoInfo.estimatedFrameCount,
                                                         frameRate: videoInfo.frameRate,
                                                         durationSeconds: videoInfo.durationSeconds))
            progress.message = switch mode {
            case .keyframe: "Scoring and grouping keyframes…"
            case .quick: "Scoring and grouping one frame a second…"
            case .full: "Scoring and grouping every frame…"
            }
            await acceptProgress(progress, token: token)
            let sampled = AVFoundationFrameSampler().sample(videoInfo.url, analysisWidth: analysisWidth,
                                                             mode: mode, onProgress: { _ in })
            let writer = ThumbnailWriterBridge(cache: thumbnailCache,
                                                videoStem: videoInfo.url.deletingPathExtension().lastPathComponent)
            let pipeline = FrameScorePipeline(scorer: VisionFrameScorer(minimumFaceAreaFraction: settings.minimumFaceAreaFraction),
                                              thumbnailStore: writer, analysisWidth: analysisWidth,
                                              workerCount: FrameProcessingConcurrency(
                                                width: album?.items.allSatisfy(\.isStill) == true ? analysisWidth : videoInfo.width,
                                                height: album?.items.allSatisfy(\.isStill) == true ? analysisWidth : videoInfo.height,
                                                analysisWidth: analysisWidth).scoringWorkers,
                                              cacheSuffix: "-\(token.uuidString)",
                                              cachedResults: cachedAnalysisResults)
            let scoredFrames = pipeline.run(frames: sampled,
                                            minimumFaceAreaFraction: settings.minimumFaceAreaFraction,
                                            onProgress: { _ in })
            let grouping = IncrementalFrameGrouper(settings: settings.grouping)
            var lastPulse = started
            var lastPhotosUpdate = started
            if album != nil { PhotosMedia.shared.resetSkippedItems(videoInfo.url) }
            for try await frame in scoredFrames {
                try Task.checkCancellation()
                analyzed.append(frame)
                // The scoring producer continues its bounded Vision workers
                // while this background consumer groups each ordered result.
                try grouping.append(frame)
                progress.processed += 1
                progress.groupedFrames = grouping.processed
                progress.groupsIdentified = grouping.linkedGroups
                progress.facesProcessed += frame.faceCount
                if frame.hasSignificantFace { progress.framesWithFaces += 1 }
                let now = Date()
                // Start with a useful batch rather than mounting a singleton
                // grid that is immediately replaced by the rest of the album.
                // Small albums publish in full through completeAnalysis below.
                if album != nil && analyzed.count >= 64 && now.timeIntervalSince(lastPhotosUpdate) >= 0.75 {
                    let liveGroups = try grouping.snapshot()
                    let liveFrames = Self.assignGroups(liveGroups, to: analyzed)
                    await acceptLivePhotos(frames: liveFrames, groups: liveGroups, token: token)
                    lastPhotosUpdate = now
                }
                if now.timeIntervalSince(lastPulse) >= 0.25 {
                    progress.elapsed = now.timeIntervalSince(started)
                    await acceptProgress(progress, token: token)
                    lastPulse = now
                }
            }
            try Task.checkCancellation()
            progress.phase = .grouping
            // Grouping knows its own total — it is the number of frames the
            // scoring pass just produced — so a keyframe pass becomes
            // determinate the moment it gets here.
            progress.isIndeterminate = false
            progress.total = progress.processed
            progress.elapsed = Date().timeIntervalSince(started)
            progress.message = "Finishing groups…"
            await acceptProgress(progress, token: token)
            let groups = try grouping.finish()
            try Task.checkCancellation()
            var groupByMember: [String: FrameGroupID] = [:]
            for group in groups {
                for member in group.memberIDs { groupByMember[member] = group.id }
            }
            let finalFrames = analyzed.map { frame -> AnalyzedFrame in
                var copy = frame
                copy.groupID = groupByMember[frame.id.raw]
                return copy
            }
            progress.phase = .complete
            progress.groupsIdentified = groups.count
            progress.groupedFrames = analyzed.count
            progress.elapsed = Date().timeIntervalSince(started)
            let skipped = album == nil ? 0 : PhotosMedia.shared.skippedItemCount(videoInfo.url)
            progress.message = skipped == 0 ? "Complete" : "Complete · \(skipped) items unavailable locally"
            if let cacheIdentity {
                await AnalysisResultMemory.shared.remember(identity: cacheIdentity, frames: finalFrames)
            }
            await completeAnalysis(frames: finalFrames, groups: groups, progress: progress, token: token)
        } catch {
            if PhotosMedia.shared.album(url ?? existing?.url ?? URL(fileURLWithPath: "/")) != nil, !analyzed.isEmpty {
                await acceptLivePhotos(frames: analyzed, groups: [], token: token)
            }
            if let cacheIdentity, !analyzed.isEmpty {
                await AnalysisResultMemory.shared.remember(identity: cacheIdentity, frames: analyzed)
            }
            progress.phase = error is CancellationError ? .cancelled : .failed
            progress.elapsed = Date().timeIntervalSince(started)
            progress.message = error is CancellationError ? "Cancelled" : "Failed: \(error.localizedDescription)"
            await failAnalysis(progress: progress, error: error, token: token)
        }
    }

    nonisolated private static func assignGroups(_ groups: [FrameGroup], to frames: [AnalyzedFrame]) -> [AnalyzedFrame] {
        var membership: [String: FrameGroupID] = [:]
        for group in groups { for id in group.memberIDs { membership[id] = group.id } }
        return frames.map { frame in
            var copy = frame; copy.groupID = membership[frame.id.raw]; return copy
        }
    }

    /// Keep tagger output and selection stable as the scored prefix grows.
    private func preservingLiveTags(_ incoming: [AnalyzedFrame]) -> [AnalyzedFrame] {
        incoming.map { frame in
            guard let previous = frameByID[frame.id], previous.hasTaggingResult else { return frame }
            var copy = frame
            copy.tagProbabilities = previous.tagProbabilities
            copy.tagVocabularyFingerprint = previous.tagVocabularyFingerprint
            copy.tagRating = previous.tagRating
            copy.topTags = previous.topTags
            copy.tagGeneralScore = previous.tagGeneralScore
            return copy
        }
    }
    private func acceptLivePhotos(frames: [AnalyzedFrame], groups: [FrameGroup], token: UUID) {
        guard analysisGeneration == token else { return }
        self.groups = groups
        updateAnalyzedFrames(frames, isProgressive: true)
    }

    /// Both progressive and final snapshots share candidate invalidation and
    /// preserve completed tags while the album's scored prefix grows.
    func updateAnalyzedFrames(_ incoming: [AnalyzedFrame], isProgressive: Bool = false) {
        defersVisualOrder = isProgressive
        frames = preservingLiveTags(incoming)
    }

    private func acceptVideo(_ video: VideoInfo, token: UUID, prepared: PreparedMedia) -> Bool {
        guard analysisGeneration == token else { return false }
        importedMedia = prepared
        self.video = video
        return true
    }
    private func acceptAnalysisCacheIdentity(_ identity: AnalysisResultCacheIdentity?, token: UUID) {
        guard analysisGeneration == token else { return }
        analysisCacheIdentity = identity
    }
    private func acceptProgress(_ progress: AnalysisProgress, token: UUID) {
        guard analysisGeneration == token, !self.progress.isTerminal else { return }
        self.progress = progress
    }
    private func completeAnalysis(frames: [AnalyzedFrame], groups: [FrameGroup],
                                  progress: AnalysisProgress, token: UUID) async {
        guard analysisGeneration == token else { return }
        let cacheKeys = TagResultCacheKeys.make(videoURL: video?.url, frames: frames,
                                                vocabulary: tagVocabulary,
                                                analysisWidth: settings.analysisWidth,
                                                analysisMode: settings.analysisMode)
        var completedFrames = frames
        if !cacheKeys.isEmpty, let vocabulary = tagVocabulary,
           TagResultMemory.shared.rememberTaggingResults {
            let restoreToken = UUID()
            tagMemoryRestoreToken = restoreToken
            beginTagMemoryRestore(restoreToken)
            let restored = await TagResultMemory.shared.restore(
                frames,
                keys: cacheKeys,
                vocabulary: vocabulary,
                threshold: settings.taggerThreshold
            )
            guard tagMemoryRestoreToken == restoreToken else { return }
            endTagMemoryRestore(restoreToken)
            guard analysisGeneration == token else { return }
            completedFrames = restored.frames
        }
        taggingCacheKeys = cacheKeys
        taggingVocabularyFingerprint = tagVocabulary?.cacheFingerprint
        self.groups = groups
        updateAnalyzedFrames(completedFrames)
        self.progress = progress
        analysisTask = nil
        isRegrouping = false
        if !cacheKeys.isEmpty {
            await TagResultMemory.shared.remember(frames: completedFrames, keys: cacheKeys)
        }
    }
    private func failAnalysis(progress: AnalysisProgress, error: Error, token: UUID) {
        guard analysisGeneration == token else { return }
        if !(error is CancellationError) { errorMessage = error.localizedDescription }
        self.progress = progress
        analysisTask = nil
        isRegrouping = false
        if defersVisualOrder {
            defersVisualOrder = false
            recomputeVisualOrder()
        }
    }

    private func clearResults() {
        defersVisualOrder = false
        thumbnailGeneration = UUID()
        for request in thumbnailRequests.values { request.task.cancel() }
        thumbnailRequests.removeAll(keepingCapacity: false)
        thumbnailCache.removeAll()
        ThumbnailRegenerator.shared.reset()
        taggingDemandActive = false
        taggingDemandGeneration = UUID()
        taggingCandidateRestoreToken = UUID()
        taggingCandidateRestoreTask?.cancel()
        taggingCandidateRestoreTask = nil
        pendingTaggingRestoreFrames.removeAll()
        tagMemoryRestoreOperations.removeAll()
        tagMemoryRestoreToken = UUID()
        restoringTagMemory = false
        taggingCacheKeys = .none
        analysisCacheIdentity = nil
        taggingVocabularyFingerprint = nil
        taggingProgress = TaggingProgress()
        taggingQueue = []
        taggingQueueIndex = 0
        taggingQueuedIDs.removeAll()
        pendingTaggedFrames.removeAll()
        lastTaggingCandidateIDs = nil
        lastTaggingCandidateKey = nil
        displayFilterResults.removeAll()
        taggingCandidateResults.removeAll()
        matchedFrameCountForTagging = 0
        untaggedFrameCountForTagging = 0
        taggedFrameCount = 0
        frames = []
        groups = []
        selectedFrameIDs.removeAll()
        selectionAnchor = nil
        focusedFrameID = nil
        gridNavigationFrameID = nil
        // Group identifiers are derived from member identifiers, so a new analysis
        // produces new ones: a group revealed for the old video must not match one
        // in the new.
        revealedGroups = []
        frameLimits = .none
        previewFrame = nil
        presentedGridNavigation = nil
        displayedFrames = []
        displayOrderFrames = []
        tagFacetTags = []
        displayedRows = []
        beyondCapByGroup = [:]
        cappedGroupSummaries = []
    }

    private func beginTagMemoryRestore(_ token: UUID) {
        tagMemoryRestoreOperations.insert(token)
        restoringTagMemory = !tagMemoryRestoreOperations.isEmpty
    }

    private func endTagMemoryRestore(_ token: UUID) {
        tagMemoryRestoreOperations.remove(token)
        restoringTagMemory = !tagMemoryRestoreOperations.isEmpty
    }

    private func restoreRememberedTagResults() {
        guard !restoringTagMemory, !taggingProgress.isRunning,
              progress.phase == .complete, !frames.isEmpty,
              let video, let vocabulary = tagVocabulary else { return }
        let keys = TagResultCacheKeys.make(videoURL: video.url, frames: frames, vocabulary: vocabulary,
                                           analysisWidth: settings.analysisWidth,
                                           analysisMode: settings.analysisMode)
        guard !keys.isEmpty else { return }

        taggingCacheKeys = keys
        taggingVocabularyFingerprint = vocabulary.cacheFingerprint
        let requestToken = UUID()
        tagMemoryRestoreToken = requestToken
        beginTagMemoryRestore(requestToken)
        let analysisToken = analysisGeneration
        let sourceURL = video.url
        let snapshot = frames
        let threshold = settings.taggerThreshold

        Task { [weak self] in
            guard let self else { return }
            let restored = await TagResultMemory.shared.restore(
                snapshot,
                keys: keys,
                vocabulary: vocabulary,
                threshold: threshold
            )
            guard self.tagMemoryRestoreToken == requestToken else { return }
            self.endTagMemoryRestore(requestToken)
            guard self.analysisGeneration == analysisToken, self.video?.url == sourceURL,
                  !self.taggingProgress.isRunning else { return }
            if restored.restoredCount > 0 {
                self.frames = restored.frames
                self.invalidateTagSummaries()
                self.recomputeDisplayedFrames()
            }
            await TagResultMemory.shared.remember(frames: restored.frames, keys: keys)
        }
    }

    private func clearTagResultsNotMatching(_ vocabularyFingerprint: UInt64) {
        guard frames.contains(where: {
            $0.hasTaggingResult && $0.tagVocabularyFingerprint != vocabularyFingerprint
        }) else { return }
        frames = frames.map { frame in
            guard frame.hasTaggingResult,
                  frame.tagVocabularyFingerprint != vocabularyFingerprint else { return frame }
            var copy = frame
            copy.tagProbabilities = nil
            copy.tagVocabularyFingerprint = nil
            copy.tagRating = nil
            copy.topTags = nil
            copy.tagGeneralScore = nil
            return copy
        }
    }

    private func scheduleTagMemorySync(keys: TagResultCacheKeys, frames: [AnalyzedFrame]) {
        guard !keys.isEmpty, !taggingProgress.isRunning, !tagMemoryCancellationFlushPending else { return }
        Task {
            await TagResultMemory.shared.remember(frames: frames, keys: keys)
        }
    }

    private func persistTagResultsAtExit() {
        let keys: TagResultCacheKeys = taggingCacheKeys.isEmpty
            ? TagResultCacheKeys.make(videoURL: video?.url, frames: frames, vocabulary: tagVocabulary,
                                      analysisWidth: settings.analysisWidth, analysisMode: settings.analysisMode)
            : taggingCacheKeys
        guard !keys.isEmpty else { return }
        TagResultMemory.shared.flushAtTermination(
            frames: frames + Array(pendingTaggedFrames.values),
            keys: keys
        )
    }

    private func persistAnalysisResultsForCurrentVideo() {
        guard let identity = analysisCacheIdentity, !frames.isEmpty else { return }
        let snapshot = frames
        Task { await AnalysisResultMemory.shared.remember(identity: identity, frames: snapshot) }
    }

    private func persistAnalysisResultsAtExit() {
        guard let identity = analysisCacheIdentity else { return }
        AnalysisResultMemory.shared.flushAtTermination(identity: identity, frames: frames)
    }

    /// Rebuilds `displayedFrames`, `displayedRows`, `limitBounds` and
    /// `beyondCapByGroup` from the current arrangement.
    private func recomputeDisplayedFrames() {
        let grouping = settings.grouping.validated()
        let taggingSelection = taggingCandidates(grouping)
        let taggingCandidates = taggingSelection.frames
        matchedFrameCountForTagging = taggingCandidates.count
        untaggedFrameCountForTagging = taggingSelection.untaggedCount
        if lastTaggingCandidateKey != taggingSelection.key || lastTaggingCandidateIDs == nil {
            let currentTaggingCandidateIDs = Set(taggingCandidates.map(\.id))
            let additions = lastTaggingCandidateIDs.flatMap { previous -> [AnalyzedFrame]? in
                guard currentTaggingCandidateIDs.count > previous.count,
                      currentTaggingCandidateIDs.isSuperset(of: previous) else { return nil }
                return taggingCandidates.filter { !previous.contains($0.id) }
            } ?? []
            lastTaggingCandidateIDs = currentTaggingCandidateIDs
            lastTaggingCandidateKey = taggingSelection.key
            if taggingDemandActive, !additions.isEmpty {
                appendTaggingCandidates(additions)
            }
        }

        // Bounds only depend on the face filter, face threshold, and current frame
        // data. Tag changes and slider drags reuse this result instead of scanning
        // every frame again.
        let boundsKey = FrameLimitBoundsCacheKey(faceFilter: faceFilter,
                                                 minimumFaceAreaBits: grouping.minimumFaceAreaFraction.bitPattern)
        if let cachedLimitBounds, cachedLimitBoundsKey == boundsKey {
            limitBounds = cachedLimitBounds
        } else {
            let base = frames.filter { faceFilter.admits($0, grouping: grouping) }
            let bounds = FrameLimitBounds.bounds(of: base, grouping: grouping)
            cachedLimitBoundsKey = boundsKey
            cachedLimitBounds = bounds
            limitBounds = bounds
        }

        // Cache-based filters can outlive a live refresh. Resolve any repeated
        // IDs before ranking and group caps so one frame contributes once.
        var displayCandidates = uniqueFramesByID(filteredDisplayCandidates(grouping: grouping))
        if photosOrderFilter != .all,
           let albumURL = photosAlbumURL,
           let albumIdentifier = PhotosMedia.shared.collectionIdentifier(for: albumURL) {
            displayCandidates = displayCandidates.filter { frame in
                let assetIdentifier = PhotosMedia.shared.assetIdentifier(for: albumURL, seconds: frame.timeSeconds)
                let isOrdered = assetIdentifier.map {
                    PhotosOrderState.shared.isOrdered(albumIdentifier: albumIdentifier, assetIdentifier: $0)
                } ?? false
                return photosOrderFilter.includes(isOrdered: isOrdered)
            }
        }

        let arrangement = FrameArrangement.arrange(
            frames,
            order: frameOrder,
            faceFilter: faceFilter,
            limits: frameLimits,
            perGroupTopCount: settings.perGroupFrameCount,
            revealedGroups: revealedGroups,
            grouping: grouping,
            groupingEnabled: groupingEnabled,
            tagFilter: TagFilter(requiredTags: tagFilter.requiredTags,
                                 excludedTags: tagFilter.excludedTags,
                                 threshold: settings.taggerThreshold,
                                 selectionOrder: tagFilter.selectionOrder,
                                 exclusionOrder: tagFilter.exclusionOrder,
                                 scoreSortTag: tagFilter.scoreSortTag,
                                 scoreSortDescending: tagFilter.scoreSortDescending),
            vocabulary: tagVocabulary,
            filteredCandidates: displayCandidates,
            applyingVisualOrder: false
        )
        let baseFrames = uniqueFramesByID(arrangement.visible)
        visualOrderBaseFrames = baseFrames
        displayedFrames = visuallyOrderedFrames(baseFrames)
        scheduleTagSummary()
        rebuildDisplayedLayout()
        beyondCapByGroup = arrangement.beyondCapByGroup
        var earliestByGroup: [FrameGroupID: Double] = [:]
        for group in groups where arrangement.beyondCapByGroup[group.id] != nil {
            earliestByGroup[group.id] = members(of: group).lazy.map(\.timeSeconds).min()
        }
        cappedGroupSummaries = arrangement.beyondCapByGroup.map { id, count in
            CappedGroupSummary(id: id, count: count,
                               label: earliestByGroup[id].map(timecode) ?? id.raw)
        }.sorted { $0.count == $1.count ? $0.id.raw < $1.id.raw : $0.count > $1.count }
    }

    /// Visual order cannot change membership, representatives, score bounds or
    /// tag summaries. Reuse those results both for controls and route completion.
    private func recomputeVisualOrder() {
        guard let baseFrames = visualOrderBaseFrames else {
            recomputeDisplayedFrames()
            return
        }
        let ordered = visuallyOrderedFrames(baseFrames)
        // Starting a background job often leaves the current sequence unchanged.
        // Avoid making SwiftUI diff the entire album just to show the spinner.
        guard groupingEnabled || ordered.map(\.id) != displayedFrames.map(\.id) else { return }
        displayedFrames = ordered
        rebuildDisplayedLayout()
    }

    private func rebuildDisplayedLayout() {
        displayedRows = FrameArrangement.rows(of: displayedFrames)
        displayOrderFrames = groupingEnabled && !frameOrder.minimalVisualDistance.enabled
            ? displayedRows.flatMap(\.frames) : displayedFrames
        rowPosition = [:]
        rowHeightRatios = [:]
        for (row, value) in displayedRows.enumerated() {
            var heightRatio = 0.0
            for (position, frame) in value.frames.enumerated() {
                rowPosition[frame.id] = (row, position)
                let aspect = frame.thumbnailAspectRatio ?? 16.0 / 9
                heightRatio = max(heightRatio, 1 / (aspect.isFinite && aspect > 0 ? aspect : 16.0 / 9))
            }
            rowHeightRatios[value.id] = heightRatio
        }
        invalidateGridLayout()
    }

    /// Cache by the actual shown set, its seed order and feature payloads. Metadata
    /// refreshes reuse the route but still display the newest frame values. Flipping
    /// direction reverses the cached route without recalculating any distances.
    private func visuallyOrderedFrames(_ shown: [AnalyzedFrame]) -> [AnalyzedFrame] {
        guard frameOrder.minimalVisualDistance.enabled, !defersVisualOrder, shown.count > 1 else {
            if visualOrderTask != nil {
                visualOrderTask?.cancel()
                visualOrderTask = nil
                pendingVisualOrderInput = nil
                visualOrderGeneration = UUID()
            }
            if isSortingVisualDistance { isSortingVisualDistance = false }
            if shown.isEmpty {
                completedVisualOrderInput = nil
                visualOrderIDs = []
            }
            // A growing album appends in source order. Re-solving every prefix
            // repeatedly moves visible cells and invalidates their prepared pixels.
            return defersVisualOrder && frameOrder.minimalVisualDistance.enabled
                ? shown.sorted { $0.id.sampleIndex < $1.id.sampleIndex } : shown
        }
        let input = shown.map { VisualOrderInput(id: $0.id, featurePrint: $0.featurePrint) }
        if completedVisualOrderInput != input, pendingVisualOrderInput != input {
            visualOrderTask?.cancel()
            let generation = UUID()
            visualOrderGeneration = generation
            pendingVisualOrderInput = input
            isSortingVisualDistance = true
            let task = Task.detached(priority: .userInitiated) { () -> [FrameID]? in
                do {
                    // Coalesce slider drags and incoming analysis batches.
                    try await Task.sleep(for: .milliseconds(150))
                    return try MinimalVisualDistanceOrder.sorted(shown).map(\.id)
                } catch {
                    return nil
                }
            }
            visualOrderTask = task
            Task { [weak self] in
                let ids = await task.value
                guard let self, self.visualOrderGeneration == generation else { return }
                self.visualOrderTask = nil
                self.pendingVisualOrderInput = nil
                self.isSortingVisualDistance = false
                guard let ids else { return }
                self.completedVisualOrderInput = input
                self.visualOrderIDs = ids
                self.recomputeVisualOrder()
            }
        } else if completedVisualOrderInput == input, visualOrderTask != nil {
            // A pending filter change may have been undone before it finished.
            visualOrderTask?.cancel()
            visualOrderTask = nil
            pendingVisualOrderInput = nil
            visualOrderGeneration = UUID()
            isSortingVisualDistance = false
        }

        let byID = Dictionary(uniqueKeysWithValues: shown.map { ($0.id, $0) })
        let cachedIDs = Set(visualOrderIDs)
        let ordered = visualOrderIDs.compactMap { byID[$0] } + shown.filter { !cachedIDs.contains($0.id) }
        return frameOrder.minimalVisualDistance.ascending ? ordered : Array(ordered.reversed())
    }

    /// Vectors remain cached in the worker. Only metadata/vocabulary changes
    /// invalidate an input revision; ordering is deliberately absent from the key.
    private func invalidateTagSummaries() {
        tagSummaryRevision &+= 1
        tagSummaryGeneration = UUID()
        tagSummaryTask?.cancel()
        tagSummaryTask = nil
        pendingTagSummaryInput = nil
        completedTagSummaryInput = nil
        completedTagSummaryLabels = nil
        if isSummarizingTags { isSummarizingTags = false }
    }

    private func scheduleTagSummary() {
        guard let vocabulary = tagVocabulary, !displayedFrames.isEmpty else {
            tagSummaryGeneration = UUID()
            tagSummaryTask?.cancel()
            tagSummaryTask = nil
            pendingTagSummaryInput = nil
            completedTagSummaryInput = nil
            completedTagSummaryLabels = nil
            if !tagFacetTags.isEmpty { tagFacetTags = [] }
            if !tagCommonTags.isEmpty { tagCommonTags = [] }
            if isSummarizingTags { isSummarizingTags = false }
            return
        }
        let input = TagSummaryInput(ids: Set(displayedFrames.map(\.id)), revision: tagSummaryRevision,
                                    threshold: settings.taggerThreshold)
        if input == completedTagSummaryInput, let labels = completedTagSummaryLabels {
            if pendingTagSummaryInput != nil {
                tagSummaryGeneration = UUID()
                tagSummaryTask?.cancel()
                tagSummaryTask = nil
                pendingTagSummaryInput = nil
                isSummarizingTags = false
            }
            publishTagSummary(labels)
            return
        }
        guard input != pendingTagSummaryInput else { return }
        tagSummaryTask?.cancel()
        let generation = UUID()
        tagSummaryGeneration = generation
        pendingTagSummaryInput = input
        if !tagFacetTags.isEmpty { tagFacetTags = [] }
        if !tagCommonTags.isEmpty { tagCommonTags = [] }
        isSummarizingTags = true
        let shown = displayedFrames
        let sourceIDs = Set(frames.map(\.id))
        let computer = tagSummaryComputer
        let task = Task.detached(priority: .utility) { () -> TagSummary.Labels? in
            do {
                let summary = try await computer.summarize(shown, vocabulary: vocabulary,
                                                          threshold: input.threshold, retaining: sourceIDs)
                try Task.checkCancellation()
                return summary.labels(using: vocabulary)
            } catch {
                return nil
            }
        }
        tagSummaryTask = task
        Task(priority: .utility) { [weak self] in
            let labels = await task.value
            guard let self, self.tagSummaryGeneration == generation else { return }
            self.tagSummaryTask = nil
            self.pendingTagSummaryInput = nil
            self.isSummarizingTags = false
            guard let labels else { return }
            self.completedTagSummaryInput = input
            self.completedTagSummaryLabels = labels
            // Publishing a tag result must not re-rank/rebuild the photo grid.
            self.publishTagSummary(labels)
        }
    }

    private func publishTagSummary(_ labels: TagSummary.Labels) {
        let common = Set(labels.common)
        let available = labels.ranked.filter { !common.contains($0) && !tagFilter.requiredTags.contains($0) }
        if tagCommonTags != labels.common { tagCommonTags = labels.common }
        if tagFacetTags != available { tagFacetTags = available }
    }

    // MARK: - Grouping

    /// The FeaturePrint distance within which two frames count as the same shot.
    ///
    /// **The live knob for how much the app groups.** Raise it to loosen — fewer,
    /// larger groups — and lower it to tighten, which produces more and smaller
    /// ones. Seeded from the persisted settings and written back when it changes.
    ///
    /// Changing it does not re-sample or re-score anything: the frames are scored
    /// already, so re-grouping is a pass over their FeaturePrints and costs a moment,
    /// not an analysis. That is why it is a live control rather than a setting you
    /// restart for.
    @Published var groupingThreshold: Float = WaifuSettings.default.similarityThreshold {
        didSet {
            // The slider writes on every tick; re-grouping waits for the release, so
            // say so rather than letting the group count look stale.
            if groupingThreshold != settings.similarityThreshold { regroupPending = true }
        }
    }

    /// True when the grouping slider has moved and the re-group has not run yet.
    @Published private(set) var regroupPending = false

    /// Re-groups the analysed frames under the current threshold.
    ///
    /// Called when the grouping slider is *released*, not on every tick: a drag
    /// passes through dozens of values and each one would run thousands of distance
    /// computations for a result that is thrown away.
    func regroup() {
        guard !frames.isEmpty else { return }
        regroupPending = false
        if settings.similarityThreshold != groupingThreshold {
            // Persisting goes through `settings`, which also recomputes the grid.
            settings.similarityThreshold = groupingThreshold
        }
        let grouping = settings.grouping.validated(), input = frames
        analysisTask?.cancel()
        let token = UUID(); analysisGeneration = token
        let started = Date()
        var initial = AnalysisProgress(phase: .grouping, processed: input.count, total: input.count,
                                       message: "Grouping alike frames…")
        initial.facesProcessed = input.reduce(0) { $0 + $1.faceCount }
        initial.framesWithFaces = input.filter(\.hasSignificantFace).count
        // Keep the existing grid/player mounted. Only the panel's activity
        // indicator changes until the finished groups replace the old ones.
        isRegrouping = true
        regroupProgress.reset()
        let comparisonCount = input.lazy.filter { $0.featurePrint != nil && $0.timeSeconds.isFinite }.count
        let snapshot = initial
        analysisTask = Task.detached { [weak self] in
            do {
                let groups = try FrameGrouper().groupCancellable(input, settings: grouping) { count, _ in
                    Task { @MainActor in
                        self?.acceptRegroupProgress(count, total: comparisonCount, token: token)
                    }
                }
                try Task.checkCancellation()
                var groupByMember: [String: FrameGroupID] = [:]
                for group in groups {
                    for member in group.memberIDs { groupByMember[member] = group.id }
                }
                let regrouped = input.map { frame -> AnalyzedFrame in
                    var copy = frame; copy.groupID = groupByMember[frame.id.raw]; return copy
                }
                var final = snapshot
                final.phase = .complete; final.message = "Complete"
                final.groupsIdentified = groups.count; final.groupedFrames = input.count
                final.elapsed = Date().timeIntervalSince(started)
                await self?.finishRegroup(frames: regrouped, groups: groups, progress: final, token: token)
            } catch {
                var final = snapshot
                final.phase = error is CancellationError ? .cancelled : .failed
                final.message = error is CancellationError ? "Cancelled" : error.localizedDescription
                final.elapsed = Date().timeIntervalSince(started)
                await self?.failAnalysis(progress: final, error: error, token: token)
            }
        }
    }

    private func finishRegroup(frames: [AnalyzedFrame], groups: [FrameGroup],
                               progress: AnalysisProgress, token: UUID) async {
        guard analysisGeneration == token else { return }
        revealedGroups = []
        await completeAnalysis(frames: frames, groups: groups, progress: progress, token: token)
    }

    private func acceptRegroupProgress(_ count: Int, total: Int, token: UUID) {
        guard analysisGeneration == token, isRegrouping else { return }
        regroupProgress.update(processed: count, total: total)
    }

    /// Moves the focus to the neighbouring frame, taking the preview with it.
    ///
    /// The *rule* lives here rather than in the grid because two views need it: the
    /// grid's keyboard handling, and the preview overlay — whose own zoom buttons take
    /// the keyboard focus when clicked, so the arrows would otherwise stop working
    /// exactly when you have set a view up and want to step through it. One definition,
    /// so stepping never depends on who happens to hold focus.
    ///
    /// In one-group-per-line mode this stays inside the current group's line and stops
    /// at its ends; in the flow grid it walks the group-preserving displayed order.
    func focusNeighbour(_ delta: Int) {
        guard uncover == nil else { return }
        let order = navigationOrder
        let frames = order.frames
        guard !frames.isEmpty else { return }
        guard let current = focusedFrameID,
              order.indexByID[current] != nil else {
            focus(on: frames[0])
            return
        }
        guard let target = order.neighbour(of: current, by: delta) else { return }
        focus(on: target)
    }

    /// Moves focus to the first frame of the next or previous group.
    func focusGroupEdge(forward: Bool) {
        guard uncover == nil, groupingEnabled else { return }
        let order = navigationOrder
        let frames = order.frames
        guard !frames.isEmpty else { return }
        guard let current = focusedFrameID,
              let index = order.indexByID[current] else {
            focus(on: frames[0])
            return
        }
        let group = frames[index].groupID
        var target: Int?
        if forward {
            for i in (index + 1)..<frames.count where frames[i].groupID != group {
                target = i
                break
            }
        } else {
            for i in stride(from: index - 1, through: 0, by: -1) where frames[i].groupID != group {
                var start = i
                while start > 0, frames[start - 1].groupID == frames[i].groupID { start -= 1 }
                target = start
                break
            }
        }
        guard let target else { return }
        focus(on: frames[target])
    }

    /// Home and End move to the first or last thumbnail in the current displayed
    /// result, regardless of whether the grid is flat or arranged in group rows.
    func focusDisplayedBoundary(forward: Bool) {
        guard uncover == nil else { return }
        let frames = navigationOrder.frames
        guard let frame = forward ? frames.last : frames.first else { return }
        focus(on: frame)
    }

    /// Space: show the focused frame, or close the preview if it is already up.
    func togglePreview() {
        guard uncover == nil else { return }
        guard previewFrame == nil else {
            previewFrame = nil
            return
        }
        guard let id = focusedFrameID, let frame = frame(id: id) else { return }
        showPreview(frame)
    }

    /// Opens a frame in the preview and makes it the sole selected frame.
    /// Context-menu preview actions can target a frame other than the current
    /// focus, so they must take the same selection path as keyboard navigation.
    func showPreview(_ frame: AnalyzedFrame) {
        guard uncover == nil else { return }
        if previewFrame == nil {
            // Keep the sequence the user opened, even if a pending visual route
            // finishes or the thumbnail stage commits behind the overlay.
            previewNavigation = navigationOrder
        }
        gridNavigationFrameID = nil
        focus(on: frame, scrollIntoView: false)
        previewFrame = frame
    }

    /// Programmatic navigation selects a frame and asks the grid to reveal it.
    /// Opening preview selects without moving the grid behind the overlay.
    func focus(on frame: AnalyzedFrame, scrollIntoView: Bool = true) {
        focusedFrameID = frame.id
        selectedFrameIDs = [frame.id]
        selectionAnchor = frame.id
        if previewFrame != nil { previewFrame = frame }
        else if scrollIntoView { requestGridScroll(to: frame.id) }
    }

    // MARK: Settings persistence

    private func persistSettings() {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let data = try? encoder.encode(settings) {
            try? data.write(to: settingsURL, options: .atomic)
        }
    }

    private func loadSettings() {
        guard let data = try? Data(contentsOf: settingsURL),
              let loaded = try? JSONDecoder().decode(WaifuSettings.self, from: data) else { return }
        settings = loaded.validated()
    }
}

// MARK: - Errors

private enum AnalysisError: Error, LocalizedError {
    case noVideo

    var errorDescription: String? {
        switch self {
        case .noVideo: return "No video loaded"
        }
    }
}

private struct PendingTagFrame: Sendable {
    let frame: AnalyzedFrame
    let source: TaggingImageSource
}

private struct PreparedTagFrame: Sendable {
    let frame: AnalyzedFrame
    let image: CGImage
}

// MARK: - Thumbnail bridge

/// Bridges `ThumbnailCache` (thread-safe memory store) to `ThumbnailWriter`
/// (protocol, `CGImage`, key only). The video stem is fixed at pipeline creation.
private struct ThumbnailWriterBridge: ThumbnailWriter, Sendable {
    private let cache: ThumbnailCache
    private let videoStem: String

    init(cache: ThumbnailCache, videoStem: String) {
        self.cache = cache
        self.videoStem = videoStem
    }

    func write(_ image: CGImage, key: String) async throws {
        // The scoring pipeline calls this once per decoded frame. Drain AppKit's
        // autoreleased bitmap/JPEG intermediates each time instead of retaining
        // them for the lifetime of a long-running analysis task.
        let jpegData = try autoreleasepool {
            try ThumbnailCache.encodeJPEG(image)
        }
        try cache.store(jpegData, forKey: key, videoStem: videoStem)
    }
}
