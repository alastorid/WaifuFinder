import SwiftUI
import Combine

/// Gesture-rate updates stay out of the session/search/metadata observation tree.
@MainActor final class CrystalOrbNavigation: ObservableObject {
    @Published var zoom: CGFloat = 1
    @Published var actualPixels = false
    @Published var fillViewport = false
    @Published var pan = CGSize.zero
    @Published var center: CGPoint?
}

@MainActor
final class WaifuCrystalOrbModel: ObservableObject {
    enum Mode: String, CaseIterable { case sideBySide = "Side by Side", swipe = "Swipe", toggle = "Toggle", difference = "Difference" }
    struct Pair {
        let queryID: UUID
        let frame: AnalyzedFrame
        let target: CGImage
        let candidate: CGImage
        var difference: CrystalOrbDifference?
        var differenceIsOriginal = false
        var alignment: CrystalOrbAlignment
        let targetIsOriginal: Bool
        let candidateIsOriginal: Bool
        let candidateNativeSize: CGSize
        var isOriginal: Bool { targetIsOriginal && candidateIsOriginal }
    }
    @Published private(set) var isOpen = false
    @Published private(set) var query: CrystalOrbQuery?
    @Published private(set) var matches: [CrystalOrbMatch] = []
    @Published private(set) var selectedID: FrameID?
    @Published private(set) var pair: Pair?
    /// Nil while there is nothing worth occupying the title-bar status slot.
    @Published private(set) var status: String?
    @Published private(set) var searching = false
    @Published private(set) var loading = false
    @Published private(set) var importing = false
    @Published private(set) var comparisonError: String?
    @Published var candidateMetadata: PhotoMetadata?
    @Published var metadataLoading = false
    @Published var mode: Mode = .sideBySide {
        didSet {
            if oldValue != mode { maximizedPane = nil }
            if mode != .toggle { holdingToggle = false }
            if mode == .difference { opacityControlFocused = false }
        }
    }
    @Published var redOverlay = false
    @Published var opacity = 0.65
    @Published var swipe = 0.5
    @Published var showCandidate = false
    @Published var holdingToggle = false
    var opacityControlFocused = false
    var inspectionControlFocused = false
    let navigation = CrystalOrbNavigation()
    var zoom: CGFloat { get { navigation.zoom } set { navigation.zoom = newValue } }
    var actualPixels: Bool { get { navigation.actualPixels } set { navigation.actualPixels = newValue } }
    var fillViewport: Bool { get { navigation.fillViewport } set { navigation.fillViewport = newValue } }
    var pan: CGSize { get { navigation.pan } set { navigation.pan = newValue } }
    var navigationCenter: CGPoint? { get { navigation.center } set { navigation.center = newValue } }
    @Published var cursor: CGPoint?
    @Published var roi: CGRect? { didSet { scheduleMeasurements() } }
    @Published var selectingRegion = false
    @Published var alignmentEditing = false
    @Published var alignmentLocked = false {
        didSet { if alignmentLocked { registrationTask?.cancel(); registrationTicket = UUID(); registering = false; alignmentEditing = false } }
    }
    @Published var alignmentEnabled = true { didSet { if oldValue != alignmentEnabled { rebuildAlignment() } } }
    @Published var registration = CrystalOrbRegistration() {
        didSet {
            guard oldValue != registration else { return }
            guard registration.x.isFinite, registration.y.isFinite, registration.scale.isFinite, registration.degrees.isFinite,
                  registration.scale >= 0.05, registration.scale <= 20 else { registration = oldValue; return }
            if registering { registrationTask?.cancel(); registrationTicket = UUID(); registering = false }
            rebuildAlignment()
        }
    }
    @Published var differenceSettings = CrystalOrbDifferenceSettings() { didSet { if oldValue != differenceSettings { refreshVisualization() } } }
    @Published private(set) var measurements: CrystalOrbMeasurements?
    @Published private(set) var pixel: CrystalOrbPixel?
    @Published private(set) var registering = false
    @Published private(set) var alignmentMessage = "Centered · No registration"
    @Published var maximizedPane: String?
    @Published var differenceCollapsed = false
    @Published var showCrosshair = true
    @Published var showPixelGrid = false
    @Published var showOverview = false
    private var registrationTask: Task<Void, Never>?
    private var measurementTask: Task<Void, Never>?
    private var pixelTask: Task<Void, Never>?
    private var pixelTicket = UUID()
    private var registrationTicket = UUID()
    private var registrationByFrame: [FrameID: CrystalOrbRegistration] = [:]
    private func cancelInspection() {
        registrationTask?.cancel(); registering = false; registrationTicket = UUID()
        measurementTask?.cancel(); pixelTask?.cancel(); pixelTask = nil; pixelTicket = UUID()
        measurements = nil; pixel = nil
    }
    /// Hard ceiling on offered candidates; the likeness band usually cuts earlier.
    nonisolated static let candidateLimit = 33
    /// Keeps matches whose Vision distance stays within this multiple of the
    /// best match (plus absolute slack for near-duplicate searches).
    nonisolated static let likenessRelativeBand = 1.35
    nonisolated static let likenessAbsoluteBand = 0.02
    private var followsBestMatch = true
    private var queryOriginalUnavailable = false
    private var candidateOriginalUnavailable = false
    private var originalRequested = false
    var canLoadOriginal: Bool {
        selectedID != nil && pair?.candidateIsOriginal != true && (!originalRequested || candidateOriginalUnavailable)
    }
    private var selectedMatch: CrystalOrbMatch?
    private(set) var generation: UUID?
    private final class CachedPair: NSObject {
        let value: Pair
        init(_ value: Pair) { self.value = value }
    }
    private let pairCache = NSCache<NSString, CachedPair>()
    private var metadataCache: [FrameID: PhotoMetadata] = [:]

    init() {
        pairCache.countLimit = 3
        pairCache.totalCostLimit = 160 * 1024 * 1024
    }

    private let engine = CrystalOrbSearchService()
    private var publicationGate = CrystalOrbPublicationGate()
    private var pendingRanking: (matches: [CrystalOrbMatch], done: Int, total: Int, ticket: UUID)?
    private var rankingTask: Task<Void, Never>?
    private var searchTask: Task<Void, Never>?
    private var comparisonTask: Task<Void, Never>?
    private var thumbnailTask: Task<Void, Never>?
    private var differenceTask: Task<Void, Never>?
    private var differenceTicket = UUID()
    private var metadataTask: Task<Void, Never>?
    private var importTask: Task<Void, Never>?
    private var searchTicket = UUID()
    private var comparisonTicket = UUID()
    private var importTicket = UUID()
    private(set) var dropTicket = UUID()

    func beginDrop() -> UUID { dropTicket = UUID(); return dropTicket }
    func beginImageDrop(session: WaifuSession) {
        open(session: session)
        importTask?.cancel(); importTicket = UUID()
        cancelSearch()
        comparisonTask?.cancel(); thumbnailTask?.cancel(); differenceTask?.cancel()
        comparisonTicket = UUID(); differenceTicket = UUID()
        importing = true; loading = false; status = "Preparing image"
    }

    func failImageDrop(ticket: UUID, message: String) {
        guard dropTicket == ticket else { return }
        importing = false; status = message
    }

    private weak var session: WaifuSession?

    /// The gallery offers every match close enough to the best one — a ranked
    /// list is sorted, so a single prefix-while cut is exact.
    nonisolated static func alikeMatches(_ ranked: [CrystalOrbMatch]) -> [CrystalOrbMatch] {
        guard let best = ranked.first else { return [] }
        let cutoff = best.distance * likenessRelativeBand + likenessAbsoluteBand
        return Array(ranked.prefix(candidateLimit).prefix { $0.distance <= cutoff })
    }
    var visibleMatches: [CrystalOrbMatch] { Self.alikeMatches(matches) }
    var selected: CrystalOrbMatch? { matches.first { $0.id == selectedID } ?? selectedMatch }
    var candidateIsPhoto: Bool {
        guard let source = session?.video?.url, let frame = selected?.frame else { return false }
        return PhotosMedia.shared.photoItemURL(for: source, seconds: frame.timeSeconds) != nil
    }
    var pairIsCurrent: Bool { pair?.queryID == query?.id && pair?.frame.id == selectedID }

    func open(session: WaifuSession) {
        self.session = session
        if !isOpen { session.previewFrame = nil; session.closeUncover() }
        session.crystalOrbIsOpen = true
        isOpen = true
        sourceChanged(session: session)
    }

    func close() {
        cancelInspection()
        cancelSearch()
        importTask?.cancel(); importTicket = UUID(); dropTicket = UUID()
        comparisonTask?.cancel(); comparisonTicket = UUID()
        thumbnailTask?.cancel(); differenceTask?.cancel(); differenceTicket = UUID()
        metadataTask?.cancel()
        pairCache.removeAllObjects(); metadataCache.removeAll()
        pair = nil; loading = false; metadataLoading = false
        opacityControlFocused = false; inspectionControlFocused = false
        holdingToggle = false
        importing = false
        isOpen = false
        session?.crystalOrbIsOpen = false
        session?.finishCrystalOrbNavigation()
        // Release the potentially large external bitmap; reusable features remain in the engine.
        query = nil; matches = []; selectedID = nil; selectedMatch = nil; candidateMetadata = nil
        status = nil
    }

    func sourceChanged(session: WaifuSession) {
        self.session = session
        guard generation != session.thumbnailGeneration else { return }
        generation = session.thumbnailGeneration
        cancelInspection(); registrationByFrame.removeAll()
        cancelSearch()
        comparisonTask?.cancel(); comparisonTicket = UUID()
        thumbnailTask?.cancel(); differenceTask?.cancel(); differenceTicket = UUID()
        metadataTask?.cancel()
        pairCache.removeAllObjects(); metadataCache.removeAll()
        matches = []; selectedID = nil; selectedMatch = nil; pair = nil; candidateMetadata = nil
        loading = false; metadataLoading = false; comparisonError = nil
        if isOpen, query != nil { search() }
    }

    func importImage(url: URL? = nil, data: Data? = nil, session: WaifuSession) {
        dropTicket = UUID()
        open(session: session)
        importTask?.cancel()
        cancelSearch()
        comparisonTask?.cancel(); comparisonTicket = UUID()
        thumbnailTask?.cancel(); differenceTask?.cancel(); differenceTicket = UUID()
        metadataTask?.cancel()
        // Keep the displayed query, candidate and gallery intact until decoding
        // succeeds. Failed replacements leave the previous workspace usable.
        importing = true
        queryOriginalUnavailable = false
        metadataLoading = false; comparisonError = nil
        status = "Preparing image"; loading = false
        let ticket = UUID(); importTicket = ticket
        let queryID = UUID()
        importTask = Task {
            do {
                let preview = try await CrystalOrbQueryDecoder.shared.read(url: url, data: data, maxPixelSize: 2048, id: queryID)
                guard !Task.isCancelled, importTicket == ticket, isOpen else { return }
                pairCache.removeAllObjects(); registrationByFrame.removeAll(); cancelInspection()
                query = preview; matches = []; selectedID = nil; selectedMatch = nil; candidateMetadata = nil
                importing = false
                search()
                if !preview.isOriginal {
                    let original = try await CrystalOrbQueryDecoder.shared.read(url: url, data: data, id: queryID)
                    guard !Task.isCancelled, importTicket == ticket, query?.id == queryID, isOpen else { return }
                    query = original
                    if let current = pair, current.queryID == queryID {
                        publishImage(current.candidate, original: current.candidateIsOriginal,
                                     frame: current.frame, ticket: comparisonTicket)
                    }
                }
            } catch {
                guard !Task.isCancelled, importTicket == ticket else { return }
                importing = false
                // A decoded preview remains usable if the original is unavailable.
                if query?.id != queryID { status = error.localizedDescription }
                else { queryOriginalUnavailable = true; loading = false; comparisonError = error.localizedDescription }
            }
        }
    }

    func paste(session: WaifuSession) -> Bool {
        guard let input = CrystalOrbImageSource.clipboard() else { return false }
        importImage(url: input.0, data: input.1, session: session)
        return true
    }

    func chooseFile(session: WaifuSession) {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = false
        panel.begin { [weak self] response in
            if response == .OK, let url = panel.url { self?.importImage(url: url, session: session) }
        }
    }

    private func cancelSearch() {
        searchTask?.cancel(); searchTicket = UUID(); searching = false
        rankingTask?.cancel(); rankingTask = nil; pendingRanking = nil
        publicationGate = CrystalOrbPublicationGate()
    }

    func search() {
        cancelSearch()
        guard !importing, let query, let session else { return }
        guard let source = session.video?.url else { status = "Load a video or open a Photos album to search."; return }
        let frames = session.frames, generation = session.thumbnailGeneration
        guard !frames.isEmpty else { status = "No analyzed candidates yet. Analyze this source, then search again."; return }
        let ticket = UUID(); searchTicket = ticket
        searching = true; followsBestMatch = true; status = "Searching for similar images"
        searchTask = Task {
            do {
                let result = try await engine.search(query: query.image, queryID: query.id, frames: frames, sourceURL: source, generation: generation, resultLimit: Self.candidateLimit,
                    partial: { [weak self] ranked, done, total in
                        await self?.publishMatches(ranked, done: done, total: total, ticket: ticket)
                    }) { _, _ in }
                guard !Task.isCancelled, searchTicket == ticket, session.thumbnailGeneration == generation else { return }
                publishMatches(Array(result.prefix(Self.candidateLimit)), done: frames.count, total: frames.count, ticket: ticket)
                await rankingTask?.value
                guard !Task.isCancelled, searchTicket == ticket, session.thumbnailGeneration == generation else { return }
                searching = false
                // A finished search needs no title-bar text: the orb itself shows
                // the matches. Only an unreadable source stays as a status.
                status = result.isEmpty ? "No readable candidates in this source." : nil
            } catch {
                guard !Task.isCancelled, searchTicket == ticket else { return }
                searching = false; status = error.localizedDescription
            }
        }
    }

    private func publishMatches(_ ranked: [CrystalOrbMatch], done: Int, total: Int, ticket: UUID) {
        guard searchTicket == ticket, isOpen, session?.thumbnailGeneration == generation else { return }
        let top = Array(ranked.prefix(Self.candidateLimit))
        guard top != matches else {
            pendingRanking = nil; rankingTask?.cancel(); rankingTask = nil
            return
        }
        pendingRanking = (top, done, total, ticket)
        guard rankingTask == nil else { return }
        let delay = publicationGate.delay(for: top, at: ProcessInfo.processInfo.systemUptime) ?? 0
        if delay <= 0 { commitRanking() }
        else {
            rankingTask = Task {
                do { try await Task.sleep(for: .seconds(delay)) } catch { return }
                guard !Task.isCancelled, searchTicket == ticket else { return }
                rankingTask = nil
                commitRanking()
            }
        }
    }

    private func commitRanking() {
        guard let next = pendingRanking, next.ticket == searchTicket,
              isOpen, session?.thumbnailGeneration == generation else { return }
        pendingRanking = nil
        guard matches != next.matches else { return }
        publicationGate.published(next.matches, at: ProcessInfo.processInfo.systemUptime)
        matches = next.matches
        status = "Searching · \(matches.count) closest so far · \(next.done) / \(next.total)"
        if followsBestMatch, let best = matches.first { select(best.id, automatic: true) }
    }

    func select(_ id: FrameID, automatic: Bool = false) {
        guard !importing, let match = matches.first(where: { $0.id == id })
                ?? (selectedMatch?.id == id ? selectedMatch : nil), let query, let session,
              let source = session.video?.url, session.thumbnailGeneration == generation else { return }
        if !automatic { followsBestMatch = false }
        if automatic, selectedID == id { return }
        let upgrade = !automatic && selectedID == id && pair?.candidateIsOriginal != true && !originalRequested
        if selectedID == id, !upgrade, loading || (pairIsCurrent && comparisonError == nil) { return }
        originalRequested = true
        if selectedID != id {
            cancelInspection()
            registrationByFrame[id] = registrationByFrame[id] ?? .init()
        }
        selectedID = id; selectedMatch = match
        session.selectCrystalOrbCandidate(id)
        comparisonTask?.cancel(); thumbnailTask?.cancel(); differenceTask?.cancel(); metadataTask?.cancel()
        let ticket = UUID(); comparisonTicket = ticket; differenceTicket = UUID()
        loading = true; candidateOriginalUnavailable = false
        comparisonError = nil; candidateMetadata = metadataCache[id]
        if let cached = pairCache.object(forKey: id.raw as NSString)?.value, cached.queryID == query.id, query.isOriginal {
            pair = cached; loading = false
            registration = registrationByFrame[id] ?? .init()
            if pair?.difference?.settings != differenceSettings { refreshVisualization() }
            scheduleMeasurements()
        } else {
            let identity = session.thumbnailGeneration.uuidString + ":" + match.frame.thumbnailKey
            let cachedPreview = GridThumbnailLoader.shared.cachedImage(identity: identity)
            if let cachedPreview { publishImage(cachedPreview, original: false, frame: match.frame, ticket: ticket) }
            thumbnailTask = Task {
                if cachedPreview == nil, let data = await session.thumbnailData(for: match.frame), !Task.isCancelled,
                   let pixels = await GridThumbnailLoader.shared.image(data: data, identity: identity),
                   !Task.isCancelled, comparisonTicket == ticket {
                    publishImage(pixels, original: false, frame: match.frame, ticket: ticket)
                }
                guard !Task.isCancelled, comparisonTicket == ticket,
                      let photo = PhotosMedia.shared.photoItemURL(for: source, seconds: match.frame.timeSeconds) else { return }
                do {
                    _ = try await PhotosMedia.shared.inspectionPreview(photo) { [weak self] pixels in
                        Task { @MainActor [weak self] in
                            self?.publishImage(pixels, original: false, frame: match.frame, ticket: ticket)
                        }
                    }
                } catch { /* Retain the best available preview while the original proceeds. */ }
            }
            // The best preview paints first; original acquisition follows the
            // existing Photos network preference automatically.
            if originalRequested {
                comparisonTask = Task {
                do {
                    let image = try await PhotosMedia.shared.image(source, seconds: match.frame.timeSeconds)
                    guard !Task.isCancelled, comparisonTicket == ticket, session.video?.url == source else { return }
                    publishImage(image, original: true, frame: match.frame, ticket: ticket)
                    loadDetailedMetadata()
                } catch {
                    guard !Task.isCancelled, comparisonTicket == ticket else { return }
                    candidateOriginalUnavailable = true
                    loading = false
                    comparisonError = error.localizedDescription
                }
            }
            }
        }
        if let photo = PhotosMedia.shared.photoItemURL(for: source, seconds: match.frame.timeSeconds) {
            metadataLoading = true
            metadataTask = Task {
                let basic = await PhotoMetadataService.shared.basic(for: photo)
                guard !Task.isCancelled, comparisonTicket == ticket else { return }
                candidateMetadata = basic
                if let basic { metadataCache[id] = basic }
                metadataLoading = false
            }
        } else { metadataLoading = false }
    }

    /// Replaces only pixels. Geometry uses declared native dimensions from the
    /// beginning, so a thumbnail-to-original upgrade cannot change inspection state.
    private func publishImage(_ image: CGImage, original: Bool, frame: AnalyzedFrame, ticket: UUID) {
        guard comparisonTicket == ticket, selectedID == frame.id, let query, let session,
              let source = session.video, session.thumbnailGeneration == generation else { return }
        if registering {
            registrationTask?.cancel(); registrationTicket = UUID(); registering = false
            alignmentMessage = "Image quality changed · Auto Align can refine the new pixels"
        }
        let samePair = pair?.queryID == query.id && pair?.frame.id == frame.id
        if !original, samePair, let current = pair {
            if current.candidateIsOriginal { return }
            if current.targetIsOriginal == query.isOriginal,
               max(image.width, image.height) <= max(current.candidate.width, current.candidate.height) { return }
        }
        let native = original ? CGSize(width: image.width, height: image.height)
            : CrystalOrbImageSource.candidateSize(source: source, frame: frame)
        let correction = registrationByFrame[frame.id] ?? .init()
        if registration != correction { registration = correction }
        let alignment = CrystalOrbAlignment(target: query.nativeSize, candidate: native, registration: alignmentEnabled ? correction : .init())
        let oldDifference = samePair && pair?.alignment == alignment ? pair?.difference : nil
        let oldQuality = oldDifference != nil && pair?.differenceIsOriginal == true
        pair = Pair(queryID: query.id, frame: frame, target: query.image, candidate: image,
                    difference: oldDifference, differenceIsOriginal: oldQuality, alignment: alignment,
                    targetIsOriginal: query.isOriginal, candidateIsOriginal: original, candidateNativeSize: native)
        prepareDifference(ticket: ticket)
    }

    private func prepareDifference(ticket: UUID) {
        guard let input = pair else { return }
        differenceTask?.cancel()
        let revision = UUID(); differenceTicket = revision
        let settings = differenceSettings
        measurements = nil; pixel = nil; measurementTask?.cancel(); pixelTask?.cancel(); pixelTask = nil; pixelTicket = UUID()
        loading = true
        differenceTask = Task {
            do {
                let difference = try await CrystalOrbPairRenderer.shared.render(target: input.target,
                    candidate: input.candidate, alignment: input.alignment, settings: settings)
                guard !Task.isCancelled, comparisonTicket == ticket, differenceTicket == revision,
                      pair?.queryID == input.queryID, pair?.frame.id == input.frame.id else { return }
                var completed = input; completed.difference = difference; completed.differenceIsOriginal = input.isOriginal
                pair = completed
                scheduleMeasurements()
                if let cursor { sample(at: cursor) }
                loading = (!completed.targetIsOriginal && !queryOriginalUnavailable)
                    || (originalRequested && !completed.candidateIsOriginal && !candidateOriginalUnavailable)
                if completed.isOriginal {
                    let cost = completed.candidate.bytesPerRow * completed.candidate.height
                        + difference.difference.bytesPerRow * difference.difference.height
                        + difference.redMask.bytesPerRow * difference.redMask.height
                    pairCache.setObject(CachedPair(completed), forKey: completed.frame.id.raw as NSString, cost: cost)
                }
            } catch {
                guard !Task.isCancelled, comparisonTicket == ticket, differenceTicket == revision else { return }
                loading = false; comparisonError = error.localizedDescription
            }
        }
    }

    func originalAccessChanged() {
        guard let id = selectedID, pair?.candidateIsOriginal != true else { return }
        originalRequested = false
        select(id)
    }
    func revealCandidateInPhotos() {
        guard let source = session?.video?.url, let frame = selected?.frame,
              let photo = PhotosMedia.shared.photoItemURL(for: source, seconds: frame.timeSeconds) else { return }
        PhotosAppIntegration.reveal(photo)
    }
    func loadDetailedMetadata() {
        guard let session, let match = selected, let source = session.video?.url,
              let photo = PhotosMedia.shared.photoItemURL(for: source, seconds: match.frame.timeSeconds) else { return }
        metadataTask?.cancel()
        let ticket = comparisonTicket
        metadataLoading = true
        metadataTask = Task {
            do {
                let metadata = try await PhotoMetadataService.shared.load(for: photo)
                guard !Task.isCancelled, comparisonTicket == ticket else { return }
                candidateMetadata = metadata
                metadataCache[match.id] = metadata
            } catch { /* Existing basic metadata remains usable. */ }
            if comparisonTicket == ticket { metadataLoading = false }
        }
    }

    func step(_ delta: Int) {
        let ladder = visibleMatches
        guard !ladder.isEmpty else { return }
        let index = selectedID.flatMap { id in ladder.firstIndex { $0.id == id } } ?? 0
        let next = min(max(0, index + delta), ladder.count - 1)
        select(ladder[next].id)
    }

    func resetZoom() { zoom = 1; actualPixels = false; fillViewport = false; pan = .zero; navigationCenter = nil }

    func center(on point: CGPoint) {
        if navigationCenter != point { navigationCenter = point }
        if pan != .zero { pan = .zero }
    }
    func zoomToRegion(_ region: CGRect) {
        guard !region.isNull, !region.isEmpty else { return }
        center(on: CGPoint(x: region.midX, y: region.midY)); actualPixels = false; fillViewport = false
        let canvas = pair?.alignment.size ?? region.size
        zoom = max(1, min(128, min(canvas.width/max(1,region.width),canvas.height/max(1,region.height))*0.8))
    }
    func resetAlignment() {
        guard !alignmentLocked else { return }
        alignmentMessage = "Centered · No registration"; registration = .init()
    }
    private func rebuildAlignment() {
        guard var input = pair, pairIsCurrent else { return }
        registrationByFrame[input.frame.id] = registration
        alignmentMessage = alignmentEnabled ? String(format: "Manual · X %+.2f px · Y %+.2f px · %.3f× · %.2f°", registration.x, registration.y, registration.scale, registration.degrees) : "Registration bypassed"
        input.alignment = CrystalOrbAlignment(target: input.alignment.targetSize, candidate: input.candidateNativeSize,
            registration: alignmentEnabled ? registration : .init())
        // Keep the previous visualization while refining; consumers explicitly
        // distinguish it from measurements under the new transform.
        pair = input
        prepareDifference(ticket: comparisonTicket)
    }
    private func refreshVisualization() {
        guard let input = pair else { return }
        guard let prior = input.difference, prior.alignment == input.alignment, input.differenceIsOriginal == input.isOriginal else { prepareDifference(ticket: comparisonTicket); return }
        differenceTask?.cancel(); measurementTask?.cancel(); pixelTask?.cancel(); pixelTask = nil; pixelTicket = UUID()
        let revision = UUID(); differenceTicket = revision
        let ticket = comparisonTicket, settings = differenceSettings
        measurements = nil; pixel = nil
        differenceTask = Task {
            do {
                let output = try await CrystalOrbPairRenderer.shared.remap(prior, settings: settings)
                guard !Task.isCancelled, differenceTicket == revision, comparisonTicket == ticket else { return }
                var updated = input; updated.difference = output; pair = updated
                loading = (!updated.targetIsOriginal && !queryOriginalUnavailable)
                    || (originalRequested && !updated.candidateIsOriginal && !candidateOriginalUnavailable)
                scheduleMeasurements(); if let cursor { sample(at: cursor) }
            } catch { /* Preserve the last useful visualization. */ }
        }
    }
    private func scheduleMeasurements() {
        measurementTask?.cancel()
        guard let difference = pair?.difference, pairIsCurrent else { measurements = nil; return }
        let region = roi, ticket = comparisonTicket
        measurementTask = Task {
            do {
                let result = try await CrystalOrbAnalysis.shared.measure(difference, roi: region)
                guard !Task.isCancelled, comparisonTicket == ticket, pair?.difference?.id == result.differenceID else { return }
                measurements = result
            } catch { /* Cancelled measurements are never published. */ }
        }
    }
    /// Hover probe in comparison coordinates. Both the target and candidate
    /// panes draw the crosshair from this shared cursor; nil or a point outside
    /// the shared canvas clears it. Snapping to pixel centers keeps mouse-move
    /// publications bounded by comparison pixels, not screen points.
    func hover(at point: CGPoint?) {
        guard showCrosshair, let point, let extent = pair?.alignment.extent,
              extent.insetBy(dx: -0.5, dy: -0.5).contains(point) else {
            if cursor != nil { cursor = nil }
            return
        }
        let snapped = CGPoint(x: floor(point.x) + 0.5, y: floor(point.y) + 0.5)
        if cursor != snapped { sample(at: snapped) }
    }
    func sample(at point: CGPoint) {
        cursor = point
        guard pixelTask == nil else { return }
        guard let difference = pair?.difference, pairIsCurrent, difference.alignment == pair?.alignment,
              difference.settings == differenceSettings else { pixel = nil; return }
        let ticket = UUID(); pixelTicket = ticket
        pixelTask = Task {
            do {
                try await Task.sleep(for: .milliseconds(30))
                guard let latest = cursor else { pixelTask = nil; return }
                let value = try await CrystalOrbPixelReader.shared.sample(difference, point: latest)
                guard !Task.isCancelled, pixelTicket == ticket, pair?.difference?.id == difference.id else { return }
                pixel = value
                pixelTask = nil
            } catch { if pixelTicket == ticket { pixelTask = nil } }
        }
    }
    func autoAlign() {
        guard let input = pair, pairIsCurrent, !alignmentLocked else { return }
        followsBestMatch = false
        registrationTask?.cancel(); let revision = UUID(); registrationTicket = revision
        let ticket = comparisonTicket
        registering = true; alignmentMessage = "Estimating translation…"
        registrationTask = Task {
            do {
                let result = try await CrystalOrbRegistrar.shared.align(target: input.target, candidate: input.candidate,
                    geometry: input.alignment, query: input.queryID, frame: input.frame.id, generation: generation)
                guard !Task.isCancelled, registrationTicket == revision, comparisonTicket == ticket,
                      pair?.target === input.target, pair?.candidate === input.candidate else { return }
                registering = false
                if let result {
                    alignmentEnabled = true
                    registration = result.registration
                    alignmentMessage = String(format: "Aligned · X %+.2f px · Y %+.2f px · Correlation %.3f", registration.x, registration.y, result.correlation)
                } else { alignmentMessage = "No reliable translation · Previous alignment retained" }
            } catch { if registrationTicket == revision { registering = false; alignmentMessage = "Alignment unavailable · Previous alignment retained" } }
        }
    }


    func handleKey(_ event: NSEvent) -> Bool {
        guard isOpen else { return false }
        if inspectionControlFocused, event.keyCode != 53 { return false }
        if event.keyCode == 49, mode == .toggle,
           event.modifierFlags.intersection([.command, .control, .option]).isEmpty {
            holdingToggle = event.type == .keyDown
            return true
        }
        guard event.type == .keyDown else { return false }
        if event.keyCode == 53 { close(); return true }
        guard event.modifierFlags.intersection([.command, .control, .option]).isEmpty else { return false }
        switch event.keyCode {
        case 123: if opacityControlFocused { opacity = max(0, opacity - 0.01) } else { step(-1) }
        case 124: if opacityControlFocused { opacity = min(1, opacity + 0.01) } else { step(1) }
        default:
            switch event.charactersIgnoringModifiers?.lowercased() {
            case "s": mode = .sideBySide
            case "w": mode = .swipe
            case "t": mode = .toggle
            case "d": mode = .difference
            case "r": redOverlay.toggle()
            default: return false
            }
        }
        return true
    }
}
