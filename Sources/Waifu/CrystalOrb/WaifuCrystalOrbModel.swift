import SwiftUI
import Combine

@MainActor
final class WaifuCrystalOrbModel: ObservableObject {
    enum Mode: String, CaseIterable { case sideBySide = "Side by Side", swipe = "Swipe", toggle = "Toggle", difference = "Difference" }
    struct Pair {
        let queryID: UUID
        let frame: AnalyzedFrame
        let target: CGImage
        let candidate: CGImage
        var difference: CrystalOrbDifference?
    }
    @Published private(set) var isOpen = false
    @Published private(set) var query: CrystalOrbQuery?
    @Published private(set) var matches: [CrystalOrbMatch] = []
    @Published private(set) var selectedID: FrameID?
    @Published private(set) var pair: Pair?
    @Published private(set) var status = "Drop an image to find visually similar frames or photos."
    @Published private(set) var searching = false
    @Published private(set) var loading = false
    @Published private(set) var importing = false
    @Published private(set) var comparisonError: String?
    @Published var candidateMetadata: PhotoMetadata?
    @Published var metadataLoading = false
    @Published var mode: Mode = .sideBySide {
        didSet {
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
    @Published var zoom: CGFloat = 1
    @Published var actualPixels = false
    @Published var pan = CGSize.zero
    @Published var limit = 24
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
    private var searchTask: Task<Void, Never>?
    private var comparisonTask: Task<Void, Never>?
    private var metadataTask: Task<Void, Never>?
    private var importTask: Task<Void, Never>?
    private var searchTicket = UUID()
    private var comparisonTicket = UUID()
    private var importTicket = UUID()
    private(set) var dropTicket = UUID()

    func beginDrop() -> UUID { dropTicket = UUID(); return dropTicket }
    private weak var session: WaifuSession?

    var visibleMatches: [CrystalOrbMatch] { Array(matches.prefix(limit)) }
    var selected: CrystalOrbMatch? { matches.first { $0.id == selectedID } }
    var pairIsCurrent: Bool { pair?.queryID == query?.id && pair?.frame.id == selectedID }

    func open(session: WaifuSession) {
        self.session = session
        if !isOpen { session.previewFrame = nil; session.closeUncover() }
        session.crystalOrbIsOpen = true
        isOpen = true
        sourceChanged(session: session)
    }

    func close() {
        cancelSearch()
        importTask?.cancel(); importTicket = UUID(); dropTicket = UUID()
        comparisonTask?.cancel(); comparisonTicket = UUID()
        metadataTask?.cancel()
        pairCache.removeAllObjects(); metadataCache.removeAll()
        pair = nil; loading = false; metadataLoading = false
        opacityControlFocused = false
        holdingToggle = false
        importing = false
        isOpen = false
        session?.crystalOrbIsOpen = false
        session?.finishCrystalOrbNavigation()
        // Release the potentially large external bitmap; reusable features remain in the engine.
        query = nil; matches = []; selectedID = nil; candidateMetadata = nil
        status = "Drop an image to find visually similar frames or photos."
    }

    func sourceChanged(session: WaifuSession) {
        self.session = session
        guard generation != session.thumbnailGeneration else { return }
        generation = session.thumbnailGeneration
        cancelSearch()
        comparisonTask?.cancel(); comparisonTicket = UUID()
        metadataTask?.cancel()
        pairCache.removeAllObjects(); metadataCache.removeAll()
        matches = []; selectedID = nil; pair = nil; candidateMetadata = nil
        loading = false; metadataLoading = false; comparisonError = nil
        if isOpen, query != nil { search() }
    }

    func importImage(url: URL? = nil, data: Data? = nil, session: WaifuSession) {
        dropTicket = UUID()
        open(session: session)
        importTask?.cancel()
        cancelSearch()
        comparisonTask?.cancel(); comparisonTicket = UUID()
        metadataTask?.cancel()
        // Keep the displayed query, candidate and gallery intact until decoding
        // succeeds. Failed replacements leave the previous workspace usable.
        importing = true
        metadataLoading = false; comparisonError = nil
        status = "Reading query image…"; loading = false
        let ticket = UUID(); importTicket = ticket
        importTask = Task {
            do {
                let result = try await CrystalOrbQueryDecoder.shared.read(url: url, data: data)
                guard !Task.isCancelled, importTicket == ticket, isOpen else { return }
                pairCache.removeAllObjects()
                query = result; matches = []; selectedID = nil; candidateMetadata = nil
                importing = false
                resetZoom(); search()
            } catch {
                guard !Task.isCancelled, importTicket == ticket else { return }
                importing = false
                status = error.localizedDescription
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
    }

    func search() {
        cancelSearch()
        guard !importing, let query, let session else { return }
        guard let source = session.video?.url else { status = "Load a video or open a Photos album to search."; return }
        let frames = session.frames, generation = session.thumbnailGeneration
        guard !frames.isEmpty else { status = "No analyzed candidates yet. Analyze this source, then search again."; return }
        let ticket = UUID(); searchTicket = ticket
        searching = true; status = "Preparing visual index…"; limit = 24
        searchTask = Task {
            do {
                let result = try await engine.search(query: query.image, queryID: query.id, frames: frames, sourceURL: source, generation: generation) { [weak self] done, total in
                    await self?.updateProgress(done: done, total: total, ticket: ticket)
                }
                guard !Task.isCancelled, searchTicket == ticket, session.thumbnailGeneration == generation else { return }
                matches = result; searching = false
                let unavailable = frames.count - result.count
                let unindexed = PhotosMedia.shared.album(source).map { max(0, $0.items.count - frames.count) } ?? 0
                status = "\(result.count) candidates · Search complete" + (unavailable > 0 ? " · \(unavailable) unavailable" : "")
                    + (unindexed > 0 ? " · \(unindexed) not analyzed" : "")
                if let first = result.first { select(first.id) }
                else { status = "No readable candidates in this source." }
            } catch {
                guard !Task.isCancelled, searchTicket == ticket else { return }
                searching = false; status = error.localizedDescription
            }
        }
    }

    private func updateProgress(done: Int, total: Int, ticket: UUID) {
        guard searchTicket == ticket else { return }
        status = "Indexing and comparing \(done) / \(total)…"
    }

    func select(_ id: FrameID) {
        guard !importing, let match = matches.first(where: { $0.id == id }), let query, let session,
              let source = session.video?.url, session.thumbnailGeneration == generation else { return }
        // Repeated clicks and arrow presses at a strip boundary must not restart
        // an original read or recompute an already loaded aligned pair.
        if selectedID == id, loading || (pairIsCurrent && comparisonError == nil) { return }
        selectedID = id
        session.selectCrystalOrbCandidate(id)
        comparisonTask?.cancel(); metadataTask?.cancel()
        let ticket = UUID(); comparisonTicket = ticket
        loading = true; comparisonError = nil; candidateMetadata = metadataCache[id]
        if let cached = pairCache.object(forKey: id.raw as NSString)?.value, cached.queryID == query.id {
            pair = cached
            loading = false
        } else {
            comparisonTask = Task {
                do {
                    let image = try await PhotosMedia.shared.image(source, seconds: match.frame.timeSeconds)
                    guard !Task.isCancelled, comparisonTicket == ticket, session.video?.url == source else { return }
                    let ready = Pair(queryID: query.id, frame: match.frame, target: query.image, candidate: image)
                    // Commit the original and its bounded difference preview together.
                    // Keep the complete previous pair visible during the handoff.
                    do {
                        let difference = try await CrystalOrbPairRenderer.shared.render(target: query.image, candidate: image)
                        guard !Task.isCancelled, comparisonTicket == ticket else { return }
                        let completed = Pair(queryID: query.id, frame: match.frame, target: query.image, candidate: image, difference: difference)
                        let cost = image.bytesPerRow * image.height + difference.difference.bytesPerRow * difference.difference.height
                            + difference.redMask.bytesPerRow * difference.redMask.height
                        pairCache.setObject(CachedPair(completed), forKey: id.raw as NSString, cost: cost)
                        pair = completed
                        loading = false
                    } catch {
                        guard !Task.isCancelled, comparisonTicket == ticket else { return }
                        pair = ready
                        comparisonError = error.localizedDescription; loading = false
                    }
                } catch {
                    guard !Task.isCancelled, comparisonTicket == ticket else { return }
                    comparisonError = error.localizedDescription; loading = false
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
                if comparisonTicket == ticket { metadataLoading = false }
            }
        } else { metadataLoading = false }
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
        guard !matches.isEmpty else { return }
        let index = selectedID.flatMap { id in matches.firstIndex { $0.id == id } } ?? 0
        let next = min(max(0, index + delta), matches.count - 1)
        limit = max(limit, next + 1)
        select(matches[next].id)
    }

    func resetZoom() { zoom = 1; actualPixels = false; pan = .zero }

    func handleKey(_ event: NSEvent) -> Bool {
        guard isOpen else { return false }
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
