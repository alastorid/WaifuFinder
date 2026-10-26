import CoreGraphics
import Foundation
import SwiftUI

/// Zoom works like Photos.app: a ladder of target cell widths. Zooming picks the
/// next rung; the column count falls out of the target and the viewport width.
enum GridZoom {
    static let steps: [CGFloat] = [70, 90, 110, 130, 160, 200, 250, 310]
    static let defaultValue: CGFloat = 130

    static func clamped(_ width: CGFloat) -> CGFloat {
        min(max(width, steps.first ?? defaultValue), steps.last ?? defaultValue)
    }

    /// The nearest rung, so pinches and restored preferences land on the ladder.
    static func snapped(_ width: CGFloat) -> CGFloat {
        clamped(steps.min(by: { abs($0 - width) < abs($1 - width) }) ?? defaultValue)
    }

    static func zoomIn(from width: CGFloat) -> CGFloat {
        clamped(steps.first(where: { $0 > width + 0.5 }) ?? steps.last ?? defaultValue)
    }

    static func zoomOut(from width: CGFloat) -> CGFloat {
        clamped(steps.last(where: { $0 < width - 0.5 }) ?? steps.first ?? defaultValue)
    }
}

/// Marks the transaction that carries a zoom commit, so containers that strip
/// animations from the grid let exactly this one through.
struct GridZoomTransactionKey: TransactionKey {
    static let defaultValue = false
}

struct GridStageKey: Hashable, Sendable {
    let revision: UInt64
    let generation: UUID
    let width: CGFloat
    let columns: Int
    let groupColumns: Int
    let grouped: Bool
    let photosSource: Bool
    let square: Bool
    var targetWidth: CGFloat = GridZoom.defaultValue

    var tokenPrefix: String {
        "\(revision):\(generation):\(width):\(grouped):\(square):"
    }
}

struct GridStageInput: Sendable {
    let key: GridStageKey
    let sourceURL: URL?
    let rows: [FrameRow]
    let frames: [AnalyzedFrame]
    let beyondCap: [FrameGroupID: Int]
    let revealed: Set<FrameGroupID>
}

struct GridStageItem: Identifiable, Sendable {
    let frame: AnalyzedFrame
    let aspect: CGFloat
    var id: FrameID { frame.id }
}

struct GridStageRow: Identifiable, Sendable {
    let id: String
    let items: [GridStageItem]
    let strip: Bool
    let groupID: FrameGroupID?
    let beyond: Int
    let revealed: Bool
    let cellWidth: CGFloat
    let contentHeight: CGFloat
    let originY: CGFloat
    /// The gap drawn below this row. Group strips separate their sections; flat
    /// thumbnail rows sit one pixel apart, matching the cells inside them.
    let spacing: CGFloat
    var height: CGFloat { contentHeight + (strip ? 16 : 0) }
}

/// Geometry and order are immutable until the entire viewport batch commits.
/// Building rows and measuring their extents happens on a worker, not while
/// SwiftUI is laying out a scrolling frame.
struct GridStageSnapshot: Sendable {
    let key: GridStageKey
    let navigation: FrameNavigationOrder
    let sourceURL: URL?
    let rows: [GridStageRow]
    let contentHeight: CGFloat
    let rowIndexByToken: [String: Int]
    let rowIndexByFrameID: [FrameID: Int]

    static func make(_ input: GridStageInput) -> Self {
        let key = input.key
        // Flat cells fill their row; the zoom target is the floor, and the cap
        // keeps a single column from growing past one large thumbnail.
        let flatWidth = min(max((key.width - 2 - CGFloat(key.columns - 1)) / CGFloat(key.columns), key.targetWidth),
                            max(220, key.targetWidth))
        let groupWidth = min(max((key.width - 2 - CGFloat(key.groupColumns - 1)) / CGFloat(key.groupColumns), 80), 220)
        var result: [GridStageRow] = []
        var top: CGFloat = 1
        let prefix = key.generation.uuidString + ":"

        func append(id: String, frames: ArraySlice<AnalyzedFrame>, strip: Bool, group: FrameGroupID?) {
            let spacing: CGFloat = strip ? 12 : 1
            let width = strip ? groupWidth : flatWidth
            let items = frames.map { frame in
                let cached = frame.thumbnailAspectRatio == nil
                    ? GridThumbnailLoader.shared.cachedImage(identity: prefix + frame.thumbnailKey) : nil
                let aspect: CGFloat
                if let ratio = frame.thumbnailAspectRatio { aspect = CGFloat(ratio) }
                else if let cached { aspect = CGFloat(cached.width) / CGFloat(cached.height) }
                else { aspect = 16.0 / 9 }
                return GridStageItem(frame: frame, aspect: aspect.isFinite && aspect > 0 ? aspect : 16.0 / 9)
            }
            let beyond = group.flatMap { input.beyondCap[$0] } ?? 0
            let revealed = group.map { input.revealed.contains($0) } ?? true
            var height = items.map { GridThumbnailLayout.size(width: width, sourceAspect: $0.aspect, square: key.square).height }.max() ?? 0
            if strip && beyond > 0 && !revealed { height = max(height, width * (key.square ? 1 : 9.0 / 16)) }
            let row = GridStageRow(id: id, items: items, strip: strip, groupID: group,
                                   beyond: beyond, revealed: revealed, cellWidth: width,
                                   contentHeight: height, originY: top, spacing: spacing)
            result.append(row)
            top += row.height + spacing
        }

        func chunks(_ frames: [AnalyzedFrame], prefix: String) {
            for start in stride(from: 0, to: frames.count, by: key.columns) {
                if Task.isCancelled { break }
                append(id: prefix + String(start / key.columns),
                       frames: frames[start..<min(start + key.columns, frames.count)], strip: false, group: nil)
            }
        }

        if key.grouped {
            for row in input.rows {
                if Task.isCancelled { break }
                if row.group == nil && key.photosSource { chunks(row.frames, prefix: row.id + ":") }
                else if !row.frames.isEmpty { append(id: row.id, frames: row.frames[...], strip: true, group: row.group) }
            }
        } else { chunks(input.frames, prefix: "flat:") }
        let tokenPrefix = key.tokenPrefix
        let indices = Dictionary(uniqueKeysWithValues: result.enumerated().map { (tokenPrefix + $0.element.id, $0.offset) })
        let frameIndices = Dictionary(uniqueKeysWithValues: result.enumerated().flatMap { index, row in
            row.items.map { ($0.id, index) }
        })
        // The rows the grid draws, as ranges over the navigation sequence: strips
        // and the chunked lines of an ungrouped run. `result`'s items follow
        // `input.frames`/`input.rows` order, so the ranges index that sequence.
        var lineRanges: [Range<Int>] = []
        if key.grouped {
            var start = 0
            for row in result {
                guard !row.items.isEmpty else { continue }
                lineRanges.append(start..<start + row.items.count)
                start += row.items.count
            }
        }
        return Self(key: key, navigation: FrameNavigationOrder(frames: input.frames, rows: input.rows,
                                                               groupRows: key.grouped,
                                                               lineRanges: key.grouped ? lineRanges : nil),
                    sourceURL: input.sourceURL, rows: result,
                    contentHeight: result.isEmpty ? 2 : top - (result.last?.spacing ?? 1) + 1, rowIndexByToken: indices,
                    rowIndexByFrameID: frameIndices)
    }

    /// Project the NEW order into the OLD viewport, including scroll clamping
    /// after filtering and each horizontal group's current scroll position.
    func visibleItems(in viewport: CGRect, horizontalOffsets: [String: CGFloat]) -> [GridStageItem] {
        guard !rows.isEmpty, viewport.height > 0 else { return [] }
        let top = min(max(0, viewport.minY), max(0, contentHeight - viewport.height))
        let bottom = top + viewport.height
        var low = 0, high = rows.count
        while low < high {
            let middle = (low + high) / 2
            if rows[middle].originY + rows[middle].height <= top { low = middle + 1 }
            else { high = middle }
        }
        var result: [GridStageItem] = []
        for index in low..<rows.count {
            let row = rows[index]
            if row.originY >= bottom { break }
            if !row.strip { result.append(contentsOf: row.items); continue }
            let stride = row.cellWidth + 1
            let extra = row.beyond > 0 && !row.revealed ? 1 : 0
            let width = CGFloat(row.items.count + extra) * stride - 1
            let left = min(max(0, horizontalOffsets[row.id] ?? 0), max(0, width - viewport.width))
            let first = min(row.items.count, max(0, Int(floor(left / stride))))
            let end = min(row.items.count, max(first, Int(ceil((left + viewport.width) / stride))))
            result.append(contentsOf: row.items[first..<end])
        }
        return result
    }
}

private final class GridPreparedImages {
    let owner = UUID()
    private let loader = GridThumbnailLoader.shared
    private var images: [String: CGImage] = [:]

    deinit { releaseAll() }
    func contains(_ identity: String) -> Bool { images[identity] != nil }
    func pin(_ image: CGImage, identity: String) {
        guard images[identity] == nil else { return }
        images[identity] = image
        loader.retainVisibleImage(image, identity: identity, owner: owner)
    }
    func keepOnly(_ wanted: Set<String>) {
        for identity in images.keys.filter({ !wanted.contains($0) }) {
            images.removeValue(forKey: identity)
            loader.releaseVisibleImage(identity: identity, owner: owner)
        }
    }
    func settle() {
        for identity in images.keys { loader.finishReveal(identity: identity) }
    }
    func releaseAll() {
        for identity in images.keys { loader.releaseVisibleImage(identity: identity, owner: owner) }
        images.removeAll()
    }
}

/// Holds the last displayed layout while preparing its replacement. A deadline
/// commits ready pixels together; unfinished cells then use the ordinary late
/// scroll-loading fade. Superseded requests never become a displayed baseline.
@MainActor
final class GridThumbnailStage: ObservableObject {
    @Published private(set) var current: GridStageSnapshot?
    /// Runs in the same main-actor transaction as the visible layout handoff.
    var onCommit: ((GridStageSnapshot) -> Void)?
    private(set) var previous: GridStageSnapshot?
    var isPreparing: Bool { request != nil }

    private final class Request {
        let ticket = UUID()
        let key: GridStageKey
        let deadline: ContinuousClock.Instant
        /// Zoom commits animate the reflow; scroll and filter commits must not.
        let animated: Bool
        let load: @MainActor (AnalyzedFrame) async -> CGImage?
        let pixels = GridPreparedImages()
        var snapshot: GridStageSnapshot?
        var wanted: [String: AnalyzedFrame] = [:]
        var attempted: Set<String> = []
        var jobs: [String: Task<Void, Never>] = [:]
        var build: Task<GridStageSnapshot, Never>?
        var install: Task<Void, Never>?
        var timeout: Task<Void, Never>?

        init(key: GridStageKey, deadline: ContinuousClock.Instant, animated: Bool,
             load: @escaping @MainActor (AnalyzedFrame) async -> CGImage?) {
            self.key = key; self.deadline = deadline; self.animated = animated; self.load = load
        }
        func cancel() {
            build?.cancel(); install?.cancel(); timeout?.cancel()
            for job in jobs.values { job.cancel() }
            jobs.removeAll()
        }
    }

    private var request: Request?
    private var viewport = CGRect.zero
    private var horizontalOffsets: [String: CGFloat] = [:]
    private var handoffs: [UUID: Task<Void, Never>] = [:]
    private let preparationLimit: Duration
    private let readTime: @MainActor () -> ContinuousClock.Instant

    init(preparationLimit: Duration = .milliseconds(300),
         now: @escaping @MainActor () -> ContinuousClock.Instant = { ContinuousClock.now }) {
        self.preparationLimit = preparationLimit
        self.readTime = now
    }

    func submit(_ input: GridStageInput, viewportSize: CGSize, animated: Bool = false,
                load: @escaping @MainActor (AnalyzedFrame) async -> CGImage?) {
        guard request?.key != input.key else { return }
        request?.cancel()
        request = nil
        guard current?.key != input.key else { return }
        Diag.trace(.grid, message: "stage submit revision=\(input.key.revision) frames=\(input.frames.count) viewport=\(viewportSize) animated=\(animated)")
        if viewport.height <= 0 { viewport = CGRect(origin: .zero, size: viewportSize) }
        if let current, current.key.generation != input.key.generation {
            viewport = CGRect(origin: .zero, size: viewportSize)
            horizontalOffsets.removeAll()
        }
        let next = Request(key: input.key, deadline: readTime().advanced(by: preparationLimit),
                           animated: animated, load: load)
        request = next
        let ticket = next.ticket
        let work = Task.detached(priority: .userInitiated) { GridStageSnapshot.make(input) }
        next.build = work
        next.install = Task { [weak self] in
            let snapshot = await work.value
            guard !Task.isCancelled, let self, self.request?.ticket == ticket else { return }
            self.request?.snapshot = snapshot
            self.refresh()
        }
        let deadline = next.deadline
        next.timeout = Task { [weak self] in
            do { try await ContinuousClock().sleep(until: deadline) }
            catch { return }
            guard let self, self.request?.ticket == ticket else { return }
            self.refresh()
        }
    }

    func updateViewport(_ rect: CGRect) {
        guard rect != viewport else { return }
        viewport = rect
        if request?.snapshot != nil { refresh() }
    }

    func updateHorizontalOffset(row: String, offset: CGFloat) {
        guard horizontalOffsets[row] != offset else { return }
        horizontalOffsets[row] = offset
        if request?.snapshot != nil { refresh() }
    }

    func stop() {
        request?.cancel(); request = nil
        for task in handoffs.values { task.cancel() }
        handoffs.removeAll()
    }

    private func refresh() {
        guard let request, let snapshot = request.snapshot else { return }
        let prefix = snapshot.key.generation.uuidString + ":"
        let items = snapshot.visibleItems(in: viewport, horizontalOffsets: horizontalOffsets)
        request.wanted = Dictionary(items.map { (prefix + $0.frame.thumbnailKey, $0.frame) }, uniquingKeysWith: { first, _ in first })
        let wanted = Set(request.wanted.keys)
        request.pixels.keepOnly(wanted)
        for identity in request.jobs.keys.filter({ !wanted.contains($0) }) {
            request.jobs.removeValue(forKey: identity)?.cancel()
        }
        for identity in wanted where !request.pixels.contains(identity) {
            if let image = GridThumbnailLoader.shared.cachedImage(identity: identity) {
                request.pixels.pin(image, identity: identity)
            }
        }
        if wanted.allSatisfy({ request.pixels.contains($0) }) || readTime() >= request.deadline {
            commit(request, snapshot: snapshot)
            return
        }
        for (identity, frame) in request.wanted where request.jobs.count < 4
            && !request.pixels.contains(identity) && request.jobs[identity] == nil && !request.attempted.contains(identity) {
            let ticket = request.ticket
            let load = request.load
            request.jobs[identity] = Task { [weak self] in
                let image = await load(frame)
                guard !Task.isCancelled, let self, let active = self.request, active.ticket == ticket else { return }
                active.jobs.removeValue(forKey: identity)
                active.attempted.insert(identity)
                if let image, active.wanted[identity] != nil { active.pixels.pin(image, identity: identity) }
                self.refresh()
            }
        }
    }

    private func commit(_ request: Request, snapshot: GridStageSnapshot) {
        guard self.request?.ticket == request.ticket else { return }
        Diag.trace(.grid, message: "stage commit revision=\(snapshot.key.revision) frames=\(snapshot.navigation.frames.count) rows=\(snapshot.rows.count) wanted=\(request.wanted.count) ready=\(request.wanted.keys.filter { request.pixels.contains($0) }.count)")
        request.cancel()
        self.request = nil
        previous = current
        request.pixels.settle()
        if request.animated {
            // The reflow rides a normal animation; the transaction key lets the
            // containers that strip grid animations wave it through.
            var transaction = Transaction(animation: .snappy(duration: 0.28))
            transaction[GridZoomTransactionKey.self] = true
            withTransaction(transaction) {
                onCommit?(snapshot)
                current = snapshot
            }
        } else {
            var transaction = Transaction(animation: nil)
            transaction.disablesAnimations = true
            withTransaction(transaction) {
                onCommit?(snapshot)
                current = snapshot
            }
        }
        // Keep prepared pixels leased until the newly mounted cells acquire
        // their own leases; NSCache eviction cannot tear the atomic handoff.
        let lease = request.pixels
        let ticket = request.ticket
        handoffs[ticket] = Task { [weak self, lease] in
            try? await Task.sleep(for: .milliseconds(350))
            lease.releaseAll()
            self?.handoffs.removeValue(forKey: ticket)
        }
    }
}
