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

    var geometryKey: Self {
        Self(revision: 0, generation: generation, width: width, columns: columns,
             groupColumns: groupColumns, grouped: grouped, photosSource: photosSource,
             square: square, targetWidth: targetWidth)
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

    func rekeyed(_ key: GridStageKey) -> Self {
        Self(key: key, navigation: navigation, sourceURL: sourceURL, rows: rows,
             contentHeight: contentHeight,
             rowIndexByToken: Dictionary(uniqueKeysWithValues: rows.enumerated().map { (key.tokenPrefix + $0.element.id, $0.offset) }),
             rowIndexByFrameID: rowIndexByFrameID)
    }

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
        var start = 0
        for row in result {
            guard !row.items.isEmpty else { continue }
            lineRanges.append(start..<start + row.items.count)
            start += row.items.count
        }
        return Self(key: key, navigation: FrameNavigationOrder(frames: input.frames, rows: input.rows,
                                                               groupRows: key.grouped,
                                                               lineRanges: lineRanges),
                    sourceURL: input.sourceURL, rows: result,
                    contentHeight: result.isEmpty ? 2 : top - (result.last?.spacing ?? 1) + 1, rowIndexByToken: indices,
                    rowIndexByFrameID: frameIndices)
    }

    /// Remove frames from the committed rows immediately. This is a projection
    /// of the current geometry, not a new arrangement: it exists so a deletion
    /// can shift surviving cells left in the same transaction instead of waiting
    /// for the background ranking and group-cap pass.
    func removing(_ ids: Set<FrameID>) -> Self? {
        guard !ids.isEmpty,
              rows.contains(where: { row in row.items.contains { ids.contains($0.id) } }) else { return nil }
        var result: [GridStageRow] = []
        var top: CGFloat = 1

        func append(_ id: String, _ items: [GridStageItem], strip: Bool, group: FrameGroupID?,
                    beyond: Int, revealed: Bool, cellWidth: CGFloat, spacing: CGFloat) {
            guard !items.isEmpty else { return }
            var height = items.map {
                GridThumbnailLayout.size(width: cellWidth, sourceAspect: $0.aspect, square: key.square).height
            }.max() ?? 0
            if strip && beyond > 0 && !revealed {
                height = max(height, cellWidth * (key.square ? 1 : 9.0 / 16))
            }
            result.append(GridStageRow(id: id, items: items, strip: strip, groupID: group,
                beyond: beyond, revealed: revealed, cellWidth: cellWidth, contentHeight: height,
                originY: top, spacing: spacing))
            top += height + (strip ? 16 : 0) + spacing
        }

        var chunkWidth = rows.first?.cellWidth ?? 0
        func appendChunks(_ items: [GridStageItem], prefix: String) {
            let count = max(1, key.columns)
            var start = 0
            while start < items.count {
                let end = min(start + count, items.count)
                append(prefix + String(start / count), Array(items[start..<end]),
                       strip: false, group: nil, beyond: 0, revealed: true,
                       cellWidth: chunkWidth, spacing: 1)
                start = end
            }
        }

        if key.grouped {
            var buffered: [GridStageItem] = []
            var bufferedPrefix: String?
            func flush() {
                guard let prefix = bufferedPrefix, !buffered.isEmpty else { return }
                appendChunks(buffered, prefix: prefix)
                buffered.removeAll(keepingCapacity: true)
                bufferedPrefix = nil
            }
            for row in rows {
                let items = row.items.filter { !ids.contains($0.id) }
                if row.strip {
                    flush()
                    append(row.id, items, strip: true, group: row.groupID,
                           beyond: row.beyond, revealed: row.revealed,
                           cellWidth: row.cellWidth, spacing: row.spacing)
                    continue
                }
                let prefix = row.id.lastIndex(of: ":").map { String(row.id[...$0]) } ?? row.id + ":"
                if bufferedPrefix != prefix {
                    flush()
                    bufferedPrefix = prefix
                }
                chunkWidth = row.cellWidth
                buffered.append(contentsOf: items)
            }
            flush()
        } else {
            appendChunks(rows.flatMap { $0.items }.filter { !ids.contains($0.id) }, prefix: "flat:")
        }

        let frames = result.flatMap { $0.items.map(\.frame) }
        let navigationRows = result.map { FrameRow(group: $0.groupID, frames: $0.items.map(\.frame)) }
        var lineRanges: [Range<Int>] = []
        var lineStart = 0
        for row in result {
            lineRanges.append(lineStart..<lineStart + row.items.count)
            lineStart += row.items.count
        }
        let navigation = FrameNavigationOrder(frames: frames, rows: navigationRows,
                                              groupRows: key.grouped, lineRanges: lineRanges)
        let tokenPrefix = key.tokenPrefix
        let indices = Dictionary(uniqueKeysWithValues: result.enumerated().map { (tokenPrefix + $0.element.id, $0.offset) })
        let frameIndices = Dictionary(uniqueKeysWithValues: result.enumerated().flatMap { index, row in
            row.items.map { ($0.id, index) }
        })
        return Self(key: key, navigation: navigation, sourceURL: sourceURL, rows: result,
                    contentHeight: result.isEmpty ? 2 : top - (result.last?.spacing ?? 1) + 1,
                    rowIndexByToken: indices, rowIndexByFrameID: frameIndices)
    }

    /// Use the same content bounds for thumbnail preparation and the actual
    /// ScrollView handoff. Clamping preparation alone leaves lazy rows offscreen.
    func clampedViewport(_ viewport: CGRect) -> CGRect {
        var result = viewport
        result.origin.y = min(max(0, viewport.minY), max(0, contentHeight - viewport.height))
        return result
    }

    /// Project the NEW order into the OLD viewport, including scroll clamping
    /// after filtering and each horizontal group's current scroll position.
    func visibleItems(in viewport: CGRect, horizontalOffsets: [String: CGFloat]) -> [GridStageItem] {
        visibleRowRange(in: viewport).flatMap {
            items(inRow: $0, width: viewport.width, horizontalOffsets: horizontalOffsets)
        }
    }

    func visibleRowRange(in viewport: CGRect) -> Range<Int> {
        guard !rows.isEmpty, viewport.height > 0 else { return 0..<0 }
        let top = clampedViewport(viewport).minY
        let bottom = top + viewport.height
        var low = 0, high = rows.count
        while low < high {
            let middle = (low + high) / 2
            if rows[middle].originY + rows[middle].height <= top { low = middle + 1 }
            else { high = middle }
        }
        var end = low
        while end < rows.count, rows[end].originY < bottom { end += 1 }
        return low..<end
    }

    func items(inRow index: Int, width viewportWidth: CGFloat,
               horizontalOffsets: [String: CGFloat]) -> ArraySlice<GridStageItem> {
        let row = rows[index]
        guard row.strip else { return row.items[...] }
        let stride = row.cellWidth + 1
        let extra = row.beyond > 0 && !row.revealed ? 1 : 0
        let width = CGFloat(row.items.count + extra) * stride - 1
        let left = min(max(0, horizontalOffsets[row.id] ?? 0), max(0, width - viewportWidth))
        let first = min(row.items.count, max(0, Int(floor(left / stride))))
        let end = min(row.items.count, max(first, Int(ceil((left + viewportWidth) / stride))))
        return row.items[first..<end]
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

private struct GridDeletionSpeculation: Sendable {
    let input: GridStageInput
    let snapshot: GridStageSnapshot
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
        let immediate: Bool
        let load: @MainActor (AnalyzedFrame) async -> CGImage?
        let pixels = GridPreparedImages()
        /// Deletions queued while this newer layout is still preparing. They are
        /// projected away both from the visible grid now and from the incoming
        /// snapshot when it commits, so a fast second Delete cannot resurrect a
        /// cell that the pre-deletion layout still contains.
        var removals: Set<FrameID> = []
        var snapshot: GridStageSnapshot?
        var wanted: [String: AnalyzedFrame] = [:]
        var attempted: Set<String> = []
        var jobs: [String: Task<Void, Never>] = [:]
        var build: Task<GridStageSnapshot, Never>?
        var install: Task<Void, Never>?
        var timeout: Task<Void, Never>?

        init(key: GridStageKey, deadline: ContinuousClock.Instant, animated: Bool, immediate: Bool,
             load: @escaping @MainActor (AnalyzedFrame) async -> CGImage?) {
            self.key = key; self.deadline = deadline; self.animated = animated; self.immediate = immediate; self.load = load
        }
        func cancel() {
            build?.cancel(); install?.cancel(); timeout?.cancel()
            for job in jobs.values { job.cancel() }
            jobs.removeAll()
        }
    }

    private var request: Request?
    private var speculations: [GridDeletionSpeculation] = []
    private var speculationTask: Task<GridDeletionSpeculation?, Never>?
    private var frozenSpeculationTask: Task<GridDeletionSpeculation?, Never>?
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

    /// Prepare the likely single-selection deletion without changing the grid.
    /// The result is used only if the actual arrangement still matches exactly.
    func prepareDeletion(_ input: GridStageInput, removing ids: Set<FrameID>,
                         build: @escaping @Sendable (GridStageInput) -> GridStageSnapshot = { GridStageSnapshot.make($0) }) {
        speculationTask?.cancel()
        guard !ids.isEmpty else { return }
        let work = Task.detached(priority: .utility) { () -> GridDeletionSpeculation? in
            guard !Task.isCancelled else { return nil }
            let frames = input.frames.filter { !ids.contains($0.id) }
            let rows = input.rows.map { FrameRow(group: $0.group, frames: $0.frames.filter { !ids.contains($0.id) }) }
                .filter { !$0.frames.isEmpty }
            let projected = GridStageInput(key: input.key, sourceURL: input.sourceURL, rows: rows,
                frames: frames, beyondCap: input.beyondCap, revealed: input.revealed)
            guard !Task.isCancelled else { return nil }
            let snapshot = build(projected)
            guard !Task.isCancelled else { return nil }
            return GridDeletionSpeculation(input: projected, snapshot: snapshot)
        }
        speculationTask = work
        Task { [weak self] in
            guard let result = await work.value, !work.isCancelled, let self else { return }
            self.speculations.append(result)
            if self.speculations.count > 4 { self.speculations.removeFirst() }
        }
    }

    /// Keep the selected photo's in-flight work alive when focus moves to its
    /// replacement; the next deletion submission awaits this exact task.
    func freezeDeletionPreparation() {
        frozenSpeculationTask?.cancel()
        frozenSpeculationTask = speculationTask
        speculationTask = nil
    }

    func preparedDeletion(for input: GridStageInput) -> GridStageSnapshot? {
        guard let candidate = speculations.last(where: { Self.matches($0.input, input) }) else { return nil }
        return candidate.snapshot.rekeyed(input.key)
    }

    private nonisolated static func matches(_ candidate: GridStageInput, _ input: GridStageInput) -> Bool {
        candidate.key.geometryKey == input.key.geometryKey && candidate.sourceURL == input.sourceURL
            && candidate.frames == input.frames && candidate.rows == input.rows
            && candidate.beyondCap == input.beyondCap && candidate.revealed == input.revealed
    }

    /// Paint the deletion now: the committed rows lose the deleted cells and
    /// surviving cells shift into their places in one animated transaction. The
    /// session's authoritative arrangement replaces this projection when its
    /// background ranking and group-cap calculation finishes.
    func applyImmediateDeletion(removing ids: Set<FrameID>) {
        guard let snapshot = current, let projected = snapshot.removing(ids) else { return }
        if let request, request.key.generation == snapshot.key.generation {
            // Any request submitted after the displayed layout is newer than it.
            // Keep it from restoring frames this deletion removes when it commits;
            // geometry-only resizes and zooms take this path too.
            request.removals.formUnion(ids)
            if let staged = request.snapshot {
                request.snapshot = staged.removing(ids) ?? staged
            }
        }
        previous = current
        Diag.trace(.grid, message: "immediate deletion removed=\(ids.count) rows=\(projected.rows.count)")
        var transaction = Transaction(animation: .easeInOut(duration: 0.33))
        transaction[GridZoomTransactionKey.self] = true
        withTransaction(transaction) {
            onCommit?(projected)
            current = projected
        }
    }

    func submit(_ input: GridStageInput, viewportSize: CGSize, animated: Bool = false, immediate: Bool = false,
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
                           animated: animated, immediate: immediate, load: load)
        request = next
        let ticket = next.ticket
        let frozen = immediate ? frozenSpeculationTask : nil
        if immediate { frozenSpeculationTask = nil }
        next.install = Task { [weak self] in
            let awaited = await withTaskCancellationHandler { await frozen?.value } onCancel: { frozen?.cancel() }
            guard !Task.isCancelled, let self, self.request?.ticket == ticket else { return }
            let candidates = immediate ? self.speculations + (awaited.map { [$0] } ?? []) : []
            let work = Task.detached(priority: .userInitiated) {
                if let match = candidates.last(where: { Self.matches($0.input, input) }) {
                    return match.snapshot.rekeyed(input.key)
                }
                return GridStageSnapshot.make(input)
            }
            self.request?.build = work
            let snapshot = await withTaskCancellationHandler { await work.value } onCancel: { work.cancel() }
            guard !Task.isCancelled, self.request?.ticket == ticket else { return }
            let removals = self.request?.removals ?? []
            self.request?.snapshot = removals.isEmpty ? snapshot : (snapshot.removing(removals) ?? snapshot)
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
        speculationTask?.cancel(); speculationTask = nil
        frozenSpeculationTask?.cancel(); frozenSpeculationTask = nil
        speculations.removeAll()
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
        if request.immediate || wanted.allSatisfy({ request.pixels.contains($0) }) || readTime() >= request.deadline {
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
            var transaction = Transaction(animation: .easeInOut(duration: 0.33))
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
