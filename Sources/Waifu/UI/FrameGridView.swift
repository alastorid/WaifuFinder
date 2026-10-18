import Foundation
import SwiftUI

/// The frame grid — the candidates.
///
/// Two layouts over the same group-preserving order, switched by "One group per line":
///
///   * **Rows**: one line per group, each a horizontal strip of that group's frames,
///     with the frames that belong to no group on a line of their own at the end.
///     A group is a run of near-identical frames, and seeing one group at a time is
///     how you choose between them.
///   * **Flow**: one adaptive grid where each group's frames stay together, with
///     ungrouped frames after the groups.
///
/// Cells show the decoded source frame number. A thick blue border sits inside
/// selected thumbnails; keyboard browsing moves the selection with the focus.
///
/// Keyboard (when focused): ←/→ walk the list, ↑/↓ jump group boundaries, **Space
/// opens the zoomable preview** (it no longer toggles selection — selecting has its
/// own click, and the preview is what a frame deserves when you stop on it), Home and
/// ⌘←/↑ focus the first frame, End and ⌘→/↓ focus the last, and Return selects.
struct FrameGridView: View, Equatable {
    /// The root observes the full session for toolbar and player state. Keep this
    /// grid itself keyed only to arrangement changes so a focus/selection update
    /// doesn't diff every item in a large filtered result.
    let session: WaifuSession
    let gridRevision: UInt64
    let squarePhotoThumbnails: Bool

    /// Called when the user clicks a frame (single-click replaces selection,
    /// modifier clicks toggle or extend selection; Return also selects).
    let onFrameClick: (AnalyzedFrame, ClickKind) -> Void

    nonisolated static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.gridRevision == rhs.gridRevision && lhs.squarePhotoThumbnails == rhs.squarePhotoThumbnails
    }

    /// Whether the grid has keyboard focus.
    @FocusState private var isFocused: Bool
    @StateObject private var thumbnailPrefetcher = GridThumbnailPrefetcher()

    private let spacing: CGFloat = 1
    private let padding: CGFloat = 1
    private let minCellWidth: CGFloat = 130
    private let maxCellWidth: CGFloat = 220

    private var usesSquarePhotoThumbnails: Bool {
        session.photosAlbumURL != nil && squarePhotoThumbnails
    }

    var body: some View {
        GeometryReader { geometry in
            ScrollViewReader { proxy in
                ScrollView(.vertical, showsIndicators: true) {
                    switchLayout(in: geometry.size.width)
                }
                .coordinateSpace(name: "frame-grid-viewport")
                .focusable()
                .focusEffectDisabled()
                .focused($isFocused)
                .onKeyPress(keys: [.leftArrow, .rightArrow, .upArrow, .downArrow, .home, .end, .return]) { press in
                    handleKeyPress(press)
                }
                .onKeyPress(keys: ["a"]) { press in
                    guard press.modifiers.contains(.command),
                          !press.modifiers.contains(.option), !press.modifiers.contains(.control),
                          session.previewFrame == nil, session.uncover == nil else { return .ignored }
                    session.selectAllDisplayedFrames()
                    return .handled
                }
                .onChange(of: session.frames.isEmpty) { _, empty in
                    if !empty { isFocused = true }
                }
                .onAppear { configureThumbnailPrefetch(width: geometry.size.width) }
                .onDisappear { thumbnailPrefetcher.stop() }
                .onChange(of: gridRevision) { _, _ in
                    configureThumbnailPrefetch(width: geometry.size.width)
                }
                .onChange(of: geometry.size.width) { _, width in
                    configureThumbnailPrefetch(width: width)
                }
                .onChange(of: session.usesGroupRows) { _, _ in
                    configureThumbnailPrefetch(width: geometry.size.width)
                }
                .onChange(of: session.thumbnailGeneration) { _, _ in
                    configureThumbnailPrefetch(width: geometry.size.width)
                }
                .onPreferenceChange(GridRowViewportPreferenceKey.self) { rowViewports in
                    let visible = Set(rowViewports.compactMap { token, rect -> String? in
                        rect.maxY > 0 && rect.minY < geometry.size.height ? token : nil
                    })
                    thumbnailPrefetcher.setVisibleRows(visible)
                }
                .onReceive(session.gridScrollRequests) { id in
                    guard session.previewFrame == nil, session.uncover == nil, id == session.focusedFrameID,
                          id == session.gridNavigationFrameID else { return }
                    scroll(to: id, proxy: proxy)
                }
                // @Published emits before its setter commits. Scrolling here can
                // force a render; defer it so that render sees the closed preview.
                .onReceive(session.previewFrameUpdates.dropFirst()) { frame in
                    guard frame == nil, session.previewFrame == nil else { return }
                    restoreKeyboardFocus(reason: "preview closed")
                }
                // Uncover can move focus into its controls and closes independently
                // of previewFrame, so restore grid navigation on this path too.
                .onReceive(session.$uncover.dropFirst().receive(on: DispatchQueue.main)) { uncover in
                    guard uncover == nil, session.uncover == nil else { return }
                    restoreKeyboardFocus(reason: "uncover closed")
                }
            }
        }
        .background(Color(nsColor: .textBackgroundColor))
        .overlay {
            if session.displayedFrames.isEmpty && !session.frames.isEmpty {
                emptyState
            }
        }
        .onAppear { if !session.frames.isEmpty { isFocused = true } }
    }

    // MARK: - Layout

    @ViewBuilder
    private func switchLayout(in width: CGFloat) -> some View {
        let rows = renderRows(in: width)
        let prefetchPrefix = thumbnailPlanKey(width: width).tokenPrefix
        let rowSpacing: CGFloat = session.usesGroupRows ? 12 : spacing
        LazyVStack(alignment: .leading, spacing: rowSpacing) {
            ForEach(rows) { row in
                let token = prefetchPrefix + row.id
                if let group = row.group {
                    rowView(group, width: width, prefetchToken: token)
                        .id(row.id)
                } else {
                    HStack(spacing: spacing) {
                        ForEach(row.frames) { frame in
                            cell(frame, width: cellWidth(in: width))
                                .id(frame.id)
                        }
                    }
                    .background {
                        GeometryReader { proxy in
                            Color.clear.preference(
                                key: GridRowViewportPreferenceKey.self,
                                value: [token: proxy.frame(in: .named("frame-grid-viewport"))]
                            )
                        }
                    }
                    .id(row.id)
                }
            }
        }
        .padding(padding)
    }

    private func rowView(_ row: FrameRow, width: CGFloat, prefetchToken: String) -> some View {
        let columns = rowColumnCount
        let available = width - padding * 2 - spacing * CGFloat(columns - 1)
        let cellWidth = min(max(available / CGFloat(columns), 80), maxCellWidth)
        let beyond = row.group.flatMap { session.beyondCapByGroup[$0] } ?? 0
        let revealed = row.group.map { session.revealedGroups.contains($0) } ?? true
        let squarePhotos = usesSquarePhotoThumbnails

        return VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Text(row.group == nil
                     ? "\(row.frames.count) ungrouped"
                     : "\(row.frames.count) frame\(row.frames.count == 1 ? "" : "s")")
                if beyond > 0 && !revealed {
                    Text("· \(beyond) more")
                        .foregroundStyle(.tertiary)
                }
            }
            .font(.system(size: 10))
            .foregroundStyle(.secondary)

            // One line per group: a wide group scrolls sideways rather than wrapping
            // into the next group's line.
            ScrollViewReader { proxy in
                ScrollView(.horizontal, showsIndicators: false) {
                    LazyHStack(spacing: spacing) {
                        ForEach(row.frames) { frame in
                            cell(frame, width: cellWidth)
                                .id(frame.id)
                        }
                        if let group = row.group, beyond > 0, !revealed {
                            moreTile(count: beyond, width: cellWidth, group: group, square: squarePhotos)
                        }
                    }
                }
                .frame(height: cellWidth * CGFloat(max(squarePhotos ? 1 : session.rowHeightRatio(for: row.id),
                    beyond > 0 && !revealed ? (squarePhotos ? 1 : 9.0 / 16) : 0)))
                .onAppear {
                    if let id = session.gridNavigationFrameID, id == session.focusedFrameID,
                       session.previewFrame == nil, session.uncover == nil, session.rowID(for: id) == row.id {
                        proxy.scrollTo(id, anchor: .center)
                    }
                }
                .onReceive(session.gridScrollRequests) { id in
                    guard session.previewFrame == nil, session.uncover == nil, id == session.focusedFrameID,
                          id == session.gridNavigationFrameID else { return }
                    if session.rowID(for: id) == row.id {
                        proxy.scrollTo(id, anchor: .center)
                    }
                }
            }
        }
        .background {
            GeometryReader { proxy in
                Color.clear.preference(
                    key: GridRowViewportPreferenceKey.self,
                        value: [prefetchToken: proxy.frame(in: .named("frame-grid-viewport"))]
                )
            }
        }
    }

    /// How many cells a row is sized for: the per-group cap, so a group that is at the
    /// cap fills its line exactly and a bigger one (revealed) scrolls.
    private var rowColumnCount: Int {
        max(2, min(session.settings.perGroupFrameCount, 8))
    }

    private func cellWidth(in width: CGFloat) -> CGFloat {
        let usable = width - padding * 2
        let count = max(1, Int(usable / (minCellWidth + spacing)))
        let raw = (usable - spacing * CGFloat(count - 1)) / CGFloat(count)
        return min(max(raw, minCellWidth), maxCellWidth)
    }

    private func columnCount(in width: CGFloat) -> Int {
        max(1, Int((width - padding * 2) / (minCellWidth + spacing)))
    }

    private func chunked(_ frames: [AnalyzedFrame], by count: Int) -> [[AnalyzedFrame]] {
        guard count > 0, !frames.isEmpty else { return [] }
        return stride(from: 0, to: frames.count, by: count).map { start in
            Array(frames[start..<min(start + count, frames.count)])
        }
    }

    /// Makes the vertical rows explicit so the prefetch window follows the same
    /// row boundaries as the grid. Group strips stay one row; ungrouped Photos and
    /// the flat layout use the current adaptive column count.
    private func renderRows(in width: CGFloat) -> [GridRenderRow] {
        let columns = columnCount(in: width)
        if session.usesGroupRows {
            var result: [GridRenderRow] = []
            for groupRow in session.displayedRows {
                if groupRow.group == nil && session.photosAlbumURL != nil {
                    for (index, frames) in chunked(groupRow.frames, by: columns).enumerated() {
                        result.append(GridRenderRow(id: "\(groupRow.id):\(index)", frames: frames, group: nil))
                    }
                } else if !groupRow.frames.isEmpty {
                    result.append(GridRenderRow(id: groupRow.id,
                                                frames: groupRow.frames, group: groupRow))
                }
            }
            return result
        }

        return chunked(session.displayOrderFrames, by: columns).enumerated().map { index, frames in
            GridRenderRow(id: "flat:\(index)", frames: frames, group: nil)
        }
    }

    private func thumbnailPlanKey(width: CGFloat) -> GridThumbnailPrefetcher.PlanKey {
        GridThumbnailPrefetcher.PlanKey(
            revision: gridRevision,
            generation: session.thumbnailGeneration,
            columns: columnCount(in: width),
            grouped: session.usesGroupRows,
            photosSource: session.photosAlbumURL != nil
        )
    }

    private func configureThumbnailPrefetch(width: CGFloat) {
        let key = thumbnailPlanKey(width: width)
        guard thumbnailPrefetcher.needsPlan(key) else { return }
        let rows = renderRows(in: width)
        thumbnailPrefetcher.install(rows: rows, key: key, session: session)
    }

    // MARK: - Cells

    private func cell(_ frame: AnalyzedFrame, width: CGFloat) -> some View {
        FrameCell(
            session: session,
            frame: frame,
            thumbnailKey: frame.thumbnailKey,
            generation: session.thumbnailGeneration,
            width: width,
            isPhotosSource: session.photosAlbumURL != nil,
            squareThumbnail: usesSquarePhotoThumbnails
        )
        .gesture(
            TapGesture().modifiers([.command, .shift]).onEnded { click(frame, .additiveRange) }
                .exclusively(before: TapGesture().modifiers(.shift).onEnded { click(frame, .range) })
                .exclusively(before: TapGesture().modifiers(.command).onEnded { click(frame, .toggle) })
                .exclusively(before: TapGesture().onEnded { click(frame, .single) })
        )
        .onTapGesture(count: 2) { click(frame, .double) }
        .contextMenu { contextMenu(for: frame) }
    }

    private func click(_ frame: AnalyzedFrame, _ kind: ClickKind) {
        if !isFocused { isFocused = true }
        onFrameClick(frame, kind)
    }

    /// Restore arrow/Return navigation after an overlay control took focus.
    /// Space/Escape are handled by the persistent window monitor.
    private func restoreKeyboardFocus(reason: String) {
        guard !session.frames.isEmpty, session.uncover == nil else { return }
        isFocused = true
        Diag.trace(.grid, .focus, message: "restore focus (\(reason)) | \(Diag.responderDescription())")
    }

    /// The tail of a capped group: what was left out, and a way to see it.
    private func moreTile(count: Int, width: CGFloat, group: FrameGroupID, square: Bool) -> some View {
        Button {
            session.revealedGroups.insert(group)
        } label: {
            VStack(spacing: 2) {
                Text("+\(count)")
                    .font(.system(size: 15, weight: .medium, design: .rounded))
                Text("more")
                    .font(.system(size: 10))
            }
            .frame(width: width, height: width * (square ? 1 : 9 / 16))
            .foregroundStyle(.secondary)
            .background(Color.secondary.opacity(0.08),
                        in: RoundedRectangle(cornerRadius: 3, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: 3, style: .continuous)
                    .strokeBorder(Color.secondary.opacity(0.25), lineWidth: 1)
            }
        }
        .buttonStyle(.plain)
        .help("Show every frame of this group")
    }

    // MARK: - Keyboard

    /// Grid navigation is active only while neither overlay is up.
    private func handleKeyPress(_ press: KeyPress) -> KeyPress.Result {
        guard session.previewFrame == nil, session.uncover == nil else { return .ignored }
        let result: KeyPress.Result
        let commandOnly = press.modifiers.contains(.command)
            && !press.modifiers.contains(.shift)
            && !press.modifiers.contains(.option)
            && !press.modifiers.contains(.control)
        switch press.key {
        case .leftArrow:
            if commandOnly { session.focusDisplayedBoundary(forward: false) }
            else { session.focusNeighbour(-1) }
            result = .handled
        case .rightArrow:
            if commandOnly { session.focusDisplayedBoundary(forward: true) }
            else { session.focusNeighbour(1) }
            result = .handled
        case .upArrow:
            if commandOnly { session.focusDisplayedBoundary(forward: false) }
            else { session.focusGroupEdge(forward: false) }
            result = .handled
        case .downArrow:
            if commandOnly { session.focusDisplayedBoundary(forward: true) }
            else { session.focusGroupEdge(forward: true) }
            result = .handled
        case .home:
            session.focusDisplayedBoundary(forward: false)
            result = .handled
        case .end:
            session.focusDisplayedBoundary(forward: true)
            result = .handled
        case .return:
            guard let id = session.focusedFrameID, let frame = session.frame(id: id) else { return .handled }
            click(frame, .single)
            result = .handled
        default:
            return .ignored
        }
        return Diag.keyResult("grid.onKeyPress", press, result)
    }

    /// Group strips scroll by row; a Photos ungrouped section is a whole grid and
    /// must scroll by individual photo instead. Avoid animating lazy measurement.
    private func scroll(to id: FrameID?, proxy: ScrollViewProxy) {
        guard let id else { return }
        let target = FrameGridScrollTarget.resolve(frame: id, rowID: session.rowID(for: id),
            group: session.frame(id: id)?.groupID, groupRows: session.usesGroupRows,
            photosSource: session.photosAlbumURL != nil)
        var transaction = Transaction(animation: nil)
        transaction.disablesAnimations = true
        withTransaction(transaction) {
            switch target {
            case .row(let row): proxy.scrollTo(row, anchor: .center)
            case .frame(let frame): proxy.scrollTo(frame, anchor: .center)
            }
        }
    }

    /// Shown when the arrangement hides everything — most often the `Selected`
    /// filter before anything has been selected.
    private var emptyState: some View {
        let isOrderFilter = session.photosAlbumURL != nil && session.photosOrderFilter != .all
        return VStack(spacing: 8) {
            Image(systemName: "line.3.horizontal.decrease.circle")
                .font(.system(size: 32))
                .foregroundStyle(.tertiary)
            Text(isOrderFilter ? "No photos match this order state" : "Nothing matches these limits")
                .font(.headline)
                .foregroundStyle(.secondary)
            Text(isOrderFilter ? "Click the order filter to show all photos, or use Sync order on a photo." : "\(session.frames.count) frames analysed — widen a range in the panel")
                .font(.callout)
                .foregroundStyle(.tertiary)
        }
    }

    /// Builds the context menu for a frame.
    @ViewBuilder
    private func contextMenu(for frame: AnalyzedFrame) -> some View {
        let isSelected = session.selectedFrameIDs.contains(frame.id)

        Button("Preview") { session.showPreview(frame) }
        Button(isSelected ? "Deselect" : "Select") { session.select(frame.id) }

        if let groupID = frame.groupID,
           let group = session.groups.first(where: { $0.id == groupID }) {
            Divider()
            Button("Select All in Group") { session.selectAll(in: group) }
        }

        if let sourceURL = session.video?.url,
           let photoURL = PhotosMedia.shared.photoItemURL(for: sourceURL, seconds: frame.timeSeconds) {
            Divider()
            Button("Sync order") { session.syncPhotosOrder(for: frame) }
                .disabled(session.isSyncingPhotosOrder || session.photosAlbumURL.flatMap { PhotosMedia.shared.album($0) }?.canReorder != true)
                .help("Write the displayed order to Photos. With multiple photos selected, sync only that selection in its displayed order.")
            Button("Reset order state") { session.resetPhotosOrderState() }
            Button("Show in Photos") { PhotosAppIntegration.reveal(photoURL) }
        }

        Divider()
        Button("Copy Timecode") {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(timecode(frame.timeSeconds), forType: .string)
        }
    }
}

private struct GridRenderRow: Identifiable {
    let id: String
    let frames: [AnalyzedFrame]
    let group: FrameRow?
}

private struct GridRowViewportPreferenceKey: PreferenceKey {
    static let defaultValue: [String: CGRect] = [:]

    static func reduce(value: inout [String: CGRect],
                       nextValue: () -> [String: CGRect]) {
        value.merge(nextValue(), uniquingKeysWith: { _, newest in newest })
    }
}

/// Keeps decoded thumbnail pixels warm for three rows on either side of the
/// rows currently in the viewport. Source reads and image decoding stay off the
/// main thread; visible cells own their loading and fade-in state.
@MainActor
private final class GridThumbnailPrefetcher: ObservableObject {
    struct PlanKey: Hashable {
        let revision: UInt64
        let generation: UUID
        let columns: Int
        let grouped: Bool
        let photosSource: Bool

        var tokenPrefix: String {
            "\(revision):\(generation.uuidString):\(columns):\(grouped):\(photosSource):"
        }
    }

    private struct Job {
        let id: UUID
        let task: Task<Void, Never>
    }

    private(set) var planKey: PlanKey?
    private var rows: [[AnalyzedFrame]] = []
    private var rowIndexByToken: [String: Int] = [:]
    private var visibleRowTokens: Set<String> = []
    private var jobs: [String: Job] = [:]
    private var refreshTask: Task<Void, Never>?
    private weak var session: WaifuSession?

    func needsPlan(_ key: PlanKey) -> Bool { planKey != key }

    func install(rows: [GridRenderRow], key: PlanKey, session: WaifuSession) {
        guard planKey != key else { return }
        planKey = key
        self.rows = rows.map(\.frames)
        self.session = session
        rowIndexByToken.removeAll(keepingCapacity: true)
        rowIndexByToken.reserveCapacity(rows.count)
        let prefix = key.tokenPrefix
        for (index, row) in rows.enumerated() {
            rowIndexByToken[prefix + row.id] = index
        }
        scheduleRefresh()
    }

    func setVisibleRows(_ tokens: Set<String>) {
        guard tokens != visibleRowTokens else { return }
        visibleRowTokens = tokens
        scheduleRefresh()
    }

    func stop() {
        refreshTask?.cancel()
        refreshTask = nil
        for job in jobs.values { job.task.cancel() }
        jobs.removeAll(keepingCapacity: false)
        visibleRowTokens.removeAll(keepingCapacity: false)
        rows.removeAll(keepingCapacity: false)
        rowIndexByToken.removeAll(keepingCapacity: false)
        planKey = nil
        session = nil
    }

    private func scheduleRefresh() {
        guard refreshTask == nil else { return }
        refreshTask = Task { [weak self] in
            await Task.yield()
            guard let self, !Task.isCancelled else { return }
            self.refreshTask = nil
            self.refresh()
        }
    }

    private func refresh() {
        guard let key = planKey, let session, !rows.isEmpty else { return }
        // Warm the first screen before geometry preferences have arrived. Once
        // scrolling starts, the viewport replaces this initial window.
        let measuredRows = Set(visibleRowTokens.compactMap { rowIndexByToken[$0] })
        let visibleRows = measuredRows.isEmpty ? Set(0..<min(rows.count, 3)) : measuredRows

        var neighborRows = Set<Int>()
        for visible in visibleRows {
            for index in max(0, visible - 3)...min(rows.count - 1, visible + 3)
                where !visibleRows.contains(index) {
                neighborRows.insert(index)
            }
        }
        let orderedRows = visibleRows.sorted() + neighborRows.sorted { lhs, rhs in
            let leftDistance = visibleRows.map { abs($0 - lhs) }.min() ?? .max
            let rightDistance = visibleRows.map { abs($0 - rhs) }.min() ?? .max
            return leftDistance == rightDistance ? lhs < rhs : leftDistance < rightDistance
        }

        // Group strips can be much longer than the viewport. Prefetch one screen
        // of each adjacent strip; the horizontal lazy stack loads the rest as it
        // comes into view. Flat rows already contain at most one screen of cells.
        var wanted: [String: AnalyzedFrame] = [:]
        let maximum = max(1, key.columns * (visibleRows.count + 6))
        for index in orderedRows {
            let candidateFrames = key.grouped ? Array(rows[index].prefix(key.columns)) : rows[index]
            for frame in candidateFrames {
                let identity = key.generation.uuidString + ":" + frame.thumbnailKey
                wanted[identity] = frame
                if wanted.count >= maximum { break }
            }
            if wanted.count >= maximum { break }
        }

        let obsoleteJobs = jobs.filter { wanted[$0.key] == nil }
        for (identity, job) in obsoleteJobs {
            job.task.cancel()
            jobs.removeValue(forKey: identity)
        }
        for (identity, frame) in wanted where jobs[identity] == nil
            && GridThumbnailLoader.shared.cachedImage(identity: identity) == nil {
            let jobID = UUID()
            let task = Task(priority: .utility) { [weak self, weak session] in
                if let session,
                   let data = await session.thumbnailData(for: frame, retainRegeneratedThumbnail: false),
                   !Task.isCancelled {
                    _ = await GridThumbnailLoader.shared.image(data: data, identity: identity)
                }
                self?.finish(identity: identity, jobID: jobID)
            }
            jobs[identity] = Job(id: jobID, task: task)
        }
    }

    private func finish(identity: String, jobID: UUID) {
        guard jobs[identity]?.id == jobID else { return }
        jobs.removeValue(forKey: identity)
    }
}

/// A single frame cell: picture, source ordinal, and inward selection border.
private struct FrameCell: View {
    let session: WaifuSession
    @ObservedObject private var photosOrderState = PhotosOrderState.shared
    let frame: AnalyzedFrame
    let thumbnailKey: String
    let generation: UUID
    let width: CGFloat
    let isPhotosSource: Bool
    let squareThumbnail: Bool
    @StateObject private var thumbnail: GridThumbnailPresentation
    @State private var isSelected: Bool
    @State private var isHovered = false

    init(session: WaifuSession, frame: AnalyzedFrame, thumbnailKey: String, generation: UUID, width: CGFloat,
         isPhotosSource: Bool, squareThumbnail: Bool) {
        self.session = session; self.frame = frame; self.thumbnailKey = thumbnailKey
        self.generation = generation; self.width = width
        self.isPhotosSource = isPhotosSource; self.squareThumbnail = squareThumbnail
        let identity = generation.uuidString + ":" + thumbnailKey
        _thumbnail = StateObject(wrappedValue: GridThumbnailPresentation(identity: identity))
        _isSelected = State(initialValue: session.selectedFrameIDs.contains(frame.id))
    }

    private var thumbnailIdentity: String {
        generation.uuidString + ":" + thumbnailKey
    }

    var body: some View {
        let visibleThumbnail = thumbnail.image(for: thumbnailIdentity)
        let aspect = frame.thumbnailAspectRatio ?? visibleThumbnail.map { Double($0.size.width / $0.size.height) } ?? (16.0 / 9)
        let isSquare = isPhotosSource && squareThumbnail
        let height = isSquare ? width : width / CGFloat(aspect.isFinite && aspect > 0 ? aspect : 16.0 / 9)
        let isOrdered = isPhotosSource
            ? session.photosAlbumURL.flatMap { albumURL in
                guard let albumIdentifier = PhotosMedia.shared.collectionIdentifier(for: albumURL),
                      let assetIdentifier = PhotosMedia.shared.assetIdentifier(for: albumURL, seconds: frame.timeSeconds) else { return nil }
                return photosOrderState.isOrdered(albumIdentifier: albumIdentifier, assetIdentifier: assetIdentifier)
            } ?? false
            : false
        Group {
            if let nsImage = visibleThumbnail {
                if isPhotosSource {
                    PhotosThumbnailImage(image: nsImage, sourceAspect: CGFloat(aspect), square: isSquare)
                        .frame(width: width, height: height)
                } else {
                    Image(nsImage: nsImage)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(width: width, height: height)
                        .clipped()
                }
            } else {
                Color.clear.frame(width: width, height: height)
            }
        }
        .frame(width: width, height: height)
        .background(Color(nsColor: .windowBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 3, style: .continuous))
        .overlay(alignment: .topLeading) {
            if !isPhotosSource { frameNumberOverlay }
        }
        .overlay(alignment: .topTrailing) {
            if isOrdered {
                PhotosOrderIcon(filter: .ordered)
                    .scaleEffect(0.65)
                    .frame(width: 14, height: 14)
                    .foregroundStyle(.white)
                    .shadow(color: .black.opacity(0.85), radius: 2)
                    .padding(5)
                    .help("Order synced")
                    .accessibilityLabel("Order synced")
            }
        }
        .overlay(alignment: .bottomLeading) {
            if isPhotosSource,
               let albumURL = session.photosAlbumURL,
               let photoURL = PhotosMedia.shared.photoItemURL(for: albumURL, seconds: frame.timeSeconds) {
                PhotosFavoriteButton(url: photoURL, isHovered: isHovered)
                    .id(photoURL)
                    .padding(5)
            }
        }
        .onHover { isHovered = $0 }
        // strokeBorder consumes picture space inward without changing cell geometry.
        .overlay {
            RoundedRectangle(cornerRadius: 3, style: .continuous)
                .strokeBorder(isSelected ? Color.blue : Color.primary.opacity(0.12),
                              lineWidth: isSelected ? 4 : 1)
                .allowsHitTesting(false)
        }
        .help(isPhotosSource
              ? (visibleThumbnail.map { "\(Int($0.size.width)) × \(Int($0.size.height)) pixels · " } ?? "") + "art \(String(format: "%.2f", frame.artScore))"
              : "\(timecode(frame.timeSeconds)) · art \(String(format: "%.2f", frame.artScore))")
        // Fade only the opacity of newly loaded content; layout, hover, and
        // selection stay immediate. Cached cells start ready and do not refade.
        .animation(visibleThumbnail == nil ? nil : .easeIn(duration: 0.33)) { content in
            content.opacity(visibleThumbnail == nil ? 0 : 1)
        }
        .allowsHitTesting(visibleThumbnail != nil)
        .accessibilityHidden(visibleThumbnail == nil)
        // Progress, tagging, and player updates must not redraw every thumbnail.
        // Only cells whose selection membership changed need a new border.
        .onReceive(session.$selectedFrameIDs.map { $0.contains(frame.id) }.removeDuplicates()) {
            isSelected = $0
        }
        .task(id: thumbnailIdentity) {
            await thumbnail.prepare(identity: thumbnailIdentity) {
                guard let data = await session.thumbnailData(for: frame), !Task.isCancelled else { return nil }
                return await GridThumbnailLoader.shared.image(data: data, identity: thumbnailIdentity)
            }
        }
        .onDisappear { thumbnail.disappear(identity: thumbnailIdentity) }

    }

    /// Zero-based decoded source ordinal, also used in exports and diagnostics.
    private var frameNumberOverlay: some View {
        Text("\(frame.id.sampleIndex)")
            .font(.system(size: 10, weight: .medium, design: .monospaced))
            .foregroundStyle(.white)
            .padding(.horizontal, 3)
            .padding(.vertical, 1)
            .background(Color.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 2))
            .padding(3)
    }

}

/// Click kind for frame interactions.
enum ClickKind {
    case single
    case double
    case toggle
    case range
    case additiveRange
}

/// Safe array access.
private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
