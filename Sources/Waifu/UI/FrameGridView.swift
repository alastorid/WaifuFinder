import Combine
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
///
/// Zoom (⌘+/⌘−/⌘0, pinch, or the panel buttons) walks `GridZoom`'s ladder of target
/// cell widths: zooming in means fewer, larger thumbnails per row; zooming out packs
/// more, smaller ones. The reflow is committed animated, unlike scroll and filter
/// layouts, which must swap without animation.
struct FrameGridView: View {
    /// The asynchronous thumbnail stage must be allowed to redraw independently
    /// of the session revision. An equality gate here can retain an empty tree
    /// after the stage has committed its decoded rows.
    let session: WaifuSession
    let gridRevision: UInt64
    let squareThumbnails: Bool

    /// The zoom rung the flat grid builds its column count from.
    @Binding var zoomTarget: CGFloat

    /// Called when the user clicks a frame (single-click replaces selection,
    /// modifier clicks toggle or extend selection; Return also selects).
    let onFrameClick: (AnalyzedFrame, ClickKind) -> Void

    /// Whether the grid has keyboard focus.
    @FocusState private var isFocused: Bool
    @StateObject private var thumbnailStage = GridThumbnailStage()
    @State private var committedNavigation: FrameID?
    @State private var verticalScrollPosition = ScrollPosition()
    @State private var horizontalScrollPositions: [String: ScrollPosition] = [:]
    @State private var scrollVisibility = FrameGridScrollVisibility()
    @StateObject private var thumbnailPrefetcher = GridThumbnailPrefetcher()
    /// The rung a pinch started from, so cumulative magnification scales one base.
    @State private var pinchBaseWidth: CGFloat?

    private let spacing: CGFloat = 1
    private let padding: CGFloat = 1

    private var usesSquareThumbnails: Bool {
        squareThumbnails
    }

    var body: some View {
        GeometryReader { geometry in
            Group {
                ScrollView(.vertical, showsIndicators: true) {
                    stagedLayout
                }
                .scrollPosition($verticalScrollPosition)
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
                          session.previewFrame == nil, session.uncover == nil, !session.crystalOrbIsOpen else { return .ignored }
                    session.selectAllDisplayedFrames()
                    return .handled
                }
                .onChange(of: session.frames.isEmpty) { _, empty in
                    if !empty { isFocused = true }
                }
                .onAppear {
                    thumbnailStage.onCommit = { [weak session] snapshot in
                        session?.commitGridNavigation(snapshot.navigation, generation: snapshot.key.generation)
                    }
                    if let snapshot = thumbnailStage.current {
                        session.commitGridNavigation(snapshot.navigation, generation: snapshot.key.generation)
                    }
                    prepareStage(size: geometry.size)
                    if let snapshot = thumbnailStage.current {
                        thumbnailPrefetcher.install(snapshot: snapshot, session: session)
                    }
                }
                .onDisappear { thumbnailStage.stop(); thumbnailPrefetcher.stop() }
                .onReceive(session.gridLayoutUpdates) { _ in
                    prepareStage(size: geometry.size)
                }
                .onChange(of: geometry.size) { _, _ in
                    prepareStage(size: geometry.size)
                }
                .onChange(of: session.usesGroupRows) { _, _ in
                    prepareStage(size: geometry.size)
                }
                .onChange(of: session.thumbnailGeneration) { _, _ in
                    prepareStage(size: geometry.size)
                }
                .onChange(of: squareThumbnails) { _, _ in prepareStage(size: geometry.size) }
                .onChange(of: zoomTarget) { _, _ in
                    prepareStage(size: geometry.size, animated: true)
                }
                // A pinch snaps its scaled base to the zoom ladder; each rung the
                // gesture crosses commits an animated reflow like Photos.app.
                .simultaneousGesture(
                    MagnifyGesture()
                        .onChanged { value in
                            let base = pinchBaseWidth ?? zoomTarget
                            pinchBaseWidth = base
                            let candidate = GridZoom.snapped(base * value.magnification)
                            if candidate != zoomTarget { zoomTarget = candidate }
                        }
                        .onEnded { _ in pinchBaseWidth = nil }
                )
                .onScrollGeometryChange(for: CGRect.self) { $0.visibleRect } action: { _, rect in
                    scrollVisibility.viewport = rect
                    thumbnailStage.updateViewport(rect)
                }
                .onChange(of: thumbnailStage.current?.key) { old, _ in
                    if let snapshot = thumbnailStage.current {
                        thumbnailPrefetcher.install(snapshot: snapshot, session: session)
                        if old?.generation != snapshot.key.generation {
                            verticalScrollPosition.scrollTo(y: 0)
                        }
                    }
                    if let id = session.gridNavigationFrameID, isNavigationTarget(id) {
                        committedNavigation = id
                        scroll(to: id)
                    }
                }
                .onPreferenceChange(GridRowViewportPreferenceKey.self) { rowViewports in
                    let visible = Set(rowViewports.compactMap { token, rect -> String? in
                        rect.maxY > 0 && rect.minY < geometry.size.height ? token : nil
                    })
                    thumbnailPrefetcher.setVisibleRows(visible)
                }
                .onReceive(session.gridScrollRequests) { id in
                    guard isNavigationTarget(id) else { return }
                    committedNavigation = id
                    scroll(to: id)
                }
                .onChange(of: session.focusedFrameID) { _, id in
                    // A focus publication can redraw the grid before a queued
                    // scroll event arrives. Reconcile the committed selection too.
                    guard let id, isNavigationTarget(id) else { return }
                    committedNavigation = id
                    scroll(to: id)
                }
                // @Published emits before its setter commits. Scrolling here can
                // force a render; defer it so that render sees the closed preview.
                .onReceive(session.$crystalOrbIsOpen.dropFirst().receive(on: DispatchQueue.main)) { active in
                    if !active, !session.crystalOrbIsOpen { restoreKeyboardFocus(reason: "Crystal Orb closed") }
                }
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
            FrameGridStatusView(session: session, stage: thumbnailStage)
        }
        .onAppear { if !session.frames.isEmpty { isFocused = true } }
    }

    // MARK: - Layout

    @ViewBuilder
    private var stagedLayout: some View {
        let _ = Diag.trace(.grid, message: "grid render input=\(gridRevision) revision=\(thumbnailStage.current?.key.revision.description ?? "nil") rows=\(thumbnailStage.current?.rows.count ?? 0)")
        if let snapshot = thumbnailStage.current {
            LazyVStack(alignment: .leading, spacing: snapshot.key.grouped ? 12 : spacing) {
                ForEach(snapshot.rows) { row in
                    let token = snapshot.key.tokenPrefix + row.id
                    Group {
                        if row.strip { rowView(row, snapshot: snapshot) }
                        else {
                            HStack(spacing: spacing) {
                                ForEach(row.items) { item in
                                    cell(item, row: row, snapshot: snapshot).id(item.id)
                                }
                            }
                            .frame(height: row.contentHeight)
                        }
                    }
                    .background {
                        GeometryReader { proxy in
                            Color.clear.preference(key: GridRowViewportPreferenceKey.self,
                                value: [token: proxy.frame(in: .named("frame-grid-viewport"))])
                        }
                    }
                    .id(row.id)
                }
            }
            .padding(padding)
        }
    }

    private func rowView(_ row: GridStageRow, snapshot: GridStageSnapshot) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Text(row.groupID == nil ? "\(row.items.count) ungrouped"
                     : "\(row.items.count) frame\(row.items.count == 1 ? "" : "s")")
                if row.beyond > 0 && !row.revealed {
                    Text("· \(row.beyond) more").foregroundStyle(.tertiary)
                }
            }
            .font(.system(size: 10))
            .foregroundStyle(.secondary)
            .frame(height: 12)
            Group {
                ScrollView(.horizontal, showsIndicators: false) {
                    LazyHStack(spacing: spacing) {
                        ForEach(row.items) { item in
                            cell(item, row: row, snapshot: snapshot).id(item.id)
                        }
                        if let group = row.groupID, row.beyond > 0, !row.revealed {
                            moreTile(count: row.beyond, width: row.cellWidth, group: group, square: snapshot.key.square)
                        }
                    }
                }
                .frame(height: row.contentHeight)
                .scrollPosition(Binding(
                    get: { horizontalScrollPositions[row.id] ?? ScrollPosition() },
                    set: { horizontalScrollPositions[row.id] = $0 }))
                .onScrollGeometryChange(for: CGRect.self) { $0.visibleRect } action: { _, rect in
                    scrollVisibility.horizontalViewports[row.id] = rect
                    thumbnailStage.updateHorizontalOffset(row: row.id, offset: rect.minX)
                    thumbnailPrefetcher.setHorizontalOffset(row: row.id, offset: rect.minX)
                }
                .onAppear {
                    scrollHorizontally(to: committedNavigation, in: row)
                }
                .onChange(of: committedNavigation) { _, id in
                    scrollHorizontally(to: id, in: row)
                }
                .onChange(of: snapshot.key) { _, _ in
                    scrollHorizontally(to: committedNavigation, in: row)
                }
                .onReceive(session.gridScrollRequests) { id in
                    scrollHorizontally(to: id, in: row)
                }
            }
        }
    }

    /// How many cells a row is sized for: the per-group cap, so a group that is at the
    /// cap fills its line exactly and a bigger one (revealed) scrolls.
    private var rowColumnCount: Int {
        max(2, min(session.settings.perGroupFrameCount, 8))
    }

    private func columnCount(in width: CGFloat) -> Int {
        max(1, Int((width - padding * 2) / (zoomTarget + spacing)))
    }

    private func prepareStage(size: CGSize, animated: Bool = false) {
        let generation = session.thumbnailGeneration
        let key = GridStageKey(revision: session.gridRevision, generation: generation, width: size.width,
            columns: columnCount(in: size.width), groupColumns: rowColumnCount,
            grouped: session.usesGroupRows, photosSource: session.photosAlbumURL != nil,
            square: usesSquareThumbnails, targetWidth: zoomTarget)
        let input = GridStageInput(key: key, sourceURL: session.photosAlbumURL ?? session.video?.url,
            rows: session.displayedRows, frames: session.displayOrderFrames,
            beyondCap: session.beyondCapByGroup, revealed: session.revealedGroups)
        thumbnailStage.submit(input, viewportSize: size, animated: animated) { [weak session] frame in
            await GridThumbnailWorkPriority.shared.withVisibleWork {
                guard let session, session.thumbnailGeneration == generation,
                      let data = await session.thumbnailData(for: frame, retainRegeneratedThumbnail: false),
                      !Task.isCancelled, session.thumbnailGeneration == generation else { return nil }
                return await GridThumbnailLoader.shared.image(data: data, identity: generation.uuidString + ":" + frame.thumbnailKey)
            }
        }
    }

    // MARK: - Cells

    private func cell(_ item: GridStageItem, row: GridStageRow, snapshot: GridStageSnapshot) -> some View {
        let frame = item.frame
        return FrameCell(
            session: session,
            frame: frame,
            thumbnailKey: frame.thumbnailKey,
            generation: snapshot.key.generation,
            width: row.cellWidth,
            sourceAspect: item.aspect,
            sourceURL: snapshot.sourceURL,
            isPhotosSource: snapshot.key.photosSource,
            squareThumbnail: snapshot.key.square,
            topTags: session.topTagsByFrame[frame.id] ?? [],
            topTagFilterActive: session.settings.tagAdvancedMode && session.settings.tagTopCount < 99
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
    }

    // MARK: - Keyboard

    /// Grid navigation is active only while neither overlay is up.
    private func handleKeyPress(_ press: KeyPress) -> KeyPress.Result {
        guard session.previewFrame == nil, session.uncover == nil, !session.crystalOrbIsOpen else { return .ignored }
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

    private func isNavigationTarget(_ id: FrameID) -> Bool {
        session.previewFrame == nil && session.uncover == nil && !session.crystalOrbIsOpen
            && id == session.focusedFrameID && id == session.gridNavigationFrameID
    }

    private func scrollHorizontally(to id: FrameID?, in row: GridStageRow) {
        guard let id, isNavigationTarget(id),
              let index = row.items.firstIndex(where: { $0.id == id }) else { return }
        let rect = scrollVisibility.horizontalViewports[row.id] ?? .zero
        let left = CGFloat(index) * (row.cellWidth + spacing)
        guard let offset = FrameGridScrollVisibility.revealOffset(lower: left, upper: left + row.cellWidth,
            visibleLower: rect.minX, visibleUpper: rect.maxX) else { return }
        var position = horizontalScrollPositions[row.id] ?? ScrollPosition()
        position.scrollTo(x: offset)
        horizontalScrollPositions[row.id] = position
    }

    /// Scroll to the measured row offset, including rows whose lazy cells are not mounted.
    private func scroll(to id: FrameID?) {
        guard let id, let snapshot = thumbnailStage.current,
              let index = snapshot.rowIndexByFrameID[id] else { return }
        let row = snapshot.rows[index]
        let rect = scrollVisibility.viewport
        guard let offset = FrameGridScrollVisibility.revealOffset(lower: row.originY, upper: row.originY + row.height,
            visibleLower: rect.minY, visibleUpper: rect.maxY) else { return }
        var transaction = Transaction(animation: nil)
        transaction.disablesAnimations = true
        withTransaction(transaction) {
            verticalScrollPosition.scrollTo(y: offset)
        }
    }

    /// Builds the context menu for a frame.
    @ViewBuilder
    private func contextMenu(for frame: AnalyzedFrame) -> some View {
        let isSelected = session.selectedFrameIDs.contains(frame.id)
        let photoURL = session.video.flatMap {
            PhotosMedia.shared.photoItemURL(for: $0.url, seconds: frame.timeSeconds)
        }

        if let groupID = frame.groupID,
           let group = session.groups.first(where: { $0.id == groupID }) {
            Button("Select All in Group") { session.selectAll(in: group) }
        }

        if session.offersFocusedRangeAnalysis {
            Button("Analyze ±3 Seconds Range") { session.analyzeFocusedRange(around: frame) }
                .disabled(!session.canAnalyzeFocusedRange)
        }

        if let photoURL {
            Button("View in Photos.app") { PhotosAppIntegration.reveal(photoURL) }
        } else if !isSelected {
            Button("Select") { session.select(frame.id) }
        }

        if photoURL != nil {
            Divider()
            Button("Set Album Order in Photos.app") { session.syncPhotosOrder(for: frame) }
                .disabled(session.isSyncingPhotosOrder || session.photosAlbumURL.flatMap { PhotosMedia.shared.album($0) }?.canReorder != true)
            Button("Set Keywords from Tags") { session.writePhotosTags(for: frame) }
                .disabled(session.isWritingPhotosTags || session.tagVocabulary == nil || frame.tagProbabilities == nil || frame.tagRating == nil)
        }

        Divider()
        Button("Copy Timecode") {
            NSPasteboard.general.clearContents()
            let sourceTime = session.video.map { VideoCollection.resolve($0.url, seconds: frame.timeSeconds).seconds } ?? frame.timeSeconds
            NSPasteboard.general.setString(timecode(sourceTime), forType: .string)
        }
    }
}

private struct GridRowViewportPreferenceKey: PreferenceKey {
    static let defaultValue: [String: CGRect] = [:]

    static func reduce(value: inout [String: CGRect],
                       nextValue: () -> [String: CGRect]) {
        value.merge(nextValue(), uniquingKeysWith: { _, newest in newest })
    }
}

/// A single frame cell: picture, source ordinal, and inward selection border.
private struct FrameCell: View {
    let session: WaifuSession
    let frame: AnalyzedFrame
    let thumbnailKey: String
    let generation: UUID
    let width: CGFloat
    let sourceAspect: CGFloat
    let sourceURL: URL?
    let isPhotosSource: Bool
    let squareThumbnail: Bool
    let topTags: [String]
    let topTagFilterActive: Bool
    @StateObject private var thumbnail: GridThumbnailPresentation
    @State private var isSelected: Bool
    @State private var isThumbnailVisible = false
    @State private var isHovered = false
    @State private var showsFrameNumber = false

    init(session: WaifuSession, frame: AnalyzedFrame, thumbnailKey: String, generation: UUID, width: CGFloat,
         sourceAspect: CGFloat, sourceURL: URL?, isPhotosSource: Bool, squareThumbnail: Bool,
         topTags: [String], topTagFilterActive: Bool) {
        self.session = session; self.frame = frame; self.thumbnailKey = thumbnailKey
        self.generation = generation; self.width = width
        self.sourceAspect = sourceAspect; self.sourceURL = sourceURL
        self.isPhotosSource = isPhotosSource; self.squareThumbnail = squareThumbnail
        self.topTags = topTags
        self.topTagFilterActive = topTagFilterActive
        let identity = generation.uuidString + ":" + thumbnailKey
        _thumbnail = StateObject(wrappedValue: GridThumbnailPresentation(identity: identity))
        _isSelected = State(initialValue: session.selectedFrameIDs.contains(frame.id))
    }

    private var thumbnailIdentity: String {
        generation.uuidString + ":" + thumbnailKey
    }

    var body: some View {
        let visibleThumbnail = thumbnail.image(for: thumbnailIdentity)
        let isSquare = squareThumbnail
        let height = GridThumbnailLayout.size(width: width, sourceAspect: sourceAspect, square: isSquare).height
        GridThumbnailImage(image: visibleThumbnail, identity: thumbnailIdentity,
                           cropToFill: isSquare, isVisible: isThumbnailVisible)
        .frame(width: width, height: height)
        .contentShape(Rectangle())
        .clipShape(RoundedRectangle(cornerRadius: 3, style: .continuous))
        .overlay(alignment: .topLeading) {
            if visibleThumbnail != nil {
                thumbnailLabel(height: height)
                .padding(3)
                .frame(maxWidth: width, maxHeight: height, alignment: .topLeading)
                .clipped()
            }
        }
        .overlay(alignment: .bottomLeading) {
            if visibleThumbnail != nil, isPhotosSource,
               let albumURL = sourceURL,
               let photoURL = PhotosMedia.shared.photoItemURL(for: albumURL, seconds: frame.timeSeconds) {
                PhotosFavoriteButton(url: photoURL, isHovered: isHovered) {
                    favoriteTargets(for: photoURL)
                }
                    .id(photoURL)
                    .padding(5)
            }
        }
        .onHover { isHovered = $0 }
        .onChange(of: topTagFilterActive) { _, _ in showsFrameNumber = false }
        .onChange(of: thumbnailIdentity) { _, _ in showsFrameNumber = false }
        // strokeBorder consumes picture space inward without changing cell geometry.
        .overlay {
            if visibleThumbnail != nil {
                RoundedRectangle(cornerRadius: 3, style: .continuous)
                    .strokeBorder(isSelected ? Color.blue : Color.primary.opacity(0.12),
                                  lineWidth: isSelected ? 4 : 1)
                    .allowsHitTesting(false)
            }
        }
        // Keep the empty slot measurable so late arrivals can detect stage entry.
        .allowsHitTesting(visibleThumbnail != nil)
        .accessibilityElement(children: .contain)
        .accessibilityLabel(isPhotosSource ? "Photo thumbnail" : "Frame thumbnail")
        .accessibilityIdentifier("frame-thumbnail-\(frame.id.sampleIndex)")
        .accessibilityHidden(visibleThumbnail == nil)
        .onScrollVisibilityChange(threshold: 0.01) { isThumbnailVisible = $0 }
        // Progress, tagging, and player updates must not redraw every thumbnail.
        // Only cells whose selection membership changed need a new border.
        .onReceive(session.$selectedFrameIDs.map { $0.contains(frame.id) }.removeDuplicates()) {
            isSelected = $0
        }
        .task(id: isThumbnailVisible ? thumbnailIdentity : nil) {
            guard isThumbnailVisible else {
                thumbnail.disappear(identity: thumbnailIdentity)
                return
            }
            await thumbnail.prepare(identity: thumbnailIdentity) {
                await GridThumbnailWorkPriority.shared.withVisibleWork {
                    guard session.thumbnailGeneration == generation,
                          let data = await session.thumbnailData(for: frame), !Task.isCancelled,
                          session.thumbnailGeneration == generation else { return nil }
                    return await GridThumbnailLoader.shared.image(data: data, identity: thumbnailIdentity)
                }
            }
        }
        .onDisappear { thumbnail.disappear(identity: thumbnailIdentity) }

    }

    private func favoriteTargets(for clickedURL: URL) -> [URL] {
        guard let albumURL = sourceURL,
              session.selectedFrameIDs.count > 1,
              session.selectedFrameIDs.contains(frame.id) else {
            return [clickedURL]
        }

        var seen = Set<String>()
        let selectedURLs = session.displayOrderFrames
            .filter { session.selectedFrameIDs.contains($0.id) }
            .compactMap { PhotosMedia.shared.photoItemURL(for: albumURL, seconds: $0.timeSeconds) }
            .filter { url in
                guard let identifier = PhotosMedia.shared.item(url)?.asset.localIdentifier else { return false }
                return seen.insert(identifier).inserted
            }

        guard let clickedIdentifier = PhotosMedia.shared.item(clickedURL)?.asset.localIdentifier else {
            return [clickedURL]
        }
        if selectedURLs.contains(where: {
            PhotosMedia.shared.item($0)?.asset.localIdentifier == clickedIdentifier
        }) {
            return selectedURLs
        }
        return selectedURLs + [clickedURL]
    }

    @ViewBuilder
    private func thumbnailLabel(height: CGFloat) -> some View {
        if topTagFilterActive && !topTags.isEmpty {
            Button {
                showsFrameNumber.toggle()
            } label: {
                if showsFrameNumber {
                    frameNumberOverlay
                } else {
                    topTagBadges(height: height)
                }
            }
            .buttonStyle(.plain)
            .accessibilityLabel(showsFrameNumber ? "Show Top tags" : "Show frame number")
            .accessibilityIdentifier("frame-label-toggle-\(frame.id.sampleIndex)")
        } else if !isPhotosSource {
            frameNumberOverlay
        }
    }

    /// Zero-based decoded source ordinal, also used in exports and diagnostics.
    private var frameNumberOverlay: some View {
        Text("\(sourceURL.flatMap { VideoCollection.get($0) } == nil ? frame.id.sampleIndex : VideoCollection.localIndex(frame.id.sampleIndex))")
            .font(.system(size: 10, weight: .medium, design: .monospaced))
            .foregroundStyle(.white)
            .padding(.horizontal, 3)
            .padding(.vertical, 1)
            .background(Color.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 2))
    }

    /// Keep labels inside small thumbnails; the +N badge counts the winners
    /// that did not fit.
    private func topTagBadges(height: CGFloat) -> some View {
        let rows = max(1, min(3, Int((height - 6) / 16)))
        let shown = Array(topTags.prefix(rows))
        return VStack(alignment: .leading, spacing: 2) {
            ForEach(Array(shown.enumerated()), id: \.element) { index, tag in
                HStack(spacing: 3) {
                    Text(tag).lineLimit(1)
                    if index == shown.count - 1, topTags.count > shown.count {
                        Text("+\(topTags.count - shown.count)").fixedSize()
                    }
                }
                .font(.system(size: 10, weight: .medium))
                .foregroundStyle(.white)
                .padding(.horizontal, 4)
                .padding(.vertical, 1)
                .background(topTagColor(tag), in: RoundedRectangle(cornerRadius: 3))
                .accessibilityLabel("Top tag: \(tag)")
            }
        }
        .frame(maxWidth: max(0, width - 6), alignment: .leading)
    }

    /// Stable hues with an opaque, dark background in both appearance modes.
    private func topTagColor(_ tag: String) -> Color {
        let hash = tag.utf8.reduce(UInt64(14_695_981_039_346_656_037)) { ($0 ^ UInt64($1)) &* 1_099_511_628_211 }
        return Color(hue: Double(hash % 360) / 360, saturation: 0.65, brightness: 0.32)
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
