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
        if session.usesGroupRows {
            LazyVStack(alignment: .leading, spacing: 12) {
                ForEach(session.displayedRows) { row in
                    if row.group == nil && session.photosAlbumURL != nil {
                        // Photos that did not join a similarity group are still
                        // independent images; keep them in an adaptive grid rather
                        // than stretching them across a single horizontal strip.
                        LazyVGrid(columns: gridColumns(for: width), spacing: spacing) {
                            ForEach(row.frames) { frame in
                                cell(frame, width: cellWidth(in: width))
                                    .id(frame.id)
                            }
                        }
                        .id(row.id)
                    } else {
                        rowView(row, width: width)
                            .id(row.id)
                    }
                }
            }
            .padding(padding)
        } else {
            LazyVGrid(columns: gridColumns(for: width), spacing: spacing) {
                ForEach(session.displayOrderFrames) { frame in
                    cell(frame, width: cellWidth(in: width))
                        .id(frame.id)
                }
            }
            .padding(padding)
        }
    }

    private func rowView(_ row: FrameRow, width: CGFloat) -> some View {
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
    }

    /// How many cells a row is sized for: the per-group cap, so a group that is at the
    /// cap fills its line exactly and a bigger one (revealed) scrolls.
    private var rowColumnCount: Int {
        max(2, min(session.settings.perGroupFrameCount, 8))
    }

    /// Adaptive column width for the flat layout.
    private func gridColumns(for width: CGFloat) -> [GridItem] {
        let usable = width - padding * 2
        let count = max(1, Int(usable / (minCellWidth + spacing)))
        return Array(repeating: GridItem(.fixed(cellWidth(in: width)), spacing: spacing), count: count)
    }

    private func cellWidth(in width: CGFloat) -> CGFloat {
        let usable = width - padding * 2
        let count = max(1, Int(usable / (minCellWidth + spacing)))
        let raw = (usable - spacing * CGFloat(count - 1)) / CGFloat(count)
        return min(max(raw, minCellWidth), maxCellWidth)
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

/// A single frame cell: picture, source ordinal, and inward selection border.
private struct FrameCell: View {
    @ObservedObject var session: WaifuSession
    @ObservedObject private var photosOrderState = PhotosOrderState.shared
    let frame: AnalyzedFrame
    let thumbnailKey: String
    let generation: UUID
    let width: CGFloat
    let isPhotosSource: Bool
    let squareThumbnail: Bool
    @State private var thumbnail: NSImage?
    @State private var isHovered = false

    init(session: WaifuSession, frame: AnalyzedFrame, thumbnailKey: String, generation: UUID, width: CGFloat,
         isPhotosSource: Bool, squareThumbnail: Bool) {
        self.session = session; self.frame = frame; self.thumbnailKey = thumbnailKey
        self.generation = generation; self.width = width
        self.isPhotosSource = isPhotosSource; self.squareThumbnail = squareThumbnail
        let identity = generation.uuidString + ":" + thumbnailKey
        let pixels = GridThumbnailLoader.shared.cachedImage(identity: identity)
        _thumbnail = State(initialValue: pixels.map { NSImage(cgImage: $0, size: NSSize(width: $0.width, height: $0.height)) })
    }

    private var thumbnailIdentity: String {
        generation.uuidString + ":" + thumbnailKey
    }

    var body: some View {
        let cached = GridThumbnailLoader.shared.cachedImage(identity: thumbnailIdentity)
        let visibleThumbnail = thumbnail ?? cached.map { NSImage(cgImage: $0, size: NSSize(width: $0.width, height: $0.height)) }
        let aspect = frame.thumbnailAspectRatio ?? visibleThumbnail.map { Double($0.size.width / $0.size.height) } ?? (16.0 / 9)
        let isSquare = isPhotosSource && squareThumbnail
        let height = isSquare ? width : width / CGFloat(aspect.isFinite && aspect > 0 ? aspect : 16.0 / 9)
        let isSelected = session.selectedFrameIDs.contains(frame.id)
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
                Rectangle()
                    .fill(Color.secondary.opacity(0.15))
                    .frame(width: width, height: height)
                    .overlay { ProgressView().controlSize(.small) }
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
        .task(id: thumbnailIdentity) {
            guard let data = await session.thumbnailData(for: frame) else { return }
            let image = await GridThumbnailLoader.shared.image(data: data, identity: thumbnailIdentity)
            guard !Task.isCancelled else { return }
            thumbnail = image.map { NSImage(cgImage: $0, size: NSSize(width: $0.width, height: $0.height)) }
        }
        .onDisappear { thumbnail = nil }

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
