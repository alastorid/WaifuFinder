import Foundation

/// An immutable navigation sequence shared with a committed thumbnail layout.
/// Preparing a replacement layout must not change where its visible cells lead.
struct FrameNavigationOrder: Sendable {
    let frames: [AnalyzedFrame]
    let indexByID: [FrameID: Int]
    private let rowByID: [FrameID: Int]
    /// The index ranges, over `frames`, of the lines the grid actually draws.
    ///
    /// A committed staged layout knows them exactly: one per group strip, plus the
    /// chunked lines a long ungrouped run is split into. Without it (the fallback
    /// order before the first commit), the arrangement's own rows are the closest
    /// approximation. Committed flat layouts also supply their wrapped lines for
    /// Shift+↑/↓ selection; ordinary arrows retain their group navigation.
    private let lineRanges: [Range<Int>]
    private let groupRows: Bool

    init(frames: [AnalyzedFrame], rows: [FrameRow], groupRows: Bool, lineRanges: [Range<Int>]? = nil) {
        self.groupRows = groupRows
        self.frames = groupRows ? rows.flatMap(\.frames) : frames
        indexByID = Dictionary(uniqueKeysWithValues: self.frames.enumerated().map { ($0.element.id, $0.offset) })
        rowByID = groupRows
            ? Dictionary(uniqueKeysWithValues: rows.enumerated().flatMap { index, row in
                row.frames.map { ($0.id, index) }
            }) : [:]
        if let lineRanges {
            self.lineRanges = lineRanges
        } else if groupRows {
            var ranges: [Range<Int>] = []
            var start = 0
            for row in rows where !row.frames.isEmpty {
                ranges.append(start..<start + row.frames.count)
                start += row.frames.count
            }
            self.lineRanges = ranges
        } else {
            self.lineRanges = []
        }
    }

    /// Preserve the preview's captured order and row boundaries after deletion.
    func removing(_ removed: Set<FrameID>) -> Self {
        let survivors = frames.filter { !removed.contains($0.id) }
        let rows = Dictionary(grouping: survivors, by: { rowByID[$0.id] ?? 0 })
            .sorted { $0.key < $1.key }
            .map { FrameRow(group: $0.value.first?.groupID, frames: $0.value) }
        var start = 0
        let ranges = lineRanges.compactMap { range -> Range<Int>? in
            let count = frames[range].filter { !removed.contains($0.id) }.count
            guard count > 0 else { return nil }
            defer { start += count }
            return start..<start + count
        }
        return Self(frames: survivors, rows: rows, groupRows: groupRows, lineRanges: ranges)
    }

    func neighbour(of id: FrameID, by delta: Int) -> AnalyzedFrame? {
        guard let index = indexByID[id] else { return nil }
        let next = index + delta
        guard frames.indices.contains(next),
              !groupRows || rowByID[id] == rowByID[frames[next].id] else { return nil }
        return frames[next]
    }

    /// One drawn line up or down, landing on the cell nearest the column the focus
    /// left. `nil` at the ends, and in layouts whose lines are not known.
    func verticalNeighbour(of id: FrameID, forward: Bool) -> AnalyzedFrame? {
        guard let index = indexByID[id],
              let line = lineRanges.firstIndex(where: { $0.contains(index) }) else { return nil }
        let target = line + (forward ? 1 : -1)
        guard lineRanges.indices.contains(target) else { return nil }
        let range = lineRanges[target]
        let column = index - lineRanges[line].lowerBound
        return frames[range.lowerBound + min(column, range.count - 1)]
    }
}
