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
    /// approximation. Flat layouts leave it empty: their lines rewrap with the
    /// window and carry no group meaning, so ↑/↓ keep jumping group boundaries.
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
