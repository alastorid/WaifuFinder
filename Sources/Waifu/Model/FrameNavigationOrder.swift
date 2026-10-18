import Foundation

/// An immutable navigation sequence shared with a committed thumbnail layout.
/// Preparing a replacement layout must not change where its visible cells lead.
struct FrameNavigationOrder: Sendable {
    let frames: [AnalyzedFrame]
    let indexByID: [FrameID: Int]
    private let rowByID: [FrameID: Int]
    private let groupRows: Bool

    init(frames: [AnalyzedFrame], rows: [FrameRow], groupRows: Bool) {
        self.groupRows = groupRows
        self.frames = groupRows ? rows.flatMap(\.frames) : frames
        indexByID = Dictionary(uniqueKeysWithValues: self.frames.enumerated().map { ($0.element.id, $0.offset) })
        rowByID = groupRows
            ? Dictionary(uniqueKeysWithValues: rows.enumerated().flatMap { index, row in
                row.frames.map { ($0.id, index) }
            }) : [:]
    }

    func neighbour(of id: FrameID, by delta: Int) -> AnalyzedFrame? {
        guard let index = indexByID[id] else { return nil }
        let next = index + delta
        guard frames.indices.contains(next),
              !groupRows || rowByID[id] == rowByID[frames[next].id] else { return nil }
        return frames[next]
    }
}
