import Foundation

struct FrameGridScrollTarget: Equatable {
    let rowID: String

    /// The outer LazyVStack knows its row IDs even before it mounts their cells.
    /// Resolve against the displayed snapshot, including individual Photos chunks.
    static func resolve(frame: FrameID, snapshot: GridStageSnapshot) -> Self? {
        guard let index = snapshot.rowIndexByFrameID[frame] else { return nil }
        return Self(rowID: snapshot.rows[index].id)
    }
}

/// Scroll geometry is recorded without publishing every scroll tick to the grid.
@MainActor
final class FrameGridScrollVisibility {
    var viewport = CGRect.zero
    var horizontalViewports: [String: CGRect] = [:]

    nonisolated static func needsReveal(lower: CGFloat, upper: CGFloat,
                            visibleLower: CGFloat, visibleUpper: CGFloat,
                            margin: CGFloat = 0) -> Bool {
        revealOffset(lower: lower, upper: upper, visibleLower: visibleLower, visibleUpper: visibleUpper, margin: margin) != nil
    }

    /// Return the smallest offset that exposes both edges, independent of lazy cell mounting.
    nonisolated static func revealOffset(lower: CGFloat, upper: CGFloat,
                                        visibleLower: CGFloat, visibleUpper: CGFloat,
                                        margin: CGFloat = 0) -> CGFloat? {
        let extent = visibleUpper - visibleLower
        guard extent > 0 else { return max(0, lower) }
        if upper - lower > extent, lower == visibleLower { return nil }
        // Keep a navigation buffer inside the viewport. Reduce it when the
        // viewport cannot fit the item plus both margins to avoid oscillation.
        let inset = min(max(0, margin), max(0, (extent - (upper - lower)) / 2))
        if lower < visibleLower + inset { return max(0, lower - inset) }
        if upper > visibleUpper - inset {
            return max(0, upper - lower > extent ? lower : upper - extent + inset)
        }
        return nil
    }
}
