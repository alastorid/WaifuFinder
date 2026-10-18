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
