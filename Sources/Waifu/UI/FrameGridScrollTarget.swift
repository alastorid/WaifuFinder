import Foundation

enum FrameGridScrollTarget: Equatable {
    case frame(FrameID)
    case row(String)

    static func resolve(frame: FrameID, rowID: String?, group: FrameGroupID?,
                        groupRows: Bool, photosSource: Bool) -> Self {
        if groupRows, let rowID, !(photosSource && group == nil) { return .row(rowID) }
        return .frame(frame)
    }
}
