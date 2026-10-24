import Foundation

/// Group-row actions target the complete cached group, including capped/filtered
/// members. Other Photos layouts target only the clicked photo. The current
/// multiselection never expands a SAFIRE request.
enum LineageGroupTarget {
    static func frames(anchor:AnalyzedFrame,frames:[AnalyzedFrame],groups:[FrameGroup],photosSource:Bool,usesGroupRows:Bool) -> [AnalyzedFrame]? {
        guard photosSource else { return nil }
        guard usesGroupRows,let groupID=anchor.groupID,
              let group=groups.first(where:{$0.id==groupID}),group.memberIDs.contains(anchor.id.raw) else { return [anchor] }
        let members=Set(group.memberIDs)
        let result=frames.filter{members.contains($0.id.raw)}
        return result.contains(where:{$0.id==anchor.id}) ? result : [anchor]
    }
}
