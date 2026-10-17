import Foundation

/// Selection follows the rendered order, including reordered group rows and caps.
struct FrameSelection {
    var ids: Set<FrameID> = []
    var anchor: FrameID?

    mutating func click(_ id: FrameID, order: [FrameID], command: Bool = false, shift: Bool = false) {
        guard order.contains(id) else { return }
        if shift, let anchor, let start = order.firstIndex(of: anchor), let end = order.firstIndex(of: id) {
            let range = Set(order[min(start, end)...max(start, end)])
            ids = command ? ids.union(range) : range
            // Keep the anchor so another Shift-click can shorten or reverse the range.
        } else {
            if command {
                if !ids.insert(id).inserted { ids.remove(id) }
            } else { ids = [id] }
            anchor = id
        }
    }

    mutating func selectAll(order: [FrameID], focused: FrameID?) {
        ids = Set(order)
        anchor = focused.flatMap { ids.contains($0) ? $0 : nil } ?? order.first
    }
}
