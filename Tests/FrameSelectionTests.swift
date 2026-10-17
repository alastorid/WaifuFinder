import Foundation
import Combine

func registerFrameSelectionTests() {
    Registry.shared.add(suite: "Frame selection", TestCase(name: "pointer clicks select without requesting scrolling or rebuilding grid geometry", knownBug: nil) {
        let directory = try Temporary.directory("pointer-scroll")
        try await Task { @MainActor in
            let session = WaifuSession(settingsURL: directory.appendingPathComponent("settings.json"))
            var scrolls: [FrameID] = []
            let subscription = session.gridScrollRequests.sink { id in
                MainActor.assumeIsolated { scrolls.append(id) }
            }
            let revision = session.gridRevision
            for index in 0..<400 {
                let id = FrameID(sampleIndex: index)
                session.focusFromPointer(id)
                session.selectFrame(id)
                session.focusFromPointer(id)
                session.selectFrame(id, command: true)
            }
            try await Task.sleep(for: .milliseconds(30))
            check(scrolls.isEmpty, "clicking or toggling thumbnails must not move the viewport")
            checkNil(session.gridNavigationFrameID, "pointer selection cancels pending navigation intent")
            checkEqual(session.gridRevision, revision, "selection keeps the existing layout geometry")
            withExtendedLifetime(subscription) {}
        }.value
    })
    Registry.shared.add(suite: "Frame selection", TestCase(name: "navigation and preview dismissal scroll once without replaying into newly mounted rows", knownBug: nil) {
        let directory = try Temporary.directory("navigation-scroll")
        try await Task { @MainActor in
            let session = WaifuSession(settingsURL: directory.appendingPathComponent("settings.json"))
            let frame = FrameFixture.make(id: "f30")
            var scrolls: [FrameID] = []
            var changing = false
            let subscription = session.gridScrollRequests.sink { id in
                MainActor.assumeIsolated {
                    check(!changing, "scrolling is deferred until focus and selection commit")
                    checkEqual(session.focusedFrameID, id, "scrolling sees committed focus")
                    scrolls.append(id)
                }
            }
            changing = true
            session.focus(on: frame)
            changing = false
            try await Task.sleep(for: .milliseconds(20))
            checkEqual(scrolls, [frame.id], "keyboard navigation requests one reveal")
            scrolls.removeAll()
            session.showPreview(frame)
            try await Task.sleep(for: .milliseconds(20))
            check(scrolls.isEmpty, "opening preview does not move the hidden grid")
            session.previewFrame = nil
            try await Task.sleep(for: .milliseconds(20))
            checkEqual(scrolls, [frame.id], "closing preview reveals the focused photo once")
            var lateScrolls: [FrameID] = []
            let newlyMountedRow = session.gridScrollRequests.sink { id in
                MainActor.assumeIsolated { lateScrolls.append(id) }
            }
            try await Task.sleep(for: .milliseconds(20))
            check(lateScrolls.isEmpty, "lazy row mounting must not replay focus or an initial nil preview")
            session.focusFromPointer(frame.id)
            checkNil(session.gridNavigationFrameID, "a subsequent click clears old navigation intent")
            withExtendedLifetime((subscription, newlyMountedRow)) {}
        }.value
    })
    Registry.shared.add(suite: "Frame selection", TestCase(name: "ungrouped Photos navigation targets the photo instead of centering its entire grid section", knownBug: nil) {
        let frame = FrameID(sampleIndex: 9_500)
        let group = FrameGroupID(raw: "group")
        checkEqual(FrameGridScrollTarget.resolve(frame: frame, rowID: "ungrouped", group: nil,
            groupRows: true, photosSource: true), .frame(frame), "large ungrouped photo grids scroll to the individual photo")
        checkEqual(FrameGridScrollTarget.resolve(frame: frame, rowID: "group", group: group,
            groupRows: true, photosSource: true), .row("group"), "horizontal photo strips scroll by row")
        checkEqual(FrameGridScrollTarget.resolve(frame: frame, rowID: "ungrouped", group: nil,
            groupRows: true, photosSource: false), .row("ungrouped"), "video strips retain row navigation")
        checkEqual(FrameGridScrollTarget.resolve(frame: frame, rowID: "group", group: group,
            groupRows: false, photosSource: true), .frame(frame), "flow layout always navigates by photo")
    })
    let order = (0..<6).map { FrameID(sampleIndex: $0) }
    Registry.shared.add(suite: "Frame selection", TestCase(name: "plain clicks replace selection and command clicks toggle individual frames", knownBug: nil) {
        var selection = FrameSelection(ids: Set(order))
        selection.click(order[2], order: order)
        checkEqual(selection.ids, [order[2]], "plain click clears all previous selections")
        selection.click(order[2], order: order)
        checkEqual(selection.ids, [order[2]], "clicking selected frame keeps it selected")
        selection.click(order[4], order: order, command: true)
        checkEqual(selection.ids, [order[2], order[4]], "command adds without clearing")
        selection.click(order[2], order: order, command: true)
        checkEqual(selection.ids, [order[4]], "command removes just the clicked frame")
        checkEqual(selection.anchor, order[2], "last physical selection click sets the anchor")
    })
    Registry.shared.add(suite: "Frame selection", TestCase(name: "shift range uses visual order and a stable anchor", knownBug: nil) {
        let visual = [order[5], order[2], order[4], order[0], order[3], order[1]]
        var selection = FrameSelection()
        selection.click(order[2], order: visual)
        selection.click(order[3], order: visual, shift: true)
        checkEqual(selection.ids, [order[2], order[4], order[0], order[3]], "inclusive visual range")
        selection.click(order[4], order: visual, shift: true)
        checkEqual(selection.ids, [order[2], order[4]], "subsequent shift click shortens the range")
        selection.click(order[5], order: visual, shift: true)
        checkEqual(selection.ids, [order[5], order[2]], "range can reverse across its anchor")
        selection.click(order[1], order: visual, command: true)
        selection.click(order[3], order: visual, command: true, shift: true)
        checkEqual(selection.ids, [order[5], order[2], order[3], order[1]], "command shift adds a range")
    })
    Registry.shared.add(suite: "Frame selection", TestCase(name: "select all excludes hidden frames and stale anchors cannot select hidden ranges", knownBug: nil) {
        var selection = FrameSelection(ids: [order[0]], anchor: order[0])
        let visible = [order[1], order[3], order[5]]
        selection.selectAll(order: visible, focused: order[3])
        checkEqual(selection.ids, Set(visible), "select all is exactly the displayed subset")
        checkEqual(selection.anchor, order[3], "focused visible frame anchors later range selection")
        selection.click(order[5], order: visible, shift: true)
        checkEqual(selection.ids, [order[3], order[5]], "shift after select all follows the focused frame")
        selection = FrameSelection(ids: [order[0]], anchor: order[0])
        selection.click(order[3], order: visible, shift: true)
        checkEqual(selection.ids, [order[3]], "hidden anchor starts a fresh selection")
        selection.click(order[0], order: visible)
        checkEqual(selection.ids, [order[3]], "hidden click cannot mutate the selection")
        selection.selectAll(order: [], focused: order[0])
        check(selection.ids.isEmpty && selection.anchor == nil, "empty view clears selection and anchor")
    })
}
