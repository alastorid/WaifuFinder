import Foundation
import Combine

func registerFrameSelectionTests() {
    Registry.shared.add(suite: "Frame selection", TestCase(name: "shift arrows grow shrink and reverse a range across displayed thumbnail rows", knownBug: nil) {
        let directory = try Temporary.directory("shift-arrow-range")
        try await Task { @MainActor in
            let session = WaifuSession(settingsURL: directory.appendingPathComponent("settings.json"))
            session.frameOrder = .only([.time])
            let frames = (1...8).map { FrameFixture.make(id: String(format: "f%06d", $0), time: Double($0)) }
            session.updateAnalyzedFrames(frames)
            session.groupingEnabled = false
            let generation = session.thumbnailGeneration
            let key = GridStageKey(revision: session.gridRevision, generation: generation, width: 400,
                columns: 3, groupColumns: 3, grouped: false, photosSource: true, square: true)
            let snapshot = GridStageSnapshot.make(GridStageInput(key: key, sourceURL: nil,
                rows: session.displayedRows, frames: session.displayOrderFrames,
                beyondCap: [:], revealed: []))
            session.commitGridNavigation(snapshot.navigation, generation: generation)
            session.selectFrame(frames[1].id)
            session.focusNeighbour(1, extendingSelection: true)
            checkEqual(session.selectedFrameIDs, Set(frames[1...2].map(\.id)), "shift right extends from anchor")
            session.focusGroupEdge(forward: true, extendingSelection: true)
            checkEqual(session.selectedFrameIDs, Set(frames[1...5].map(\.id)), "shift down includes intervening thumbnails")
            session.focusGroupEdge(forward: false, extendingSelection: true)
            checkEqual(session.selectedFrameIDs, Set(frames[1...2].map(\.id)), "shift up shrinks toward anchor")
            session.focusNeighbour(-1, extendingSelection: true)
            session.focusNeighbour(-1, extendingSelection: true)
            checkEqual(session.selectedFrameIDs, Set(frames[0...1].map(\.id)), "shift left reverses across anchor")
            session.focusNeighbour(1)
            checkEqual(session.selectedFrameIDs, [frames[1].id], "ordinary arrow selects a single frame again")
        }.value
    })

    Registry.shared.add(suite: "Frame selection", TestCase(name: "pointer selection requests visibility checks without rebuilding grid geometry", knownBug: nil) {
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
            checkEqual(scrolls, (0..<400).map { FrameID(sampleIndex: $0) },
                       "each newly selected thumbnail requests a visibility check; deselection does not")
            checkNil(session.gridNavigationFrameID, "pointer selection cancels pending navigation intent")
            checkEqual(session.gridRevision, revision, "selection keeps the existing layout geometry")
            withExtendedLifetime(subscription) {}
        }.value
    })
    Registry.shared.add(suite: "Frame selection", TestCase(name: "navigation and preview dismissal scroll once without replaying into newly mounted rows", knownBug: nil) {
        let directory = try Temporary.directory("navigation-scroll")
        try await Task { @MainActor in
            let session = WaifuSession(settingsURL: directory.appendingPathComponent("settings.json"))
            let frame = FrameFixture.make(id: "f30", time: 30)
            let preceding = FrameFixture.make(id: "f29", time: 29)
            session.frameOrder = .only([.time])
            session.updateAnalyzedFrames([preceding, frame])
            session.focusFromPointer(preceding.id)
            session.selectFrame(preceding.id)
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
            session.focusNeighbour(1)
            changing = false
            try await Task.sleep(for: .milliseconds(20))
            checkEqual(scrolls, [frame.id], "keyboard navigation requests one reveal")
            checkEqual(session.selectedFrameIDs, [frame.id], "the reveal targets the thumbnail selected by the arrow")
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
    Registry.shared.add(suite: "Frame selection", TestCase(name: "offscreen navigation targets concrete lazy rows in flow, Photos chunks and strips", knownBug: nil) {
        let frames = (0..<10_000).map { index in
            AnalyzedFrame(sampleIndex: index, timeSeconds: Double(index), artScore: 0,
                faceScore: nil, faceCount: 0, faceAreaFraction: nil, featurePrint: nil,
                thumbnailAspectRatio: 1)
        }
        let frame = frames[9_500].id
        let group = FrameGroupID(raw: "group")
        let ungrouped = FrameRow(group: nil, frames: frames)
        func snapshot(grouped: Bool, photos: Bool, rows: [FrameRow]) -> GridStageSnapshot {
            GridStageSnapshot.make(GridStageInput(
                key: GridStageKey(revision: 1, generation: UUID(), width: 263, columns: 2,
                    groupColumns: 2, grouped: grouped, photosSource: photos, square: true),
                sourceURL: nil, rows: rows, frames: frames, beyondCap: [:], revealed: []))
        }
        let flow = snapshot(grouped: false, photos: true, rows: [])
        let photos = snapshot(grouped: true, photos: true, rows: [ungrouped])
        let strip = snapshot(grouped: true, photos: true, rows: [FrameRow(group: group, frames: frames)])
        let video = snapshot(grouped: true, photos: false, rows: [ungrouped])
        checkEqual(FrameGridScrollTarget.resolve(frame: frame, snapshot: flow)?.rowID,
                   "flat:4750", "flow targets a row known to the outer LazyVStack before its cells mount")
        checkEqual(FrameGridScrollTarget.resolve(frame: frame, snapshot: photos)?.rowID,
                   ungrouped.id + ":4750", "ungrouped Photos target the selected photo's chunk, not the entire section")
        checkEqual(FrameGridScrollTarget.resolve(frame: frame, snapshot: strip)?.rowID,
                   group.raw, "horizontal photo strips reveal their containing row first")
        checkEqual(FrameGridScrollTarget.resolve(frame: frame, snapshot: video)?.rowID,
                   ungrouped.id, "video strips retain row navigation")
        checkNil(FrameGridScrollTarget.resolve(frame: FrameID(sampleIndex: 10_000), snapshot: flow),
                 "a target absent from the displayed snapshot waits for a later layout")
    })
    Registry.shared.add(suite: "Frame selection", TestCase(name: "visibility checks reveal clipped thumbnails and leave visible items in place", knownBug: nil) {
        await Task { @MainActor in
            func needs(_ lower: CGFloat, _ upper: CGFloat) -> Bool {
                FrameGridScrollVisibility.needsReveal(lower: lower, upper: upper,
                    visibleLower: 100, visibleUpper: 400)
            }
            check(!needs(100, 230), "an item aligned with the leading edge stays put")
            check(!needs(270, 400), "an item aligned with the trailing edge stays put")
            check(needs(99, 229), "a clipped leading edge needs revealing")
            check(needs(271, 401), "a clipped trailing edge needs revealing")
            check(needs(500, 630), "an offscreen item needs revealing")
            check(needs(50, 450), "an oversized clipped item reveals its leading edge")
            check(!needs(100, 500), "an oversized item already aligned at the leading edge stays put")
            check(needs(450, 850), "an oversized offscreen item still needs revealing")
            check(needs(99.9, 229.9), "even a fractional leading-edge clip needs revealing")
            check(needs(270.1, 400.1), "even a fractional trailing-edge clip needs revealing")
            func offset(_ lower: CGFloat, _ upper: CGFloat) -> CGFloat? {
                FrameGridScrollVisibility.revealOffset(lower: lower, upper: upper,
                    visibleLower: 100, visibleUpper: 400)
            }
            checkEqual(offset(99, 229), 99, "reveal the leading edge with the smallest scroll")
            checkEqual(offset(271, 401), 101, "reveal the trailing edge with the smallest scroll")
            checkEqual(offset(9000, 9130), 8830, "far-off lazy rows use absolute offsets without cell IDs")
            checkEqual(offset(50, 450), 50, "oversized clipped thumbnails align their leading edge")

        }.value
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
