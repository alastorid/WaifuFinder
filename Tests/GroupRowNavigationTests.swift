import Foundation

// ↑/↓ navigation in one-group-per-line mode. The lines the grid draws are the
// navigation: a group strip is one line, and a Photos album's ungrouped run is
// split into chunked lines. Stepping by group boundaries stranded the arrows
// inside that run, where every frame shares `groupID == nil`.
func registerGroupRowNavigationTests() {
    Registry.shared.add(suite: "Group row navigation", TestCase(name: "vertical neighbour walks drawn lines and keeps the column", knownBug: nil) {
        let a = FrameGroupID(raw: "A"), b = FrameGroupID(raw: "B")
        var frames = [
            FrameFixture.make(id: "f000001", time: 1, groupID: a),
            FrameFixture.make(id: "f000002", time: 2, groupID: a),
            FrameFixture.make(id: "f000003", time: 3, groupID: a),
            FrameFixture.make(id: "f000004", time: 4, groupID: a),
            FrameFixture.make(id: "f000005", time: 5, groupID: b),
            FrameFixture.make(id: "f000006", time: 6, groupID: b),
        ]
        for i in 7...12 { frames.append(FrameFixture.make(id: String(format: "f%06d", i), time: Double(i))) }
        let rows = [FrameRow(group: a, frames: Array(frames[0..<4])),
                    FrameRow(group: b, frames: Array(frames[4..<6])),
                    FrameRow(group: nil, frames: Array(frames[6..<12]))]
        let order = FrameNavigationOrder(frames: frames, rows: rows, groupRows: true)

        checkNil(order.verticalNeighbour(of: FrameID(sampleIndex: 2), forward: false)?.id,
                 "up from the first line has no line above")
        checkEqual(order.verticalNeighbour(of: FrameID(sampleIndex: 1), forward: true)?.id,
                   frames[4].id, "down from a strip lands on the next line at the same column")
        checkEqual(order.verticalNeighbour(of: FrameID(sampleIndex: 2), forward: true)?.id,
                   frames[5].id, "a column past the next line's end clamps to its last cell")
        checkNil(order.verticalNeighbour(of: FrameID(sampleIndex: 11), forward: true)?.id,
                 "down from the last frame has nowhere to go")

        // Flat layouts keep group-boundary jumps: their lines rewrap with the window.
        let flat = FrameNavigationOrder(frames: frames, rows: rows, groupRows: false)
        checkNil(flat.verticalNeighbour(of: FrameID(sampleIndex: 1), forward: true)?.id,
                 "a flow grid has no fixed lines to step")
    })

    Registry.shared.add(suite: "Group row navigation", TestCase(name: "committed photos layout walks the chunked ungrouped lines", knownBug: nil) {
        let directory = try Temporary.directory("group-nav-photos")
        try await Task { @MainActor in
            let session = WaifuSession(settingsURL: directory.appendingPathComponent("settings.json"))
            session.frameOrder = .only([.time])
            let a = FrameGroupID(raw: "A")
            var frames = [
                FrameFixture.make(id: "f000001", time: 1, groupID: a),
                FrameFixture.make(id: "f000002", time: 2, groupID: a),
            ]
            for i in 3...10 { frames.append(FrameFixture.make(id: String(format: "f%06d", i), time: Double(i))) }
            session.updateAnalyzedFrames(frames)
            session.groupingEnabled = true
            session.groupRows = true

            // A photos-source layout: the group is one strip, the ungrouped run is
            // chunked into two-wide lines — columns 2, matching the fixture key.
            let generation = session.thumbnailGeneration
            let key = GridStageKey(revision: session.gridRevision, generation: generation, width: 263,
                columns: 2, groupColumns: 2, grouped: true, photosSource: true, square: true, targetWidth: 130)
            let snapshot = GridStageSnapshot.make(GridStageInput(key: key, sourceURL: nil,
                rows: session.displayedRows, frames: session.displayOrderFrames,
                beyondCap: [:], revealed: []))
            checkEqual(snapshot.rows.map(\.id), ["A", "\u{0}ungrouped:0", "\u{0}ungrouped:1", "\u{0}ungrouped:2", "\u{0}ungrouped:3"],
                       "the ungrouped run is split into two-wide chunk lines")
            session.commitGridNavigation(snapshot.navigation, generation: generation)

            @MainActor func step(_ id: String, forward: Bool) -> String? {
                session.focusFromPointer(FrameID(sampleIndex: FrameFixture.sampleIndex(of: id)))
                session.focusGroupEdge(forward: forward)
                return session.focusedFrameID.map(\.description)
            }
            checkEqual(step("f000002", forward: true), "f000004",
                       "down from the strip enters the first chunk line at the same column")
            checkEqual(step("f000003", forward: true), "f000005",
                       "down crosses into the next chunk line instead of dead-ending")
            checkEqual(step("f000006", forward: false), "f000004",
                       "up returns to the previous chunk line at the same column")
            checkEqual(step("f000004", forward: false), "f000002",
                       "up from the first chunk line returns to the strip")
            checkEqual(step("f000003", forward: true), "f000005",
                       "repeat presses keep walking lines downward")
        }.value
    })
}
