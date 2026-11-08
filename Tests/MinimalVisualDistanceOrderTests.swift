import Foundation
import Vision

func registerMinimalVisualDistanceOrderTests() {
    let suite = "frame arrangement: visual distance"

    Registry.shared.add(suite: suite, TestCase(name: "nine visual matches reconcile a deep scroll with the actual prepared thumbnail rows", knownBug: nil) {
        let directory = try Temporary.directory("visual-nine-viewport")
        let a = try JSONEncoder().encode(await makeFeaturePrintObservation(0))
        let b = try JSONEncoder().encode(await makeFeaturePrintObservation(4))
        try await Task { @MainActor in
            let session = WaifuSession(settingsURL: directory.appendingPathComponent("settings.json"))
            session.frameOrder = .only([.time])
            let frames = (0..<90).map { index in
                FrameFixture.make(id: String(format: "f%06d", index), time: Double(index),
                                  featurePrint: index.isMultiple(of: 2) ? a : b)
            }
            session.updateAnalyzedFrames(frames)
            session.frameOrder.setEnabled(true, for: .minimalVisualDistance)
            session.frameLimits.time.low = 81
            for _ in 0..<500 {
                if !session.isComputingVisualDistance { break }
                try await Task.sleep(for: .milliseconds(10))
            }
            check(!session.isComputingVisualDistance, "nine real Vision prints complete their asynchronous route")
            checkEqual(Set(session.displayOrderFrames.map(\.id)), Set(frames.suffix(9).map(\.id)), "the route keeps all nine matches")
            let key = GridStageKey(revision: session.gridRevision, generation: session.thumbnailGeneration,
                width: 800, columns: 5, groupColumns: 5, grouped: session.usesGroupRows, photosSource: true, square: true)
            let snapshot = GridStageSnapshot.make(GridStageInput(key: key, sourceURL: nil,
                rows: session.displayedRows, frames: session.displayOrderFrames, beyondCap: [:], revealed: []))
            let visibility = FrameGridScrollVisibility()
            visibility.viewport = CGRect(x: 0, y: 2_000, width: 800, height: 600)
            // Reproduce the old mismatch: preparation already finds nine items,
            // while not one drawn row intersects the retained real scroll rect.
            checkEqual(snapshot.visibleItems(in: visibility.viewport, horizontalOffsets: [:]).count, 9, "preparation clamps internally")
            check(!snapshot.rows.contains { $0.originY + $0.height > visibility.viewport.minY && $0.originY < visibility.viewport.maxY }, "the old real viewport is entirely below the nine matching thumbnails")
            checkEqual(visibility.reconcile(snapshot: snapshot, viewportSize: visibility.viewport.size, reset: false), 0,
                       "same-album filtering explicitly scrolls the short result back into view")
            check(snapshot.rows.contains { $0.originY + $0.height > visibility.viewport.minY && $0.originY < visibility.viewport.maxY }, "actual and prepared viewports now intersect the matching rows")
            checkNil(visibility.reconcile(snapshot: snapshot, viewportSize: visibility.viewport.size, reset: false), "stable geometry does not repeatedly reset scrolling")
        }.value
    })

    Registry.shared.add(suite: suite, TestCase(name: "single filtered match is published immediately with MVD enabled", knownBug: nil) {
        let directory = try Temporary.directory("visual-single-match")
        try await Task { @MainActor in
            let session = WaifuSession(settingsURL: directory.appendingPathComponent("settings.json"))
            let frames = (0..<12).map { FrameFixture.make(id: String(format: "f%06d", $0), time: Double($0)) }
            session.updateAnalyzedFrames(frames)
            session.frameLimits.time.low = 11
            check(!session.isComputingVisualDistance, "a singleton needs no route")
            checkEqual(session.displayOrderFrames.map(\.id), [frames[11].id], "the one match reaches the grid immediately")
            try await Task.sleep(for: .milliseconds(250))
            checkEqual(session.displayOrderFrames.map(\.id), [frames[11].id], "an obsolete multi-photo route cannot replace the singleton")
            session.frameLimits.time.low = 12
            check(session.displayOrderFrames.isEmpty, "zero matches clear the grid")
            session.frameLimits.time.low = 11
            checkEqual(session.displayOrderFrames.map(\.id), [frames[11].id], "returning from zero to one match stays visible")
            session.settings.tagTopCount = 3
            session.settings.taggerThreshold = 0.8
            session.tagFilter = TagFilter(requiredTags: ["test"], scoreSortTag: "test")
            let revision = session.tagControlsResetRevision
            session.resetTagControls()
            checkEqual(session.settings.tagTopCount, WaifuSettings.default.tagTopCount, "Clear resets the Top slider")
            checkEqual(session.settings.taggerThreshold, WaifuSettings.default.taggerThreshold, "Clear resets the threshold")
            checkEqual(session.tagFilter, TagFilter(), "Clear resets every tag filter control")
            checkEqual(session.tagControlsResetRevision, revision + 1, "the quick-filter view receives a complete reset")
        }.value
    })
    Registry.shared.add(suite: suite, TestCase(name: "native route verification reuses shared edges without changing route quality", knownBug: nil) {
        let seed = [0,1,2,3,4,5]
        var calls = 0
        let result = try MinimalVisualDistanceOrder.verifiedRoute(seed: seed, proposed: Array(seed.reversed()), count: seed.count) {
            calls += 1
            return Double(abs($0 - $1))
        }
        checkEqual(result, seed, "equal-cost routes retain deterministic orientation")
        checkEqual(calls, seed.count - 1, "shared/reversed edges are measured only once")
    })
    Registry.shared.add(suite: suite, TestCase(name: "a scrambled line becomes a shortest open path", knownBug: nil) {
        let positions = [0.0, 10, 1, 9, 2, 8, 3, 7, 4, 6, 5]
        let route = try MinimalVisualDistanceOrder.path(count: positions.count) {
            abs(positions[$0] - positions[$1])
        }
        let cost = zip(route, route.dropFirst()).reduce(0.0) { $0 + abs(positions[$1.0] - positions[$1.1]) }
        checkEqual(cost, 10, "the open path spans the line once, with no return edge")
        checkEqual(Set(route), Set(positions.indices), "every photo occurs exactly once")
        checkEqual(route.count, positions.count, "no repeated or missing photos")
        let repeated = try MinimalVisualDistanceOrder.path(count: positions.count) {
            abs(positions[$0] - positions[$1])
        }
        checkEqual(route, repeated, "forward orientation is deterministic")
    })

    Registry.shared.add(suite: suite, TestCase(name: "large paths avoid an all-pairs distance matrix", knownBug: nil) {
        let count = 4_096
        let positions = (0..<count).map { Double(($0 * 7_919) % count) }
        var comparisons = 0
        let route = try MinimalVisualDistanceOrder.path(count: count) {
            comparisons += 1
            return abs(positions[$0] - positions[$1])
        }
        let cost = zip(route, route.dropFirst()).reduce(0.0) { $0 + abs(positions[$1.0] - positions[$1.1]) }
        checkEqual(cost, Double(count - 1), "hierarchical joins preserve the shortest path on a line")
        checkEqual(Set(route), Set(0..<count), "partitioning retains every photo")
        check(comparisons < count * 400, "distance work stays below the quadratic pair count: \(comparisons)")
    })

    Registry.shared.add(suite: suite, TestCase(name: "equal and unavailable distances preserve a stable order", knownBug: nil) {
        for distance: Double? in [0, nil, .nan, -.infinity] {
            let route = try MinimalVisualDistanceOrder.path(count: 70) { _, _ in distance }
            checkEqual(route, Array(0..<70), "unusable or equal metrics fall back to the supplied order")
        }
        checkEqual(try MinimalVisualDistanceOrder.path(count: 0) { _, _ in nil }, [], "empty path")
        checkEqual(try MinimalVisualDistanceOrder.path(count: 1) { _, _ in nil }, [0], "single photo")
    })

    Registry.shared.add(suite: suite, TestCase(name: "obsolete path work responds to cancellation", knownBug: nil) {
        let task = Task.detached { () -> Bool in
            do {
                _ = try MinimalVisualDistanceOrder.path(count: 2_000) { left, right in
                    withUnsafeCurrentTask { $0?.cancel() }
                    return Double(abs(left - right))
                }
                return false
            } catch is CancellationError {
                return true
            } catch {
                return false
            }
        }
        check(await task.value, "a superseded route stops before finishing its distance work")
    })

    Registry.shared.add(suite: suite, TestCase(name: "visual ordering is enabled by default and cannot filter", knownBug: nil) {
        check(FrameOrder.defaultOrder.minimalVisualDistance.enabled, "default on")
        check(!FrameSortKey.minimalVisualDistance.supportsFiltering, "no visual range control")
        var limits = FrameLimits.none
        limits.setLimit(FrameLimit(low: 5, high: 6), for: .minimalVisualDistance)
        check(limits.isEmpty, "a visual limit cannot exclude photos")
        check(limits.limit(for: .minimalVisualDistance).isUnlimited, "the visual range is always unlimited")
        checkNil(FrameLimitBounds(time: 0...1, art: 0...1, face: nil).range(for: .minimalVisualDistance), "no visual slider span")
        var onlyVisual = FrameOrder.only([.minimalVisualDistance])
        onlyVisual.setEnabled(false, for: .minimalVisualDistance)
        check(onlyVisual.minimalVisualDistance.enabled, "the last active order stays enabled")
    })

    Registry.shared.add(suite: suite, TestCase(name: "Vision route preserves membership and reverses exactly after filters and caps", knownBug: nil) {
        let observationA = try await makeFeaturePrintObservation(0)
        let observationB = try await makeFeaturePrintObservation(2)
        let observationC = try await makeFeaturePrintObservation(4)
        let payloadA = try JSONEncoder().encode(observationA)
        let payloadB = try JSONEncoder().encode(observationB)
        let payloadC = try JSONEncoder().encode(observationC)
        let group = FrameGroupID(raw: "alternates")
        let frames = [
            FrameFixture.make(id: "f000000", time: 0, artScore: 1, featurePrint: payloadA),
            FrameFixture.make(id: "f000001", time: 1, artScore: 0.9, featurePrint: payloadB, groupID: group),
            FrameFixture.make(id: "f000002", time: 2, artScore: 0.8, featurePrint: payloadA),
            FrameFixture.make(id: "f000003", time: 3, artScore: 0.7, featurePrint: payloadC),
            FrameFixture.make(id: "f000004", time: 4, artScore: 0.6, featurePrint: payloadA, groupID: group),
            FrameFixture.make(id: "f000005", time: 5, artScore: 0.5, featurePrint: Data("corrupt".utf8)),
            FrameFixture.make(id: "f000006", time: 6, artScore: 0.4),
            FrameFixture.make(id: "f000007", time: 7, artScore: 0.1, featurePrint: payloadA)
        ]
        var order = FrameOrder.only([.art, .minimalVisualDistance])
        var limits = FrameLimits.none
        limits.art.low = 0.3
        let forward = FrameArrangement.arrange(frames, order: order, limits: limits,
            perGroupTopCount: 1, grouping: .default)
        checkEqual(Set(forward.visible.map(\.id)), Set(frames.prefix(7).filter { $0.id != frames[4].id }.map(\.id)),
                   "filtering and score-based representatives run before visual ordering")
        checkEqual(forward.beyondCapByGroup[group], 1, "cap accounting survives visual ordering")
        checkEqual(Array(forward.visible.suffix(2)).map(\.id), [frames[5].id, frames[6].id], "corrupt and absent prints remain visible at the end")
        let valid = forward.visible.compactMap { $0.featurePrint.flatMap(FeaturePrintCodec.decode) }
        var cost = 0.0
        for index in 1..<valid.count { cost += try valid[index - 1].distance(to: valid[index]) }
        let baseline = try observationA.distance(to: observationB) * 2 + observationA.distance(to: observationC)
        check(cost <= baseline + 1e-9, "visual path never worsens the existing adjacent distance")
        order.toggleDirection(of: .minimalVisualDistance)
        let reverse = FrameArrangement.arrange(frames, order: order, limits: limits,
            perGroupTopCount: 1, grouping: .default)
        checkEqual(reverse.visible.map(\.id), Array(forward.visible.reversed()).map(\.id), "reverse follows the identical route backward")
        checkEqual(reverse.beyondCapByGroup, forward.beyondCapByGroup, "direction never changes group representatives")
    })

    Registry.shared.add(suite: suite, TestCase(name: "visual controls keep a tagged album off the synchronous rebuild path", knownBug: nil) {
        let directory = try Temporary.directory("visual-toggle-responsiveness")
        try await Task { @MainActor in
            let session = WaifuSession(settingsURL: directory.appendingPathComponent("settings.json"))
            session.frameOrder = .only([.time, .ratingGeneral])
            var probabilities = (0..<TagVocabulary.headCount).map { Float($0 % 101) / 10_100 }
            probabilities[0] = 0.95
            if let generalIndex = session.tagVocabulary?.generalIndices.first {
                probabilities[generalIndex] = 0.9
            }
            let payload = TagProbabilityCodec.encode(probabilities)!
            let frames = (0..<1_800).map { index in
                var frame = FrameFixture.make(id: String(format: "f%06d", index), time: Double(index), artScore: 1)
                frame.tagProbabilities = payload
                return frame
            }
            session.updateAnalyzedFrames(frames)
            // Let the background summary publish before measuring visual work.
            if session.tagVocabulary != nil {
                for _ in 0..<500 {
                    if !session.tagCommonTags.isEmpty { break }
                    try await Task.sleep(for: .milliseconds(20))
                }
                check(!session.tagCommonTags.isEmpty, "the fixture's shared tag is indexed")
            }
            let clock = ContinuousClock()
            // Full vectors make accidental full decoding for rating keys visible.
            // Both controls and ordinary rebuilds must stay responsive; the latter
            // no longer provides a deliberately slow relative timing baseline.
            let rebuildStart = clock.now
            session.faceFilter = .all
            let rebuildTime = rebuildStart.duration(to: clock.now)
            let revision = session.gridRevision
            let commonTags = session.tagCommonTags
            let facetTags = session.tagFacetTags
            let toggleStart = clock.now
            session.frameOrder.setEnabled(true, for: .minimalVisualDistance)
            let toggleTime = toggleStart.duration(to: clock.now)
            print("    tagged album rebuild: \(rebuildTime); visual toggle: \(toggleTime)")
            check(rebuildTime < .seconds(1), "rating keys must not decode 1,800 full probability vectors")
            check(toggleTime < .milliseconds(100), "enabling visual order stays responsive")
            checkEqual(session.gridRevision, revision, "starting work keeps an unchanged grid mounted")
            check(session.isComputingVisualDistance, "the route is scheduled after the checkbox handler returns")
            check(!session.isSortingVisualDistance, "quick routes do not flash a busy indicator")
            checkEqual(session.displayedFrames.map(\.id), frames.map(\.id), "pending route keeps the published grid stable")
            for _ in 0..<250 {
                if !session.isComputingVisualDistance { break }
                try await Task.sleep(for: .milliseconds(20))
            }
            check(!session.isComputingVisualDistance, "background route completes")
            checkEqual(session.displayedFrames.map(\.id), frames.map(\.id), "missing features retain every photo in score order")
            checkEqual(session.gridRevision, revision, "an unchanged route does not rebuild the grid on completion")
            checkEqual(session.tagCommonTags, commonTags, "visual routing reuses common tags")
            checkEqual(session.tagFacetTags, facetTags, "visual routing reuses tag summaries")
            let reverseStart = clock.now
            session.frameOrder.toggleDirection(of: .minimalVisualDistance)
            check(reverseStart.duration(to: clock.now) < .milliseconds(100), "reverse stays responsive")
            checkEqual(session.displayOrderFrames.map(\.id), frames.reversed().map(\.id), "navigation reverses with the cached route")
            let disableStart = clock.now
            session.frameOrder.setEnabled(false, for: .minimalVisualDistance)
            check(disableStart.duration(to: clock.now) < .milliseconds(100), "disabling visual order stays responsive")
            checkEqual(session.displayOrderFrames.map(\.id), frames.map(\.id), "disabling restores score order")
            session.updateAnalyzedFrames(Array(frames.suffix(7)))
            session.frameOrder.setEnabled(true, for: .minimalVisualDistance)
            checkEqual(Set(session.displayedFrames.map(\.id)), Set(frames.suffix(7).map(\.id)), "frame replacement invalidates the cached album")
            session.frameOrder.setEnabled(false, for: .minimalVisualDistance)
        }.value
    })

    Registry.shared.add(suite: suite, TestCase(name: "session publishes only the current shown set and reverses its cached route", knownBug: nil) {
        let directory = try Temporary.directory("visual-order")
        let payloadA = try JSONEncoder().encode(await makeFeaturePrintObservation(0))
        let payloadB = try JSONEncoder().encode(await makeFeaturePrintObservation(4))
        try await Task { @MainActor in
            let session = WaifuSession(settingsURL: directory.appendingPathComponent("settings.json"))
            check(!session.groupingEnabled, "grouping defaults off now that visual sequencing is the default")
            let frames = (0..<12).map {
                FrameFixture.make(id: String(format: "f%06d", $0), time: Double($0), artScore: Float($0),
                                  featurePrint: $0 % 3 == 0 ? payloadA : payloadB,
                                  groupID: FrameGroupID(raw: "group-\($0 % 2)"))
            }
            check(!session.usesGroupRows, "the default visual path is shown as a continuous grid")
            session.updateAnalyzedFrames(frames)
            session.frameLimits.time.low = 5
            for _ in 0..<250 {
                if !session.isComputingVisualDistance { break }
                try await Task.sleep(for: .milliseconds(20))
            }
            check(!session.isComputingVisualDistance, "background route completes")
            let forward = session.displayOrderFrames.map(\.id)
            checkEqual(Set(forward), Set(frames.suffix(7).map(\.id)), "obsolete unfiltered work cannot restore hidden photos")
            let expected = try MinimalVisualDistanceOrder.sorted(Array(frames.suffix(7)))
            checkEqual(forward, expected.map(\.id), "the background result is published without regrouping the global path")
            checkEqual(session.displayedFrames.map(\.id), forward, "rendering and navigation share the same visual order")
            session.frameOrder.toggleDirection(of: .minimalVisualDistance)
            checkEqual(session.displayOrderFrames.map(\.id), Array(forward.reversed()), "reverse is immediate")
            check(!session.isComputingVisualDistance, "reverse reuses its cached route")
            session.frameLimits.time.low = 9
            check(session.isComputingVisualDistance, "a new shown set schedules another route")
            session.frameLimits.time.low = 5
            check(!session.isComputingVisualDistance, "undoing a pending filter change restores the cached route")
            checkEqual(session.displayOrderFrames.map(\.id), Array(forward.reversed()), "cache restoration keeps the chosen direction")
            session.frameLimits.time.low = 9
            // Visual distance is the only signal on by default, and the last
            // remaining signal cannot be switched off. Enable the fallback the
            // user reaches for before turning the visual route off.
            session.frameOrder.setEnabled(true, for: .time)
            session.frameOrder.setEnabled(false, for: .minimalVisualDistance)
            check(!session.frameOrder.minimalVisualDistance.enabled, "the visual route can be switched off once another signal is on")
            check(!session.isComputingVisualDistance, "turning the control off cancels the pending route")
            check(!session.usesGroupRows, "turning off the global sort does not enable grouping")
            session.groupingEnabled = true
            check(session.usesGroupRows, "group rows resume when grouping is explicitly enabled")
            try await Task.sleep(for: .milliseconds(250))
            checkEqual(session.displayedFrames.map(\.id), frames.suffix(3).map(\.id), "canceled work cannot replace the current score order")
        }.value
    })
}
