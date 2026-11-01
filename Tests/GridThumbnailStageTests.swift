import AppKit
import Foundation
import Combine

private func stageFrame(_ index: Int) -> AnalyzedFrame {
    AnalyzedFrame(sampleIndex: index, timeSeconds: Double(index), artScore: 0,
        faceScore: nil, faceCount: 0, faceAreaFraction: nil, featurePrint: nil,
        thumbnailAspectRatio: 1)
}

private func stageInput(_ revision: UInt64, _ frames: [AnalyzedFrame], generation: UUID,
                        grouped: Bool = false, rows: [FrameRow] = []) -> GridStageInput {
    GridStageInput(key: GridStageKey(revision: revision, generation: generation, width: 263,
        columns: 2, groupColumns: 2, grouped: grouped, photosSource: true, square: true),
        sourceURL: nil, rows: rows, frames: frames, beyondCap: [:], revealed: [])
}

@MainActor
private func stageWait(_ condition: () -> Bool) async {
    let deadline = ContinuousClock.now.advanced(by: .seconds(2))
    while !condition(), ContinuousClock.now < deadline { try? await Task.sleep(for: .milliseconds(2)) }
    check(condition(), "asynchronous viewport preparation reached its expected state")
}

@MainActor
private final class StageReads {
    private var pending: [FrameID: CheckedContinuation<CGImage?, Never>] = [:]
    var waiting: Set<FrameID> { Set(pending.keys) }
    func read(_ frame: AnalyzedFrame) async -> CGImage? {
        await withCheckedContinuation { pending[frame.id] = $0 }
    }
    func finish(_ id: FrameID, _ image: CGImage?) { pending.removeValue(forKey: id)?.resume(returning: image) }
    func finishAll() {
        let values = Array(pending.values); pending.removeAll()
        for value in values { value.resume(returning: nil) }
    }
}

func registerGridThumbnailStageTests() {
    let suite = "thumbnail staging"
    Registry.shared.add(suite: suite, TestCase(name: "deletion awaits in-flight preparation without blocking the UI actor", knownBug: nil) {
        await Task { @MainActor in
            let stage = GridThumbnailStage(), generation = UUID()
            let frames = (3000..<3005).map(stageFrame)
            let gate = DispatchSemaphore(value: 0), events = TickLog()
            stage.prepareDeletion(stageInput(1, frames, generation: generation), removing: [frames[0].id]) { input in
                events.record(1)
                gate.wait()
                return GridStageSnapshot.make(input)
            }
            await stageWait { events.all == [1] }
            stage.freezeDeletionPreparation()
            let remaining = Array(frames.dropFirst())
            stage.submit(stageInput(2, remaining, generation: generation),
                         viewportSize: CGSize(width: 263, height: 300), animated: true, immediate: true) { _ in nil }
            // This timer must run while the worker is blocked: the main actor
            // waits asynchronously, and the grid cannot install a duplicate.
            try? await Task.sleep(for: .milliseconds(35))
            checkNil(stage.current, "the grid waits for the existing preparation")
            gate.signal()
            await stageWait { stage.current?.key.revision == 2 }
            checkEqual(stage.current?.navigation.frames.map(\.id), remaining.map(\.id), "the prepared grid is committed")
            checkEqual(events.all, [1], "the selected layout is prepared once")
            stage.stop()
        }.value
    })
    Registry.shared.add(suite: suite, TestCase(name: "prepared deletion is reused only for the matching arrangement", knownBug: nil) {
        await Task { @MainActor in
            let stage = GridThumbnailStage(), generation = UUID()
            let frames = (2000..<2005).map(stageFrame)
            let original = stageInput(10, frames, generation: generation)
            stage.prepareDeletion(original, removing: [frames[1].id])
            let remaining = [frames[0]] + Array(frames.dropFirst(2))
            let expected = stageInput(11, remaining, generation: generation)
            await stageWait { stage.preparedDeletion(for: expected) != nil }
            let prepared = stage.preparedDeletion(for: expected)
            checkEqual(prepared?.key.revision, 11, "the prepared layout adopts the current revision")
            checkEqual(prepared?.navigation.frames.map(\.id), remaining.map(\.id), "the prepared order removes the selected photo")
            checkNil(stage.preparedDeletion(for: stageInput(11, Array(remaining.reversed()), generation: generation)),
                     "changed sort order cannot reuse stale geometry")
            checkNil(stage.preparedDeletion(for: stageInput(11, remaining, generation: UUID())),
                     "a different source generation cannot reuse the prepared layout")
            stage.stop()
        }.value
    })
    Registry.shared.add(suite: suite, TestCase(name: "deletion reflow commits without waiting for missing thumbnail reads", knownBug: nil) {
        await Task { @MainActor in
            let generation = UUID()
            let stage = GridThumbnailStage(preparationLimit: .seconds(5))
            let frames = (1000..<1004).map(stageFrame)
            let reads = StageReads()
            stage.submit(stageInput(1, frames, generation: generation),
                         viewportSize: CGSize(width: 263, height: 300), immediate: true) { frame in
                await reads.read(frame)
            }
            await stageWait { stage.current?.key.revision == 1 }
            let remaining = Array(frames.dropFirst())
            stage.submit(stageInput(2, remaining, generation: generation),
                         viewportSize: CGSize(width: 263, height: 300), animated: true, immediate: true) { frame in
                await reads.read(frame)
            }
            await stageWait { stage.current?.key.revision == 2 }
            checkEqual(stage.current?.navigation.frames.map(\.id), remaining.map(\.id),
                       "the removed photo leaves the grid even when its replacements are not decoded")
            check(reads.waiting.isEmpty, "an immediate deletion does not wait for image reads")
            reads.finishAll()
            stage.stop()
        }.value
    })
    Registry.shared.add(suite: suite, TestCase(name: "an immediate deletion shifts surviving cells into the committed rows", knownBug: nil) {
        let frames = (0..<5).map(stageFrame)
        let snapshot = GridStageSnapshot.make(stageInput(1, frames, generation: UUID()))
        let projected = snapshot.removing([frames[1].id])
        checkEqual(projected?.rows[0].items.map(\.id), [frames[0].id, frames[2].id],
                   "the next cell slides into the deleted cell’s place")
        checkEqual(projected?.rows[1].items.map(\.id), [frames[3].id, frames[4].id],
                   "later chunks reflow without leaving a hole")
        checkEqual(projected?.navigation.frames.map(\.id), frames.filter { $0.id != frames[1].id }.map(\.id),
                   "arrow navigation immediately drops the deleted photo")
        checkEqual(projected?.rows[1].originY, 132, "row origins contract after the removed content")
        checkEqual(projected?.contentHeight, 263, "the scroll extent matches the projected rows")
        checkNil(snapshot.removing([frames[1].id]).flatMap { $0.removing([FrameID(sampleIndex: 9999)]) },
                 "a snapshot without a queued cell is left untouched")
        checkEqual(snapshot.removing(Set(frames.map(\.id)))?.contentHeight, 2,
                   "deleting every row leaves an empty grid")
    })
    Registry.shared.add(suite: suite, TestCase(name: "an immediate grouped deletion shrinks only the affected strip", knownBug: nil) {
        let frames = (0..<5).map(stageFrame)
        let group = FrameGroupID(raw: "group")
        let rows = [FrameRow(group: group, frames: Array(frames.prefix(3))),
                    FrameRow(group: nil, frames: Array(frames.suffix(2)))]
        let snapshot = GridStageSnapshot.make(stageInput(1, frames, generation: UUID(), grouped: true, rows: rows))
        let projected = snapshot.removing([frames[1].id])
        checkEqual(projected?.rows.map(\.strip), [true, false], "empty strips are dropped and later rows remain")
        checkEqual(projected?.rows[0].items.map(\.id), [frames[0].id, frames[2].id],
                   "the group’s surviving cells close the hole")
        checkEqual(projected?.rows[1].originY, 159, "the following section moves up with the strip")
        checkEqual(projected?.navigation.neighbour(of: frames[2].id, by: 1), nil,
                   "navigation still stops at the group strip boundary")
        checkEqual(projected?.navigation.neighbour(of: frames[3].id, by: -1), nil,
                   "the ungrouped run retains its own line boundary")
    })
    Registry.shared.add(suite: suite, TestCase(name: "the stage paints an immediate deletion before the authoritative layout", knownBug: nil) {
        await Task { @MainActor in
            let generation = UUID()
            let stage = GridThumbnailStage(preparationLimit: .seconds(5))
            let frames = (0..<4).map(stageFrame)
            stage.submit(stageInput(1, frames, generation: generation),
                         viewportSize: CGSize(width: 263, height: 300), immediate: true) { _ in nil }
            await stageWait { stage.current?.key.revision == 1 }
            stage.applyImmediateDeletion(removing: [frames[0].id])
            checkEqual(stage.current?.navigation.frames.map(\.id), Array(frames.dropFirst()).map(\.id),
                       "the deleted photo leaves the committed grid synchronously")
            stage.stop()
        }.value
    })
    Registry.shared.add(suite: suite, TestCase(name: "a second delete cannot be undone by a newer layout that was already preparing", knownBug: nil) {
        let pixels = try ImageFixture.cgImage(width: 16, height: 16)
        await Task { @MainActor in
            let generation = UUID()
            let stage = GridThumbnailStage(preparationLimit: .seconds(5))
            let frames = (0..<4).map(stageFrame)
            stage.submit(stageInput(1, frames, generation: generation),
                         viewportSize: CGSize(width: 263, height: 300), immediate: true) { _ in nil }
            await stageWait { stage.current?.key.revision == 1 }
            let reads = StageReads()
            let remaining = Array(frames.dropFirst())
            stage.submit(stageInput(2, remaining, generation: generation),
                         viewportSize: CGSize(width: 263, height: 300), immediate: false) { frame in
                await reads.read(frame)
            }
            await stageWait { !reads.waiting.isEmpty }
            stage.applyImmediateDeletion(removing: [frames[1].id])
            checkEqual(stage.current?.navigation.frames.map(\.id), [frames[0].id, frames[2].id, frames[3].id],
                       "the second deleted photo leaves the visible grid immediately")
            for id in reads.waiting { reads.finish(id, pixels) }
            await stageWait { stage.current?.key.revision == 2 }
            checkEqual(stage.current?.navigation.frames.map(\.id), [frames[2].id, frames[3].id],
                       "the preparing layout is projected too and cannot restore the deleted photo")
            stage.stop()
        }.value
    })
    Registry.shared.add(suite: suite, TestCase(name: "visible thumbnails preempt bounded offscreen prefetching which resumes at the new viewport in both modes", knownBug: nil) {
        let directory = try Temporary.directory("thumbnail-prefetch-priority")
        await Task { @MainActor in
            for photosSource in [true, false] {
                let priority = GridThumbnailWorkPriority()
                let session = WaifuSession(settingsURL: directory.appendingPathComponent("settings.json"))
                let frames = (0..<40).map(stageFrame)
                let key = GridStageKey(revision: 1, generation: UUID(), width: 263, columns: 2,
                    groupColumns: 2, grouped: false, photosSource: photosSource, square: true)
                let snapshot = GridStageSnapshot.make(GridStageInput(key: key, sourceURL: nil,
                    rows: [], frames: frames, beyondCap: [:], revealed: []))
                var started: [FrameID] = []
                var cancelled = 0
                let prefetcher = GridThumbnailPrefetcher(priority: priority) { _, frame, _ in
                    started.append(frame.id)
                    do { try await Task.sleep(for: .seconds(30)) }
                    catch { cancelled += 1 }
                }
                prefetcher.install(snapshot: snapshot, session: session)
                prefetcher.setVisibleRows([key.tokenPrefix + snapshot.rows[0].id])
                await stageWait { started.count == 2 }
                checkEqual(started.count, 2, "speculative loading has a bounded concurrency")
                let reads = StageReads()
                let visible = Task { await priority.withVisibleWork { await reads.read(frames[20]) } }
                await stageWait { cancelled == 2 && reads.waiting.contains(frames[20].id) }
                prefetcher.setVisibleRows([key.tokenPrefix + snapshot.rows[10].id])
                for _ in 0..<20 { await Task.yield() }
                checkEqual(started.count, 2, "prefetching stays suspended while visible work is pending")
                reads.finishAll()
                _ = await visible.value
                await stageWait { started.count == 4 }
                checkEqual(Array(started.suffix(2)), [frames[20].id, frames[21].id],
                           "prefetching resumes in the newly visible rows before their neighbors")
                prefetcher.stop()
                await stageWait { cancelled == 4 }
            }
        }.value
    })
    Registry.shared.add(suite: suite, TestCase(name: "cancelled video thumbnail reads leave the newly visible frame available", knownBug: nil) {
        let directory = try Temporary.directory("thumbnail-scroll-cancel")
        let url = directory.appendingPathComponent("source.mp4")
        try VideoFixture.writeVideo(to: url, duration: 0.5, width: 128, height: 96, frameRate: 12)
        let obsolete = Task { try await ThumbnailRegenerator.shared.jpegData(for: url, timeSeconds: 0, analysisWidth: 64) }
        obsolete.cancel()
        do {
            _ = try await obsolete.value
            check(false, "an offscreen request must report cancellation")
        } catch is CancellationError {} catch {
            check(false, "cancelled reads report cancellation rather than a decode error: \(error)")
        }
        let data = try await ThumbnailRegenerator.shared.jpegData(for: url, timeSeconds: 0.25, analysisWidth: 64)
        check(!data.isEmpty, "the next visible frame decodes after cancellation")
    })
    Registry.shared.add(suite: suite, TestCase(name: "offscreen consumers cancel source work without interrupting another visible consumer", knownBug: nil) {
        await Task { @MainActor in
            var started = false
            var cancelled = false
            let source = Task<Data?, Never> { @MainActor in
                started = true
                do { try await Task.sleep(for: .seconds(30)) }
                catch { cancelled = true }
                return nil
            }
            let request = ThumbnailRequest(id: UUID(), task: source, retainAfterCompletion: false)
            let first = Task { await request.value() }
            let second = Task { await request.value() }
            await stageWait { started }
            // Let both consumers register before simulating a scroll departure.
            for _ in 0..<10 { await Task.yield() }
            first.cancel()
            for _ in 0..<10 { await Task.yield() }
            check(!source.isCancelled, "a remaining visible consumer keeps the source read alive")
            second.cancel()
            await stageWait { cancelled }
            check(source.isCancelled, "the last offscreen consumer cancels the underlying read")
            checkNil(await first.value, "an abandoned cell receives no result")
            checkNil(await second.value, "all abandoned consumers complete")
        }.value
    })
    Registry.shared.add(suite: suite, TestCase(name: "an already cancelled consumer cannot start an orphan thumbnail read", knownBug: nil) {
        await Task { @MainActor in
            let source = Task<Data?, Never> {
                try? await Task.sleep(for: .seconds(30))
                return nil
            }
            let request = ThumbnailRequest(id: UUID(), task: source, retainAfterCompletion: false)
            let consumer = Task { @MainActor in await request.value() }
            consumer.cancel()
            checkNil(await consumer.value, "a cancelled consumer returns immediately")
            check(source.isCancelled, "no source read survives without a consumer")
            _ = await source.value
        }.value
    })
    Registry.shared.add(suite: suite, TestCase(name: "an album grid mounted after loading receives the latest layout without another mutation", knownBug: nil) {
        let directory = try Temporary.directory("late-album-grid")
        let pixels = try ImageFixture.cgImage(width: 16, height: 16)
        await Task { @MainActor in
            let session = WaifuSession(settingsURL: directory.appendingPathComponent("settings.json"))
            session.frameOrder = .only([.time])
            let frames = (0..<120).map(stageFrame)
            session.updateAnalyzedFrames(frames)
            let stage = GridThumbnailStage()
            defer { stage.stop() }
            let subscription = session.gridLayoutUpdates.sink { revision in
                MainActor.assumeIsolated {
                    checkEqual(revision, session.gridRevision, "the replay names the committed revision")
                    stage.submit(stageInput(revision, session.displayOrderFrames, generation: session.thumbnailGeneration),
                                 viewportSize: CGSize(width: 263, height: 130)) { _ in pixels }
                }
            }
            await stageWait { stage.current?.navigation.frames.count == frames.count }
            checkEqual(stage.current?.navigation.frames.map(\.id), frames.map(\.id),
                       "a late subscriber paints the whole album without needing another event")
            withExtendedLifetime(subscription) {}
        }.value
    })
    Registry.shared.add(suite: suite, TestCase(name: "album loading and stale empty snapshots cannot claim filters rejected the photos", knownBug: nil) {
        checkEqual(FrameGridStatus.resolve(snapshotHasRows: false, snapshotGenerationMatches: true,
            snapshotRevisionMatches: false, analysisRunning: false, displayedCount: 120,
            analyzedCount: 120, filtersActive: false), .loading,
            "the initial empty snapshot is being replaced by real photos, not a filter failure")
        checkEqual(FrameGridStatus.resolve(snapshotHasRows: false, snapshotGenerationMatches: true,
            snapshotRevisionMatches: true, analysisRunning: true, displayedCount: 0,
            analyzedCount: 0, filtersActive: false), .loading,
            "an album awaiting its first batch shows loading")
        checkEqual(FrameGridStatus.resolve(snapshotHasRows: false, snapshotGenerationMatches: true,
            snapshotRevisionMatches: true, analysisRunning: false, displayedCount: 0,
            analyzedCount: 120, filtersActive: true), .filtered,
            "only a settled, actually filtered empty result shows the filter warning")
        checkEqual(FrameGridStatus.resolve(snapshotHasRows: false, snapshotGenerationMatches: true,
            snapshotRevisionMatches: true, analysisRunning: false, displayedCount: 0,
            analyzedCount: 0, filtersActive: false), .empty,
            "an album with no available photos must not suggest changing nonexistent limits")
        checkEqual(FrameGridStatus.resolve(snapshotHasRows: true, snapshotGenerationMatches: true,
            snapshotRevisionMatches: false, analysisRunning: true, displayedCount: 240,
            analyzedCount: 240, filtersActive: false), .content,
            "existing album content stays visible while the next batch prepares")
        checkEqual(FrameGridStatus.resolve(snapshotHasRows: true, snapshotGenerationMatches: false,
            snapshotRevisionMatches: true, analysisRunning: true, displayedCount: 0,
            analyzedCount: 0, filtersActive: false), .loading,
            "a previous album's content does not determine the new album's status")
    })
    Registry.shared.add(suite: suite, TestCase(name: "a growing album delivers its final committed layout and defers repeated visual routes", knownBug: nil) {
        let directory = try Temporary.directory("album-grid-publication")
        let pixels = try ImageFixture.cgImage(width: 16, height: 16)
        let payloadA = try JSONEncoder().encode(await makeFeaturePrintObservation(0))
        let payloadB = try JSONEncoder().encode(await makeFeaturePrintObservation(4))
        await testAlbumGridPublication(directory: directory, pixels: pixels, payloads: [payloadA, payloadB])
    })
    Registry.shared.add(suite: suite, TestCase(name: "target viewport uses stale scroll position and clamps a shorter filter", knownBug: nil) {
        let frames = (0..<20).map(stageFrame)
        let snapshot = GridStageSnapshot.make(stageInput(1, frames, generation: UUID()))
        let viewport = CGRect(x: 0, y: 264, width: 263, height: 130)
        checkEqual(snapshot.visibleItems(in: viewport, horizontalOffsets: [:]).map(\.id),
                   [frames[4].id, frames[5].id], "prepare the new order at the old viewport, not its first row")
        let filtered = GridStageSnapshot.make(stageInput(2, Array(frames.prefix(4)), generation: UUID()))
        checkEqual(filtered.visibleItems(in: viewport, horizontalOffsets: [:]).map(\.id),
                   [frames[2].id, frames[3].id], "short filtering prepares the final clamped viewport")
        let group = FrameGroupID(raw: "horizontal")
        let strip = GridStageSnapshot.make(stageInput(3, frames, generation: UUID(), grouped: true,
            rows: [FrameRow(group: group, frames: frames)]))
        checkEqual(strip.visibleItems(in: CGRect(x: 0, y: 0, width: 263, height: 140),
                    horizontalOffsets: [group.raw: 393]).map(\.id),
                   [frames[3].id, frames[4].id, frames[5].id], "a side-scrolled strip warms its visible columns")
    })
    Registry.shared.add(suite: suite, TestCase(name: "reorder holds displayed order until every viewport image is prepared", knownBug: nil) {
        let pixels = try ImageFixture.cgImage(width: 16, height: 16)
        await testAtomicStage(pixels)
    })
    Registry.shared.add(suite: suite, TestCase(name: "deadline commits ready subset; stale preparation never replaces displayed baseline", knownBug: nil) {
        let pixels = try ImageFixture.cgImage(width: 16, height: 16)
        await testStageDeadline(pixels)
    })
    Registry.shared.add(suite: suite, TestCase(name: "scrolling retargets a pending batch without restarting its deadline", knownBug: nil) {
        await testStageScroll()
    })
    Registry.shared.add(suite: suite, TestCase(name: "preview follows the visible route through a staged reversal and stays stable until closed", knownBug: nil) {
        let pixels = try ImageFixture.cgImage(width: 16, height: 16)
        let directory = try Temporary.directory("staged-preview-route")
        await testStagePreviewNavigation(pixels, directory: directory)
    })
    Registry.shared.add(suite: suite, TestCase(name: "committed group rows keep navigation inside each original row", knownBug: nil) {
        let frames = (0..<4).map(stageFrame)
        let rows = [FrameRow(group: FrameGroupID(raw: "a"), frames: Array(frames.prefix(2))),
                    FrameRow(group: nil, frames: Array(frames.suffix(2)))]
        let snapshot = GridStageSnapshot.make(stageInput(1, frames, generation: UUID(), grouped: true, rows: rows))
        checkEqual(snapshot.navigation.neighbour(of: frames[0].id, by: 1)?.id, frames[1].id, "forward within a strip")
        checkNil(snapshot.navigation.neighbour(of: frames[1].id, by: 1), "stop at strip boundary")
        checkEqual(snapshot.navigation.neighbour(of: frames[3].id, by: -1)?.id, frames[2].id, "ungrouped Photos chunks retain their original row")
        checkNil(snapshot.navigation.neighbour(of: frames[2].id, by: -1), "stop at the ungrouped row's start")
    })
    Registry.shared.add(suite: suite, TestCase(name: "group strips keep the section gap while ungrouped photo rows sit one pixel apart", knownBug: nil) {
        let frames = (0..<8).map(stageFrame)
        let rows = [FrameRow(group: FrameGroupID(raw: "g"), frames: Array(frames.prefix(3))),
                    FrameRow(group: nil, frames: Array(frames.suffix(5)))]
        let snapshot = GridStageSnapshot.make(stageInput(1, frames, generation: UUID(), grouped: true, rows: rows))
        checkEqual(snapshot.rows.count, 4, "the group is one strip and the ungrouped photos chunk into flat rows")
        checkEqual(snapshot.rows[0].spacing, CGFloat(12), "group strips keep the section gap below them")
        checkEqual(snapshot.rows.dropFirst().map(\.spacing), [CGFloat](repeating: 1, count: 3),
                   "ungrouped photo rows sit one pixel apart like the flat grid")
        var top: CGFloat = 1
        for row in snapshot.rows {
            checkEqual(row.originY, top, "row origins accumulate each row's own gap")
            top += row.height + row.spacing
        }
        checkEqual(snapshot.contentHeight, top - (snapshot.rows.last?.spacing ?? 1) + 1,
                   "the content ends one pixel below the last row")
        let flat = GridStageSnapshot.make(stageInput(2, frames, generation: UUID()))
        checkEqual(flat.rows.map(\.spacing), [CGFloat](repeating: 1, count: 4),
                   "the flat grid keeps one-pixel gaps everywhere")
    })
    Registry.shared.add(suite: suite, TestCase(name: "zoom walks the cell-width ladder and commits the zoomed layout animated", knownBug: nil) {
        let pixels = try ImageFixture.cgImage(width: 16, height: 16)
        await Task { @MainActor in
            checkEqual(GridZoom.zoomIn(from: 130), 160, "zoom in steps up to the next larger thumbnails")
            checkEqual(GridZoom.zoomOut(from: 130), 110, "zoom out steps down to more, smaller thumbnails")
            checkEqual(GridZoom.zoomIn(from: GridZoom.steps.last ?? 0), GridZoom.steps.last ?? 0, "zoom in stops at the top rung")
            checkEqual(GridZoom.zoomOut(from: GridZoom.steps.first ?? 0), GridZoom.steps.first ?? 0, "zoom out stops at the bottom rung")
            checkEqual(GridZoom.snapped(155), 160, "pinch magnification snaps to the nearest rung")

            let generation = UUID()
            let zoomedKey = GridStageKey(revision: 2, generation: generation, width: 263,
                columns: 1, groupColumns: 2, grouped: false, photosSource: true, square: true,
                targetWidth: 250)
            let zoomed = GridStageSnapshot.make(GridStageInput(key: zoomedKey, sourceURL: nil,
                rows: [], frames: (0..<4).map(stageFrame), beyondCap: [:], revealed: []))
            checkEqual(zoomed.rows.count, 4, "a zoomed-in target puts one thumbnail on each row")
            checkEqual(zoomed.rows[0].cellWidth, 250, "the flat cell width follows the zoom target")

            let stage = GridThumbnailStage(preparationLimit: .seconds(2))
            defer { stage.stop() }
            let frames = (0..<4).map(stageFrame)
            let size = CGSize(width: 263, height: 130)
            stage.submit(stageInput(1, frames, generation: generation), viewportSize: size) { _ in pixels }
            await stageWait { stage.current != nil }
            stage.submit(GridStageInput(key: zoomedKey, sourceURL: nil, rows: [], frames: frames,
                beyondCap: [:], revealed: []), viewportSize: size, animated: true) { _ in pixels }
            await stageWait { stage.current?.key == zoomedKey }
            checkEqual(stage.current?.rows.first?.cellWidth ?? 0, 250,
                       "the animated commit delivers the zoomed layout like any other")
        }.value
    })
}

@MainActor
private func testAlbumGridPublication(directory: URL, pixels: CGImage, payloads: [Data]) async {
    let session = WaifuSession(settingsURL: directory.appendingPathComponent("settings.json"))
    session.frameOrder = .only([.art, .minimalVisualDistance])
    let frames = (0..<120).map { index in
        FrameFixture.make(id: String(format: "f%06d", index), time: Double(index), artScore: Float(index),
                          featurePrint: payloads[index % payloads.count])
    }
    let stage = GridThumbnailStage(), generation = session.thumbnailGeneration
    defer { stage.stop() }
    stage.onCommit = { session.commitGridNavigation($0.navigation, generation: $0.key.generation) }
    var changing = false
    var revisions: [UInt64] = []
    let subscription = session.gridLayoutUpdates.sink { _ in
        MainActor.assumeIsolated {
            check(!changing, "grid delivery happens after the mutation finishes")
            revisions.append(session.gridRevision)
            stage.submit(stageInput(session.gridRevision, session.displayOrderFrames, generation: generation),
                         viewportSize: CGSize(width: 263, height: 130)) { _ in pixels }
        }
    }
    changing = true
    session.updateAnalyzedFrames(Array(frames.prefix(1)), isProgressive: true)
    changing = false
    await stageWait { stage.current?.navigation.frames.count == 1 }
    changing = true
    session.updateAnalyzedFrames(Array(frames.prefix(80)), isProgressive: true)
    changing = false
    check(!session.isSortingVisualDistance, "a partial album does not launch another visual route")
    checkEqual(session.displayOrderFrames.map(\.id), frames.prefix(80).map(\.id),
               "growing results append in source order instead of shuffling the whole viewport")
    await stageWait { stage.current?.navigation.frames.count == 80 }
    changing = true
    session.updateAnalyzedFrames(frames)
    changing = false
    await stageWait { !session.isComputingVisualDistance && stage.current?.key.revision == session.gridRevision }
    checkEqual(stage.current?.navigation.frames.count, frames.count, "the grid cannot remain stuck on the initial photo or a prefix")
    checkEqual(stage.current?.navigation.frames.map(\.id), session.displayOrderFrames.map(\.id),
               "the last asynchronous route reaches the displayed layout without another UI event")
    check(revisions.contains(session.gridRevision), "the final layout revision is delivered explicitly")
    withExtendedLifetime(subscription) {}
}

@MainActor
private func testStagePreviewNavigation(_ pixels: CGImage, directory: URL) async {
    let session = WaifuSession(settingsURL: directory.appendingPathComponent("settings.json"))
    session.frameOrder = .only([.time])
    let frames = (0..<4).map(stageFrame)
    session.updateAnalyzedFrames(frames)
    let generation = session.thumbnailGeneration
    let stage = GridThumbnailStage(preparationLimit: .seconds(2)), reads = StageReads()
    defer { stage.stop(); reads.finishAll() }
    stage.onCommit = { session.commitGridNavigation($0.navigation, generation: $0.key.generation) }
    let size = CGSize(width: 263, height: 130)
    stage.submit(stageInput(1, frames, generation: generation), viewportSize: size) { _ in pixels }
    await stageWait { stage.current != nil }

    session.frameOrder.setEnabled(true, for: .minimalVisualDistance)
    session.frameOrder.toggleDirection(of: .minimalVisualDistance)
    await stageWait { !session.isComputingVisualDistance }
    checkEqual(session.displayOrderFrames.map(\.id), frames.reversed().map(\.id), "model already has the reversed visual route")
    stage.submit(stageInput(2, session.displayOrderFrames, generation: generation), viewportSize: size) { await reads.read($0) }
    await stageWait { reads.waiting.count == 2 }
    checkEqual(stage.current?.key.revision, 1, "the old route is still visible while replacement pixels load")
    if let snapshot = stage.current {
        checkEqual(FrameGridScrollTarget.resolve(frame: frames[0].id, snapshot: snapshot)?.rowID,
                   "flat:0", "arrow reveals use the currently mounted layout while a replacement is pending")
    }
    session.showPreview(frames[0])
    session.focusNeighbour(-1)
    checkEqual(session.previewFrame?.id, frames[0].id, "the first visible thumbnail has no backward neighbour")
    session.focusNeighbour(1)
    checkEqual(session.previewFrame?.id, frames[1].id, "forward follows the thumbnails the user clicked")

    for id in Array(reads.waiting) { reads.finish(id, pixels) }
    await stageWait { stage.current?.key.revision == 2 }
    if let snapshot = stage.current {
        checkEqual(FrameGridScrollTarget.resolve(frame: frames[0].id, snapshot: snapshot)?.rowID,
                   "flat:1", "the reveal target moves to the new row when its layout commits")
    }
    session.focusNeighbour(1)
    checkEqual(session.previewFrame?.id, frames[2].id, "a hidden grid commit cannot reverse an open preview")
    session.focusNeighbour(-1)
    checkEqual(session.previewFrame?.id, frames[1].id, "backward retraces the same sequence")
    session.focusDisplayedBoundary(forward: true)
    checkEqual(session.previewFrame?.id, frames[3].id, "end uses the preview's sequence")
    session.previewFrame = nil
    session.showPreview(frames[3])
    session.focusNeighbour(1)
    checkEqual(session.previewFrame?.id, frames[2].id, "reopening uses the newly displayed reverse sequence")
    session.previewFrame = nil
    session.commitGridNavigation(FrameNavigationOrder(frames: frames, rows: [], groupRows: false), generation: UUID())
    session.focusDisplayedBoundary(forward: false)
    checkEqual(session.focusedFrameID, frames[3].id, "an obsolete source cannot replace committed navigation")
    session.frameOrder.setEnabled(false, for: .minimalVisualDistance)
}

@MainActor
private func testAtomicStage(_ pixels: CGImage) async {
    let generation = UUID(), reads = StageReads()
    let stage = GridThumbnailStage(preparationLimit: .seconds(2))
    defer { stage.stop(); reads.finishAll() }
    let frames = (0..<4).map(stageFrame)
    let size = CGSize(width: 263, height: 130)
    stage.submit(stageInput(1, Array(frames.prefix(2)), generation: generation), viewportSize: size) { _ in pixels }
    await stageWait { stage.current?.key.revision == 1 }
    stage.submit(stageInput(2, Array(frames.suffix(2).reversed()), generation: generation), viewportSize: size) { await reads.read($0) }
    await stageWait { reads.waiting.count == 2 }
    checkEqual(stage.current?.rows.first?.items.map(\.id), [frames[0].id, frames[1].id], "old viewport remains displayed")
    reads.finish(frames[2].id, pixels)
    await stageWait { GridThumbnailLoader.shared.cachedImage(identity: generation.uuidString + ":" + frames[2].thumbnailKey) != nil }
    checkEqual(stage.current?.key.revision, 1, "one ready image cannot partially switch the viewport")
    reads.finish(frames[3].id, pixels)
    await stageWait { stage.current?.key.revision == 2 }
    checkEqual(stage.current?.rows.first?.items.map(\.id), [frames[3].id, frames[2].id], "all target pixels switch with their new order")
    checkEqual(stage.previous?.key.revision, 1, "transition baseline is the actual displayed order")
}

@MainActor
private func testStageDeadline(_ pixels: CGImage) async {
    let generation = UUID(), reads = StageReads()
    let stage = GridThumbnailStage()
    defer { stage.stop(); reads.finishAll() }
    let frames = (0..<8).map(stageFrame), size = CGSize(width: 263, height: 130)
    stage.submit(stageInput(1, Array(frames.prefix(2)), generation: generation), viewportSize: size) { _ in pixels }
    await stageWait { stage.current != nil }
    stage.submit(stageInput(2, Array(frames[2..<4]), generation: generation), viewportSize: size) { await reads.read($0) }
    await stageWait { reads.waiting.count == 2 }
    stage.submit(stageInput(3, Array(frames[4..<6]), generation: generation), viewportSize: size) { await reads.read($0) }
    await stageWait { reads.waiting.count == 4 }
    reads.finish(frames[2].id, pixels); reads.finish(frames[3].id, pixels)
    reads.finish(frames[4].id, pixels)
    await stageWait { GridThumbnailLoader.shared.cachedImage(identity: generation.uuidString + ":" + frames[4].thumbnailKey) != nil }
    checkEqual(stage.current?.key.revision, 1, "superseded and partial batches cannot replace the displayed baseline")
    await stageWait { stage.current?.key.revision == 3 }
    checkEqual(stage.previous?.key.revision, 1, "timeout transition skips superseded intermediate orders")
    checkNil(GridThumbnailLoader.shared.cachedImage(identity: generation.uuidString + ":" + frames[5].thumbnailKey), "unfinished pixels remain absent at the deadline")
    checkNotNil(GridThumbnailLoader.shared.cachedImage(identity: generation.uuidString + ":" + frames[4].thumbnailKey), "ready pixels survive the atomic handoff")
    reads.finish(frames[5].id, pixels)
    await Task.yield()
    checkEqual(stage.current?.key.revision, 3, "late canceled work cannot change the committed order")
}

@MainActor
private func testStageScroll() async {
    var time = ContinuousClock.now
    let generation = UUID(), reads = StageReads()
    let stage = GridThumbnailStage(now: { time })
    defer { stage.stop(); reads.finishAll() }
    let frames = (0..<12).map(stageFrame), size = CGSize(width: 263, height: 130)
    stage.submit(stageInput(1, frames, generation: generation), viewportSize: size) { await reads.read($0) }
    await stageWait { reads.waiting.count == 2 }
    time = time.advanced(by: .milliseconds(180))
    stage.updateViewport(CGRect(x: 0, y: 264, width: 263, height: 130))
    await stageWait { reads.waiting.contains(frames[4].id) && reads.waiting.contains(frames[5].id) }
    check(stage.current == nil, "scrolling during preparation does not prematurely show the batch")
    // Trigger a refresh at the original deadline. A scroll that reset the
    // deadline to 480 ms would fail this check, regardless of runner load.
    time = time.advanced(by: .milliseconds(120))
    stage.updateViewport(CGRect(x: 0, y: 265, width: 263, height: 130))
    check(stage.current != nil, "viewport updates retain the original 300 ms deadline")
}
