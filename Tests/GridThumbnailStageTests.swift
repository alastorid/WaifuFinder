import AppKit
import Foundation

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
    let generation = UUID(), reads = StageReads(), stage = GridThumbnailStage()
    defer { stage.stop(); reads.finishAll() }
    let frames = (0..<12).map(stageFrame), size = CGSize(width: 263, height: 130)
    let started = ContinuousClock.now
    stage.submit(stageInput(1, frames, generation: generation), viewportSize: size) { await reads.read($0) }
    await stageWait { reads.waiting.count == 2 }
    try? await Task.sleep(for: .milliseconds(180))
    stage.updateViewport(CGRect(x: 0, y: 264, width: 263, height: 130))
    await stageWait { reads.waiting.contains(frames[4].id) && reads.waiting.contains(frames[5].id) }
    check(stage.current == nil, "scrolling during preparation does not prematurely show the batch")
    await stageWait { stage.current != nil }
    check(started.duration(to: .now) < .milliseconds(450), "viewport updates retain the original 300 ms deadline")
}
