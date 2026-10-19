import Combine
import Foundation
import SwiftUI

/// Keeps decoded thumbnail pixels warm for three rows on either side of the
/// rows currently in the viewport. Source reads and image decoding stay off the
/// main thread; visible cells own their loading and fade-in state.
@MainActor
final class GridThumbnailPrefetcher: ObservableObject {
    typealias PlanKey = GridStageKey

    private struct Job {
        let id: UUID
        let task: Task<Void, Never>
    }

    private(set) var planKey: PlanKey?
    private var rows: [GridStageRow] = []
    private var rowIndexByToken: [String: Int] = [:]
    private var visibleRowTokens: Set<String> = []
    private var jobs: [String: Job] = [:]
    private var refreshTask: Task<Void, Never>?
    private weak var session: WaifuSession?
    private var attempted: Set<String> = []
    private var horizontalOffsets: [String: CGFloat] = [:]
    private let priority: GridThumbnailWorkPriority
    private var prioritySubscription: AnyCancellable?
    private let load: @MainActor (WaifuSession, AnalyzedFrame, String) async -> Void

    init(priority: GridThumbnailWorkPriority = .shared,
         load: @escaping @MainActor (WaifuSession, AnalyzedFrame, String) async -> Void = { session, frame, identity in
             if let data = await session.thumbnailData(for: frame, retainRegeneratedThumbnail: false), !Task.isCancelled {
                 _ = await GridThumbnailLoader.shared.image(data: data, identity: identity)
             }
         }) {
        self.priority = priority
        self.load = load
        prioritySubscription = priority.$visibleWorkCount.sink { [weak self] count in
            MainActor.assumeIsolated {
                guard let self else { return }
                if count > 0 { self.cancelJobs() }
                else { self.scheduleRefresh() }
            }
        }
    }

    private func cancelJobs() {
        for job in jobs.values { job.task.cancel() }
        jobs.removeAll()
    }

    func setHorizontalOffset(row: String, offset: CGFloat) {
        guard horizontalOffsets[row] != offset else { return }
        horizontalOffsets[row] = offset
        cancelJobs()
        attempted.removeAll()
        scheduleRefresh()
    }

    func install(snapshot: GridStageSnapshot, session: WaifuSession) {
        guard planKey != snapshot.key else { return }
        cancelJobs()
        attempted.removeAll()
        if planKey?.generation != snapshot.key.generation { horizontalOffsets.removeAll() }
        planKey = snapshot.key
        // These immutable arrays/indices were built off-main. Installation only
        // takes shared storage references, even for a very large photo library.
        rows = snapshot.rows
        rowIndexByToken = snapshot.rowIndexByToken
        self.session = session
        scheduleRefresh()
    }

    func setVisibleRows(_ tokens: Set<String>) {
        guard tokens != visibleRowTokens else { return }
        cancelJobs()
        attempted.removeAll()
        visibleRowTokens = tokens
        scheduleRefresh()
    }

    func stop() {
        refreshTask?.cancel()
        refreshTask = nil
        for job in jobs.values { job.task.cancel() }
        jobs.removeAll(keepingCapacity: false)
        visibleRowTokens.removeAll(keepingCapacity: false)
        rows.removeAll(keepingCapacity: false)
        rowIndexByToken.removeAll(keepingCapacity: false)
        attempted.removeAll()
        horizontalOffsets.removeAll()
        planKey = nil
        session = nil
    }

    private func scheduleRefresh() {
        guard refreshTask == nil else { return }
        refreshTask = Task { [weak self] in
            await Task.yield()
            guard let self, !Task.isCancelled else { return }
            self.refreshTask = nil
            self.refresh()
        }
    }

    private func refresh() {
        guard priority.visibleWorkCount == 0, let key = planKey, let session, !rows.isEmpty else { return }
        // Warm the first screen before geometry preferences have arrived. Once
        // scrolling starts, the viewport replaces this initial window.
        let measuredRows = Set(visibleRowTokens.compactMap { rowIndexByToken[$0] })
        let visibleRows = measuredRows.isEmpty ? Set(0..<min(rows.count, 3)) : measuredRows

        var neighborRows = Set<Int>()
        for visible in visibleRows {
            for index in max(0, visible - 3)...min(rows.count - 1, visible + 3)
                where !visibleRows.contains(index) {
                neighborRows.insert(index)
            }
        }
        let orderedRows = visibleRows.sorted() + neighborRows.sorted { lhs, rhs in
            let leftDistance = visibleRows.map { abs($0 - lhs) }.min() ?? .max
            let rightDistance = visibleRows.map { abs($0 - rhs) }.min() ?? .max
            return leftDistance == rightDistance ? lhs < rhs : leftDistance < rightDistance
        }

        // Warm nearby horizontal cells around each strip's current offset too.
        // Preserve distance order and bound concurrent speculative reads to two.
        var wanted: [(String, AnalyzedFrame)] = []
        var wantedIDs: Set<String> = []
        let maximum = max(1, key.columns * (visibleRows.count + 6))
        for index in orderedRows {
            let row = rows[index]
            let first = row.strip ? min(max(0, row.items.count - 1), max(0, Int((horizontalOffsets[row.id] ?? 0) / (row.cellWidth + 1)))) : 0
            let candidateFrames = row.strip
                ? Array(row.items[max(0, first - key.columns)..<min(row.items.count, first + key.columns * 2)])
                : row.items
            for item in candidateFrames {
                let frame = item.frame
                let identity = key.generation.uuidString + ":" + frame.thumbnailKey
                if wantedIDs.insert(identity).inserted { wanted.append((identity, frame)) }
                if wanted.count >= maximum { break }
            }
            if wanted.count >= maximum { break }
        }

        let obsoleteJobs = jobs.filter { !wantedIDs.contains($0.key) }
        for (identity, job) in obsoleteJobs {
            job.task.cancel()
            jobs.removeValue(forKey: identity)
        }
        for (identity, frame) in wanted where jobs.count < 2 && jobs[identity] == nil && !attempted.contains(identity)
            && GridThumbnailLoader.shared.cachedImage(identity: identity) == nil {
            let jobID = UUID()
            let task = Task(priority: .utility) { [weak self, weak session] in
                guard let self, let session, !Task.isCancelled, self.priority.visibleWorkCount == 0 else { return }
                await self.load(session, frame, identity)
                self.finish(identity: identity, jobID: jobID)
            }
            jobs[identity] = Job(id: jobID, task: task)
        }
    }

    private func finish(identity: String, jobID: UUID) {
        guard jobs[identity]?.id == jobID else { return }
        jobs.removeValue(forKey: identity)
        attempted.insert(identity)
        scheduleRefresh()
    }
}
