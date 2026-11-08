import Foundation

/// One app-owned cloud request at a time. Viewing preempts background work;
/// background work preempts speculative preload. Interrupted work is requeued,
/// while cancellation from its owner removes it permanently. An active visible
/// thumbnail keeps its transfer when a higher-priority request arrives.
actor PhotosCloudQueue {
    static let shared = PhotosCloudQueue()
    enum Priority: Int, Sendable {
        case viewing, selectedThumbnail, thumbnail, background, preload
        var isVisibleThumbnail: Bool { self == .selectedThumbnail || self == .thumbnail }
    }

    /// A speculative request can become visible without restarting its transfer.
    final class PriorityTicket: @unchecked Sendable {
        private let lock = NSLock()
        private var value: Priority
        init(_ priority: Priority) { value = priority }
        var priority: Priority { lock.withLock { value } }

        func update(_ priority: Priority) {
            let changed = lock.withLock {
                guard value != priority else { return false }
                value = priority
                return true
            }
            if changed { Task { await PhotosCloudQueue.shared.prioritiesChanged() } }
        }
    }
    private struct Job: Sendable {
        let id: UUID
        let basePriority: Priority
        let ticket: PriorityTicket?
        var priority: Priority { ticket?.priority ?? basePriority }
        let start: @Sendable () -> Task<Void, Never>
        let cancel: @Sendable () -> Void
    }
    private struct Active {
        let job: Job
        let task: Task<Void, Never>
        var interrupted = false
        var cancelled = false
    }
    private var pending: [Job] = []
    private var active: Active?

    func perform<Value: Sendable>(priority: Priority, ticket: PriorityTicket? = nil,
                                 operation: @escaping @Sendable () async throws -> Value) async throws -> Value {
        let id = UUID()
        return try await withTaskCancellationHandler {
            try Task.checkCancellation()
            return try await withCheckedThrowingContinuation { continuation in
                let job = Job(id: id, basePriority: priority, ticket: ticket, start: {
                    Task {
                        let result: Result<Value, Error>
                        do { result = .success(try await operation()) }
                        catch { result = .failure(error) }
                        await self.complete(id) { continuation.resume(with: result) }
                    }
                }, cancel: { continuation.resume(throwing: CancellationError()) })
                pending.append(job)
                prioritiesChanged()
            }
        } onCancel: { Task { await self.cancel(id) } }
    }

    private func prioritiesChanged() {
        if let running = active, !running.job.priority.isVisibleThumbnail,
           pending.contains(where: { $0.priority.rawValue < running.job.priority.rawValue }) {
            active?.interrupted = true
            running.task.cancel()
        }
        startNext()
    }

    private func startNext() {
        guard active == nil, let first = pending.indices.min(by: {
            pending[$0].priority.rawValue < pending[$1].priority.rawValue
        }) else { return }
        let job = pending.remove(at: first)
        active = Active(job: job, task: job.start())
    }

    private func complete(_ id: UUID, deliver: @Sendable () -> Void) {
        guard let running = active, running.job.id == id else { return }
        active = nil
        if running.cancelled { running.job.cancel() }
        else if running.interrupted { pending.insert(running.job, at: 0) }
        else { deliver() }
        startNext()
    }

    private func cancel(_ id: UUID) {
        if let index = pending.firstIndex(where: { $0.id == id }) {
            pending.remove(at: index).cancel()
        } else if active?.job.id == id {
            active?.cancelled = true
            active?.task.cancel()
        }
    }
}

/// Three items in each direction, nearest first, alternating right and left.
/// The caller supplies the same ordering used by preview navigation.
enum PhotosPreviewNeighbours {
    static func indices(current: Int, count: Int) -> [Int] {
        guard (0..<count).contains(current) else { return [] }
        return (1...3).flatMap { [current + $0, current - $0] }
            .filter { (0..<count).contains($0) }
    }
}
