import Foundation

/// One app-owned cloud request at a time. Viewing preempts background work;
/// background work preempts speculative preload. Interrupted work is requeued,
/// while cancellation from its owner removes it permanently.
actor PhotosCloudQueue {
    static let shared = PhotosCloudQueue()
    enum Priority: Int, Sendable { case viewing, thumbnail, background, preload }
    private struct Job: Sendable {
        let id: UUID
        let priority: Priority
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

    func perform<Value: Sendable>(priority: Priority,
                                 operation: @escaping @Sendable () async throws -> Value) async throws -> Value {
        let id = UUID()
        return try await withTaskCancellationHandler {
            try Task.checkCancellation()
            return try await withCheckedThrowingContinuation { continuation in
                let job = Job(id: id, priority: priority, start: {
                    Task {
                        let result: Result<Value, Error>
                        do { result = .success(try await operation()) }
                        catch { result = .failure(error) }
                        await self.complete(id) { continuation.resume(with: result) }
                    }
                }, cancel: { continuation.resume(throwing: CancellationError()) })
                pending.append(job)
                if let running = active, priority.rawValue < running.job.priority.rawValue {
                    active?.interrupted = true
                    running.task.cancel()
                }
                startNext()
            }
        } onCancel: { Task { await self.cancel(id) } }
    }

    private func startNext() {
        guard active == nil, let priority = pending.map(\.priority.rawValue).min(),
              let index = pending.firstIndex(where: { $0.priority.rawValue == priority }) else { return }
        let job = pending.remove(at: index)
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

/// The next item, the previous item, then the rest of the next seven items.
/// The caller supplies the same ordering used by preview navigation.
enum PhotosPreviewNeighbours {
    static func indices(current: Int, count: Int) -> [Int] {
        guard (0..<count).contains(current) else { return [] }
        return ([current + 1, current - 1] + (2...7).map { current + $0 })
            .filter { (0..<count).contains($0) }
    }
}
