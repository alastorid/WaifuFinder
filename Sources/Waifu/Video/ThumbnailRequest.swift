import Foundation

/// A source read can serve a cell and a staged layout together. Stop the read
/// when its last consumer leaves, while preserving work still needed on screen.
@MainActor
final class ThumbnailRequest {
    let id: UUID
    let task: Task<Data?, Never>
    var retainAfterCompletion: Bool
    private var consumers: Set<UUID> = []

    init(id: UUID, task: Task<Data?, Never>, retainAfterCompletion: Bool) {
        self.id = id
        self.task = task
        self.retainAfterCompletion = retainAfterCompletion
    }

    func value() async -> Data? {
        guard !Task.isCancelled else {
            if consumers.isEmpty { task.cancel() }
            return nil
        }
        let consumer = UUID()
        consumers.insert(consumer)
        defer { release(consumer) }
        return await withTaskCancellationHandler {
            let data = await task.value
            return Task.isCancelled ? nil : data
        } onCancel: {
            Task { @MainActor in self.release(consumer) }
        }
    }

    private func release(_ consumer: UUID) {
        guard consumers.remove(consumer) != nil else { return }
        if consumers.isEmpty { task.cancel() }
    }
}
