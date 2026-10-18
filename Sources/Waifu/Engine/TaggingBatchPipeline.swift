import Foundation

/// Bounded tagging preparation and processing. Local inference prefetches one
/// batch; remote inference uses an ordered window of asynchronous requests.
enum TaggingBatchPipeline {
    /// Keeps a bounded window of asynchronous operations alive, committing their
    /// results in input order by default, or on completion for independent updates.
    /// Preparation stays serial because `next` owns the
    /// mutable source queue. Empty decisions wait for all commits, so newly added
    /// candidates are not mistaken for a completed run.
    static func runPipelined<Element: Sendable, Output: Sendable>(
        maximumInFlight: Int,
        commitInOrder: Bool = true,
        next: @escaping @Sendable () async throws -> [Element]?,
        whenEmpty: @escaping @Sendable () async -> Bool,
        process: @escaping @Sendable ([Element]) async throws -> Output,
        commit: @escaping @Sendable (Output) async throws -> Void
    ) async throws {
        precondition(maximumInFlight > 0)
        try await withThrowingTaskGroup(of: (Int, Result<Output, Error>).self) { group in
            var submitted = 0
            var committed = 0
            var ready: [Int: Result<Output, Error>] = [:]
            var draining = false
            var ended = false
            var preparationError: Error?
            while true {
                try Task.checkCancellation()
                while !draining, submitted - committed < maximumInFlight {
                    do {
                        guard let batch = try await next() else {
                            draining = true
                            ended = true
                            break
                        }
                        try Task.checkCancellation()
                        guard !batch.isEmpty else { draining = true; break }
                        let index = submitted
                        submitted += 1
                        group.addTask {
                            do {
                                try Task.checkCancellation()
                                return (index, .success(try await process(batch)))
                            } catch { return (index, .failure(error)) }
                        }
                    } catch {
                        // Commit already submitted batches before surfacing a
                        // later preparation failure, as the local pipeline does.
                        preparationError = error
                        draining = true
                    }
                }
                if committed < submitted {
                    guard let (index, result) = try await group.next() else { return }
                    if commitInOrder {
                        ready[index] = result
                        while let result = ready.removeValue(forKey: committed) {
                            try Task.checkCancellation()
                            try await commit(result.get())
                            committed += 1
                        }
                    } else {
                        // Frame updates are independent. Retire completed batches immediately
                        // so faster workers can refill their queues behind a slower worker.
                        try Task.checkCancellation()
                        try await commit(result.get())
                        committed += 1
                    }
                } else {
                    if let preparationError { throw preparationError }
                    if ended { return }
                    guard await whenEmpty() else { return }
                    draining = false
                }
            }
        }
    }

    static func run<Element: Sendable>(
        next: @escaping @Sendable () async throws -> [Element]?,
        whenEmpty: @escaping @Sendable () async -> Bool,
        consume: @escaping @Sendable ([Element]) async throws -> Void
    ) async throws {
        var batch = try await next()
        while let current = batch {
            try Task.checkCancellation()
            if current.isEmpty {
                guard await whenEmpty() else { return }
                batch = try await next()
                continue
            }
            // Structured lifetime: cancellation or a consumer error cancels and
            // joins the pending fetch before the caller releases its model.
            async let following = next()
            try await consume(current)
            // Preserve completed results even if preparing the next batch fails.
            batch = try await following
        }
    }
}
