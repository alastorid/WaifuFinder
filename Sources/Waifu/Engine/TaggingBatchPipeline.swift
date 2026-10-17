import Foundation

/// Fetches at most one following batch while consuming the current batch.
/// Empty-queue decisions run only after the previous batch has been committed.
enum TaggingBatchPipeline {
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
