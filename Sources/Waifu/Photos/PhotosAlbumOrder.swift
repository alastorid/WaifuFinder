import Foundation
import Photos

/// Reorders only the requested members in their existing album slots. Hidden
/// photos, filtered results, and videos keep their positions and membership.
enum PhotosAlbumOrder {
    struct Move: Equatable, Sendable {
        let from: Int
        let to: Int
    }

    static func moves(current: [String], ordered: [String]) throws -> [Move] {
        let requested = Set(ordered)
        guard requested.count == ordered.count,
              requested.isSubset(of: Set(current)) else {
            throw PhotosMedia.failure("The album changed. Reopen it before syncing its order.")
        }
        var desired = current
        var next = 0
        for index in current.indices where requested.contains(current[index]) {
            desired[index] = ordered[next]
            next += 1
        }
        var working = current
        var moves: [Move] = []
        for index in desired.indices where working[index] != desired[index] {
            guard let source = working[(index + 1)...].firstIndex(of: desired[index]) else {
                throw PhotosMedia.failure("Could not prepare the album order.")
            }
            moves.append(Move(from: source, to: index))
            working.insert(working.remove(at: source), at: index)
        }
        return moves
    }
}

/// A fresh PhotoKit snapshot is required on every attempt: a failed verification
/// may follow a successful commit, so replaying the original moves is unsafe.
enum PhotosOrderSync {
    enum VerificationError: LocalizedError {
        case orderNotVisible
        var errorDescription: String? { "Photos has not confirmed the requested album order. Try again shortly." }
    }

    static func run(
        operation: @escaping @Sendable () async throws -> Void,
        wait: @escaping @Sendable (TimeInterval) async throws -> Void = { delay in
            try await Task.sleep(for: .seconds(delay))
        }
    ) async throws {
        let delays: [TimeInterval] = [0.25, 0.5, 1, 1.5, 2, 2, 2]
        for attempt in 0...delays.count {
            try Task.checkCancellation()
            do {
                try await operation()
                return
            } catch {
                try Task.checkCancellation()
                guard attempt < delays.count, isRetryable(error) else { throw error }
                try await wait(delays[attempt])
            }
        }
    }

    static func isRetryable(_ error: Error) -> Bool {
        if error is VerificationError { return true }
        let error = error as NSError
        guard error.domain == PHPhotosErrorDomain else { return false }
        return [PHPhotosError.Code.internalError, .operationInterrupted, .networkError]
            .contains { $0.rawValue == error.code }
    }
}
