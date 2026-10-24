import Foundation

/// A busy inference lane is a temporary wait, not a reason to disable entry or
/// mark a photograph as analyzed. Cancellation and permanent failures propagate.
@MainActor enum SafireJobWait {
    static func run<Value:Sendable>(onWaiting:@MainActor ()->Void,
        wait:@MainActor () async throws -> Void = { try await Task.sleep(for:.milliseconds(500)) },
        operation:@MainActor () async throws -> Value) async throws -> Value {
        var announced=false
        while true {
            try Task.checkCancellation()
            do { return try await operation() }
            catch RemoteError.busy {
                if !announced { onWaiting();announced=true }
                try await wait()
            }
        }
    }
}
