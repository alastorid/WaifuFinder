import Foundation

struct RemoteArenaRequest: Codable, Sendable {
    let revision: String
    let images: [Data]
    func validate() throws {
        guard revision == ArenaJudgeConfiguration.revision, images.count == 2,
              images.allSatisfy({ !$0.isEmpty && $0.count <= 16 * 1024 * 1024 }) else {
            throw RemoteError.invalidRequest("Unsupported PhotoArena request")
        }
    }
}
extension RemoteServerSession {
    func handleArena(_ envelope: RemoteEnvelope) async {
        do {
            let request = try JSONDecoder().decode(RemoteArenaRequest.self, from: envelope.data)
            try request.validate()
            await ModelLibrary.shared.inUseArena(true)
            let result: ArenaJudgeResult
            do { result = try await catalog.arena(request) }
            catch { await ModelLibrary.shared.inUseArena(false); throw error }
            await ModelLibrary.shared.inUseArena(false)
            try Task.checkCancellation()
            try await connection.send(.arenaResult, payload: [UInt8](JSONEncoder().encode(result)), streamID: envelope.streamID)
        } catch { try? await connection.sendError(envelope, Task.isCancelled ? .cancelled : Self.describe(error)) }
        await resources.finish(streamID: envelope.streamID)
    }
}
extension RemoteClient {
    var offersArena: Bool { status.isConnected && capabilities?.arenaRevision == ArenaJudgeConfiguration.revision }
    func arena(_ images: [Data]) async throws -> ArenaJudgeResult {
        guard offersArena, let connection else { throw RemoteError.modelUnavailable("No matching PhotoArena server") }
        let request = RemoteArenaRequest(revision: ArenaJudgeConfiguration.revision, images: images)
        try request.validate()
        let stream = try await connection.beginRequest()
        do {
            let result = try await withTaskCancellationHandler {
                try Task.checkCancellation()
                try await connection.send(.arenaRequest, payload: [UInt8](JSONEncoder().encode(request)), streamID: stream)
                guard let response = await connection.next(streamID: stream) else { throw RemoteError.framing("PhotoArena stream ended") }
                if response.isError { throw RemoteConnection.remoteError(from: response) }
                guard response.opcode == .arenaResult else { throw RemoteError.framing("Unexpected PhotoArena response") }
                let result = try JSONDecoder().decode(ArenaJudgeResult.self, from: response.data)
                try result.validate()
                return result
            } onCancel: { Task { await connection.cancelRequest(stream) } }
            await connection.endRequest()
            return result
        } catch { await connection.endRequest(); throw error }
    }
}

/// Captured once per judging action. A remote failure is surfaced, never retried
/// on another machine or silently redirected to local inference.
struct ArenaInferenceBackend: Sendable {
    let label: String
    let isRemote: Bool
    let infer: @Sendable ([Data]) async throws -> ArenaJudgeResult

    @MainActor
    static func preferred(clients: [RemoteClient]) -> Self {
        if let client = clients.first(where: { $0.offersArena }) {
            return Self(label: "Remote · " + client.status.label, isRemote: true,
                        infer: { try await client.arena($0) })
        }
        return Self(label: "This Mac", isRemote: false, infer: { try await ArenaJudge.local($0) })
    }
}
