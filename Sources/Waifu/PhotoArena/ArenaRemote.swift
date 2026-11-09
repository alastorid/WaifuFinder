import Foundation

struct RemoteArenaRequest: Codable, Sendable {
    let revision: String
    let images: [Data]
    var group: ArenaGroupPayload? = nil
    func validate() throws {
        if let group {
            guard revision == ArenaGroupPayload.revision, images.isEmpty else { throw RemoteError.invalidRequest("Unsupported PhotoArena group request") }
            try group.validate(); return
        }
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
            do {
                if let group = request.group { result = try await catalog.arenaGroup(group) }
                else { result = try await catalog.arena(request) }
            }
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
    var offersArenaReview: Bool { status.isConnected && capabilities?.arenaReviewRevision == ArenaGroupPayload.revision }
    func arenaGroup(_ payload: ArenaGroupPayload) async throws -> ArenaJudgeResult {
        guard offersArenaReview, let connection else { throw RemoteError.modelUnavailable("Update the inference server for multi-source PhotoArena review") }
        let request = RemoteArenaRequest(revision: ArenaGroupPayload.revision, images: [], group: payload)
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
                guard let review = result.review else { throw RemoteError.framing("Missing group review") }
                try review.validate(count: payload.sourceIDs.count)
                return result
            } onCancel: { Task { await connection.cancelRequest(stream) } }
            await connection.endRequest(); return result
        } catch { await connection.endRequest(); throw error }
    }
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

struct ArenaReviewBackend: Sendable {
    let label: String
    let isRemote: Bool
    let review: @Sendable (ArenaAnalysis, [ArenaReviewJob], [ArenaReviewRecord], @escaping @Sendable (ArenaReviewRecord) -> Void) async throws -> [ArenaReviewRecord]
    @MainActor static func preferred(clients: [RemoteClient]) -> Self {
        if let client = clients.first(where: \.offersArenaReview) {
            return Self(label: "Remote · " + client.status.label, isRemote: true, review: { analysis, jobs, previous, progress in
                var records: [ArenaReviewRecord] = []
                for job in jobs {
                    try Task.checkCancellation()
                    let preparation = Task.detached { try ArenaGroupEvidence.prepare(analysis, job: job) }
                    let payload = try await withTaskCancellationHandler {
                        try await preparation.value
                    } onCancel: { preparation.cancel() }
                    try Task.checkCancellation()
                    let key = try ArenaGroupEvidence.key(payload, job: job)
                    let record: ArenaReviewRecord
                    if let saved = previous.last(where: { $0.cacheKey == key && $0.sourceIDs == payload.sourceIDs }),
                       let assessment = saved.result.review, (try? assessment.validate(count: payload.sourceIDs.count)) != nil { record = saved }
                    else {
                        let result = try await client.arenaGroup(payload)
                        record = ArenaReviewRecord(job: job, sourceIDs: payload.sourceIDs, cacheKey: key, result: result)
                    }
                    records.append(record); progress(record)
                }
                return records
            })
        }
        return Self(label: "This Mac", isRemote: false, review: { analysis, jobs, previous, progress in
            try await ArenaJudge.review(analysis: analysis, jobs: jobs, previous: previous, onResult: progress)
        })
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
