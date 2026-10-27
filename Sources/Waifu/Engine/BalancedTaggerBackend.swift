import CoreGraphics

/// Reserves a lane before suspending, so simultaneous submissions see its queue depth.
actor BalancedTaggerBackend: TaggerBackend {
    struct Lane: Sendable {
        let backend: any TaggerBackend
        let capacity: Int
    }

    private let lanes: [Lane]
    private var depths: [Int]
    private var cursor = 0
    nonisolated let maximumInFlight: Int

    init(lanes: [Lane]) {
        precondition(!lanes.isEmpty && lanes.allSatisfy { $0.capacity > 0 })
        self.lanes = lanes
        depths = Array(repeating: 0, count: lanes.count)
        maximumInFlight = lanes.reduce(0) { $0 + $1.capacity }
    }

    func tag(batch: [CGImage]) async throws -> [TagResult] {
        try await perform { try await $0.tag(batch: batch) }
    }

    func tag(sources: [TaggingImageSource]) async throws -> [TagResult] {
        try await perform { try await $0.tag(sources: sources) }
    }

    private func perform(_ work: @Sendable (any TaggerBackend) async throws -> [TagResult]) async throws -> [TagResult] {
        try Task.checkCancellation()
        // The pipeline limits submissions to the total lane capacity.
        let candidates = (0..<lanes.count).map { (cursor + $0) % lanes.count }
            .filter { depths[$0] < lanes[$0].capacity }
        guard let index = candidates.min(by: { depths[$0] < depths[$1] }) else { throw RemoteError.busy }
        depths[index] += 1
        cursor = (index + 1) % lanes.count
        defer { depths[index] -= 1 }
        return try await work(lanes[index].backend)
    }
}
