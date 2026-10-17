import Foundation
import Vision

/// One worker owns this state. Each newly scored frame compares with earlier
/// neighbours within 120 seconds, visiting the same edges as the batch pass.
/// Union-find membership is independent of edge order; presentation waits for EOF.
final class IncrementalFrameGrouper {
    enum InputError: Error { case outOfOrder }
    private struct Entry {
        let payload: Data?
        let observation: FeaturePrintObservation?
    }
    private let settings: GroupingSettings
    private let distance: (@Sendable (Data, Data) -> Double?)?
    private var members: [(id: String, time: Double)] = []
    private var parent: [Int] = []
    private var sizes: [Int] = []
    private var rank: [Int] = []
    private var active: [Int: Entry] = [:]
    private var windowStart = 0
    private var lastInput: (time: Double, id: String)?
    private(set) var processed = 0
    private(set) var linkedGroups = 0
    var retainedPrintCount: Int { active.count }

    init(settings: GroupingSettings, distance: (@Sendable (Data, Data) -> Double?)? = nil) {
        self.settings = settings.validated()
        self.distance = distance
    }

    func append(_ frame: AnalyzedFrame) throws {
        try Task.checkCancellation()
        guard frame.timeSeconds.isFinite else {
            processed += 1
            return
        }
        if let last = lastInput, (frame.timeSeconds, frame.id.raw) < (last.time, last.id) {
            throw InputError.outOfOrder
        }
        lastInput = (frame.timeSeconds, frame.id.raw)
        let index = members.count
        while windowStart < index, frame.timeSeconds - members[windowStart].time > 120 {
            active.removeValue(forKey: windowStart)
            windowStart += 1
        }
        guard let payload = frame.featurePrint else {
            processed += 1
            return
        }
        members.append((frame.id.raw, frame.timeSeconds))
        parent.append(index); sizes.append(1); rank.append(0)
        let observation = distance == nil ? FeaturePrintCodec.decode(payload) : nil
        active[index] = Entry(payload: distance == nil ? nil : payload, observation: observation)
        if distance != nil || observation != nil {
            for previous in windowStart..<index {
                if previous % 256 == 0 { try Task.checkCancellation() }
                if find(previous) == find(index) { continue }
                guard let entry = active[previous] else { continue }
                let separation: Double?
                if let distance, let earlier = entry.payload {
                    separation = distance(earlier, payload)
                } else if let earlier = entry.observation, let observation {
                    // Preserve chronological argument order, even though Vision's
                    // distance is symmetric.
                    separation = try? earlier.distance(to: observation)
                } else { separation = nil }
                if let separation, separation <= Double(settings.similarityThreshold) {
                    union(previous, index)
                }
            }
        }
        processed += 1
    }

    func finish() throws -> [FrameGroup] {
        try Task.checkCancellation()
        active.removeAll(keepingCapacity: false)
        return try snapshot()
    }

    /// Rank a live prefix without discarding the comparison window for later frames.
    func snapshot() throws -> [FrameGroup] {
        try Task.checkCancellation()
        var components: [Int: [String]] = [:]
        for index in members.indices {
            if index % 256 == 0 { try Task.checkCancellation() }
            components[find(index), default: []].append(members[index].id)
        }
        return FrameGrouper.finalize(components: Array(components.values),
                                    dates: Dictionary(uniqueKeysWithValues: members.map { ($0.id, $0.time) }),
                                    maximumSize: settings.maximumGroupSize)
    }

    private func find(_ node: Int) -> Int {
        var root = node
        while parent[root] != root { root = parent[root] }
        var walk = node
        while parent[walk] != root {
            let next = parent[walk]; parent[walk] = root; walk = next
        }
        return root
    }

    private func union(_ a: Int, _ b: Int) {
        let rootA = find(a), rootB = find(b)
        guard rootA != rootB else { return }
        if sizes[rootA] > 1 { linkedGroups -= 1 }
        if sizes[rootB] > 1 { linkedGroups -= 1 }
        linkedGroups += 1
        let joined = sizes[rootA] + sizes[rootB]
        if rank[rootA] < rank[rootB] {
            parent[rootA] = rootB; sizes[rootB] = joined
        } else {
            parent[rootB] = rootA; sizes[rootA] = joined
            if rank[rootA] == rank[rootB] { rank[rootA] += 1 }
        }
    }
}
