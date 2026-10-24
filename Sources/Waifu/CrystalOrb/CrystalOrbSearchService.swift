import Foundation
import CoreGraphics
import Vision

struct CrystalOrbMatch: Identifiable, Sendable, Equatable {
    let frame: AnalyzedFrame
    let distance: Double
    var id: FrameID { frame.id }

    static func ranked(_ values: [Self]) -> [Self] {
        values.filter { $0.distance.isFinite && $0.distance >= 0 }.sorted {
            $0.distance == $1.distance ? $0.id.raw < $1.id.raw : $0.distance < $1.distance
        }
    }
}

/// A serial, session-scoped revision-2 index. It reuses the application's persisted
/// Vision payloads; missing/corrupt entries use analysis renditions, never originals.
/// No SwiftUI or presentation state crosses this boundary.
actor CrystalOrbSearchService {
    private var source: UUID?
    private var features: [FrameID: FeaturePrintObservation] = [:]
    private var payloads: [FrameID: Data] = [:]
    private var index: WaifuVisionVectorIndex?
    private var indexedIDs: [FrameID] = []
    private var completedFrames: [AnalyzedFrame] = []
    private var completedFrameMap: [String:AnalyzedFrame] = [:]
    private var queryFeature: (UUID, FeaturePrintObservation)?
    private var ticket = UUID()
    private let vectors = WaifuVectorSearch()
    private var rankingCache: (request: UUID, revision: UInt64, count: Int, limit: Int, matches: [CrystalOrbMatch])?
    private(set) var queryComputations = 0
    private(set) var readableCount = 0
    private(set) var backend = "Vision reference"
    var cachedCount: Int { features.count }
    private(set) var computations = 0

    func search(query: CGImage, queryID: UUID, frames: [AnalyzedFrame], sourceURL: URL, generation: UUID,
                resultLimit: Int? = nil,
                partial: @Sendable ([CrystalOrbMatch], Int, Int) async -> Void = { _, _, _ in },
                progress: @Sendable (Int, Int) async -> Void) async throws -> [CrystalOrbMatch] {
        try Task.checkCancellation()
        let request = UUID(); ticket = request; rankingCache = nil
        // Live rankings follow the requested result limit; callers without one
        // keep the compact nineteen-item publication.
        let window = max(19, resultLimit ?? 0)
        if source != generation {
            source = generation; features.removeAll(); payloads.removeAll()
            index = nil; indexedIDs = []; completedFrames = []; completedFrameMap = [:]; computations = 0
        }
        let queryObservation: FeaturePrintObservation
        if let cached = queryFeature, cached.0 == queryID { queryObservation = cached.1 }
        else {
            queryObservation = try await GenerateImageFeaturePrintRequest(.revision2).perform(on: query)
            try check(request)
            queryFeature = (queryID, queryObservation); queryComputations += 1
        }
        // Common warm path: immutable session arrays share storage, so equality
        // can skip rebuilding dictionaries and touching every cached row. Value
        // equality also catches metadata/payload changes before taking this path.
        if !completedFrames.isEmpty, frames == completedFrames, index?.observations.count == frames.count {
            let result = try await ranked(queryObservation, frames: completedFrameMap, k: resultLimit ?? frames.count, ticket: request)
            try check(request)
            await partial(Array(result.prefix(window)),frames.count,frames.count)
            await progress(frames.count,frames.count)
            return result
        }
        // A changed payload invalidates its decoded observation even if the source
        // generation stayed the same. Appended frames reuse all existing rows.
        let ids = frames.map(\.id)
        var changed = false
        for frame in frames {
            if let data = frame.featurePrint, features[frame.id] != nil, payloads[frame.id] != data {
                features[frame.id] = nil; payloads[frame.id] = nil; changed = true
            }
        }
        if changed || !ids.starts(with: indexedIDs) { index = nil }
        if index == nil { index = WaifuVisionVectorIndex(source: generation) }
        indexedIDs = ids
        let frameMap = Dictionary(frames.map { ($0.id.raw,$0) }, uniquingKeysWith: { first,_ in first })
        var lastPublication = ContinuousClock.now
        var prepared = 0
        for (position, frame) in frames.enumerated() {
            try check(request)
            var feature = features[frame.id]
            if feature == nil, let data = frame.featurePrint {
                feature = FeaturePrintCodec.decode(data)
                if feature != nil { payloads[frame.id] = data }
            }
            if let feature {
                features[frame.id] = feature; index?.append(id: frame.id.raw, observation: feature)
            }
            prepared = position + 1
            if position == 0 || (position % 256 == 0 && lastPublication.duration(to: .now) >= .milliseconds(500)) {
                let top = try await ranked(queryObservation, frames: frameMap, k: window, ticket: request)
                try check(request)
                await partial(top, prepared, frames.count); lastPublication = .now
            }
        }
        // Search available persisted features before doing any missing-image work.
        var top = try await ranked(queryObservation, frames: frameMap, k: window, ticket: request)
        try check(request)
        if !top.isEmpty { await partial(top, prepared, frames.count) }
        for (position, frame) in frames.enumerated() where features[frame.id] == nil {
            try check(request)
            do {
                let image = try await PhotosMedia.shared.analysisImage(sourceURL, seconds: frame.timeSeconds)
                try check(request)
                let feature = try await GenerateImageFeaturePrintRequest(.revision2).perform(on: image)
                try check(request)
                features[frame.id] = feature; index?.append(id: frame.id.raw, observation: feature); computations += 1
            } catch is CancellationError { throw CancellationError() }
            catch { /* Unavailable thumbnails remain counted in the summary. */ }
            try check(request)
            if lastPublication.duration(to: .now) >= .milliseconds(500) || position == frames.count-1 {
                top = try await ranked(queryObservation, frames: frameMap, k: window, ticket: request)
                try check(request)
                await partial(top, position+1, frames.count); lastPublication = .now
            }
            await progress(position+1, frames.count)
        }
        top = try await ranked(queryObservation, frames: frameMap, k: resultLimit ?? frames.count, ticket: request)
        try check(request)
        if index?.observations.count == frames.count { completedFrames = frames; completedFrameMap = frameMap }
        await partial(Array(top.prefix(window)), frames.count, frames.count)
        await progress(frames.count,frames.count)
        return top
    }
    private func check(_ request: UUID) throws {
        try Task.checkCancellation()
        guard ticket == request else { throw CancellationError() }
    }
    private func ranked(_ query: FeaturePrintObservation, frames: [String: AnalyzedFrame], k: Int, ticket request: UUID) async throws -> [CrystalOrbMatch] {
        guard let index else { return [] }
        let count = index.observations.count
        let limit = min(max(0,k),count)
        guard limit > 0 else { readableCount = 0; return [] }
        let revision = index.packed?.revision ?? 0
        if let cache=rankingCache, cache.request == request, cache.revision == revision, cache.count == count, cache.limit >= limit {
            return Array(cache.matches.prefix(limit))
        }
        var candidates: Set<String>
        var accelerated = false
        var readable = 0
        if let packed = index.packed, let q = index.validatedQuery(query) {
            let shortlist = min(packed.count, max(64,limit+1))
            let result = try await vectors.search(packed, query: q, k: shortlist)
            try check(request)
            backend = result.backend.rawValue
            accelerated = true; readable = packed.count
            candidates = Set(result.hits.map(\.id))
            // Float reduction order can reorder near-ties. Include every row in
            // a conservative error band around the cutoff, then score in Vision.
            if result.hits.count > limit, let boundary = result.hits.dropFirst(limit-1).first,
               let last = result.hits.last, Double(last.distance-boundary.distance) <= 4e-5 * max(1,Double(boundary.distance)) {
                let distances = try WaifuVectorCPU.distances(packed, query: q)
                let cutoff = Double(boundary.distance) + 4e-5 * max(1,Double(boundary.distance))
                for row in distances.indices where Double(distances[row]) <= cutoff { candidates.insert(packed.ids[row]) }
            }
            candidates.formUnion(index.unrepresentable)
        } else { backend = "Vision reference"; candidates = Set(index.observations.keys) }
        var best: [CrystalOrbMatch] = []
        // Native Double scores remain the public API, including exact tie order.
        for id in candidates {
            try check(request)
            guard let feature = index.observations[id], let frame = frames[id],
                  let distance = try? query.distance(to: feature), distance.isFinite, distance >= 0 else { continue }
            if !accelerated || index.unrepresentable.contains(id) { readable += 1 }
            best.append(CrystalOrbMatch(frame: frame,distance: distance))
        }
        // Only the small reranked shortlist is sorted in the application path.
        let result: [CrystalOrbMatch]
        if best.count <= limit { result = CrystalOrbMatch.ranked(best) }
        else {
            // Bounded insertion selects K even when a large near-tie band needs
            // native reranking. Only K matches, never N, are kept sorted.
            var top: [CrystalOrbMatch] = []
            for match in best {
                var lo=0, hi=top.count
                while lo < hi {
                    let mid=(lo+hi)/2, other=top[mid]
                    if match.distance < other.distance || (match.distance == other.distance && match.id.raw < other.id.raw) { hi=mid } else { lo=mid+1 }
                }
                if lo < limit { top.insert(match,at:lo); if top.count > limit { top.removeLast() } }
            }
            result=top
        }
        readableCount = readable
        rankingCache=(request,revision,count,limit,result)
        return result
    }
}

/// Monotonic publication gate for live rankings. Search may run at full speed;
/// presentation coalesces to one distinct snapshot per second.
struct CrystalOrbPublicationGate {
    private var lastTime: TimeInterval?
    private var lastRanking: [CrystalOrbMatch] = []
    func delay(for ranking: [CrystalOrbMatch], at time: TimeInterval) -> TimeInterval? {
        guard ranking != lastRanking else { return nil }
        return lastTime.map { max(0, 1 - (time - $0)) } ?? 0
    }
    mutating func published(_ ranking: [CrystalOrbMatch], at time: TimeInterval) {
        lastRanking = ranking; lastTime = time
    }
}
