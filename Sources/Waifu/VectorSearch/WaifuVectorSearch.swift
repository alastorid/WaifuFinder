import Accelerate
import Foundation

/// Flat Float32 row-major storage; identity is independent of append order.
/// Copies use Swift COW. A revision identifies an immutable GPU upload snapshot.
struct WaifuVectorIndex: Sendable {
    enum Metric: UInt32, Sendable { case squaredEuclidean = 1, euclidean = 0 }
    let identity = UUID()
    let source: UUID
    let dimensions: Int
    let metric: Metric
    // Keep every row at the same 64-byte alignment. Accelerate may otherwise
    // choose different reduction orders for identical rows at odd dimensions.
    var rowStride: Int { (dimensions + 15) / 16 * 16 }
    private(set) var revision: UInt64 = 0
    private(set) var ids: [String] = []
    private(set) var values: [Float] = []
    private var rows: Set<String> = []
    static let maximumBytes = 256 * 1024 * 1024
    var count: Int { ids.count }
    var bytes: Int { values.count * MemoryLayout<Float>.stride }
    init?(source: UUID, dimensions: Int, metric: Metric) {
        guard dimensions > 0, dimensions <= 4096 else { return nil }
        self.source = source; self.dimensions = dimensions; self.metric = metric
    }
    @discardableResult mutating func append(id: String, vector: [Float]) -> Bool {
        guard !rows.contains(id), vector.count == dimensions, vector.allSatisfy(\.isFinite),
              bytes <= Self.maximumBytes - rowStride * 4 else { return false }
        ids.append(id); values.append(contentsOf: vector)
        values.append(contentsOf: repeatElement(0,count:rowStride-dimensions))
        rows.insert(id); revision &+= 1
        return true
    }
}

struct WaifuVectorHit: Sendable, Equatable {
    let row: Int
    let id: String
    let distance: Float
    static func precedes(_ a: Self, _ b: Self) -> Bool { a.distance == b.distance ? a.id < b.id : a.distance < b.distance }
}

/// A max heap keeps only K values; no collection-sized sort is needed.
struct WaifuVectorTopK {
    let limit: Int
    private(set) var heap: [WaifuVectorHit] = []
    init(_ k: Int) { limit = max(0,k) }
    mutating func insert(_ hit: WaifuVectorHit) {
        guard limit > 0, hit.distance.isFinite, hit.distance >= 0 else { return }
        if heap.count < limit {
            heap.append(hit)
            var i = heap.count-1
            while i > 0 {
                let p = (i-1)/2
                guard WaifuVectorHit.precedes(heap[p],heap[i]) else { break }
                heap.swapAt(p,i); i = p
            }
        } else if WaifuVectorHit.precedes(hit,heap[0]) {
            heap[0] = hit; var i = 0
            while 2*i+1 < heap.count {
                var child = 2*i+1
                if child+1 < heap.count && WaifuVectorHit.precedes(heap[child],heap[child+1]) { child += 1 }
                guard WaifuVectorHit.precedes(heap[i],heap[child]) else { break }
                heap.swapAt(i,child); i = child
            }
        }
    }
    var sorted: [WaifuVectorHit] { heap.sorted(by: WaifuVectorHit.precedes) }
}

enum WaifuVectorCPU {
    static func distances(_ index: WaifuVectorIndex, query: [Float]) throws -> [Float] {
        guard query.count == index.dimensions, query.allSatisfy(\.isFinite) else { throw WaifuVectorError.invalidQuery }
        var output = [Float](repeating: 0, count: index.count)
        try index.values.withUnsafeBufferPointer { rows in
            try query.withUnsafeBufferPointer { q in
                for row in 0..<index.count {
                    if row % 256 == 0 { try Task.checkCancellation() }
                    var squared: Float = 0
                    vDSP_distancesq(rows.baseAddress! + row * index.rowStride, 1, q.baseAddress!, 1, &squared, vDSP_Length(index.dimensions))
                    output[row] = squared.isFinite ? (index.metric == .squaredEuclidean ? max(0,squared) : sqrt(max(0,squared))) : .infinity
                }
            }
        }
        return output
    }
    static func refine(_ index: WaifuVectorIndex, query: [Float], hits: [WaifuVectorHit]) -> [WaifuVectorHit] {
        index.values.withUnsafeBufferPointer { rows in
            query.withUnsafeBufferPointer { q in
                hits.map { hit in
                    var squared: Float = 0
                    vDSP_distancesq(rows.baseAddress! + hit.row * index.rowStride,1,q.baseAddress!,1,&squared,vDSP_Length(index.dimensions))
                    return WaifuVectorHit(row:hit.row,id:hit.id,distance:index.metric == .squaredEuclidean ? max(0,squared) : sqrt(max(0,squared)))
                }.filter { $0.distance.isFinite }.sorted(by:WaifuVectorHit.precedes)
            }
        }
    }
    static func search(_ index: WaifuVectorIndex, query: [Float], k: Int) throws -> [WaifuVectorHit] {
        let values = try distances(index, query: query)
        var top = WaifuVectorTopK(min(max(0,k),index.count))
        for row in values.indices { top.insert(.init(row: row, id: index.ids[row], distance: values[row])) }
        return top.sorted
    }
}

enum WaifuVectorError: Error { case invalidQuery, unavailable, allocation, command }

/// Measurements are end-to-end warm-query seconds, including selection/readback.
/// Unknown hardware/workloads choose CPU. Explicit profiling can teach the planner
/// a GPU win without baking a size-only crossover from a different Apple chip.
struct WaifuVectorPlanner: Sendable {
    private var samples: [String: [Double]] = [:]
    private func key(_ count: Int, _ dimensions: Int, _ gpu: Bool) -> String { "\(count):\(dimensions):\(gpu)" }
    mutating func record(count: Int, dimensions: Int, gpu: Bool, seconds: Double) {
        guard seconds.isFinite, seconds > 0 else { return }
        let k = key(count,dimensions,gpu)
        samples[k,default:[]].append(seconds)
        if samples[k]!.count > 7 { samples[k]!.removeFirst() }
    }
    func usesMetal(count: Int, dimensions: Int, resident: Bool) -> Bool {
        guard resident, let cpu=samples[key(count,dimensions,false)], let gpu=samples[key(count,dimensions,true)], cpu.count >= 3, gpu.count >= 3 else { return false }
        let a=cpu.sorted()[cpu.count/2], b=gpu.sorted()[gpu.count/2]
        return b < a * 0.85
    }
}

actor WaifuVectorSearch {
    enum Backend: String, Sendable { case automatic, accelerate, metal, metalReadback, metalScalar }
    struct Result: Sendable { let hits: [WaifuVectorHit]; let backend: Backend; let timing: WaifuVectorTiming? }
    private let metalFactory: @Sendable () throws -> WaifuMetalVectorSearch
    init(metalFactory: @escaping @Sendable () throws -> WaifuMetalVectorSearch = { try WaifuMetalVectorSearch() }) {
        self.metalFactory = metalFactory
    }
    private var gpu: WaifuMetalVectorSearch?
    private var initializationAttempted = false
    private var planner = WaifuVectorPlanner()
    private var residentKey: String?
    // M3 measurements at 1k/10k x 768 favor CPU. Do not extrapolate a GPU
    // crossover beyond the measured range or initialize Metal for a tiny query.
    func search(_ index: WaifuVectorIndex, query: [Float], k: Int = 19, backend: Backend = .automatic) async throws -> Result {
        try Task.checkCancellation()
        guard query.count == index.dimensions, query.allSatisfy(\.isFinite) else { throw WaifuVectorError.invalidQuery }
        if index.count == 0 || k <= 0 { return Result(hits: [], backend: .accelerate, timing: nil) }
        let key = "\(index.identity):\(index.revision)"
        let warm = residentKey == key
        let useMetal = backend != .accelerate && (backend != .automatic || planner.usesMetal(count: index.count, dimensions: index.dimensions, resident: warm))
        let start = ProcessInfo.processInfo.systemUptime
        if useMetal {
            if !initializationAttempted { initializationAttempted = true; gpu = try? metalFactory() }
            if let gpu {
                do {
                    let shortlist = min(index.count,max(64,k+1))
                    let result = try await gpu.search(index, query: query, k: min(64,shortlist), readback: backend == .metalReadback, scalar: backend == .metalScalar)
                    var hits = WaifuVectorCPU.refine(index,query:query,hits:result.0)
                    if hits.count < min(k,index.count) { throw WaifuVectorError.unavailable }
                    if index.count > result.0.count, let boundary=hits.prefix(k).last, let edge=result.0.last {
                        // Bound ordinary Float reduction error conservatively. A
                        // crowded cutoff uses a complete canonical CPU scan.
                        let band = Float(index.dimensions) * Float.ulpOfOne * max(1,abs(boundary.distance)) * 4
                        if edge.distance - boundary.distance <= band {
                            hits = try WaifuVectorCPU.search(index,query:query,k:k)
                        }
                    }
                    hits = Array(hits.prefix(k))
                    residentKey = key
                    if warm, backend != .metalScalar, backend != .metalReadback {
                        planner.record(count: index.count, dimensions: index.dimensions, gpu: true, seconds: ProcessInfo.processInfo.systemUptime-start)
                    }
                    return Result(hits: hits, backend: backend == .automatic ? .metal : backend, timing: result.1)
                } catch is CancellationError { throw CancellationError() }
                catch { /* An unavailable GPU never makes a valid CPU query fail. */ }
            }
        }
        let hits = try WaifuVectorCPU.search(index, query: query, k: k)
        planner.record(count: index.count, dimensions: index.dimensions, gpu: false, seconds: ProcessInfo.processInfo.systemUptime-start)
        return Result(hits: hits, backend: .accelerate, timing: nil)
    }
    /// Each query has its own lease; batch callers can reuse the same resident index.
    func searchBatch(_ index: WaifuVectorIndex, queries: [[Float]], k: Int = 19, backend: Backend = .automatic) async throws -> [Result] {
        var results: [Result] = []
        for query in queries { results.append(try await search(index, query: query, k: k, backend: backend)) }
        return results
    }
}

struct WaifuVectorTiming: Sendable {
    let upload: Double
    let distanceGPU: Double
    let selectionGPU: Double
    let commandWall: Double
    let readback: Double
    let allocatedBytes: Int
}
