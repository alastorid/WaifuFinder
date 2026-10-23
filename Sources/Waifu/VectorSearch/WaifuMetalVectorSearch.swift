import Foundation
import Metal

/// Immutable resident uploads and exclusive scratch leases make actor reentrancy,
/// source replacement, and cancellation safe while commands are in flight.
final class WaifuMetalVectorSearch: @unchecked Sendable {
    private let device: any MTLDevice
    private let queue: any MTLCommandQueue
    private let distance: any MTLComputePipelineState
    private let scalarDistance: any MTLComputePipelineState
    private let localTop: any MTLComputePipelineState
    private let mergeTop: any MTLComputePipelineState
    private let allocationLimit: Int
    private let lock = NSLock()
    private struct Resident {
        let identity: UUID
        let revision: UInt64
        let vectors: any MTLBuffer
        let ranks: any MTLBuffer
    }
    private struct Scratch {
        let query: any MTLBuffer
        let distances: any MTLBuffer
        let local: any MTLBuffer
        let output: any MTLBuffer
        var bytes: Int { query.length + distances.length + local.length + output.length }
    }
    private var resident: Resident?
    private var pool: [Scratch] = []
    private struct Params { var n, d, k, squared: UInt32 }
    private struct Record { var distance: Float; var row: UInt32; var rank: UInt32 }
    init(allocationLimit: Int = WaifuVectorIndex.maximumBytes + 16 * 1024 * 1024) throws {
        guard let device = MTLCreateSystemDefaultDevice(), device.hasUnifiedMemory,
              let queue = device.makeCommandQueue() else { throw WaifuVectorError.unavailable }
        let options = MTLCompileOptions(); options.mathMode = .safe
        let library = try device.makeLibrary(source: Self.source, options: options)
        func pipeline(_ name: String) throws -> any MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else { throw WaifuVectorError.unavailable }
            let state = try device.makeComputePipelineState(function: function)
            guard state.threadExecutionWidth == 32, state.maxTotalThreadsPerThreadgroup >= 256 else { throw WaifuVectorError.unavailable }
            return state
        }
        self.device = device; self.queue = queue; self.allocationLimit = max(0,allocationLimit)
        distance = try pipeline("query_distances"); scalarDistance = try pipeline("query_distances_scalar")
        localTop = try pipeline("local_top"); mergeTop = try pipeline("merge_top")
    }
    private func allocate(_ bytes: Int) throws -> any MTLBuffer {
        guard bytes > 0, bytes <= device.maxBufferLength, bytes <= allocationLimit,
              let buffer = device.makeBuffer(length: bytes, options: .storageModeShared) else { throw WaifuVectorError.allocation }
        return buffer
    }
    private func copy<T>(_ values: [T]) throws -> any MTLBuffer {
        try values.withUnsafeBytes { bytes in
            guard let address = bytes.baseAddress, bytes.count <= device.maxBufferLength, bytes.count <= allocationLimit,
                  let buffer = device.makeBuffer(bytes: address, length: bytes.count, options: .storageModeShared) else { throw WaifuVectorError.allocation }
            return buffer
        }
    }
    private func upload(_ index: WaifuVectorIndex) throws -> Resident {
        try lock.withLock {
            if let resident, resident.identity == index.identity, resident.revision == index.revision { return resident }
            let order = index.ids.indices.sorted { index.ids[$0] < index.ids[$1] }
            var ranks = [UInt32](repeating: 0, count: index.count)
            for (rank,row) in order.enumerated() { ranks[row] = UInt32(rank) }
            let result = Resident(identity: index.identity, revision: index.revision, vectors: try copy(index.values), ranks: try copy(ranks))
            resident = result
            return result
        }
    }
    private func lease(_ index: WaifuVectorIndex, k: Int) throws -> Scratch {
        let localBytes = ((index.count+511)/512) * k * 12
        if let existing: Scratch = lock.withLock({
            guard let i = pool.firstIndex(where: { $0.query.length >= index.rowStride*4 && $0.distances.length >= index.count*4 && $0.local.length >= localBytes && $0.output.length >= k*12 }) else { return nil }
            return pool.remove(at: i)
        }) { return existing }
        return Scratch(query: try allocate(index.rowStride*4), distances: try allocate(index.count*4), local: try allocate(localBytes), output: try allocate(k*12))
    }
    func search(_ index: WaifuVectorIndex, query: [Float], k requested: Int, readback: Bool = false, scalar: Bool = false) async throws -> ([WaifuVectorHit], WaifuVectorTiming) {
        try Task.checkCancellation()
        guard index.count > 0, query.count == index.dimensions, query.allSatisfy(\.isFinite), requested > 0 else { throw WaifuVectorError.invalidQuery }
        let k = min(requested,index.count)
        guard k <= 64 || readback else { throw WaifuVectorError.unavailable }
        let start = ProcessInfo.processInfo.systemUptime
        let input = try upload(index)
        let scratch = try lease(index, k: min(k,64))
        defer { lock.withLock { if pool.count < 2 { pool.append(scratch) } } }
        scratch.query.contents().initializeMemory(as: UInt8.self,repeating: 0,count: index.rowStride*4)
        query.withUnsafeBytes { scratch.query.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        let uploadTime = ProcessInfo.processInfo.systemUptime-start
        var p = Params(n: UInt32(index.count), d: UInt32(index.rowStride), k: UInt32(min(k,64)), squared: index.metric.rawValue)
        guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else { throw WaifuVectorError.command }
        encoder.setComputePipelineState(scalar ? scalarDistance : distance)
        encoder.setBuffer(input.vectors, offset: 0, index: 0); encoder.setBuffer(scratch.query, offset: 0, index: 1)
        encoder.setBuffer(scratch.distances, offset: 0, index: 2); encoder.setBytes(&p, length: MemoryLayout<Params>.stride, index: 3)
        encoder.dispatchThreadgroups(MTLSize(width: scalar ? (index.count+127)/128 : (index.count+3)/4, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        encoder.endEncoding()
        var selection: (any MTLCommandBuffer)?
        if !readback {
            guard let select = queue.makeCommandBuffer(), let local = select.makeComputeCommandEncoder() else { throw WaifuVectorError.command }
            local.setComputePipelineState(localTop)
            local.setBuffer(scratch.distances, offset: 0, index: 0); local.setBuffer(input.ranks, offset: 0, index: 1)
            local.setBuffer(scratch.local, offset: 0, index: 2); local.setBytes(&p, length: MemoryLayout<Params>.stride, index: 3)
            local.dispatchThreadgroups(MTLSize(width: (index.count+511)/512, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)); local.endEncoding()
            guard let merge = select.makeComputeCommandEncoder() else { throw WaifuVectorError.command }
            merge.setComputePipelineState(mergeTop)
            merge.setBuffer(scratch.local, offset: 0, index: 0); merge.setBuffer(scratch.output, offset: 0, index: 1)
            merge.setBytes(&p, length: MemoryLayout<Params>.stride, index: 2)
            merge.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)); merge.endEncoding()
            selection = select
        }
        let wall = ProcessInfo.processInfo.systemUptime
        // No waitUntilCompleted on the main actor or a cooperative worker thread.
        // Even cancellation waits for completion before returning the scratch lease.
        let completion = selection ?? command
        await withCheckedContinuation { (continuation: CheckedContinuation<Void,Never>) in
            completion.addCompletedHandler { _ in continuation.resume() }
            command.commit(); selection?.commit()
        }
        try Task.checkCancellation()
        guard command.status == .completed, completion.status == .completed else { throw completion.error ?? WaifuVectorError.command }
        let commandWall = ProcessInfo.processInfo.systemUptime-wall
        let readStart = ProcessInfo.processInfo.systemUptime
        let hits: [WaifuVectorHit]
        if readback {
            let output = scratch.distances.contents().bindMemory(to: Float.self, capacity: index.count)
            var top = WaifuVectorTopK(k)
            for row in 0..<index.count { top.insert(.init(row: row, id: index.ids[row], distance: output[row])) }
            hits = top.sorted
        } else {
            let output = scratch.output.contents().bindMemory(to: Record.self, capacity: k)
            hits = (0..<k).compactMap { position in
                let record = output[position], row = Int(record.row)
                guard row < index.count, record.distance.isFinite else { return nil }
                return WaifuVectorHit(row: row, id: index.ids[row], distance: record.distance)
            }
        }
        return (hits, WaifuVectorTiming(upload: uploadTime, distanceGPU: command.gpuEndTime-command.gpuStartTime,
            selectionGPU: selection.map { $0.gpuEndTime-$0.gpuStartTime } ?? 0, commandWall: commandWall,
            readback: ProcessInfo.processInfo.systemUptime-readStart, allocatedBytes: input.vectors.length+input.ranks.length+scratch.bytes))
    }
    private static let source = #"""
    #include <metal_stdlib>
    using namespace metal;
    struct Params { uint n,d,k,squared; };
    struct Record { float distance; uint row, rank; };
    inline bool less_record(Record a, Record b) { return a.distance != b.distance ? a.distance < b.distance : a.rank < b.rank; }
    inline Record empty_record() { return {INFINITY,0xffffffffu,0xffffffffu}; }
    kernel void query_distances(device const float* rows [[buffer(0)]], device const float* query [[buffer(1)]], device float* out [[buffer(2)]], constant Params& p [[buffer(3)]],
        uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
        uint row = group*4+simd;
        if (row >= p.n) return;
        float total = 0;
        for (uint d=lane; d<p.d; d+=32) { float delta = rows[row*p.d+d]-query[d]; total += delta*delta; }
        float value = simd_sum(total);
        if (lane == 0) out[row] = isfinite(value) ? (p.squared ? max(0.0f,value) : sqrt(max(0.0f,value))) : INFINITY;
    }
    kernel void query_distances_scalar(device const float* rows [[buffer(0)]], device const float* query [[buffer(1)]], device float* out [[buffer(2)]], constant Params& p [[buffer(3)]], uint row [[thread_position_in_grid]]) {
        if (row >= p.n) return;
        float4 total(0.0f); uint d=0;
        for (; d+3<p.d; d+=4) { float4 delta = float4(rows[row*p.d+d],rows[row*p.d+d+1],rows[row*p.d+d+2],rows[row*p.d+d+3])-float4(query[d],query[d+1],query[d+2],query[d+3]); total += delta*delta; }
        float value = total.x+total.y+total.z+total.w;
        for (;d<p.d;++d) { float delta=rows[row*p.d+d]-query[d]; value+=delta*delta; }
        out[row] = isfinite(value) ? (p.squared ? max(0.0f,value) : sqrt(max(0.0f,value))) : INFINITY;
    }
    // Sort only bounded 512-row tiles, retaining K records from each tile.
    kernel void local_top(device const float* distances [[buffer(0)]], device const uint* ranks [[buffer(1)]], device Record* out [[buffer(2)]], constant Params& p [[buffer(3)]],
        uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
        threadgroup Record scratch[512];
        for (uint i=tid;i<512;i+=256) { uint row=group*512+i; scratch[i]=row<p.n ? Record{distances[row],row,ranks[row]} : empty_record(); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint size=2;size<=512;size<<=1) {
            for (uint stride=size>>1;stride>0;stride>>=1) {
                for (uint i=tid;i<512;i+=256) {
                    uint partner=i^stride;
                    if (partner>i) {
                        Record a=scratch[i], b=scratch[partner]; bool ascending=(i&size)==0;
                        if (ascending ? less_record(b,a) : less_record(a,b)) { scratch[i]=b; scratch[partner]=a; }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        if (tid<p.k) out[group*p.k+tid]=scratch[tid];
    }
    // Hierarchical merge scans only tile winners. Every output removes one winner.
    kernel void merge_top(device const Record* input [[buffer(0)]], device Record* output [[buffer(1)]], constant Params& p [[buffer(2)]], uint tid [[thread_index_in_threadgroup]]) {
        threadgroup Record scratch[128];
        uint count=((p.n+511)/512)*p.k;
        for (uint position=0;position<p.k;++position) {
            Record best=empty_record();
            for (uint i=tid;i<count;i+=128) {
                Record value=input[i]; bool used=false;
                for (uint old=0;old<position;++old) if (output[old].row==value.row) { used=true; break; }
                if (!used && less_record(value,best)) best=value;
            }
            scratch[tid]=best; threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint step=64;step>0;step>>=1) {
                if (tid<step && less_record(scratch[tid+step],scratch[tid])) scratch[tid]=scratch[tid+step];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid==0) output[position]=scratch[0];
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        }
    }
    """#
}
