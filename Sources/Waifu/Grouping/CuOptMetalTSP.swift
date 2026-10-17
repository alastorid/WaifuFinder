// SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Modified for WaifuFinder: Metal/Swift port of cuOpt's two_opt.cu candidate
// evaluation, ranked reduction and snapshot move execution, specialized to a
// symmetric open path. See ThirdParty/cuopt/NOTICE and docs/CuOptMetal.md.

import Foundation
import Metal

/// Each solve owns its buffers. The shared object only holds immutable pipelines
/// and a thread-safe command queue, so canceled and replacement jobs cannot share
/// route state. Shader source is embedded to keep app/test resource handling equal.
final class CuOptMetalTSP: @unchecked Sendable {
    static let shared: CuOptMetalTSP? = try? CuOptMetalTSP()
    static let denseLimit = 4_096 // 64 MiB, not an unbounded n² allocation.
    static let vectorLimit = 16_384

    private let device: any MTLDevice
    private let queue: any MTLCommandQueue
    private let distancesPipeline: any MTLComputePipelineState
    private let densePipeline: any MTLComputePipelineState
    private let sparsePipeline: any MTLComputePipelineState
    private let reductionPipeline: any MTLComputePipelineState
    private let applyPipeline: any MTLComputePipelineState

    private struct Parameters {
        var count: UInt32
        var dimensions: UInt32
        var tours: UInt32
        var dense: UInt32
        var candidates: UInt32
        var squared: UInt32
    }

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice(), device.hasUnifiedMemory,
              let queue = device.makeCommandQueue() else { throw TSPError.metalUnavailable }
        let options = MTLCompileOptions()
        options.mathMode = .safe
        let library = try device.makeLibrary(source: Self.source, options: options)
        func pipeline(_ name: String) throws -> any MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else { throw TSPError.metalUnavailable }
            let pipeline = try device.makeComputePipelineState(function: function)
            guard pipeline.threadExecutionWidth == 32, pipeline.maxTotalThreadsPerThreadgroup >= 128 else {
                throw TSPError.metalUnavailable
            }
            return pipeline
        }
        self.device = device
        self.queue = queue
        self.distancesPipeline = try pipeline("feature_distances")
        self.densePipeline = try pipeline("find_two_opt_moves")
        self.sparsePipeline = try pipeline("find_vector_two_opt_moves")
        self.reductionPipeline = try pipeline("reduce_two_opt_moves")
        self.applyPipeline = try pipeline("execute_two_opt_moves")
    }

    private func buffer<T>(_ values: [T]) throws -> any MTLBuffer {
        guard !values.isEmpty else { throw TSPError.allocationFailed }
        return try values.withUnsafeBytes { bytes in
            guard let address = bytes.baseAddress,
                  let buffer = device.makeBuffer(bytes: address, length: bytes.count, options: .storageModeShared) else {
                throw TSPError.allocationFailed
            }
            return buffer
        }
    }

    private func allocate(_ bytes: Int) throws -> any MTLBuffer {
        guard bytes > 0, bytes <= device.maxBufferLength,
              let buffer = device.makeBuffer(length: bytes, options: .storageModeShared) else { throw TSPError.allocationFailed }
        return buffer
    }

    private func finish(_ command: any MTLCommandBuffer) throws {
        try Task.checkCancellation()
        command.commit()
        command.waitUntilCompleted()
        try Task.checkCancellation()
        guard command.status == .completed else { throw command.error ?? TSPError.commandFailed }
    }

    private func distanceBuffer(_ vectors: VisualFeatureVectors, buffer: any MTLBuffer) throws -> any MTLBuffer {
        guard vectors.count <= Self.denseLimit else { throw TSPError.allocationFailed }
        let output = try allocate(vectors.count * vectors.count * 4)
        guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
            throw TSPError.commandFailed
        }
        var parameters = Parameters(count: UInt32(vectors.count), dimensions: UInt32(vectors.dimensions), tours: 1, dense: 1, candidates: 0, squared: vectors.metric.rawValue)
        encoder.setComputePipelineState(distancesPipeline)
        encoder.setBuffer(buffer, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBytes(&parameters, length: MemoryLayout<Parameters>.stride, index: 2)
        encoder.dispatchThreadgroups(MTLSize(width: (vectors.count * vectors.count + 3) / 4, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        encoder.endEncoding()
        try finish(command)
        // Validate the actual GPU metric too, not just the CPU interpretation of
        // the raw embedding. A mismatch must never silently change grouping's metric.
        let data = output.contents().bindMemory(to: Float.self, capacity: vectors.count * vectors.count)
        for (left, right) in VisualFeatureVectors.validationPairs(count: vectors.count) {
            guard VisualFeatureVectors.matches(Double(vectors.distance(left, right)), Double(data[left * vectors.count + right])) else {
                throw TSPError.invalidMatrix
            }
        }
        return output
    }

    /// Exposed for CPU/GPU parity checks and benchmarks; production keeps this
    /// matrix in its Metal buffer without a second n² host-side copy.
    func distances(_ vectors: VisualFeatureVectors) throws -> TSPDistanceMatrix {
        let input = try buffer(vectors.values)
        let result = try distanceBuffer(vectors, buffer: input)
        return try TSPDistanceMatrix(count: vectors.count, values: Array(UnsafeBufferPointer(
            start: result.contents().bindMemory(to: Float.self, capacity: vectors.count * vectors.count),
            count: vectors.count * vectors.count)))
    }

    func solve(_ vectors: VisualFeatureVectors, seeds: [[Int]], iterations: Int = 128,
               denseLimit: Int = CuOptMetalTSP.denseLimit) throws -> [Int] {
        guard vectors.count <= Self.vectorLimit else { throw TSPError.allocationFailed }
        try CuOptTSP.validate(seeds, count: vectors.count)
        let input = try buffer(vectors.values)
        let dense = vectors.count <= min(denseLimit, Self.denseLimit)
        let matrix = dense ? try distanceBuffer(vectors, buffer: input) : try buffer([Float(0)])
        if !dense {
            // Validate this GPU's raw-vector metric without a square allocation
            // for the full collection. Both kernels call vector_distance.
            let indices = Set(VisualFeatureVectors.validationPairs(count: vectors.count).flatMap { [$0.0, $0.1] }).sorted()
            let sample = indices.flatMap { index in
                Array(vectors.values[(index * vectors.dimensions)..<((index + 1) * vectors.dimensions)])
            }
            guard let sampled = VisualFeatureVectors(count: indices.count, dimensions: vectors.dimensions, values: sample, metric: vectors.metric) else {
                throw TSPError.invalidMatrix
            }
            _ = try distanceBuffer(sampled, buffer: buffer(sample))
        }
        // Sparse mode computes only a bounded set of proposed edge distances on
        // the GPU, including both nearby and distant route positions.
        let candidates = dense ? vectors.count : 64
        let activeSeeds = dense ? seeds : Array(seeds.prefix(2))
        return try run(count: vectors.count, dimensions: vectors.dimensions, metric: vectors.metric, dense: dense,
                       candidates: candidates, seeds: activeSeeds, vectors: input, matrix: matrix,
                       iterations: dense ? iterations : min(iterations, 32), distance: vectors.distance)
    }

    func solve(_ matrix: TSPDistanceMatrix, seeds: [[Int]], iterations: Int = 128) throws -> [Int] {
        guard matrix.count <= Self.denseLimit else { throw TSPError.allocationFailed }
        try CuOptTSP.validate(seeds, count: matrix.count)
        return try run(count: matrix.count, dimensions: 0, metric: .euclidean, dense: true, candidates: matrix.count,
                       seeds: seeds, vectors: buffer([Float(0)]), matrix: buffer(matrix.values),
                       iterations: iterations, distance: { matrix[$0, $1] })
    }

    private func run(count: Int, dimensions: Int, metric: VisualFeatureVectors.Metric, dense: Bool, candidates: Int, seeds: [[Int]],
                     vectors: any MTLBuffer, matrix: any MTLBuffer, iterations: Int,
                     distance: (Int, Int) -> Float) throws -> [Int] {
        try Task.checkCancellation()
        let routeCount = count * seeds.count
        var input = try buffer(seeds.flatMap { $0.map(UInt32.init) })
        var output = try allocate(routeCount * 4)
        let moves = try allocate(routeCount * 12)
        let best = try allocate(seeds.count * 12)
        var parameters = Parameters(count: UInt32(count), dimensions: UInt32(dimensions),
                                    tours: UInt32(seeds.count), dense: dense ? 1 : 0, candidates: UInt32(candidates), squared: metric.rawValue)
        let threads = MTLSize(width: 128, height: 1, depth: 1)
        for _ in 0..<max(0, iterations) {
            try Task.checkCancellation()
            guard let command = queue.makeCommandBuffer() else { throw TSPError.commandFailed }
            guard let search = command.makeComputeCommandEncoder() else { throw TSPError.commandFailed }
            search.setComputePipelineState(dense ? densePipeline : sparsePipeline)
            search.setBuffer(matrix, offset: 0, index: 0)
            search.setBuffer(vectors, offset: 0, index: 1)
            search.setBuffer(input, offset: 0, index: 2)
            search.setBuffer(moves, offset: 0, index: 3)
            search.setBytes(&parameters, length: MemoryLayout<Parameters>.stride, index: 4)
            search.dispatchThreadgroups(MTLSize(width: routeCount, height: 1, depth: 1), threadsPerThreadgroup: threads)
            search.endEncoding()
            guard let reduce = command.makeComputeCommandEncoder() else { throw TSPError.commandFailed }
            reduce.setComputePipelineState(reductionPipeline)
            reduce.setBuffer(moves, offset: 0, index: 0)
            reduce.setBuffer(best, offset: 0, index: 1)
            reduce.setBytes(&parameters, length: MemoryLayout<Parameters>.stride, index: 2)
            reduce.dispatchThreadgroups(MTLSize(width: seeds.count, height: 1, depth: 1), threadsPerThreadgroup: threads)
            reduce.endEncoding()
            guard let apply = command.makeComputeCommandEncoder() else { throw TSPError.commandFailed }
            apply.setComputePipelineState(applyPipeline)
            apply.setBuffer(input, offset: 0, index: 0)
            apply.setBuffer(output, offset: 0, index: 1)
            apply.setBuffer(best, offset: 0, index: 2)
            apply.setBytes(&parameters, length: MemoryLayout<Parameters>.stride, index: 3)
            apply.dispatchThreads(MTLSize(width: routeCount, height: 1, depth: 1), threadsPerThreadgroup: threads)
            apply.endEncoding()
            try finish(command)
            swap(&input, &output)
            // Move stride is three 32-bit words in both Swift and Metal.
            let records = best.contents().bindMemory(to: Float.self, capacity: seeds.count * 3)
            if !(0..<seeds.count).contains(where: { records[$0 * 3] < -CuOptTSP.epsilon }) { break }
        }
        let indices = input.contents().bindMemory(to: UInt32.self, capacity: routeCount)
        var routes = seeds
        for tour in seeds.indices {
            routes.append((0..<count).map { Int(indices[tour * count + $0]) })
        }
        try CuOptTSP.validate(routes, count: count)
        func cost(_ route: [Int]) -> Double {
            zip(route, route.dropFirst()).reduce(0) { $0 + Double(distance($1.0, $1.1)) }
        }
        var winner = routes[0]
        var score = cost(winner)
        for route in routes.dropFirst() {
            let candidate = cost(route)
            if candidate < score { winner = route; score = candidate }
        }
        return winner
    }

    private static let source = #"""
    #include <metal_stdlib>
    using namespace metal;
    struct Params { uint n, dimensions, tours, dense, candidates, squared; };
    struct Move { float delta; uint first, second; };
    inline Move empty_move() { return {0.0f, 0xffffffffu, 0xffffffffu}; }
    inline Move better(Move a, Move b) {
        if (a.delta != b.delta) return a.delta < b.delta ? a : b;
        if (a.first != b.first) return a.first < b.first ? a : b;
        return a.second < b.second ? a : b;
    }
    inline Move ranked_reduce(Move value, uint tid, threadgroup Move* scratch) {
        scratch[tid] = value;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 64; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] = better(scratch[tid], scratch[tid + stride]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        return scratch[0];
    }
    inline float vector_distance(device const float* x, uint a, uint b, uint d, uint lane, uint squared) {
        float sum = 0;
        for (uint k = lane; k < d; k += 32) {
            float difference = x[a*d+k] - x[b*d+k];
            sum += difference * difference;
        }
        float distance = max(0.0f, simd_sum(sum));
        return squared ? distance : sqrt(distance);
    }
    kernel void feature_distances(device const float* x [[buffer(0)]], device float* matrix [[buffer(1)]],
                                  constant Params& p [[buffer(2)]], uint group [[threadgroup_position_in_grid]],
                                  uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
        uint pair = group * 4 + simd;
        if (pair >= p.n * p.n) return;
        uint a = pair / p.n, b = pair % p.n;
        if (b < a) return;
        float distance = vector_distance(x, a, b, p.dimensions, lane, p.squared);
        if (lane == 0) { matrix[a*p.n+b] = distance; matrix[b*p.n+a] = distance; }
    }
    // Port of cuOpt's per-node candidate search and ranked block reduction.
    // Route constraints/depot bookkeeping are removed; missing endpoint edges cost zero.
    kernel void find_two_opt_moves(device const float* matrix [[buffer(0)]], device const float* x [[buffer(1)]],
                                   device const uint* routes [[buffer(2)]], device Move* moves [[buffer(3)]],
                                   constant Params& p [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]]) {
        uint first = group % p.n, tour = group / p.n;
        device const uint* route = routes + tour*p.n;
        Move best = empty_move();
        for (uint second = first + 1 + tid; second < p.n; second += 128) {
            if (first == 0 && second == p.n-1) continue;
            float before = 0, after = 0;
            if (first > 0) {
                before += matrix[route[first-1]*p.n+route[first]];
                after += matrix[route[first-1]*p.n+route[second]];
            }
            if (second + 1 < p.n) {
                before += matrix[route[second]*p.n+route[second+1]];
                after += matrix[route[first]*p.n+route[second+1]];
            }
            float delta = after - before;
            if (delta < -1e-6f) best = better(best, {delta, first, second});
        }
        threadgroup Move scratch[128];
        best = ranked_reduce(best, tid, scratch);
        if (tid == 0) moves[group] = best;
    }
    kernel void find_vector_two_opt_moves(device const float* matrix [[buffer(0)]], device const float* x [[buffer(1)]],
                                          device const uint* routes [[buffer(2)]], device Move* moves [[buffer(3)]],
                                          constant Params& p [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
                                          uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                          uint simd [[simdgroup_index_in_threadgroup]]) {
        uint first = group % p.n, tour = group / p.n;
        device const uint* route = routes + tour*p.n;
        Move best = empty_move();
        uint remaining = p.n-first-1, slots = min(remaining, p.candidates);
        for (uint slot = simd; slot < slots; slot += 4) {
            uint offset = (remaining <= p.candidates || slot < p.candidates/2)
                ? slot : ((slot-p.candidates/2)*remaining/(p.candidates/2));
            if (slot == slots-1) offset = remaining-1;
            uint second = first+1+offset;
            if (first == 0 && second == p.n-1) continue;
            float before = 0, after = 0;
            if (first > 0) {
                before += vector_distance(x, route[first-1], route[first], p.dimensions, lane, p.squared);
                after += vector_distance(x, route[first-1], route[second], p.dimensions, lane, p.squared);
            }
            if (second+1 < p.n) {
                before += vector_distance(x, route[second], route[second+1], p.dimensions, lane, p.squared);
                after += vector_distance(x, route[first], route[second+1], p.dimensions, lane, p.squared);
            }
            float delta = after-before;
            if (lane == 0 && delta < -1e-6f) best = better(best, {delta, first, second});
        }
        threadgroup Move scratch[128];
        best = ranked_reduce(best, tid, scratch);
        if (tid == 0) moves[group] = best;
    }
    kernel void reduce_two_opt_moves(device const Move* moves [[buffer(0)]], device Move* winners [[buffer(1)]],
                                     constant Params& p [[buffer(2)]], uint tour [[threadgroup_position_in_grid]],
                                     uint tid [[thread_index_in_threadgroup]]) {
        Move best = empty_move();
        for (uint i = tid; i < p.n; i += 128) best = better(best, moves[tour*p.n+i]);
        threadgroup Move scratch[128];
        best = ranked_reduce(best, tid, scratch);
        if (tid == 0) winners[tour] = best;
    }
    // Execute only one winning move per independent tour. This eliminates overlap
    // conflicts and stale deltas; every next search observes the completed snapshot.
    kernel void execute_two_opt_moves(device const uint* input [[buffer(0)]], device uint* output [[buffer(1)]],
                                      device const Move* winners [[buffer(2)]], constant Params& p [[buffer(3)]],
                                      uint index [[thread_position_in_grid]]) {
        if (index >= p.n*p.tours) return;
        uint tour = index/p.n, position = index%p.n;
        Move move = winners[tour];
        uint source = position;
        if (move.delta < -1e-6f && position >= move.first && position <= move.second)
            source = move.first+move.second-position;
        output[index] = input[tour*p.n+source];
    }
    """#
}
