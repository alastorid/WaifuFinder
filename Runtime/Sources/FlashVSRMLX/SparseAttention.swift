// Copyright 2026 Waifu contributors. Apache-2.0.
// Semantics: official FlashVSR wan_video_dit.py, cf910c61a60733e610e9c6e8b607f80c3a6c202b.
import Foundation
import MLX

public final class FlashSparseAttention {
    private var geometry: [Int] = []
    private var local: MLXArray?
    private let kernel = MLXFast.metalKernel(
        name: "flashvsr_lcsa_online", inputNames: ["q", "k", "v", "mask"], outputNames: ["out"],
        source: """
            uint query = threadgroup_position_in_grid.x;
            uint head = threadgroup_position_in_grid.y;
            uint lane = thread_index_in_simdgroup;
            float queryValue[4], accumulator[4] = {0, 0, 0, 0};
            for (uint j=0; j<4; ++j) queryValue[j] = float(q[(query*HEADS+head)*128+lane+j*32]);
            float maximum = -INFINITY, denominator = 0;
            for (uint block=0; block<KEY_BLOCKS; ++block) {
                if (!mask[(head*QUERY_BLOCKS+query/128)*KEY_BLOCKS+block]) continue;
                for (uint token=0; token<128; ++token) {
                    uint base = ((block*128+token)*HEADS+head)*128;
                    float partial=0;
                    for (uint j=0; j<4; ++j) partial += queryValue[j]*float(k[base+lane+j*32]);
                    float score = simd_sum(partial)*0.08838834764831845f;
                    float next = max(maximum,score);
                    float previousWeight=exp(maximum-next), currentWeight=exp(score-next);
                    for (uint j=0; j<4; ++j) accumulator[j] = accumulator[j]*previousWeight+currentWeight*float(v[base+lane+j*32]);
                    denominator = denominator*previousWeight+currentWeight;
                    maximum=next;
                }
            }
            for (uint j=0; j<4; ++j) out[(query*HEADS+head)*128+lane+j*32] = T(denominator > 0 ? accumulator[j]/denominator : 0);
            """)

    public init() {}

    /// Official global top-k threshold per head/per temporal window, not per query row.
    public func blockMask(
        q: MLXArray, k: MLXArray, height: Int, width: Int, heads: Int = 12,
        temporalWindows: Int, topK: Int, localRange: Int = 11
    ) -> MLXArray {
        let spatial = height * width
        let queryBlocks = q.shape[0]
        let keyBlocks = k.shape[0]
        let dim = q.shape[2] / heads
        precondition(
            queryBlocks == temporalWindows * spatial && keyBlocks % spatial == 0 && dim == 128 && topK >= 0)
        let current = [height, width, localRange]
        if geometry != current {
            let eligibility = (0..<spatial).flatMap { i in
                (0..<spatial).map { j in
                    let dy = j / width - i / width
                    let dx = j % width - i % width
                    let half = localRange / 2
                    return dy >= -half && dy < localRange - half && dx >= -half && dx < localRange - half
                }
            }
            local = MLXArray(eligibility, [spatial, spatial])
            eval(local!)
            geometry = current
        }
        let qm = mean(q, axis: 1).asType(q.dtype).reshaped(queryBlocks, heads, dim).transposed(1, 0, 2)
        let km = mean(k, axis: 1).asType(k.dtype).reshaped(keyBlocks, heads, dim).transposed(1, 2, 0)
        let scores = (matmul(qm, km) * Float(1 / sqrt(Double(dim)))).asType(.float32)
        let allowed = tiled(local!, repetitions: [temporalWindows, keyBlocks / spatial]).expandedDimensions(
            axis: 0)
        let probabilities = softmax(which(allowed, scores, MLXArray(-Float.infinity)), axis: -1)
        let flat = probabilities.reshaped(heads * temporalWindows, spatial * keyBlocks)
        let thresholdIndex = flat.shape[1] - 1 - min(flat.shape[1] - 1, topK)
        let thresholds = sorted(flat, axis: -1)[0..., thresholdIndex..<(thresholdIndex + 1)]
        let mask = (flat .> thresholds).reshaped(heads, queryBlocks, keyBlocks)
        eval(mask)
        return mask
    }

    /// Q/K/V remain on GPU. One SIMD group/query/head performs FP32 online softmax
    /// over eligible blocks; no token-by-token dense attention matrix is allocated.
    public func apply(q: MLXArray, k: MLXArray, v: MLXArray, mask: MLXArray, heads: Int = 12) -> MLXArray {
        let queries = q.shape[0] * 128
        let keys = k.shape[0] * 128
        precondition(q.shape[1] == 128 && k.shape == v.shape && q.shape[2] == heads * 128)
        let result = kernel(
            [
                q.reshaped(queries, heads, 128), k.reshaped(keys, heads, 128), v.reshaped(keys, heads, 128),
                mask,
            ],
            template: [
                ("T", q.dtype), ("HEADS", heads), ("QUERY_BLOCKS", q.shape[0]), ("KEY_BLOCKS", k.shape[0]),
            ],
            grid: (queries * 32, heads, 1), threadGroup: (32, 1, 1), outputShapes: [[queries, heads, 128]],
            outputDTypes: [q.dtype])[0]
        eval(result)
        return result.reshaped(q.shape)
    }

    public static func reference(q: MLXArray, k: MLXArray, v: MLXArray, mask: MLXArray, heads: Int = 12)
        -> MLXArray
    {
        let query = q.reshaped(-1, heads, 128).transposed(1, 0, 2).expandedDimensions(axis: 0)
        let key = k.reshaped(-1, heads, 128).transposed(1, 0, 2).expandedDimensions(axis: 0)
        let value = v.reshaped(-1, heads, 128).transposed(1, 0, 2).expandedDimensions(axis: 0)
        let dense = repeated(repeated(mask, count: 128, axis: 1), count: 128, axis: 2).expandedDimensions(
            axis: 0)
        let scores =
            matmul(query.asType(.float32), key.asType(.float32).transposed(0, 1, 3, 2))
            * Float(1 / sqrt(128.0))
        let valid = sum(dense.asType(.float32), axis: -1, keepDims: true) .> MLXArray(0)
        let probabilities = which(
            valid, softmax(which(dense, scores, MLXArray(-Float.infinity)), axis: -1), MLXArray(0))
        return matmul(probabilities, value.asType(.float32))[0].transposed(1, 0, 2).reshaped(q.shape).asType(
            q.dtype)
    }
}
