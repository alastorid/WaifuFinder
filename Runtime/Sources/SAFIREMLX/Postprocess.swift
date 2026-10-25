// Fused GPU post-processing for the automatic 256-prompt SAFIRE path.
// The scalar reference for every coordinate here is SAFIREModel.resize
// (Prediction.swift): torch align_corners=False square bilinear 256→1024
// with clamped boundary samples. The kernel reproduces that float arithmetic
// in the same association order, thresholds at zero, extracts the stride-16
// binary sample grid used for feature pooling and accumulates integer
// thresholded areas, so no full-resolution FP32/bool intermediate is ever
// written to memory.
import Foundation
import MLX

public enum SAFIREMaskPost {
    public static let kernel = MLXFast.metalKernel(
        name: "safire_mask_post", inputNames: ["masks", "count"],
        outputNames: ["samples", "partials"], source: """
        uint i = thread_position_in_grid.x;
        uint ox = i & 1023u, oy = (i >> 10) & 1023u, b = i >> 20;
        if (b >= (uint)count) return;
        // (o+0.5)*256/1024-0.5, clamped: identical expression to resize().
        float py = (float(oy)+0.5f)*256.0f/1024.0f-0.5f;
        float px = (float(ox)+0.5f)*256.0f/1024.0f-0.5f;
        py = clamp(py, 0.0f, 255.0f); px = clamp(px, 0.0f, 255.0f);
        int y0 = int(py), x0 = int(px);
        int y1 = min(y0+1, 255), x1 = min(x0+1, 255);
        float fy = py-floor(py), fx = px-floor(px);
        uint base = b*65536u;
        float rows0 = masks[base+uint(y0)*256u+uint(x0)]*(1.0f-fy) + masks[base+uint(y1)*256u+uint(x0)]*fy;
        float rows1 = masks[base+uint(y0)*256u+uint(x1)]*(1.0f-fy) + masks[base+uint(y1)*256u+uint(x1)]*fy;
        float value = rows0*(1.0f-fx) + rows1*fx;
        float on = value > 0.0f ? 1.0f : 0.0f;
        if (oy%16u==0u && ox%16u==0u) samples[b*4096u+(oy/16u)*64u+ox/16u] = on;
        threadgroup float local[256];
        local[thread_position_in_threadgroup.x] = on;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128u; stride > 0u; stride >>= 1) {
            if (thread_position_in_threadgroup.x < stride)
                local[thread_position_in_threadgroup.x] += local[thread_position_in_threadgroup.x+stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        // Exactly one threadgroup covers each (row, 256-pixel tile) partial.
        if (thread_position_in_threadgroup.x == 0u)
            partials[b*4096u+oy*4u+(ox>>8)] = (int)local[0];
        """)

    /// For each prompt mask in `masks` ([batch,256,256] logits), returns the
    /// [batch,64,64] stride-16 binary samples and [batch,1024,4] area partials
    /// whose sum is the exact thresholded 1024×1024 area.
    public static func apply(_ masks: MLXArray, batch: Int) -> (samples: MLXArray, partials: MLXArray) {
        precondition(masks.shape == [batch, 256, 256] && masks.dtype == .float32)
        let outputs = kernel([masks, MLXArray(Int32(batch))],
            grid: (batch * 1024 * 1024, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[batch, 64, 64], [batch, 1024, 4]],
            outputDTypes: [.float32, .int32])
        return (outputs[0], outputs[1])
    }

    /// Scalar oracle mirroring resize()/predict() exactly, for validation.
    public static func reference(_ masks: [[Float]], batch: Int) -> (samples: [[Float]], areas: [Int]) {
        func position(_ o: Int) -> (Int, Int, Float) {
            let p = max(0, min((Float(o) + 0.5) * Float(256) / Float(1024) - 0.5, Float(255)))
            let base = Int(p)
            return (base, min(base + 1, 255), p - Float(base))
        }
        var samples = [[Float]](), areas = [Int]()
        for b in 0..<batch {
            var grid = [Float](repeating: 0, count: 64 * 64)
            var area = 0
            for oy in stride(from: 0, to: 1024, by: 16) {
                for ox in stride(from: 0, to: 1024, by: 16) {
                    let (y0, y1, fy) = position(oy), (x0, x1, fx) = position(ox)
                    let rows0 = masks[b][y0 * 256 + x0] * (1 - fy) + masks[b][y1 * 256 + x0] * fy
                    let rows1 = masks[b][y0 * 256 + x1] * (1 - fy) + masks[b][y1 * 256 + x1] * fy
                    if rows0 * (1 - fx) + rows1 * fx > 0 {
                        grid[(oy / 16) * 64 + ox / 16] = 1
                    }
                }
            }
            for oy in 0..<1024 {
                for ox in 0..<1024 {
                    let (y0, y1, fy) = position(oy), (x0, x1, fx) = position(ox)
                    let rows0 = masks[b][y0 * 256 + x0] * (1 - fy) + masks[b][y1 * 256 + x0] * fy
                    let rows1 = masks[b][y0 * 256 + x1] * (1 - fy) + masks[b][y1 * 256 + x1] * fy
                    if rows0 * (1 - fx) + rows1 * fx > 0 { area += 1 }
                }
            }
            samples.append(grid); areas.append(area)
        }
        return (samples, areas)
    }
}
