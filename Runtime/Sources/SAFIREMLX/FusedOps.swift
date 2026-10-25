// Fused FP32 elementwise/reduction kernels for the SAFIRE runtime.
// MLX's generic LayerNorm and erf-GELU primitives run these mid-sized
// channel-last tensors several times slower than bandwidth; the kernels here
// each make one coalesced pass. The GELU polynomial is the classical
// Abramowitz–Stegun 7.1.26 rational approximation (maximum absolute error
// 1.5e-7, an order below the runtime's declared parity tolerances).
// Scalar oracles for validation live in SAFIRENorm.reference below.
import Foundation
import MLX

public enum SAFIRENorm {
    static let erfHelper = """
    static float safire_erf(float x) {
        bool neg = x < 0.0f;
        float ax = neg ? -x : x;
        float t = 1.0f / (1.0f + 0.3275911f * ax);
        float poly = ((((1.061405429f*t - 1.453152027f)*t) + 1.421413741f)*t - 0.284496736f)*t + 0.254829592f;
        float y = 1.0f - poly*t*exp(-ax*ax);
        return neg ? -y : y;
    }
    """
    static let geluExpression = "0.5f*v*(1.0f+safire_erf(v*0.7071067811865476f))"

    static let normKernel = MLXFast.metalKernel(
        name: "safire_fused_norm_v2", inputNames: ["x", "residual", "weight", "bias", "eps", "rows", "channels"],
        outputNames: ["out"], source: """
        uint row = threadgroup_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint padded = (uint)PADDED;
        uint n = (uint)channels;
        if (row >= (uint)rows || tid >= padded) return;
        bool active = tid < n;
        uint base = row*n + tid;
        float v = 0.0f;
        if (active) {
            v = x[base];
            if (ADD_RESIDUAL) v += residual[base];
        }
        threadgroup float shared[1024];
        shared[tid] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Reduce over the padded width: inactive lanes hold exact zeros, and a
        // power-of-two stride keeps the pairwise partition lossless.
        for (uint s = padded/2; s > 0u; s >>= 1u) {
            if (tid < s) shared[tid] += shared[tid+s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float mean = shared[0]/float(n);
        // Every thread reads shared[0] above before any thread overwrites it.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        shared[tid] = active ? (v-mean)*(v-mean) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = padded/2; s > 0u; s >>= 1u) {
            if (tid < s) shared[tid] += shared[tid+s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float variance = shared[0]/float(n);
        if (active) {
            float y = (v-mean)*rsqrt(variance+eps)*weight[tid]+bias[tid];
            if (GELU) { v = y; y = \(geluExpression); }
            out[base] = y;
        }
        """, header: erfHelper)

    static let geluKernel = MLXFast.metalKernel(
        name: "safire_gelu", inputNames: ["x"], outputNames: ["out"], source: """
        uint i = thread_position_in_grid.x;
        float v = x[i];
        out[i] = \(geluExpression);
        """, header: erfHelper)

    static let maskProjectKernel = MLXFast.metalKernel(
        name: "safire_mask_project", inputNames: ["up", "hyper", "batch"], outputNames: ["masks"],
        source: """
        uint i = thread_position_in_grid.x;
        uint p = i & 65535u, b = i >> 16u;
        if (b >= (uint)batch) return;
        float total = 0.0f;
        for (uint h = 0u; h < 32u; h++) total += up[b*2097152u+p*32u+h]*hyper[b*32u+h];
        masks[i] = total;
        """)

    /// Fused residual-add + LayerNorm + affine (+ optional GELU) in one pass.
    /// Requires FP32, contiguous rows whose length equals the threadgroup size.
    public static func norm(_ x: MLXArray, residual: MLXArray?, weight: MLXArray, bias: MLXArray,
                            eps: Float, gelu: Bool = false) -> MLXArray {
        precondition(x.dtype == .float32)
        let channels = x.shape.last!
        precondition(channels <= 1024)
        let rows = x.size / channels
        // One threadgroup per row, padded to a power of two so the tree
        // reduction stays exact for channel counts like 768.
        var padded = 1
        while padded < channels { padded *= 2 }
        // grid counts total threads in MLX fast kernels; rows*padded covers
        // one padded threadgroup per row.
        return normKernel([x, residual ?? x, weight, bias, MLXArray(Float(eps)), MLXArray(Int32(rows)), MLXArray(Int32(channels))],
            template: [("ADD_RESIDUAL", residual != nil), ("GELU", gelu), ("PADDED", padded)],
            grid: (rows * padded, 1, 1), threadGroup: (padded, 1, 1),
            outputShapes: [x.shape], outputDTypes: [.float32])[0]
    }

    public static func gelu(_ x: MLXArray) -> MLXArray {
        precondition(x.dtype == .float32)
        return geluKernel([x], grid: (x.size, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [x.shape], outputDTypes: [.float32])[0]
    }

    /// Per-prompt mask projection: masks[b,p] = Σ_h up[b,p,h]·hyper[b,h],
    /// one coalesced pass instead of a batched N=1 matmul.
    public static func maskProject(_ up: MLXArray, hyper: MLXArray, batch: Int) -> MLXArray {
        precondition(up.shape == [batch, 65536, 32] && hyper.shape == [batch, 32])
        return maskProjectKernel([up, hyper, MLXArray(Int32(batch))], grid: (batch * 65536, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[batch, 65536]], outputDTypes: [.float32])[0]
    }

    /// Scalar oracle: exact reference for the fused kernels on random inputs.
    public static func referenceNorm(_ row: [Float], weight: [Float], bias: [Float],
                                     eps: Float, gelu: Bool, residual: [Float]? = nil) -> [Float] {
        func erfReference(_ x: Float) -> Float {
            let ax = abs(x)
            let t = 1 / (1 + Float(0.3275911) * ax)
            let poly = ((((Float(1.061405429)*t - Float(1.453152027))*t) + Float(1.421413741))*t - Float(0.284496736))*t + Float(0.254829592)
            var y = 1 - poly*t*exp(-ax*ax)
            if x < 0 { y = -y }
            return y
        }
        let n = row.count
        var values = row
        if let residual { for i in 0..<n { values[i] += residual[i] } }
        let mean = values.reduce(0, +) / Float(n)
        let variance = values.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Float(n)
        let scale = 1 / (variance + eps).squareRoot()
        return (0..<n).map { i in
            var y = (values[i] - mean) * scale * weight[i] + bias[i]
            if gelu { y = 0.5 * y * (1 + erfReference(y * Float(0.7071067811865476))) }
            return y
        }
    }
}

/// Fused FP32 decoder-tail upsampling GEMMs. One Metal kernel per 2×2
/// stride-2 transposed convolution replaces the MLX pipeline
/// matmul → bias add → pixel-shuffle copy (→ LayerNorm+GELU | → GELU), whose
/// extra passes move ~370 MB per 4-prompt grid chunk on this checkpoint.
///
/// Geometry and the weight "permutation". The runtime weight is [in, kh·kw·out]
/// with columns ordered j = kh·(kw·out) + kw·out + o (see `upscaleWeights` in
/// SAFIREModel.init). That natural (kh, kw, o) column order is already the
/// permutation the fused epilogue wants, so no load-time reordering is needed:
/// for a GEMM row m = (b, y, x) the four output pixels
/// out[b, 2y+kh, 2x+kw, o] read exactly the contiguous column block
/// j = g·out + o with g = 2·kh+kw. A threadgroup that owns the full row width
/// N = 4·out therefore owns complete output pixels, the pixel shuffle
/// collapses into the store address
///   out[((b·2side + 2y+kh)·2side + 2x+kw)·out + o] = acc[m, g·out + o],
/// and each threadgroup row writes two maximally contiguous runs (2·out floats
/// per output row, one per kh). Because one threadgroup holds every channel of
/// its rows' output pixels, the first stage's LayerNorm reduces inside the
/// same kernel (two-pass, like SAFIRENorm) instead of a second full pass.
///
/// Tiling. Register-blocked GEMM: one threadgroup computes a BM×N tile (full
/// row width, K = in channels), simd lanes share each A row address
/// (broadcast load) and read B as float4 vectors; accumulators live in
/// registers via generated unrolled code. M = C·4096 (stage 1) / C·16384
/// (stage 2) is always a multiple of BM, so no boundary predication is needed.
/// FP32 throughout: fp32 FMA accumulation, exact-erf GELU
/// (Abramowitz–Stegun 7.1.26) and FP32 LayerNorm statistics.
public enum SAFIREUpscale {
    private static let component = ["x", "y", "z", "w"]
    private static var cache: [String: MLXFast.MLXFastKernel] = [:]

    /// Builds the kernel body for one stage configuration. All sizes are baked
    /// into the source text (MLXFast template ints render as C++ template
    /// parameters; textual baking keeps the cache key distinct per config).
    /// tc: output columns per thread (4 or 8); one threadgroup covers
    /// 32·tc = N columns and bm/simds accumulator rows per thread.
    private static func source(name:String, hasLN:Bool, bm:Int, tg:Int, tc:Int,
                               k:Int, n:Int, side:Int, outc:Int, kunroll:Int) -> String {
        let simds=tg/32, tr=bm/simds, lanesPer=outc/tc, log2Lanes=Int(log2(Double(lanesPer)))
        precondition(n == 4*outc && 32*tc == n && k%kunroll == 0 && bm%simds == 0 && outc%tc == 0)
        var s = """
            uint tid = thread_position_in_threadgroup.x;
            uint ty = tid >> 5u, tx = tid & 31u;
            uint rowBase = threadgroup_position_in_grid.x * \(bm)u;
            const device float4* w4 = (const device float4*)weight;
            const device float4* b4 = (const device float4*)cbias;
            device float4* out4 = (device float4*)out;
            """
        for i in 0..<tr { for c in 0..<tc { s += "\nfloat a\(i)_\(c) = 0.0f;" } }
        // GEMM: out[m,n] = Σ_k x[m,k]·W[k,n]. Two inner-loop shapes, picked per
        // stage by measurement: kunroll=1 keeps one scalar A row load per owned
        // row per step (simd lanes share the address, so the load is a cheap
        // broadcast and register pressure stays low); kunroll=4 instead loads
        // each row once as a float4 and reuses it across four k steps, which
        // trades registers for a better FMA:load ratio.
        if kunroll == 1 {
            s += "\nfor (uint k = 0u; k < \(k)u; k++) {\n"
            if tc == 8 {
                s += "float4 wlo = w4[k*\(n/4)u + (tx<<1u)];\nfloat4 whi = w4[k*\(n/4)u + (tx<<1u)+1u];\n"
            } else {
                s += "float4 w0 = w4[k*\(n/4)u + tx];\n"
            }
            for i in 0..<tr {
                s += "float x\(i) = x[(rowBase + ty + \(simds)u*\(i)u)*\(k)u + k];\n"
                for c in 0..<tc {
                    let w = tc == 8 ? (c < 4 ? "wlo.\(component[c])" : "whi.\(component[c-4])")
                                    : "w0.\(component[c])"
                    s += "a\(i)_\(c) = fma(x\(i), \(w), a\(i)_\(c));\n"
                }
            }
            s += "}\n"
        } else if kunroll == 4 {
            s += "\nconst device float4* x4 = (const device float4*)x;\n"
            s += "for (uint kk = 0u; kk < \(k/kunroll)u; kk++) {\n"
            for i in 0..<tr {
                s += "float4 q\(i) = x4[(rowBase + ty + \(simds)u*\(i)u)*\(k/4)u + kk];\n"
            }
            for j in 0..<kunroll {
                let kj = "(kk*\(kunroll)u+\(j)u)"
                if tc == 8 {
                    s += "float4 wlo\(j) = w4[\(kj)*\(n/4)u + (tx<<1u)];\nfloat4 whi\(j) = w4[\(kj)*\(n/4)u + (tx<<1u)+1u];\n"
                } else {
                    s += "float4 w0\(j) = w4[\(kj)*\(n/4)u + tx];\n"
                }
                for i in 0..<tr {
                    for c in 0..<tc {
                        let w = tc == 8 ? (c < 4 ? "wlo\(j).\(component[c])" : "whi\(j).\(component[c-4])")
                                        : "w0\(j).\(component[c])"
                        s += "a\(i)_\(c) = fma(q\(i).\(component[j]), \(w), a\(i)_\(c));\n"
                    }
                }
            }
            s += "}\n"
        } else if kunroll == 16 {
            // Threadgroup-staged W: each superstep loads a BK×N weight chunk
            // into threadgroup memory once (cooperative, float4 per thread) so
            // the simd-wide broadcast reads hit threadgroup storage instead of
            // amplifying L1 traffic. A stays direct (simd lanes share the row
            // address, a cheap broadcast).
            s += """
                threadgroup float ws[\(kunroll)*\(n)];
                threadgroup float4* ws4 = (threadgroup float4*)ws;
                for (uint ks = 0u; ks < \(k/kunroll)u; ks++) {
                {
                uint f = tid;
                for (uint l = 0u; l < \(kunroll*n/(4*tg))u; l++) {
                    uint r = f >> \(Int(log2(Double(n/4))))u, c4 = f & \(n/4-1)u;
                    ws4[r*\(n/4)u + c4] = w4[(ks*\(kunroll)u + r)*\(n/4)u + c4];
                    f += \(tg)u;
                }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint j = 0u; j < \(kunroll)u; j++) {
                """
            for i in 0..<tr {
                s += "float x\(i) = x[(rowBase + ty + \(simds)u*\(i)u)*\(k)u + ks*\(kunroll)u + j];\n"
            }
            if tc == 8 {
                s += "float4 wlo = *(threadgroup float4*)(ws + j*\(n)u + (tx<<3u));\nfloat4 whi = *(threadgroup float4*)(ws + j*\(n)u + (tx<<3u)+4u);\n"
            } else {
                s += "float4 w0 = *(threadgroup float4*)(ws + j*\(n)u + (tx<<2u));\n"
            }
            for i in 0..<tr {
                for c in 0..<tc {
                    let w = tc == 8 ? (c < 4 ? "wlo.\(component[c])" : "whi.\(component[c-4])")
                                    : "w0.\(component[c])"
                    s += "a\(i)_\(c) = fma(x\(i), \(w), a\(i)_\(c));\n"
                }
            }
            s += "}\nthreadgroup_barrier(mem_flags::mem_threadgroup);\n}\n"
        }
        // Conv bias (tiled 4× at load: column j carries bias[j % outc]).
        if tc == 8 {
            s += "float4 vblo = b4[(tx<<1u)];\nfloat4 vbhi = b4[(tx<<1u)+1u];\n"
        } else {
            s += "float4 vb = b4[tx];\n"
        }
        for i in 0..<tr { for c in 0..<tc {
            let b = tc == 8 ? (c < 4 ? "vblo.\(component[c])" : "vbhi.\(component[c-4])")
                            : "vb.\(component[c])"
            s += "float v\(i)_\(c) = a\(i)_\(c) + \(b);\n"
        } }
        // Fused LayerNorm+GELU (stage 1): two-pass statistics per
        // (GEMM row, output pixel) over the pixel's outc channels, held by the
        // lanesPer threads tx = g·lanesPer…(g+1)·lanesPer.
        if hasLN {
            s += """
                threadgroup float sh[\(bm*32)];
                threadgroup float shMean[\(bm*4)];
                threadgroup float shVar[\(bm*4)];
                """
            for i in 0..<tr {
                let terms = (0..<tc).map { "v\(i)_\($0)" }.joined(separator:"+")
                s += "sh[(ty + \(simds)u*\(i)u)*32u + tx] = \(terms);\n"
            }
            s += "threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            s += "if (tid < \(bm*4)u) { uint r = tid>>2u; uint g = tid&3u; float t = 0.0f;\n"
            for j in 0..<lanesPer { s += "t += sh[r*32u + g*\(lanesPer)u + \(j)u];\n" }
            s += "shMean[r*4u+g] = t/\(Float(outc)); }\n"
            s += "threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            for i in 0..<tr {
                let terms = (0..<tc).map { c in
                    let v = "v\(i)_\(c)"; return "(\(v)-mean\(i))*(\(v)-mean\(i))"
                }.joined(separator:"+")
                s += "float mean\(i) = shMean[(ty + \(simds)u*\(i)u)*4u + (tx>>\(log2Lanes)u)];\n"
                s += "sh[(ty + \(simds)u*\(i)u)*32u + tx] = \(terms);\n"
            }
            s += "threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            s += "if (tid < \(bm*4)u) { uint r = tid>>2u; uint g = tid&3u; float t = 0.0f;\n"
            for j in 0..<lanesPer { s += "t += sh[r*32u + g*\(lanesPer)u + \(j)u];\n" }
            s += "shVar[r*4u+g] = t/\(Float(outc)); }\n"
            s += "threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        }
        // Epilogue stores. Rows map to (b, y, x); columns to the (kh, kw, o)
        // pixel offset g = n/outc — the pixel shuffle computed inline.
        let shiftB = Int(log2(Double(side*side)))
        let shiftY = Int(log2(Double(side)))
        let outSide = 2*side
        let shiftOut = Int(log2(Double(outc)))
        let shiftB2 = 2*Int(log2(Double(outSide)))+shiftOut
        let shiftRow = Int(log2(Double(outSide)))+shiftOut
        let shiftO = Int(log2(Double(tc)))
        if hasLN {
            s += "const device float4* lw4 = (const device float4*)lnw;\n"
            s += "const device float4* lb4 = (const device float4*)lnb;\n"
            s += "float4 lwlo = lw4[(tx&\(lanesPer-1)u)<<1u];\nfloat4 lwhi = lw4[((tx&\(lanesPer-1)u)<<1u)+1u];\n"
            s += "float4 lblo = lb4[(tx&\(lanesPer-1)u)<<1u];\nfloat4 lbhi = lb4[((tx&\(lanesPer-1)u)<<1u)+1u];\n"
        }
        for i in 0..<tr {
            s += """
                {
                uint r = rowBase + ty + \(simds)u*\(i)u;
                uint bb = r >> \(shiftB)u, yx = r & \(side*side-1)u;
                uint yy = yx >> \(shiftY)u, xx = yx & \(side-1)u;
                uint g = \(hasLN ? "tx>>\(log2Lanes)u" : "tx>>3u");
                uint kh = g >> 1u, kw = g & 1u;
                uint ob = (bb << \(shiftB2)u) + ((2u*yy + kh) << \(shiftRow)u) + ((2u*xx + kw) << \(shiftOut)u) + ((tx & \(lanesPer-1)u) << \(shiftO)u);
                """
            if hasLN {
                s += "float mean = shMean[((ty + \(simds)u*\(i)u)<<2u) + g];\n"
                s += "float inv = rsqrt(shVar[((ty + \(simds)u*\(i)u)<<2u) + g] + eps);\n"
            }
            for c in 0..<tc {
                let value: String
                if hasLN {
                    let lw = c < 4 ? "lwlo.\(component[c])" : "lwhi.\(component[c-4])"
                    let lb = c < 4 ? "lblo.\(component[c])" : "lbhi.\(component[c-4])"
                    value = "float v = (v\(i)_\(c) - mean)*inv*\(lw) + \(lb); y\(c) = \(SAFIRENorm.geluExpression);"
                } else {
                    value = "float v = v\(i)_\(c); y\(c) = \(SAFIRENorm.geluExpression);"
                }
                s += "float y\(c) = 0.0f; { \(value) }\n"
            }
            for q in 0..<tc/4 {
                let vals = (q*4..<q*4+4).map { "y\($0)" }.joined(separator:",")
                s += "out4[(ob>>2u)+\(q)u] = float4(\(vals));\n"
            }
            s += "}\n"
        }
        return s
    }

    private static func kernel(_ name:String, hasLN:Bool, bm:Int, tg:Int, tc:Int,
                               k:Int, n:Int, side:Int, outc:Int, kunroll:Int) -> MLXFast.MLXFastKernel {
        if let cached = cache[name] { return cached }
        let body=source(name:name,hasLN:hasLN,bm:bm,tg:tg,tc:tc,k:k,n:n,side:side,outc:outc,kunroll:kunroll)
        if let dump=ProcessInfo.processInfo.environment["WAIFU_SAFIRE_DUMP_SOURCE"] {
            try? body.write(toFile:"\(dump)/\(name).metal",atomically:true,encoding:.utf8)
        }
        let built = MLXFast.metalKernel(
            name:name, inputNames: hasLN ? ["x","weight","cbias","lnw","lnb","eps"]
                                         : ["x","weight","cbias"],
            outputNames:["out"],
            source:body,
            header:SAFIRENorm.erfHelper)
        cache[name]=built
        return built
    }

    /// Stage 1: [C,64,64,256] × W[256,256] + bias → pixel shuffle →
    /// [C,128,128,64] → LayerNorm(64)+GELU, one pass.
    public static func upscale1(_ x:MLXArray, weight:MLXArray, bias:MLXArray,
                                lnWeight:MLXArray, lnBias:MLXArray, eps:Float,
                                bm:Int=32, tg:Int=128, kunroll:Int=1) -> MLXArray {
        precondition(x.dtype == .float32 && x.ndim == 4 && x.dim(1) == 64 && x.dim(2) == 64 && x.dim(3) == 256)
        precondition(weight.shape == [256,256] && bias.shape == [256]
                     && lnWeight.shape == [64] && lnBias.shape == [64])
        let batch=x.dim(0), rows=batch*4096
        precondition(rows%bm == 0, "Row tiles must divide the batched pixel rows")
        return kernel("safire_up1_gemm_ln_\(bm)x\(tg)k\(kunroll)",hasLN:true,bm:bm,tg:tg,tc:8,k:256,n:256,side:64,outc:64,kunroll:kunroll)(
            [x,weight,bias,lnWeight,lnBias,MLXArray(Float(eps))],
            grid:(rows/bm*tg,1,1), threadGroup:(tg,1,1),
            outputShapes:[[batch,128,128,64]], outputDTypes:[.float32])[0]
    }

    /// Stage 2: [C,128,128,64] × W[64,128] + bias → pixel shuffle →
    /// [C,256,256,32] → GELU, one pass.
    public static func upscale2(_ x:MLXArray, weight:MLXArray, bias:MLXArray,
                                bm:Int=32, tg:Int=128, kunroll:Int=1) -> MLXArray {
        precondition(x.dtype == .float32 && x.ndim == 4 && x.dim(1) == 128 && x.dim(2) == 128 && x.dim(3) == 64)
        precondition(weight.shape == [64,128] && bias.shape == [128])
        let batch=x.dim(0), rows=batch*16384
        precondition(rows%bm == 0, "Row tiles must divide the batched pixel rows")
        return kernel("safire_up2_gemm_gelu_\(bm)x\(tg)k\(kunroll)",hasLN:false,bm:bm,tg:tg,tc:4,k:64,n:128,side:128,outc:32,kunroll:kunroll)(
            [x,weight,bias],
            grid:(rows/bm*tg,1,1), threadGroup:(tg,1,1),
            outputShapes:[[batch,256,256,32]], outputDTypes:[.float32])[0]
    }

    /// Scalar oracle for stage 1: x is [C][4096][256], weight [256][256] with
    /// bias tiled 4× ([256]); returns the [C][128][128][64] LN+GELU output.
    public static func reference1(_ x:[[[Float]]], weight:[[Float]], bias:[Float],
                                  lnWeight:[Float], lnBias:[Float], eps:Float) -> [[[[Float]]]] {
        func gelu(_ v:Float) -> Float {
            let ax=abs(v*Float(0.7071067811865476)), t=1/(1+Float(0.3275911)*ax)
            let poly=((((Float(1.061405429)*t - Float(1.453152027))*t) + Float(1.421413741))*t - Float(0.284496736))*t + Float(0.254829592)
            var e=1-poly*t*exp(-ax*ax); if v*Float(0.7071067811865476) < 0 { e = -e }
            return 0.5*v*(1+e)
        }
        let (side,outc,n,k)=(64,64,256,256)
        var out=[[[[Float]]]]()
        for b in 0..<x.count {
            var image=(0..<(2*side)).map { _ in (0..<(2*side)).map { _ in [Float](repeating:0,count:outc) } }
            for y in 0..<side { for xx in 0..<side {
                let row=y*side+xx
                var block=[Float](repeating:0,count:n)
                for j in 0..<n {
                    var sum=Float(0)
                    for kk in 0..<k { sum+=x[b][row][kk]*weight[kk][j] }
                    block[j]=sum+bias[j]
                }
                for g in 0..<4 {
                    let (kh,kw)=(g/2,g%2)
                    var vals=[Float](repeating:0,count:outc)
                    for o in 0..<outc { vals[o]=block[g*outc+o] }
                    let mean=vals.reduce(0,+)/Float(outc)
                    let variance=vals.reduce(0){$0+($1-mean)*($1-mean)}/Float(outc)
                    let inv=1/(variance+eps).squareRoot()
                    for o in 0..<outc {
                        image[2*y+kh][2*xx+kw][o]=gelu((vals[o]-mean)*inv*lnWeight[o]+lnBias[o])
                    }
                }
            } }
            out.append(image)
        }
        return out
    }

    /// Scalar oracle for stage 2: x is [C][16384][64], weight [64][128] with
    /// bias tiled 4× ([128]); returns the [C][256][256][32] GELU output.
    public static func reference2(_ x:[[[Float]]], weight:[[Float]], bias:[Float]) -> [[[[Float]]]] {
        func gelu(_ v:Float) -> Float {
            let ax=abs(v*Float(0.7071067811865476)), t=1/(1+Float(0.3275911)*ax)
            let poly=((((Float(1.061405429)*t - Float(1.453152027))*t) + Float(1.421413741))*t - Float(0.284496736))*t + Float(0.254829592)
            var e=1-poly*t*exp(-ax*ax); if v*Float(0.7071067811865476) < 0 { e = -e }
            return 0.5*v*(1+e)
        }
        let (side,outc,n,k)=(128,32,128,64)
        var out=[[[[Float]]]]()
        for b in 0..<x.count {
            var image=(0..<(2*side)).map { _ in (0..<(2*side)).map { _ in [Float](repeating:0,count:outc) } }
            for y in 0..<side { for xx in 0..<side {
                let row=y*side+xx
                var block=[Float](repeating:0,count:n)
                for j in 0..<n {
                    var sum=Float(0)
                    for kk in 0..<k { sum+=x[b][row][kk]*weight[kk][j] }
                    block[j]=sum+bias[j]
                }
                for g in 0..<4 {
                    let (kh,kw)=(g/2,g%2)
                    for o in 0..<outc {
                        image[2*y+kh][2*xx+kw][o]=gelu(block[g*outc+o])
                    }
                }
            } }
            out.append(image)
        }
        return out
    }
}

/// Fused relative-position bias outer-sum for the chunked global attention.
/// For query row i and flattened key j = ky*64 + kx the additive bias is
/// height[(h*rows+i)*64 + ky] + width[(h*rows+i)*64 + kx], where height/width
/// are the [1,12,rows/64,64,64] gathered-table einsum outputs. MLX's
/// broadcast-add writes this 50 MB tile at ~26 GB/s; this kernel is a coalesced
/// streaming write from the two L1-resident table rows.
public enum SAFIREBias {
    static let kernel = MLXFast.metalKernel(
        name: "safire_rel_bias_v1", inputNames: ["height", "width"],
        outputNames: ["out"], source: """
        uint gid = thread_position_in_grid.x;   // (h*rows + i)*4096 + j
        uint j = gid & 4095u;
        uint base = gid >> 12u;                 // h*rows + i
        uint tableRow = base << 6u;             // (h*rows + i)*64
        out[gid] = height[tableRow + (j >> 6u)] + width[tableRow + (j & 63u)];
        """)

    /// Outer-sum the per-query height/width tables into the dense
    /// [1,heads,rows,4096] additive-bias tile for one attention chunk.
    public static func outerSum(height: MLXArray, width: MLXArray, rows: Int, heads: Int = 12) -> MLXArray {
        let groups = rows/64
        precondition(height.shape == [1, heads, groups, 64, 64]
            && width.shape == [1, heads, groups, 64, 64], "Unexpected rel-position table shapes")
        return kernel([height, width],
            grid: (heads*rows*4096, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[1, heads, rows, 4096]], outputDTypes: [.float32])[0]
    }
}
