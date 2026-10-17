# Native WDTagger optimization — 2026-10-15

The subsequent fused QKV pass and new native defaults are documented in
[`WDTagger-native-tuning.md`](WDTagger-native-tuning.md). The measurements below
describe the earlier checkpoint; add `--unfused-qkv` to reproduce its native path.

The experimental backend uses Swift, native MLX and specialized Metal kernels.
Weight preparation reads the original upstream safetensors directly in Swift.
There is no Python interpreter, bridge or Python dependency in this path.
The app still uses its existing Core ML model.

## Implementation

- Fixed 196-token, head-dimension-32 tiled attention: FP16 matrix operands,
  FP32 accumulation and online softmax, Q retained in registers, compact
  positional bias and shifted-window region IDs. No expanded attention matrix.
- One SIMD group combines Q/K cosine normalization, learned scaling, splitting
  and packing. Attention writes directly in projection input order.
- Persistent window layouts remove separate block partition/unpartition work.
  LayerNorm combines its affine transform, residual addition and next window
  permutation. Patch merging routes directly into the next stage's layout.
- Input normalization and 4×4 patch extraction share one dispatch, writing
  directly in window order for patch embedding.
- Aligned QKV, projection, MLP and reduction matrices use tiled FP16 GEMMs with
  FP32 accumulators. Bias and optional tanh-GELU run in the GEMM epilogue.
- Whole-model MLX compilation; positional bias and routing are prepared once.
  Initial embedding and the unaligned classifier still use native MLX addMM.
- Allocator memory/cache budgets expose a throughput mode and a smaller mode.
  These are soft MLX budgets, not hard process-memory caps.

Sources: `Runtime/Sources/WDTaggerMLX/`, with a native validation executable in
`Runtime/Sources/WaifuWDTaggerValidation/`. The Metal directory includes upstream
MLX Steel attribution and its MIT license. The earlier Python MLX prototype is
a separate historical experiment.

## Cooled comparison

MacBook Air M3, 16 GB, macOS 26.5.1; MLX Swift 0.31.2. Exact retained production
input tensors: 48 natural frames and 16 synthetic inputs. Each process received
one warmup batch and three timed 64-frame sweeps. Reported speed is 64 divided
by median **prediction-only** sweep time, including GPU completion, excluding
decode, input preparation and result extraction. Peak physical footprint is
sampled every 10 ms, including model load and warmup; MB below are decimal.

| Backend | Batch | Prediction frames/s | Peak MB |
|---|---:|---:|---:|
| Core ML | 8 | 14.02 | 683.6 |
| Native Swift/Metal | 8 | 14.73 | 1246.2 |
| Native Swift/Metal, smaller budgets | 2 | 14.18 | 619.3 |
| Core ML | 2 | 13.59 | 512.8 |

Native batch 8 used a 1536 MiB memory budget and 128 MiB cache; native batch 2
used 384/64 MiB. Both started and ended at normal thermal state. GPU starting
temperatures, in the table's run order, were 36.2, 42.5, 45.5 and 47.4 °C.

Native batch 8 was 5.1% faster than Core ML batch 8 in this ordering, with 82.3%
more process memory. Native batch 2 retained 96.3% of native batch 8 throughput
with about half its footprint. Native batch 2 used 9.4% less memory than Core ML
batch 8 at similar speed in this ordering. Core ML batch 2 remains the
lowest-memory measured option.

This is one cooled ordering, not evidence of a repeatable 5% win or sustained
long-video performance. Earlier Core ML tensor replay reached 14.80 frames/s.
Normal thermal state does not remove power, clock or background-load variation.
Keep Core ML as the app default given the native candidate's memory cost and
numerical differences. No performance ceiling has been established.

## Numerical checks

All **28 native kernel checks passed**, covering input packing, window routing,
patch merge, normalization, GEMM/GELU, QKV packing and shifted/unshifted attention.
Native batches 2 and 8 produced byte-identical probabilities on all 64 inputs.

Against the independent original FP32 reference, maximum absolute probability
error was **0.0137874**, mean **0.000004321**, with two general-tag decisions
crossing the 0.35 threshold and zero rating argmax changes. Against production
Core ML batch 8, maximum error was 0.0199852, with five general-tag threshold
differences and zero rating changes. These are numerical comparisons, not a
measurement of tagging quality against human labels.

## One GPU capture

After implementation and correctness checks, inference sources were frozen and
one Metal System Trace was recorded with the M3 Performance Limiters counter set
and shader timeline configured. Timing comparisons ran separately, after cooling.
No second profile was recorded.

The saved trace contains custom native pipeline labels and 154 target GPU encoder
intervals, totaling 1082.81 ms across a 1105.88 ms activity span. It includes
startup/warmup and one warmed batch. Those numbers are encoder intervals, not
per-shader costs or ALU utilization. The shader interval table and exported GPU
counter-profile table were empty; counter metadata alone cannot identify a
compute, bandwidth or occupancy bottleneck. Consequently this capture confirms
execution but cannot rank individual shader hotspots.

Local capture: `.build/wd-native/final-native-profile.trace`. XML exports and
analysis remain beside it. Raw comparison, accuracy, kernel checks, input/weight
identity and frozen source hashes are retained in
[`wd-tagger-native-optimization.json`](wd-tagger-native-optimization.json).

## Reproduce the native candidate

Requires Xcode/Metal tools, the existing native MLX metallib and original weights.
The helper verifies the metallib checksum before building; no Python setup runs.

```sh
tools/build-native-wdtagger.sh
Runtime/.build/release/waifu-wdtagger-validation --check-kernels --report checks.json
Runtime/.build/release/waifu-wdtagger-validation \
  --prepare-from ORIGINAL.safetensors --weights native.safetensors
Runtime/.build/release/waifu-wdtagger-validation \
  --weights native.safetensors --inputs inputs.bin \
  --batch 8 --frames 64 --repeats 3 --memory-mb 1536 --cache-mb 128 \
  --report native.json --probabilities native.bin
```

`inputs.bin` is contiguous Float32 RGB `[frames,3,448,448]` from the production
preprocessor. The benchmark is not a complete image tagging CLI. The retained
corpus, weights and full traces are ignored local artifacts, not checked in.
`tools/run-wd-tagger-native-benchmarks.mjs` orchestrates the four native processes
using those local artifacts and `macmon` cooling checks. Node only orchestrates
processes and analyzes files; inference and conversion remain native.

Build-helper execution, release compilation, kernel checks, full-corpus numerical
validation, linkage inspection and diff whitespace checks passed. The executable
links native MLX/framework dependencies without a Python runtime.
