# WD SwinV2 direct ANE investigation

## Decision

The Metal implementation remains the production backend and default. This work adds an isolated direct-ANE prototype and reproducible probes. A complete block MIL graph compiles, but the M3 runtime rejects its evaluation with `0x1d`; the whole tagger is not running on direct ANE. The promising direct-ANE timings below cover individual projections, one attention window, and MLP-tail chains; they do not establish an end-to-end speedup. Core ML's CPU+ANE route did invoke ANE execution. Its batch-8 trial was slower and failed parity; the batch-1 trial was faster on this sample but still failed parity badly. No ANE or hybrid backend is selected by the application.

## What is implemented

- `WDTaggerANE` dynamically loads Apple's private `AppleNeuralEngine` framework, discovers the in-memory model/request/surface classes, compiles fixed MIL text with immutable BLOBFILE weights, evaluates over caller-owned IOSurfaces, and releases programs and surfaces. It is not referenced by the production tagger path.
- `waifu-wdtagger-ane-probe` provides fixed-checkpoint QKV, explicit SwinV2 attention, a configurable complete-block fusion attempt, MLP-fusion diagnostics, and selective W8A8 QKV experiments. Probe commands default to batch 1; larger batches remain explicit options for component sweeps. Inputs and outputs use FP16 IOSurfaces; model weights are compile-time BLOBFILE constants where applicable.
- The Swin attention graph performs cosine-normalized Q/K dot products, the checkpoint's learned query scale and relative-position bias, the shifted-window region penalty, MIL softmax, and attention-times-V inside one ANE program. It returns only the attended output, so the scores and probabilities do not round-trip through the host or GPU.
- QKV shape sweeps exercise the actual stage dimensions, including batch 1, 4, and 8 cases, row chunking, padding, compile-time weights, and dynamic-weight comparison.
- The production Metal worker and its routing remain the existing GPU reference and are unchanged by this prototype.

The bridge uses `_ANEInMemoryModelDescriptor`, `_ANEInMemoryModel`, `_ANERequest`, and `_ANEIOSurfaceObject`, with the in-memory model's private compile/evaluate selectors. Orion's runtime and maderix/ANE's checked-in `bridge/ane_bridge.m` use this same in-memory model route. In the maderix checkout studied here, `_ANEClient` appears only in API-introspection code and `_ANECompiler` has no operational bridge implementation, so there was no working client/compiler bridge to adapt. The mutable-weight training path is excluded because this tagger uses fixed inference weights. Programs are cached within the process by SHA-256 of MIL plus weight paths, offsets, and contents; the LRU retains at most two loaded programs. A same-process repeat of the masked stage-1 attention graph hit this cache in **0.156 ms**. The cache is not persisted across launches.

## Device and GPU reference

Measurements were taken on `Mac15,12` (base M3), macOS 26.5.1 (25F80), with the prepared checkpoint SHA-256 `34fdaae4d72640985f985bb447db27c227288571122ff40cb773f1299074d550`. No M4 device was available, so these results do not predict M4 behavior.

The unchanged GPU-only worker processed 64 synthetic frames at batch 8 in three warm sweeps: **16.13 images/s**, with mean measured batch latency **0.500 s** and peak worker physical footprint **662,455,304 bytes**. Prediction and IPC were included; image-file preparation was excluded. Output matched the saved GPU reference exactly. This is the system-level baseline, not a natural-image accuracy evaluation.

## Direct ANE FP16 measurements

| Probe | Stage/batch | ANE median | Metal comparison | Maximum absolute error |
|---|---:|---:|---:|---:|
| QKV projection, compile-time weights | 1 / 8 | 2.693 ms (8 row-chunk calls) | raw Metal GEMM 6.273 ms; production fused QKV 10.315 ms | 0.00391 |
| QKV projection, compile-time weights | 2 / 8 | 1.070 ms (2 row-chunk calls) | raw Metal GEMM 4.567 ms; production fused QKV 6.087 ms | 0.00781 |
| QKV projection, padded rows | 3 / 1 | 0.293 ms | raw Metal GEMM 1.355 ms; production fused QKV 1.475 ms | 0.00195 |
| QKV projection | 3 / 4 | 0.597 ms | raw Metal GEMM 4.559 ms; production fused QKV 5.146 ms | 0.00195 |
| QKV projection | 3 / 8 | 0.970 ms | raw Metal GEMM 4.678 ms; production fused QKV 5.315 ms | 0.00195 |
| QKV projection | 4 / 8 | 1.044 ms | raw Metal GEMM 7.093 ms; production fused QKV 7.370 ms | 0.00781 |

These are warm projection-component timings. The ANE measurement excludes IOSurface input writes, output reads, compilation, and the remaining block operations. The production Metal QKV comparison includes cosine normalization and learned query scaling, so that column is deliberately labeled non-like-for-like. Stage 1 and 2 use multiple ANE calls because a single full-batch graph did not fit the tested compiler shape. Stage 3 batch 1 needed padding to 800 rows; the unpadded execution was incorrect. Thus, these results are building-block evidence, not full-stage or end-to-end performance.

The one-window attention graph passed on shifted windows at the stage-1 top, right, bottom, and corner boundaries, and at representative stage-2 and stage-3 shifted windows. It also passed on an unshifted stage-4 window:

| Window | ANE median | Metal Flash32 median | Maximum absolute output error |
|---|---:|---:|---:|
| Stage 1 shifted, window 0 | 0.140 ms | 0.247 ms | 0.00195 |
| Stage 1 shifted, window 7 | 0.114 ms | 0.439 ms | 0.00122 |
| Stage 1 shifted, window 56 | 0.092 ms | 0.432 ms | 0.00624 |
| Stage 1 shifted, window 63 | 0.093 ms | 0.297 ms | 0.00079 |
| Stage 2 shifted, window 15 | 0.147 ms | 0.473 ms | 0.00061 |
| Stage 3 shifted, window 3 | 0.251 ms | 0.457 ms | 0.00054 |
| Stage 4 unshifted, window 0 | 0.537 ms | 0.817 ms | 0.00079 |

Each attention run tests a single fixed 196-token window. Its QKV values are generated synthetic activations passed through the production cosine-normalization kernel, and its learned query scale and relative bias come from the prepared checkpoint. Region masking uses the Swin shifted-window rule. Bias and mask are baked into a per-window FP16 blob. Probability-output taps and a score-only output graph were rejected by the runtime with error `0x1d`; the successful graph keeps softmax internal and compares the final attended output. This does not yet prove whole-stage speed or correctness on image-derived activations.

## Quantization and fusion experiments

The maderix synthetic convolution benchmark was compiled and run on this M3. Its five FP16/W8A8 latency ratios were 1.00x, 1.00x, 1.08x, 0.99x, and 1.01x. It uses random synthetic weights, fixed scalar scales, and does not establish WD checkpoint accuracy. Its console banner says “M4” even on this `Mac15,12`; the hardware identity above comes from the running machine, not that banner.

The WD-specific selective QKV W8A8 probe uses real checkpoint weights, symmetric per-tensor INT8 weights, FP16 bias, and a synthetic per-tensor activation scale. After switching to one immutable BLOBFILE per tensor, all outputs were finite. Stage 3 errors versus FP16 Metal were similar at batch 1/4/8 (maximum **0.0498–0.0509**, mean **0.0088**) and the warm QKV component ran in **0.325/0.616/1.012 ms** versus **1.843/4.591/6.973 ms** for the Metal FP16 projection probe. Batch 1 pads 16 rows to 800 for ANE. This is a promising component result only: activations are synthetic, per-tensor calibration is not image-derived, and no tag-probability or end-to-end W8A8 accuracy has been measured.

The MLP fusion diagnostics exposed and fixed a BLOBFILE issue: packing several different tensor sizes into one file gave incorrect values. Binding each immutable checkpoint tensor as its own BLOBFILE reduced the stage-3 batch-8 `fc1` maximum error from **428.41** to **0.00391**. With that correction, a one-tail graph (`fc1 → tanh-GELU → fc2 → LayerNorm/residual`) matched its equivalent Metal chain within **0.000855** maximum error. A two-tail graph held both MLP outputs inside one ANE program, with no host/GPU transfer between them, and matched the equivalent simplified Metal chain within **0.00110** maximum error:

| Stage-3 MLP-tail batch | ANE two-tail median | Metal two-tail median | Maximum absolute error |
|---:|---:|---:|---:|
| 1 (16 rows padded) | 2.347 ms | 6.787 ms | 0.000977 |
| 4 | 7.421 ms | 13.834 ms | 0.001099 |
| 8 | 15.113 ms | 26.236 ms | 0.001099 |

This does not represent consecutive complete Swin blocks: the attention and pre-MLP portion of each block are omitted, and the reference repeats only the MLP-tail/residual/LayerNorm chain. It validates multi-subgraph fusion and internal activation residency, not full-block or stage execution.

A separate configurable graph contains a complete one-window block: QKV projection, cosine Q/K normalization, learned query scale, relative bias, optional shifted-region mask, on-ANE softmax and value product, output projection, both residual normalizations, and the two-layer tanh-GELU MLP. It compiles and loads, but evaluation fails with the private runtime error `0x1d` on the tested M3 for stage-4 block 0 and stage-3 blocks 0 and 1 (including a shifted stage-3 window). No numerical output or timing was produced for those full-block attempts. Standalone masked attention and two-tail MLP fusion continue to pass; the combined graph remains unaccepted.

The maderix SRAM probe was run unchanged as a synthetic 1×1-convolution sweep. It reported 0.2 MB total buffers / 0.195 ms at 256 channels, 8.5 MB / 0.315 ms at 2,048 channels, 51.2 MB / 1.000 ms at 5,120 channels, 73.5 MB / 1.343 ms at 6,144 channels, and 129 MB / 4.329 ms at 8,192 channels. The measurements are noisy and show no single sharp capacity cliff; they do not measure WD stage memory or identify a usable SRAM limit.

## Memory, caching, and zero-copy findings

The prototype allocates flat IOSurfaces and hands them to `_ANEIOSurfaceObject`. Its public probe helpers explicitly copy FP16 bytes into and out of those surfaces with `memcpy` under `IOSurfaceLock`. There is no Metal texture/buffer import, shared GPU/ANE tensor view, or demonstrated GPU-to-ANE zero-copy path. IOSurface-backed ANE I/O is verified; GPU↔ANE zero-copy is not.

The maderix README advertises GPU/ANE IOSurface-sharing examples, but the named `gpu_ane_share.m` and `gpu_prefill_ane_decode.m` sources are absent from the checkout studied here. The README claim is therefore not treated as code evidence. The maderix INT8 and SRAM probes are synthetic microbenchmarks and are reported separately from WD measurements.

Compile-time fixed BLOBFILE weights reduce dynamic IOSurface input count and memory in the successful QKV probe. At stage 3 batch 8, binding weights reduced reported IOSurface allocation from 77.1 MB to 38.5 MB and the median projection time from 1.345 ms to 0.970 ms relative to dynamic weights. The two-tail experiment demonstrates activation residency inside a graph, but there is no complete-block or stage activation pool, no cache persisted across launches, and no end-to-end ANE schedule.

## Core ML portability and the ANE question

The repository already has a separate Core ML WD Tagger package. The released app package is fixed at batch 8, while `tools/convert-wd-tagger.py --batch-size 1` can build an isolated single-image graph for a trial; neither package has a flexible batch axis. The converter rebuilds the model from the same checkpoint in timm/PyTorch, exports with `torch.export`, then converts to an FP16 Core ML ML Program. It installs the Swin attention path as rank-four SDPA. This is a graph-and-weights port; it does not convert the tuned Swift/MLX implementation or carry over its hand-written Metal kernels and M3 tile choices. The high-level SwinV2 math, checkpoint weights, static batch shapes, and selected fusions can be represented in Core ML, but the Core ML compiler chooses its own kernels and operation placement. Apple provides direct PyTorch-to-Core-ML conversion and the MIL representation for this kind of graph; see [Core ML Tools' PyTorch conversion guide](https://apple.github.io/coremltools/docs-guides/source/convert-pytorch.html) and [MIL guide](https://apple.github.io/coremltools/docs-guides/source/model-intermediate-language.html).

The historical WD comparison loaded Core ML with `computeUnits = .cpuAndGPU`, which explicitly rules out the Neural Engine. It measured 14.31/14.38 images/s for Core ML versus 15.45/15.55 images/s for the then-current fused native MLX path. A later 3-sweep, batch-8 M3 comparison allowed only CPU+ANE: native measured **15.66 images/s** and Core ML measured **14.62 images/s** (Core ML throughput **6.65% lower**). Those warm rates exclude the multi-minute Core ML/ANE model-load compilation. The result does not show a speed benefit.

The parity check exposed a more serious problem. Core ML's output is an `MLMultiArray` with row stride 10,864 for a logical `[8, 10,861]` tensor; the comparison code reads using the reported strides through `withUnsafeBytes`. On the same 64 synthetic RGB inputs, CPU+ANE differed from native by **0.96944 maximum / 0.002028 mean absolute probability error**, with **697 non-rating-tag threshold flips at 0.35**, **21 rating-argmax changes**, and a minimum top-15 overlap of **2/15**. Replacing raw pointer reads and writes with Core ML's managed byte-access APIs produced the exact same probability files, ruling out that access method as the cause. A separate one-sweep diagnostic showed CPU+GPU Core ML much closer to native (**0.02256 max / 7.46×10⁻⁶ mean**, 5 non-rating-tag threshold flips, no rating-argmax changes, minimum top-15 overlap 14/15). The published Core ML package's own CPU+GPU comparison against its PyTorch reference passed with 0.00305 maximum error and 15/15 top-15 overlap; that validates the package in that tested configuration, but does not explain the CPU+ANE mismatch. Both implementations use FP16 model weights, but their activation/accumulation and operator lowering paths differ. This error is too large to attribute to ordinary FP16 rounding; its exact cause has not been isolated.

Comparing the saved CPU+GPU and CPU+ANE Core ML probability files directly on the same 64 synthetic frames gives **0.96952 maximum / 0.002028 mean absolute error**, **700 non-rating tag threshold flips at 0.35**, **21 changed rating argmaxes**, and **6.16 mean top-15 overlap** (minimum **2/15**). So the two Core ML execution choices do not currently produce interchangeable tags. This comparison is synthetic-input evidence, not natural-image accuracy. A diagnostic package exposing attention, normalization, block, and stage outputs was built, and a Swift activation probe was added, but loading that tapped package with CPU+ANE did not return within two minutes while the ANE compiler service remained busy. No intermediate-tensor result was produced; the exact first divergent operator remains unknown.

The runtime trace for the CPU+ANE attempt passed through Core ML's `ANECompilerService`, `AneInferenceOperationImplUsingAnefAPIs`, and `_ANEClient` direct evaluation path. That verifies that Core ML invoked the Neural Engine for this model; it does not prove that every layer ran there. A separate `.all` cold-load check was stopped after more than two minutes waiting in ANE compilation and produced no benchmark or output. The paired tool supports `--coreml-units cpuAndNeuralEngine` and `--coreml-units all`. Apple documents that these compute-unit choices allow scheduling on the named units, not guaranteed placement; inspect per-operation placement with a Core ML performance report or `MLComputePlan`. `.all` allows CPU, GPU, and ANE, while CPU+ANE excludes the GPU. See [MLComputeUnits](https://developer.apple.com/documentation/coreml/mlcomputeunits) and [Core ML performance analysis](https://developer.apple.com/documentation/coreml/analyzing-a-core-ml-model-s-performance-in-xcode).

Core ML can use ANE. The batch-1 trial measured a warm throughput advantage on this 12-image sample, but its severe disagreement with native Metal means that speed is not an acceptable trade. The tuned MLX graph does not transfer as a tuned implementation: weights and high-level SwinV2 math can be reused or converted, while custom Metal kernels, M3 tile choices, and MLX compilation do not carry over. Another Core ML speed trial needs to isolate and fix the CPU+ANE numerical failure, then repeat warm/cold, end-to-end, and parity measurements before considering a switch. M4 was not available.

### Freely licensed artwork and single-image ANE check

A follow-up used 12 non-synthetic anime-style illustrations from Wikimedia Commons under CC0, public-domain, CC BY, or CC BY-SA licenses. Their page, author, license, and downloaded-rendition hash are recorded in [the sample report](wd-tagger-open-license-sample.json); downloaded image files stay under ignored `.build/` and are not added to the repository. Inputs use the app's resize, white-letterbox, and color conversion path.

The first comparison ran batch 8 over 64 rows, cycling through the 12 images. Native Metal and Core ML CPU+GPU had **4.54×10⁻⁶ mean / 0.00666 maximum probability error**, **zero tag-threshold flips at 0.35**, no rating changes, and top-15 overlap of **14/15 minimum, 14.92/15 mean**. A separately exported batch-1 Core ML model then made **one `[1, 3, 448, 448]` prediction call per image**, with no duplicated rows or tail padding. Over three measured sweeps, CPU+ANE reached **17.10 images/s** median versus **14.62 images/s** for native Metal, but its outputs were far outside the parity gate: **0.98374 maximum / 0.003528 mean absolute probability error**, **303 non-rating threshold flips at 0.35**, **5 rating changes**, and minimum top-15 overlap **2/15**. The batch-1 CPU+GPU run was slower than native (**13.73 vs 15.17 images/s**) and remained close in output (**0.00726 max / 4.67×10⁻⁶ mean**, zero tag flips, no rating changes, minimum top-15 overlap 14/15). The batch-1 ANE model's Core ML output has row stride 10,864 for logical `[1, 10,861]`; stride-aware extraction is used.

The batch-1 shape is reproducible with `tools/convert-wd-tagger.py --output .build/wd-ane-baseline/upstream-batch1 --batch-size 1`, followed by `xcrun swift tools/compile-coreml-model.swift .build/wd-ane-baseline/upstream-batch1/WDTagger.mlpackage .build/wd-ane-baseline/coreml-batch1/model.mlmodelc`. The paired validator accepts `--batch-size 1`; the runner defaults to batch 1 when `--coreml-units cpuAndNeuralEngine` is selected. Direct ANE probe defaults are also batch 1. Batch 8 remains the published app package's fixed shape and the production Metal backend is unchanged.

These are backend-agreement measurements, not tagger accuracy: the sample has no ground-truth tag annotations. The batch-1 CPU+ANE route is faster in this warm three-sweep sample, but the severe output mismatch rejects it. Per-image ANE execution therefore does not make the current backend interchangeable with native Metal.

## Constraints and next implementation gate

Private APIs are unsupported and isolated in the optional probe target. If those APIs change or are unavailable, the current application still uses its existing Metal backend. There is no production `ane` or `ane-hybrid` selector, no full-model direct-ANE implementation, no stage-level hybrid benchmark, no ground-truth natural-image accuracy measurement, and no M4 result. Neither the direct-ANE nor Core ML CPU+ANE evidence is sufficient to change the backend default. The CPU+ANE numerical failure still needs an intermediate-activation diagnosis before another parity-gated speed trial.

The next gate is to isolate the first CPU+ANE activation divergence, then make a complete direct-ANE Swin block execute and match Metal before measuring stage execution and GPU↔ANE transfers. W8A8 still needs image-derived calibration plus tag-probability and end-to-end agreement before production consideration. The production Metal path remains the selected backend.

## References

- [Orion](https://github.com/mechramc/Orion) — MIT-licensed direct in-memory ANE runtime, MIL compiler frontend, BLOBFILE handling, and empirically recorded constraints.
- [maderix/ANE](https://github.com/maderix/ANE) — MIT-licensed research project with in-memory MIL examples, transformer generators, synthetic SRAM/INT8 probes, and private-API experiments.
- [Machine-readable results](wd-tagger-ane-results.json)
