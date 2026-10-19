# WD Tagger batch and runtime measurements

Generate synthetic smoke-test inputs and local comparison outputs with
`./tools/setup-wd-tagger-benchmarks.sh`; see [development setup](Development-setup.md).
Test media, tensors, predictions and compressed data archives are not committed.
These synthetic cases do not reproduce the historical corpus measured below.

Measured on October 15, 2026: Apple M3, 16 GB, macOS 26.5.1, Swift 6.3.2,
coremltools 9.0, PyTorch 2.7.0 and timm 1.0.30. The installed production model
is the rank-four SDPA release, revision `95615611e0cb63e081b6c9904ee9d4047c90e7d3`.
Full measurements and numerical comparisons: [JSON report](wd-tagger-batch-optimization.json).

## Batch size

Eight remains a reasonable default for throughput on this machine. Larger
batches did not improve throughput and used substantially more memory. Two is
the useful alternative when memory matters more than maximum throughput.

| Exported batch | Frames/second | Peak process footprint |
| --- | ---: | ---: |
| 1 | 13.88 | 428 MB |
| 2 | 13.38–14.43 | 450–464 MB |
| 4 | 13.25 | 513 MB |
| 8, production | 14.25–14.75 | 640–654 MB |
| 16 | 13.70 | 898 MB |
| 32 | 14.59 | 1,809 MB |

Ranges include independent processes started after cooling. The final paired
confirmation measured batch 2 at 13.38 fps / 464.0 MB and batch 8 at 14.64 fps /
654.1 MB: **29.1% lower peak footprint and 8.6% lower throughput**. In the earlier
cooled comparison, the throughput difference was about 2%; that smaller penalty
was not consistent across repeats. Batch 2 predictions matched production
exactly on all 64 validation inputs. Other batch shapes differed by at most
0.000023, with no threshold or rating changes relative to production.

These are separately exported, compiled models. The app's current model and
backend require `[8, 3, 448, 448]`. Reducing the queue size to two while keeping
that model still evaluates eight slots through padding. Adopting batch 2 would
require backend shape support and a model release update. Experimental packages
and compiled models remain under the ignored `.build/wd-optimization/` folder;
the production model and batch setting were retained.

## Fusion and compression

The production model already uses 24 native SDPA operations, packed QKV
projections, a BGR channel permutation folded into the first convolution, FP16
internal computation, and a reused native input buffer. The earlier SDPA change
established a roughly 22% throughput gain and 23% peak-memory reduction in its
paired measurements; see [that experiment](wd-tagger-sdpa-experiment.json).

| Additional probe, batch 8 | Frames/second | Peak footprint | Result |
| --- | ---: | ---: | --- |
| Reciprocal-square-root Q/K normalization | 14.75 | 640 MB | No established gain |
| Input scale/offset folded into first convolution | 14.02 | 641 MB | No established gain; predictions changed |
| INT8 convolution/linear weights, per channel | 13.19 | 1,945 MB | Slower, higher memory, greater prediction drift |
| INT4 convolution/linear weights, blocks of 32 | 13.69 | 1,909 MB | Slower, higher memory, substantial prediction drift |
| Core ML fast-prediction specialization | 14.50 | 646 MB | Within baseline variation |
| Low-precision GPU accumulation allowed | 14.51 | 653 MB | Within baseline variation; same predictions |

INT8 reduced package size from 287.1 to 189.3 MB, and INT4 to 146.4 MB. Those
smaller files did not translate into lower runtime memory. Apple also documents
that compression's runtime benefit depends on the representation and hardware:
[Core ML quantization performance](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-perf.html).
The exact allocations behind this model's increase were not traced.

On 48 natural video frames, maximum absolute probability error against the
independent FP32 reference was 0.00989 for production, 0.07407 for INT8 and
0.80793 for INT4. General-tag threshold disagreements with that reference were
5, 14 and 241 respectively. These compare numerical fidelity, not labeled
tagging accuracy. Compression candidates were not promoted.

Accelerate's strided byte-to-float conversion was also tested against the current
RGB packing loop. Both produced exactly the expected values. Median time per
1,000 packs was about 0.120 seconds for the current loop and 0.161 seconds for
Accelerate. The current loop was retained.

Inference accounted for roughly 97% of measured tagging time. Decode, resize,
packing and result extraction leave little headroom for a large overall gain.
Layer normalization depends on each input's statistics, so it cannot generally
be folded into fixed weights like an affine input transformation. Further kernel
fusion would need an execution trace to identify remaining materializations and
dispatch costs, followed by a validated implementation. Logical model operation
counts alone do not establish the kernels executed or a global performance peak.

## Follow-up tensor layout experiments

Two rewrites combine cyclic shift and window partition into indexed permutations.
One restores grid order after every attention call; the other keeps window order
through the block's residual, normalization and MLP. Both matched the original
Torch graph exactly on two random inputs and matched production probabilities
exactly on all 64 retained validation inputs.

| Model | First cooled run, fps / MB | Reverse-order cooled run, fps / MB |
| --- | ---: | ---: |
| Production | 13.10 / 640.7 | 13.10 / 654.3 |
| Persistent window order | 12.62 / 673.4 | 13.57 / 630.5 |
| Combined shift/partition gather | 13.08 / 697.0 | 12.35 / 654.1 |

These later comparisons support no reliable improvement. Persistent window order
reduced logical transposes from 100 to 52, reshapes from 239 to 147, and gathers
from 44 to 25, while retaining all 24 native attention operations. Those graph
reductions did not produce a consistent runtime gain. The production package
remains unchanged. Raw results: [layout report](wd-tagger-layout-optimization.json).

`tools/wd_tagger_layout.py` contains both rewrites.
`tools/inspect-wd-tagger-plan.py` records Core ML's estimated operation placement
and cost; those estimates are not executed shader timings.

The original thumbnail cache paths later became unavailable. The 64 exact native
input tensors, independent reference, and recorded candidate probabilities remain
available locally. The benchmark now accepts `--tensor-input inputs.bin` instead
of `--images JSON` for compute comparisons without depending on that cache. Such
runs exclude image decoding and must be compared using prediction-stage times.

## GPU tracing setup

Full Xcode 26.6 (17F113) is installed at `/Applications/Xcode.app`, with first-launch
setup complete. Metal Toolchain 17F109 is installed and `metal --version` reports
Apple Metal 32023.883. Tools use `DEVELOPER_DIR` explicitly; the global developer
directory selection was not changed. Xcode 26.6 supports this Mac's macOS 26.5.1;
see [Apple's system requirements](https://developer.apple.com/xcode/system-requirements/).

Available Instruments include Metal System Trace, Metal GPU Counters, Core ML,
Allocations and Time Profiler. A native tagger timeline successfully recorded at
`.build/wd-optimization/wd-baseline-timeline.trace` using a stable repository
gradient fixture. A second capture, `wd-baseline-counters.trace`, enables the
M3's Performance Limiters counter set and Shader Timeline in the **Metal
Application** instrument. Adding the separate Metal GPU Counters instrument
instead initially reported an unsupported counter profile; that failed capture
is diagnostic only. Instrumented timing is not used as benchmark throughput.
The configured capture contains sampled GPU counter values, but exporting
`metal-shader-profiler-intervals` returned zero rows. Thus this capture does not
provide per-shader timing, despite the enabled setting. The successful ordinary
timeline contains 1,033 target GPU encoder intervals, largely labeled as
MPSGraph inference rather than individual network operations. No kernel hotspot
or compute-versus-bandwidth conclusion is claimed from these exports.

To use the installed tools in a terminal:

```sh
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
xcrun xctrace list templates
xcrun metal --version
```

Use the Metal System Trace template, add Core ML, and launch the native shape
benchmark. For counters, choose **Metal Application → Counter Set → Performance
Limiters**, enable **Shader Timeline**, and save the template. The locally saved
template is `.build/wd-optimization/Waifu GPU.tracetemplate`.

Counter exports can expand a short trace into many gigabytes of temporary data.
Export one table at a time with ample free space, and prefer shader intervals to
dumping every sampled counter value. Two simultaneous exports exhausted disk
space during setup; their disposable export caches were removed and the Metal
toolchain installation then completed successfully. The original captures were
retained. Apple documents component management in
[additional Xcode components](https://developer.apple.com/documentation/xcode/downloading-and-installing-additional-xcode-components).

A minimal Metal kernel compiled and linked into a `.metallib` successfully with
the installed tools. Setup evidence and logs remain in `.build/xcode-install/`.

## MLX fused-kernel experiment

An isolated MLX 0.32.4 port uses fused LayerNorm and compiled SwinV2 blocks,
preserving shifted-window masks, positional bias, cosine normalization, patch
merge order, and all classifier heads. An eight-input FP32 smoke comparison
matched the independent Torch reference within 0.0000023, with identical top-15
heads and no general-tag threshold or rating differences.

MLX's full fused attention kernel does not support this model's 32-wide heads.
A second candidate pads Q/K/V heads with 32 zeros, forces the supported 64-wide
fused kernel, retains the original attention scale, and slices the output back
to 32. The zero extension preserves dot products in real arithmetic.

| Runtime, batch 8 | Prediction-only frames/second |
| --- | ---: |
| Core ML production, exact retained tensors | 14.80 |
| MLX FP16, compiled blocks, 32-wide heads | 9.74 |
| MLX FP16, compiled blocks, padded fused attention | 10.20 |

These runs began at normal thermal state and GPU temperatures of 51.7, 52.8 and
54.9 C respectively. Each used one warmup batch and three 64-frame sweeps,
without concurrent profiling/export work. MLX's end-of-run thermal state was not
recorded; the first sweeps were also slower (9.74 and 10.20 fps). Input transfers,
image preprocessing, file IO and initial compilation are excluded from the
prediction timings. Native tensor replay matched the previous Core ML
probabilities byte-for-byte.

The padded MLX process peaked at 2,689.6 MB physical footprint, sampled every
10 ms including model load/warmup, versus 686.4 MB for the native tensor replay.
MLX retained about 2.3 GB in its allocator cache at completion; the probe did not
tune that cache. Its allocator/RSS counters are separately recorded and must not
be confused with process physical footprint. The alternate runtime has not
demonstrated a speed or memory benefit.

Both FP16 candidates were checked on all 64 retained inputs. Neither is promoted;
their outputs differ from production. Detailed numerical comparisons and raw
timings: [MLX report](wd-tagger-mlx-optimization.json). The eight-input FP32 smoke
test does not establish full-corpus FP32 fidelity, and these numerical checks do
not measure labeled tagging accuracy.

`tools/experiment-wd-tagger-mlx.py --export-from ORIGINAL_WEIGHTS --weights
EXPERIMENT_WEIGHTS` exports fixed attention biases/masks using the existing Torch
converter. Run inference in an isolated environment with MLX and NumPy, passing
`--weights EXPERIMENT_WEIGHTS --inputs inputs.bin --report report.json
--probabilities probabilities.bin --precision float16 --batch 8 --frames 64
--repeats 3 --compile`. Add `--force-fused --pad-head-dim 64` for the padded probe.
The tool is experimental and does not install a model or change the app runtime.

Live session score vectors are separate from the benchmark process footprint.
Previously, 10,861 Float32 values plus a four-byte header used 43,448 bytes per
frame, about 434 MB per 10,000 frames before allocator overhead. Live storage now
zeros scores below 0.35, quantizes retained values to E4M3 FP8 and compresses sparse
index/value pairs with Apple's LZ4. See `TagResultMemory.md` for precision and
threshold semantics.

## Native Swift/Metal follow-up

The subsequent native implementation, fused Metal kernels, single profiling
session and cooled comparison are documented in
[`WDTagger-native-optimization.md`](WDTagger-native-optimization.md).
It measured a modest speed gain in one ordering with higher memory than Core ML;
the production backend remains Core ML.

## Validation and limits

- Corpus: 48 frames sampled across three local videos, plus 16 synthetic inputs
  including gradients, solid colors, random pixels and checkerboards.
- The independent reference uses original unfused FP32 weights and runtime BGR
  conversion. It consumes the exact tensors prepared by production Swift
  geometry, avoiding a second resize implementation in the comparison.
- Native optimized Swift benchmark: three timed 64-frame sweeps per cooled
  process, one prediction to warm up, reused buffers, CPU and GPU execution.
  Compiling models and generating references finished before timed comparisons.
- All cooled comparisons began and ended at normal thermal state. GPU starting
  temperatures were 50–55 C. An initial uninterrupted sequence entered elevated
  thermal state and is retained as diagnostic data, excluded from the table.
- These short warm measurements do not establish sustained throughput for a
  whole long video. Cooling and a nominal thermal state also do not remove all
  clock, power, or background-load variation; the confirmation shows that limit.
- Footprint is the benchmark process's physical memory, sampled every 10 ms and
  including model warmup. It excludes the app's other services and full session.
- The first regression run aborted in Core ML/Espresso's
  `EncodeMemoryBuffers`, with MPSGraph asserting
  `shape.count = 0 != strides.count = 2`. This is the same assertion signature
  documented in the earlier investigation. It ran alongside CPU reference work;
  that concurrency is context, not an established cause. An isolated rerun passed
  all **13 tagger and tagger-contract cases**, including independent concurrent
  model instances. The intermittent assertion remains unresolved; the rank-four
  encoding cannot be claimed to have conclusively fixed it.

## Reproducing

`tools/convert-wd-tagger.py --batch-size {1,2,4,8,16,32}` now exports isolated
shapes while preserving batch 8 as its production default. It still supports the
original conversion validation. `tools/experiment-wd-tagger-optimization.py`
builds the four additional graph/compression variants without installing them.

Compile `tools/wd-tagger-shape-benchmark.swift` with `-O`, the production
`WDTaggerBackend.swift`, `TaggerBackend.swift`, and `TagVocabulary.swift` sources,
and the AppKit, CoreML and CoreImage frameworks. Pass `--model PACKAGE
--compile-to OUTPUT.mlmodelc` in a separate process before benchmarking. Normal
benchmark arguments are `--model COMPILED --labels CSV --images JSON --report
JSON --repeats 3`; the images JSON is an ordered array of local file paths.

Use `--tensor-output inputs.bin` on a separate diagnostic run to retain prepared
inputs, and `--probability-output NAME-probabilities.bin` for each candidate.
Do not use the tensor-dump process's timing or footprint as a benchmark: retaining
all inputs changes its allocation profile. `tools/validate-wd-tagger-optimization.py`
compares those predictions with the independent reference and verifies cached
reference identity against input and original-weight hashes.

`tools/run-wd-tagger-optimization-benchmarks.py` runs the cooled sequence using
`macmon` for GPU temperature and the native executable's `--thermal` probe.
`tools/wd-tagger-packing-benchmark.swift` independently checks and times pixel
packing. Model weights, image lists, source tensors and predictions remain local.
