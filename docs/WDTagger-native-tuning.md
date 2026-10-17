# Native WDTagger speed pass — 2026-10-15

For a clean clone, `./tools/setup-wd-tagger-benchmarks.sh` generates synthetic inputs
and comparison outputs, downloads pinned models, and builds the benchmark tools.
These smoke-test inputs do not reproduce the historical corpus measured below.
See [development setup](Development-setup.md).

**Updated defaults and memory:** see the [2026-10-16 memory/input pass](WDTagger-native-memory.md).
The measurements and defaults below describe the earlier fusion checkpoint.

Following checkpoint `9ba0516`, the useful change is **fused QKV projection,
cosine normalization and packing**. It is now the experimental native backend's
default. The native executable also defaults to a 768 MiB soft allocation budget.
**App cutover, 2026-10-16:** the Apple Silicon app now selects this native backend.
The performance measurements below precede integration and remain prediction-only.
See [the integration checkpoint](wd-tagger-native-integration.json) for packaging
and app-adapter validation. The [main README matrix](../README.md#wd-tagger-performance-matrix)
is the quick starting point for future optimization.

## Measured result

Two fresh processes compared the complete original native model, the fused
model and production Core ML on the same batch-8 tensors. Each process ran three
64-frame rounds. Native variants shared immutable weights and routing. All six
execution orders were balanced across adjacent batches, reducing clock and
background-load drift between implementations.

| Controlled run | Original native | Fused native | Core ML | Gain over original | Gain over Core ML |
|---|---:|---:|---:|---:|---:|
| First | 15.11 fps | 15.45 fps | 14.31 fps | 2.2% | 7.9% |
| Confirmation | 15.23 fps | 15.55 fps | 14.38 fps | 2.1% | 8.1% |

The fused model won 47 of 48 batches against original native execution. All six
rounds improved over the original. These comparisons retain both backends in one
process and therefore cannot compare process memory.

Separate cooled processes confirmed the speed gain and evaluated allocation
budgets. Peak memory includes load and warmup and is sampled every 10 ms.

| Standalone configuration | Prediction fps | Peak process MB |
|---|---:|---:|
| Original native, 1536 MiB budget | 15.17 | 1259.6 |
| Fused native, 1536 MiB budget | 15.53 | 1310.6 |
| Fused native, 1024 MiB budget | 15.49 | 1218.3 |
| **Fused native, 768 MiB budget — new default** | **15.50** | **949.9** |

The selected configuration is **2.18% faster with 24.58% lower peak process
memory** than the original standalone configuration in these runs. MLX peak
active allocation fell from 1060.3 to 870.9 MB. The budget applies backpressure;
it is not a hard process-memory cap. Cache budget remained 128 MiB in this table.
Core ML's separately measured process footprint was about 684 MB, so the native
backend still uses more process memory.

All timings cover prediction and GPU completion. Decode, input preparation and
bulk result extraction are excluded. The final controlled benchmark touches
Core ML's first output scalar inside its timer to ensure completion. M3 MacBook
Air, 16 GB, macOS 26.5.1, MLX Swift 0.31.2. Selected runs began and ended at normal
thermal state; starting conditions and per-batch thermal state are recorded.

Absolute rates vary with background work, clocks and memory pressure. Historical
Core ML reached 14.80 fps. These results establish a measured inference lead on
this machine in these comparisons; they do not establish a global performance
ceiling, sustained-video throughput or whole-app tagging speed.

## What changed

Previously each of 24 blocks wrote a projected `[tokens,3×channels]` tensor and
then launched a separate kernel to normalize and pack Q/K/V. The new Metal kernel
performs those operations in the GEMM epilogue and writes the attention layout
directly.

The selected kernel preserves the original 32×64×32 GEMM tile and FP16 operands
with FP32 accumulation. Each threadgroup projects two 32-channel heads. Only
512 bytes of FP32 partial reductions cross SIMD groups; projected values remain
in registers. V bypasses the normalization work. This removes 24 dispatches and
about 1.195 GB of **logical intermediate writes plus reads per batch of eight**.
Caches affect actual DRAM traffic, which was not measured here.

Fewer intermediate tensors did not automatically reduce the allocator's peak:
the 1536 MiB fused run peaked higher. The lower scheduling budget retained the
speed gain while reducing measured peak memory. No cause beyond those observed
allocation changes is claimed.

`WDNativeTagger` defaults to fused QKV. The validation executable defaults to
batch 8, 768 MiB memory budget and 128 MiB cache. Use `--unfused-qkv` to reproduce
the original path. Weight format remains `swinv2-native-metal32-v1`, so existing
natively prepared weights remain compatible. Runtime implementation is
`swinv2-native-metal32-fused-qkv-v2`.

## Numerical validation

- **138 native kernel checks passed**, covering the selected fusion and optional
  matrix layouts, partial row tiles, packing, normalization, routing and attention.
- Both controlled fused runs and all three standalone allocation budgets produced
  byte-identical fused predictions on all 64 inputs.
- Fusion changes FP32 reduction order inside normalization, so probabilities are
  not byte-identical to the original native model. Maximum difference was
  0.00355348; mean difference was 0.000001419.
- There were **zero additional general-tag threshold decisions and zero rating
  argmax changes** relative to the original native model on the 64-input corpus.
- Against the independent original FP32 reference, maximum error was 0.013938,
  mean 0.000004455, with two general-tag threshold differences and zero rating
  changes. The original native model also had two threshold differences.

The corpus comprises 48 natural frames and 16 synthetic inputs. Threshold is
0.35. These checks measure numerical fidelity, not accuracy against human labels.

## Trials that did not establish a speed gain

- Larger matrix tiles and the native MLX matrix implementation showed no clear
  additional gain in the initial matrix benchmark.
- SIMD-row LayerNorm's first apparent gain did not survive confirmation.
- A full-tile load/store fast path measured about 0.3% slower in direct comparison.
- A 64×64×16 layout with eight SIMD groups measured 12.6% slower despite reducing
  threadgroup memory. It remains disabled.
- The first QKV fusion used a narrower matrix tile and came out about 0.4% slower.
  Preserving the original matrix geometry produced the confirmed gain above.
- Compact attention bias preserves all stored values and removes 28.34 MB of
  parameter data. Its tested configuration did not establish a speed or total
  process-memory win. It remains optional.
- Reduced SIMD synchronization showed a small normalized difference between
  separate comparison sessions; an added benefit over the original strategy was
  not established. It remains optional.

Only the geometry-preserving fusion and selected allocation budget were promoted.
Original normalization, dense attention bias and original GEMM geometry remain
the defaults. No new GPU trace was recorded during this pass.

## Reproduction and evidence

```sh
tools/build-native-wdtagger.sh --check-kernels --check-gemm-tiles --report checks.json
node tools/run-wd-tagger-paired.mjs --tile 32x64x32 --fused-qkv --compare-original --label repeat
node tools/run-wd-tagger-native-tuning.mjs --fused-speed
node tools/run-wd-tagger-native-tuning.mjs --fused-memory
```

The orchestration scripts require `macmon` and the retained local weights and
input corpus under `.build/wd-native/` and `.build/wd-optimization/`. Node launches
processes and analyzes files. Preparation and inference remain native Swift,
C++ and Metal with no Python runtime.

Raw timing samples, decisions, numerical checks, identities and source hashes are
recorded in [`wd-tagger-native-tuning.json`](wd-tagger-native-tuning.json).

## Published app resources

The prepared weights, matching labels, manifest, validation, license and attribution
are published on the Hugging Face Hub as
[alastorid/WD-SwinV2-Tagger-v3-Swift-MLX](https://huggingface.co/alastorid/WD-SwinV2-Tagger-v3-Swift-MLX).
The app descriptor [tools/wd-tagger-release.json](../tools/wd-tagger-release.json)
pins the Hub revision and every file by SHA-256.

The compiled MLX Metal kernel library is a build artifact rather than a weight, so it
is published separately as
[alastorid/WaifuRuntime-Metal](https://huggingface.co/alastorid/WaifuRuntime-Metal)
and verified by digest during the native build. The native worker and its Metal source
bundle ship inside the app; source and dependency pins are in GitHub.

Existing Core ML folders remain intact. Download the native model from Settings
→ Models; it installs as `wd-tagger-native-v2` and activates only after checksum
verification and a real native prediction. Cached tag results remain available;
previously tagged frames do not automatically rerun. Native tagging requires
the Apple Silicon build.

Maintainers can stage a release with `tools/stage-native-wdtagger-release.mjs`
using prepared weights, the matching upstream CSV and Apache-2.0 license, then
publish it with
`tools/upload-hf-bundles.py --only WD-SwinV2-Tagger-v3-Swift-MLX`, which reads the
Hub revision back and rewrites the descriptor.
The prior Core ML descriptor is retained as `tools/wd-tagger-coreml-release.json`.

The post-publication installer test passed against anonymous GitHub downloads:
all pinned files verified, native prediction and a real app-adapter tail succeeded,
repeated setup reused the verified release, and the previous Core ML folder was
preserved. The WD Tagger suites passed 13 cases; the model-management suite passed
8 cases with its unrelated opt-in PGTFormer network test skipped. The release also
includes `WaifuFinder-native-wdtagger-checkpoint.zip` built from checkpoint source
`98e03f9`, plus `SHA256SUMS` for the downloadable assets.
