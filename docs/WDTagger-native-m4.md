# WD Tagger M4 confirmation and further tuning — 2026-10-16

Start with the [performance matrix](../README.md#wd-tagger-performance-matrix).

## Decision

Keep the released native defaults: batch 8, fused QKV, dense attention bias,
32×64×32 GEMM, 448 MiB scheduling budget and 64 MiB allocator cache.
Further trials did **not establish an additional speed gain**. The current native
backend beat production Core ML on the M4 as well as the earlier M3 measurements.
This establishes the best confirmed configuration in these tests, not an absolute
hardware performance ceiling.

## M4 results

Mac mini, Apple M4, 16 GB, macOS 26.4.1. The same retained 64-image corpus, batch 8
and 448×448 inputs were used. Core ML is the production CPU + GPU model, compiled
on the target Mac. Prediction timers include GPU completion and exclude file I/O,
image preparation, byte-input construction and bulk probability extraction.

| Separate fresh process, three sweeps | Prediction fps | Peak process MB |
|---|---:|---:|
| Core ML, opening / closing | 19.526 / 20.950 | 675.3 / 685.5 |
| **Current native, opening / closing** | **23.341 / 23.355** | **671.8 / 671.4** |
| Compact bias, 448 / 64 MiB | 23.172 | 656.0 |
| Compact bias, 416 / 64 MiB | 23.580 | 639.1 |
| Dense bias, 416 / 32 MiB | 22.951 | 641.7 |
| Smaller command-buffer submission threshold | 21.976 | 686.7 |
| GEMM 64×64×32 | 23.186 | 670.6 |
| GEMM 32×128×32 | 22.687 | 671.4 |
| GEMM 64×128×32 | 22.534 | 672.4 |
| Relaxed synchronization GEMM | 23.534 | 668.6 |

MB means decimal MB. Budgets are soft allocator/scheduling controls, not memory
caps. These are benchmark-process footprints, not combined app and worker memory.
The native controls were stable, while Core ML varied by 7.3%; interpreting small
candidate speed differences from this sweep alone would be unreliable.

Three adjacent comparisons subsequently rotated all six orders of candidate,
current native and Core ML, covering 24 batch triads in each fresh process:

| Candidate | Current native fps | Candidate fps | Core ML fps | Candidate change vs current | Batches faster than current |
|---|---:|---:|---:|---:|---:|
| Compact lookup, repeat 1 | 23.337 | 23.430 | 21.029 | +0.40% | 13 / 24 |
| Compact lookup, repeat 2 | 23.853 | 23.873 | 21.381 | +0.08% | 6 / 24 |
| Compact lookup + relaxed GEMM | 23.683 | 23.689 | 21.111 | +0.02% | 15 / 24 |

The **released current kernel was 10.98–12.18% faster than Core ML** in these
interleaved comparisons. Candidate changes were too small and inconsistent to
establish a further speed win. Paired comparisons keep models resident and share
native matrices, so their memory must not be compared with standalone footprints.

## What changed in the experimental kernel

Compact attention bias stores 729 relative displacements instead of a dense
196×196 matrix per head. It removes about 28.34 MB of retained FP16 bias data.
The new optional implementation computes key offsets once per threadgroup using
a 392-byte shared lookup table, instead of recalculating division and remainder
for every score fragment. It preserves the exact stored FP16 bias values.

Both the original and revised compact implementations passed 139 kernel checks
on the M4. Every standalone candidate and all adjacent native predictions were
byte-identical to the current native baseline on all 64 inputs. The M4 baseline
also matched the validated M3 baseline byte for byte. Fidelity to the independent
FP32 reference is unchanged; this does not measure human-labeled tagging accuracy.

Two M3 adjacent checks guarded against selecting an M4-only result:

| M3 candidate | Current fps | Candidate fps | Candidate change |
|---|---:|---:|---:|
| Compact lookup | 15.702 | 15.540 | −1.03% |
| Compact lookup + relaxed GEMM | 14.486 | 14.494 | +0.05% |

The lower absolute rate in the second M3 process also shows why these comparisons
use adjacent controls. No production default was changed. Compact lookup remains
an optional tuning path. The 416 MiB compact configuration's lower memory is an
exploratory result, not a confirmed replacement for the selected default.

A separate batch-8 GEMM sweep tested real stage dimensions with rotated tile
order. Its isolated timings did not translate into a confirmed whole-model gain.
The raw report includes the individual samples, including slower alternatives.

## Reproduction and limits

The remote test ran in an isolated directory. The installed app, model settings
and existing workloads were left alone. Only one GPU benchmark ran at a time on
each machine. M4 starts required normal thermal state, GPU temperature ≤55°C and
sampled P/E scaled CPU use ≤40%. M3 adjacent starts used ≤60°C and scaled CPU use
≤40%. Background macOS media services remained a source of variation. No new GPU
trace was recorded, and these are not sustained whole-video throughput tests.

Scripts:

- [Remote standalone matrix](../tools/run-wd-tagger-remote-matrix.mjs): populate an
  isolated root with `bin/`, `data/` and a Core ML model compiled on that machine;
  run `node run-wd-tagger-remote-matrix.mjs ROOT UNIQUE-LABEL`.
- [Remote adjacent comparison](../tools/run-wd-tagger-remote-contender.mjs): run
  with `--root ROOT --label UNIQUE-LABEL`; `--bundle bin-lut` selects a copied
  executable/resource bundle containing the revised Metal source. The baseline
  matrix must exist at `results/screen/`. `--tile` selects GEMM geometry and
  `--dense` disables compact bias. Both native variants use 448 / 64 MiB.
- [Local standalone sweep](../tools/run-wd-tagger-next-pass.mjs) and
  [local adjacent comparison](../tools/run-wd-tagger-contender.mjs) retain the
  original local corpus paths and guard against overwriting existing reports.
- The native validator's `--compare-current` mode pins the released dense/fused
  control and shares immutable matrices with the candidate. `--benchmark-gemm
  --batch 8` reproduces the isolated tile sweep.

See [raw timings, configurations, identities, telemetry and quality checks](wd-tagger-native-m4.json).
This historical pass introduced no new weights or app defaults. In the clean
repository, model weights are pinned to Hugging Face by
[`tools/wd-tagger-release.json`](../tools/wd-tagger-release.json); native workers are
built from source and included in [app releases](https://github.com/alastorid/WaifuFinder/releases/latest).
Historical checkpoint archives are not republished.
