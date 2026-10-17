# cuOpt routing search on Metal

WaifuFinder ports the routing **2-opt candidate search, ranked reduction and move
execution** from NVIDIA cuOpt into Swift and Metal. It minimizes the sum of distances
between consecutive shown photos, with free endpoints and no return edge. This is
a bounded heuristic, not a guarantee of a globally optimal path or a port of the
whole cuOpt engine.

The [project page](https://alastorid.github.io/WaifuFinder/#sequence) explains the
idea with synthetic drawings. [Credits](Credits.md) identifies the upstream work
and gives the measured speedup its workload context. Grouping defaults **off**;
visual sequencing is independent and remains enabled by default.

## Pinned source audit

Audited upstream commit: `02bfe53163c3e8eba25827f548623a8ab8071f88`.

| Audited source | Findings and disposition |
| --- | --- |
| [`two_opt.cu`](https://github.com/NVIDIA/cuopt/blob/02bfe53163c3e8eba25827f548623a8ab8071f88/cpp/src/routing/local_search/two_opt.cu) | `evaluate_two_opt_cvrp_move` compares the replaced boundaries and reversed fragment cost; `find_two_opt_moves` searches candidates and reduces to ranked winners; `execute_two_opt_moves` applies a reversed fragment from a snapshot. These operations are adapted to symmetric distances and free endpoints in `CuOptTSP.swift` and `CuOptMetalTSP.swift`. Symmetry cancels the internal fragment cost. |
| [`sliding_tsp.cu`](https://github.com/NVIDIA/cuopt/blob/02bfe53163c3e8eba25827f548623a8ab8071f88/cpp/src/routing/local_search/sliding_tsp.cu) | Relocates/reverses windows up to 20 nodes, searches neighbor lists, and recycles non-overlapping moves. Audited as a possible later extension; not ported. |
| [`sliding_window.cu`](https://github.com/NVIDIA/cuopt/blob/02bfe53163c3e8eba25827f548623a8ab8071f88/cpp/src/routing/local_search/sliding_window.cu) | Permutation/window search with route constraints and a bounded search range. Not ported. |
| [`tsp_route.cuh`](https://github.com/NVIDIA/cuopt/blob/02bfe53163c3e8eba25827f548623a8ab8071f88/cpp/src/routing/route/tsp_route.cuh) | Device predecessor/successor representation using CUDA streams, RAFT spans and RMM vectors. Replaced with flat route buffers; upstream implementation not included. |

The audited files carry NVIDIA copyright and `SPDX-License-Identifier: Apache-2.0`.
The pinned root license is preserved verbatim in `ThirdParty/cuopt/LICENSE`, with
attribution and modification details in `ThirdParty/cuopt/NOTICE`. No root upstream
NOTICE is provided at this revision. The build includes both files in the app's
`ThirdPartyLicenses` resources on every architecture.

Only those Apache-licensed routing sources were inspected. A full checkout was
unnecessary and was avoided so unrelated vendored software could not enter this
repository. The port retains **no upstream binary or transitive dependency**. CUDA,
RAFT, RMM, Thrust, CUB, NVTX and the vehicle-routing solution/constraint types used
by upstream cannot be linked as this native Metal backend; their host/device
operations are replaced using Apple's system Metal, Accelerate and Vision
frameworks and Swift's standard library. No packages are added or downloaded at
build/runtime, and no GPL/LGPL component is used by this implementation or its
tests/benchmark.

## Execution and metric

1. Decode each shown photo's existing Vision feature print once. Missing/corrupt
   prints stay visible at the end of the forward sequence.
2. Accept raw Float32 embeddings only when their dimensions and originating request
   descriptors agree, and sampled Euclidean or squared Euclidean distances match Vision's native
   `FeaturePrintObservation.distance(to:)`. Other formats/metrics retain native
   Vision distance calculations. On the tested macOS SDK, revision 2 uses squared
   Euclidean distance; both CPU and Metal preserve that convention.
3. Construct a deterministic hierarchical CPU seed. Explore it, the original order,
   a rotated path and a perturbed interior as independent initial routes.
4. Small sets use the CPU reference. Larger sets calculate symmetric distances on
   Metal and perform parallel candidate evaluation and ranked reductions.
5. Apply one winning reversal per independent route, using separate input/output
   buffers. There are no simultaneous overlapping moves or stale candidate deltas.
   Recompute the candidates after each accepted snapshot.
6. Keep the best route and compare its complete cost using native Vision before
   publishing it. Reverse traverses that exact cached route backward.

The dense matrix limit is 4,096 photos (64 MiB). Larger sets, up to 16,384 photos,
use direct GPU edge distances with 64 candidate endpoints per position, two initial
routes and at most 32 iterations, without allocating a square matrix. Larger sets,
unavailable Metal, unsupported embeddings or a GPU failure retain the bounded CPU
seed/native fallback. Dense search uses at most 128 iterations. Each job owns its
buffers and checks cancellation between GPU commands. Sorting happens off the UI
thread; the inline ring remains visible until the current job finishes or is canceled.

These cutoffs were chosen for this first implementation, not imposed by cuOpt or
Metal. A 200,000-frame input is not rejected: it uses the CPU fallback. Sequencing
at that scale has not been benchmarked. A dense matrix would need roughly 149 GiB;
scalable GPU work should use bounded candidates, tiles and an explicit memory budget.

## Checks and benchmark

`./run-tests.sh --only 'frame arrangement'` includes solver permutation/objective
checks, small instances checked against brute-force open-path optima, real Vision
metric checks, CPU/Metal parity, bounded-memory candidate mode, invalid inputs,
cancellation and session/filter/reversal checks. All 38 targeted checks passed on
the available M3, including a 513-photo integration check. These are background compute tests
and do not operate the user's UI.

`./tools/benchmark-visual-sort.sh` builds an optimized standalone benchmark using
only Apple system frameworks. It compares the same four seeds and 16 search
iterations, including matrix construction, with 768-dimensional synthetic vectors
using Euclidean distances. The tests separately verify Vision's squared metric.
It excludes one-time Metal pipeline compilation and prints device, costs and metric
error. These are solver measurements, not total photo analysis/sorting latency.
Only the available Apple M3 can be measured here; M4 performance is unverified.

M3 measurements from the final implementation:

| Photos | CPU (ms) | Metal (ms) |
| ---: | ---: | ---: |
| 32 | 0.145 | 7.256 |
| 64 | 0.443 | 4.782 |
| 128 | 1.725 | 6.730 |
| 256 | 7.537 | 8.868 |
| 512 | 29.202 | 16.783 |
| 1,024 | 116.296 | 29.068 |

Both backends returned the same objective in these cases. Maximum absolute matrix
error was 0.000007629. The app uses a 512-photo crossover so small collections avoid
GPU dispatch overhead. Rerun the benchmark on other Apple Silicon hardware before
changing that threshold.
