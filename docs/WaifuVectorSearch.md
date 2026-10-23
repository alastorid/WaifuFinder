# WaifuVectorSearch

Native exact nearest-neighbor search, independent of SwiftUI. No new dependency,
CUDA runtime, remote service, or image upload is involved. Existing TSP sorting is
unchanged. Benchmarks deliberately stop at **10,000 candidates**.

## Representation and correctness

On the tested Apple M3 / installed Vision revision 2, feature prints expose 768
native-endian Float32 elements (3,072 bytes), and Vision distance agrees with
**squared Euclidean distance**. This is an observed, validated convention, not an
assumption made from compressed on-disk bytes. `FeaturePrintCodec` continues to
store the original Codable observation. Public extraction checks element type,
count, byte length, finite values, and originating request descriptor.

Each query validates sampled query-to-index distances against Vision. The reusable
GPU search API canonically rescores its compact shortlist on CPU and falls back to
a complete CPU scan when a conservative floating-point error band crowds the cutoff. Unknown
formats or a changed metric use the native reference. Accelerated shortlists are
reranked with Vision, returning its original Double distances. Near-tie bands
expand the shortlist conservatively and use identity as the secondary ordering.
Float16 storage is intentionally not used: at this scale, preserving the current
Float32 feature representation is more useful than sacrificing ranking precision.

## Pipeline

- A generation/revision keyed packed index appends unchanged frame identities once.
  Row identity is separate from vector position. Odd dimensions are zero-padded to
  64-byte row strides so identical rows receive identical Accelerate reduction order. Changed persisted payloads invalidate
  decoded observations. Missing features use existing analysis renditions.
- Accelerate computes query-to-row distance with `vDSP_distancesq`. A bounded max
  heap selects K; the app requests 19 rather than sorting all source candidates.
- Metal has two independent distance strategies: a SIMD group per row and a scalar
  thread per row with float4 accumulation. Neither allocates an N-by-N matrix.
- GPU selection sorts only bounded 512-row tiles in threadgroup memory, emits K
  winners per tile, and merges those compact lists. Identity ranks break ties.
- Device, command queue, and pipelines persist. Immutable vector/rank uploads are
  keyed by index identity/revision. Scratch leases are exclusive to an in-flight
  command and returned only after completion, including cancellation. Two idle
  leases are retained. GPU failures preserve CPU search.
- Progressive search shows available indexed candidates before generating missing
  features. Existing follow-best and two-publications-per-second UI rules remain
  in the Orb model. Source/query tickets reject stale work after every suspension.

The packed index has a 256 MiB vector budget. GPU memory is linear in N×D plus
row distances, identity ranks and tile winners. Batch API queries reuse the index;
the current implementation submits each query separately rather than claiming a
fused matrix-multiply batch kernel.

## Research and algorithm choice

The [cuVS brute-force guide](https://docs.nvidia.com/cuvs/user-guide/api-guides/indexing-guide/brute-force)
describes exhaustive distance scans plus Top-K. Its [K-selection guide](https://docs.nvidia.com/cuvs/user-guide/api-guides/other-ap-is/k-selection)
and [memory guide](https://docs.nvidia.com/cuvs/user-guide/api-guides/core-types/memory-management)
reinforce retaining small outputs and reusing resident resources. This implementation
uses original Swift/Metal code, without copying CUDA kernels or adding cuVS components.

[IVF-Flat](https://docs.nvidia.com/cuvs/user-guide/api-guides/indexing-guide/ivf-flat)
trains centroids and searches selected partitions; [CAGRA](https://docs.nvidia.com/cuvs/user-guide/api-guides/indexing-guide/cagra)
constructs/prunes a neighbor graph and traverses a bounded candidate set. HNSW
trades graph construction/memory for fast approximate CPU traversal. These add
build cost and a recall tradeoff. Exact scans below one millisecond at the requested
10k size do not justify adding any of them yet. No ANN recall claim is made.

## Measured baseline

Run `./tools/benchmark-vector-search.sh` for an optimized (`-O`) background benchmark.
The initial run used Apple M3, 8 CPU cores, 16 GB unified memory, D=768, K=19,
21 timed repetitions per size. Measurements below are median / p95 milliseconds.

| Stage | 1,000 candidates | 10,000 candidates |
|---|---:|---:|
| Vision distance loop + Top-K, repeated native corpus | 0.162 / 0.172 | 1.497 / 1.546 |
| Accelerate exact + bounded Top-K | 0.055 / 0.058 | 0.596 / 0.768 |
| Metal SIMD + GPU Top-K, warm end-to-end | 0.618 / 0.628 | 1.210 / 1.685 |
| Metal scalar float4 + GPU Top-K, warm end-to-end | 0.576 / 0.741 | 1.542 / 1.688 |
| Metal distances + CPU Top-K, warm end-to-end | 0.274 / 0.422 | 0.692 / 0.742 |
| GPU distance execution only | 0.050 / 0.054 | 0.423 / 0.745 |
| GPU selection execution only | 0.242 / 0.242 | 0.394 / 0.417 |

Index construction was 3.55 / 33.66 ms for 1k / 10k. First GPU search, including
upload/staging but excluding pipeline compilation and index build, was 1.18 / 4.75 ms.
At 10k the packed vectors occupy 30,720,000 bytes; resident GPU buffers and leased
scratch totaled 30,807,860 bytes. These are explicit buffer sizes, not process RSS.

48 generated native still fixtures were encoded by Vision (median 3.37 ms, p95
3.80 ms per image); encoding/decompressing 48 cached observations took 5.17 / 1.97 ms.
The scaling corpus repeats these real Vision vectors with deterministic small
perturbations. It is not 10k independently captured photographs. The Vision-loop
baseline repeats the original native observations and therefore measures call/scan
cost rather than the exact perturbed scaling corpus. An independent unmodified
48-observation corpus checked all 2,304 pairs: worst CPU distance error 0.0.
All 48 queries matched native Vision Top-19 identities on Metal (Recall@19 = 1.0
on that corpus). Warm scaling queries matched exact CPU Top-19 identities.

## Planner and limits

CPU is the default on unprofiled workloads: the measured M3 workload favors CPU
at both requested sizes. Explicit CPU/Metal backends support reproducible profiling.
The planner can use a resident GPU only after at least three measured CPU and GPU
queries for the same shape show a median GPU advantage of at least 15%; cold uploads
are excluded from the warm GPU profile. No crossover is extrapolated to 100k or
larger datasets. No performance claim is made for M4 without measurements there.

## Integrated Orb timing

`./tools/benchmark-vector-search.sh --orb` also compiles and benchmarks the actual
Orb search service with production sources. On a later optimized M3 run:

| Complete cached-feature search | 1,000 | 10,000 |
|---|---:|---:|
| Prior warm Vision loop + full sort, median / p95 ms | 0.340 / 0.452 | 3.322 / 4.306 |
| New integrated warm service, median / p95 ms | 0.192 / 0.267 | 1.320 / 1.559 |
| Prior cold feature-cache decode + ranking, ms | 35.58 | 342.53 |
| New cold decode + packing + query encoding + ranking, ms | 43.62 | 387.55 |

The cold baseline excludes query encoding; the new cold measurement includes it.
The same previously encoded query is reused for warm measurements on both paths.
These use a repeated 48-image native fixture corpus, not a private photo library.
Unchanged complete session arrays reuse their frame mapping and packed rows; changed
frames/payloads take the invalidation path. Native reranking retains exact IDs/scores.
Cold packing has an explicit cost; this is a warm-query improvement, not a claim
that every first search became faster. A standalone repeated run measured process
peak RSS at 163,151,872 bytes, including Vision/frameworks and both CPU/GPU buffers;
the integrated run reached 171,245,568 bytes before its Orb phase. Pipeline setup
was 189 ms in that standalone run and is avoided entirely by default CPU queries.

Source decoding, initial feature creation, index construction, main-thread result
publication and the intentional 500 ms UI coalescing interval are separate costs.
The submillisecond scan numbers must not be described as complete UI search latency.
