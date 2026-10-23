# Native WDTagger kernel review — 2026-10-23

The fused native architecture remains selected. Round 1 fixed QKV tile dispatch
and repeated source loads in normalization without establishing a throughput gain.
Round 2, recorded below, selects a measured M3 profile while retaining dense bias,
scheduling budgets, model format and the existing execution framework.

## Round 1 checkpoint

`Kernels.swift` now passes BM, BN, BK, WM, relaxed synchronization and row alignment
to `QKVProject.metal`. Fully aligned row grids also omit the epilogue's row
bounds predicate; partial grids retain the guard. Previously every tile except `occupancy` used 32×64×32.
The existing native tile choices now drive the actual matrix and epilogue geometry:

| BM×BN×BK | SIMD groups | FP32 accumulator elements/thread | Declared threadgroup bytes, including cosine partials |
|---|---:|---:|---:|
| 32×32×32 | 4 | 8 | 5,376 |
| 32×64×32 | 4 | 16 | 8,192 |
| 64×64×32 | 4 | 32 | 11,264 |
| 32×128×32 | 4 | 32 | 13,824 |
| 64×128×32 | 4 | 64 | 17,408 |
| 32×64×64 | 4 | 16 | 14,336 |
| 64×64×64 | 4 | 32 | 19,456 |
| 64×64×16 occupancy | 8 | 16 | 7,168 |

The epilogue additionally retains one FP16 value per accumulator element until
normalization completes. Larger tiles increase weight reuse within a threadgroup,
but increase register use and/or threadgroup memory. They do not provide explicit
weight reuse between threadgroups. Steel's existing vectorized loaders and K-loop
machinery remain intact.

One geometry-derived epilogue handles these choices. Complete 32-element heads
are divided across two N SIMD groups; only their FP32 square-sum partials cross
SIMD groups. BN divides C, so a tile cannot mix Q, K and V segments. For the supported
two-head input, BN=128 is capped to C=64. No `[M,3C]` intermediate is written.
The `aligned` aliases already share QKV's existing full-row load path; `relaxed`
now actually selects Steel's reduced synchronization in fused QKV. `mlx` explicitly
fell back to the fused native baseline because library GEMM cannot expose
accumulators to this epilogue. Round 2 replaces that fallback with a real unfused
MLX reference path; the historical results below retain their original meaning.

`NormRoute.metal` now keeps each thread's one to four FP32 source values through
both statistics passes and the affine epilogue, instead of expressing three loads
and conversions of each source value. The original SIMD/threadgroup reduction
partition and order remain. This is independent of the optional one-SIMD-per-row
`NormSIMD` implementation. Source-level load removal does not establish a reduction
in DRAM traffic; compiler reuse and caches can already serve repeated reads.

The existing validation executable now checks all requested QKV tiles at all four
head counts, both full and partial Stage 3 row grids, and distinct learned head
scales. `--benchmark-fused-qkv` exercises the actual fused operation at existing
stage/batch shapes; the old GEMM benchmark's `qkv` operation remains unfused GEMM.

## Round 1 numerical results

- FP16 matrix operands and FP32 accumulation; projection+bias rounds to FP16 before
  Q/K normalization. V bypasses normalization. Q normalization and learned scaling
  retain their separate FP16 rounding boundaries.
- Full-channel, two-pass FP32 LayerNorm; affine output rounds to FP16 before residual
  addition. The block remains **LayerNorm(projected) + residual**, followed by routing.
- Attention's FP32 online softmax, relative positional bias, shifted-window masks,
  direct Q/K/V layout and token-major attention output.
- Existing Steel license, attribution, fallback paths and dependency pins.

**319 native checks passed at the Round 1 checkpoint.** Every native tile's V output exactly matched its
corresponding unfused GEMM+pack output; Q/K passed the existing tolerances. Compact
and dense attention outputs matched exactly in the exercised shifted/unshifted cases.
On 64 generated RGBA inputs, all 695,104 full-model output probabilities using the
selected baseline tile were byte-identical to the previous QKV and NormRoute
kernels. Inputs were generated natively with Swift/Foundation; no new dependency
or UI interaction was needed. These synthetic inputs do not replace the historical
natural-image corpus.

See [machine-readable results](wd-tagger-native-kernel-review.json) for the measured
samples and limits. Runs were on Apple M3; M4 performance remains unverified.

## Round 1 findings and limits

**Stage 3:** 18 blocks account for 75% of block dense projection arithmetic, not
necessarily 75% of execution time. Its batch-eight fused QKV shape is M=6272,
K=512, N=1536. The first isolated sweep measured 64×64×32 at 4.507 ms versus
4.669 ms for baseline. This is a candidate for independent QKV selection, not
sufficient evidence for a default change. A confirmation sweep reversed the small
Stage 3 lead (5.177 ms versus 5.129 ms), supporting retention of the default.
A single global `gemmTile` still couples
QKV, FC1, FC2 and attention projection; the fastest isolated QKV tile need not win
for that combination. The new kernels make such comparisons meaningful.

**GEMM fast paths:** MLPGelu already has full-tile unsafe loads/stores, compile-time
M alignment, fused bias/GELU, FP32 accumulators and relaxed synchronization. Its
baseline deliberately retains safe handling: the previous aligned-path trial did
not establish a gain. QKV already used unsafe full-row loads before this pass.
Removing checks alone is not evidence of better matrix execution.

**Compact bias:** the displacement-key table is already built once per threadgroup,
with query displacement outside the K/V loop. Packing preserves exact stored FP16
values. The 376 block-heads need 28,888,832 dense bytes versus 548,208 compact bytes:
a 98.1% reduction in bias storage, saving 27.03 MiB. This is neither a 98.1% model
memory saving nor a demonstrated throughput win. Dense bias remains the default,
consistent with prior measurements; both paths remain validated.

**Projection/FC2 plus normalization:** output channels span separate GEMM
threadgroups, whereas LayerNorm needs the entire row. Adding the residual before
normalization would also change this model's postnorm ordering. A useful fusion
would need a different channel-ownership schedule or extra partial-statistics
traffic and synchronization. Retaining the current GEMM and full-row normalization
boundary avoids duplicate GEMM work and numerical changes from one-pass variance.

**Patch embedding:** normalization and extraction already write the initial window
layout directly. Patch projection uses K=48, outside the current native GEMM's
K-alignment contract; it uses library addMM. Folding input normalization into
weights would remove an established FP16 rounding boundary. A dedicated K=48
projection+norm kernel remains a possible separate optimization, with full-row
channel ownership required for normalization.

**QKV/attention and allocation:** attention consumes complete normalized head
vectors and already produces the next projection's token-major layout. The QKV
boundary materializes only the required Q/K/V operands. Merging attention with
projection would couple different work decompositions and repeat projection work
or require cross-threadgroup synchronization. MLX command-buffer input retention
and allocation backpressure also remain; removing Swift references alone does not
prove shorter GPU buffer lifetimes.

**Alternatives:** NormSIMD remains an opt-in comparison because no confirmed gain
justifies selecting it. No additional kernel family or MPS execution framework was
introduced. MPS would require buffer/command-buffer integration and compensating
for lost epilogues; no compelling advantage was identified here.

## Round 2 decisions

The detailed decision record is now in the [kernel README](../Runtime/Sources/WDTaggerMLX/Metal/README.md#optimization-decisions--rounds-1-and-2).
Round 2 promotes a small, measured profile change rather than changing precision or
merging incompatible threadgroup decompositions.

### Selected execution

Automatic selection is restricted to the exact `MTLDevice.name == "Apple M3"`,
batch eight, fused native QKV and the standard normalization implementation.
It selects **64×64×32 for Stage 3 QKV and FC2**, retains **32×64×32 for FC1,
attention projection and downsampling**, and uses **16 query rows / 32 keys** in
Flash32. Every other device or batch keeps the baseline. The classifier and K=48
patch projection keep their existing MLX library implementation. Explicit global
tiles disable automatic selection; operation/stage overrides and explicit query
rows remain available independently. No worker-protocol or prepared-weight format
changes were needed.

| Adjacent native comparison | Combined profile / baseline speed ratio | Maximum probability difference |
|---|---:|---:|
| First paired profile run | 1.0171 | 0 |
| Confirmation paired profile run | 1.0152 | 0 |
| Final direct-arithmetic source | 1.0188 | 0 |

The combined/control columns all use dense bias; compact was timed separately.
These are 64 synthetic RGBA inputs, batch eight, three rounds per process, shared
immutable weights and unchanged 448/64 MiB budgets. Each profile beat its baseline
in all three round totals in each comparison. Absolute clocks varied; the figures
support a **1.5–1.9% prediction gain for the combined profile** on this M3, not a
whole-app, natural-corpus or M4 throughput claim. Individual changes were smaller
and less consistent; only their measured combination was selected. M3 Pro/Max and
M4 performance remain unverified, and their automatic path remains baseline.

### Further concrete changes

- `GEMMTile.effectiveQKVTile` makes native alias equivalence explicit, while
  `qkvGeometry` reports the shape's effective BM/BN/BK/WM. The old `min(BN,C)`
  condition was insufficient for six heads (C=192, requested BN=128); the selected
  BN now divides C. Alignment aliases remain in ordinary GEMM, where they change
  actual load/store strategy. Their QKV specializations were already identical;
  canonicalization fixes configuration/reporting rather than GPU execution.
  The QKV sweep no longer times duplicate aliases.
- `.library` now performs actual MLX projection followed by cosine packing.
  Native QKV can be selected independently through `.qkv` overrides when other
  operations use MLX. This deliberate reference path materializes `[M,3C]`;
  the production fused path still never does.
- `IDENTITY_ROUTE` specializes NormRoute and NormSIMD without changing statistics
  or rounding order. Identity operations no longer read a route or calculate a
  routed batch index. `Stage.first` was never consumed by inference. Removing
  it, redundant stage identity tensors and identity transition tensors eliminates
  200,704 bytes of route payload, replaced by one four-byte placeholder per model.
- Flash32 hoists the shifted query's region out of the K loop. Compact key
  coordinates use `key + 13*(key/14)` with compile-time divisor 14. Both the old
  per-threadgroup coordinate initialization and its initial publication barrier
  are eliminated. A constant-memory lookup trial was replaced by direct arithmetic,
  avoiding an additional dependent coordinate read and an extra header table.

### Rejected trials and hardware limits

The complete-head QKV trial used WM=4/WN=1 instead of WM=2/WN=2, and removed the
512-byte baseline norm buffer and its barrier. Steel's fragment coordinates then
needed four 8-column fragments per head instead of two 16-column strides divided
between N groups. At the same accumulator count this changes A/B fragment reuse
and operand residency. Its full-model trial measured 14.59 fps versus 15.06 fps
for a separate preceding standalone control, with maximum probability difference 0.002138
and no threshold changes. This trial also selected BM=64 in the first three stages, so it did not isolate
SIMD ownership from matrix geometry. These standalone timings are exploratory;
they did not justify promotion. The implementation and option were removed. The selected
cooperative reduction retains the original reduction order.

Flash query-64 and key-64 variants passed correctness checks but did not earn their
resource costs. Query-64 pads 196 to 256, compared with 224 for query-32 and 208 for
query-16. Key-64 similarly increases padded keys to 256, doubles score fragments
and expands K/V scratch space; fewer online-softmax steps did not compensate.
The key-64 full-model trial's maximum probability difference was 0.002135. Both
64-row variants were removed; BK=32 and the stable per-row reduction remain.
QKV 64×128 also remains unselected: its 64 FP32 accumulators and 64 FP16 projected
values per thread are four times baseline's counts, and shared memory rises from
8 to 17 KiB. No disassembly or physical spill/residency counters were collected;
static resource counts must not be presented as observed occupancy.

The batch-eight premise needed correction: **Stage 4's M=1568 is not divisible
by 64**. Its remainder is 32, and the selected BM=32 covers it fully. The other
batch-eight stages are divisible by 64. At batch one, Stage 3 and Stage 4 have
16-row and 4-row tails respectively for BM=32. Existing tail-safe execution remains.

Whole-row projection/FC2 normalization at C=512 and BK=32 would require
`(32+512)*(32+8)*2 = 43,520` shared bytes before statistics, beyond the measured
32,768-byte M3 limit. BK=16 fits the shared tile but requires 128 FP32 accumulator
components per thread at 128 threads, plus retained FP16 values and reductions.
This is a resource/scheduling tradeoff, not an impossible fusion. No replacement
justified those costs. The model also requires `LayerNorm(projected) + residual`;
adding the residual as a GEMM addend before normalization changes its math.

Patch projection plus normalization is feasible at C=128, but needs a dedicated
K=48 loading strategy. Its batch-eight projection is about 1.23 GFLOPs, versus
about 947 GFLOPs for the block dense operations; this arithmetic ratio does not
measure its runtime. It could eliminate roughly 51.4 MB of logical projected
writes/reads. Replacing the library
kernel requires a new K specialization and a full-row reduction ownership/order.
This pass had no independent natural-image corpus to validate a new rounding
trajectory, and selected exact-matching improvements instead. The dedicated
patch kernel remains a feasible, unmeasured candidate rather than a claimed win. Existing contiguous-K
vectorized weight loads, window layouts, Q/K/V producer-consumer layout, and MLX
buffer submission semantics were retained. Offline repacking would require an
additional checkpoint layout or conversion/fallback storage without an established
execution benefit. No MPS integration or new execution framework was introduced.

### Compact bias and numerical scope

Direct arithmetic brought compact bias close to throughput parity in the native
comparison: compact/baseline median sweep ratio was 1.0040, while the two constant
lookup trials were 0.9919 and 0.9934. Those small differences do not establish a
compact throughput gain. A fresh run with the selected M3 profile measured 668.2 MB
peak process memory / 565.6 MB peak active MLX memory with dense bias, versus
650.4 / 537.1 MB with compact bias; probabilities matched exactly. Prediction
rates were 16.09 and 16.05 fps. Reverse-order confirmation measured 666.0 MB dense versus 653.0 MB compact, at
16.11 versus 16.04 fps, again with exact probabilities. The process reduction is
about 2.0–2.7%, distinct from
the 98.1% reduction in bias tensor storage. Dense bias remains the compatibility
default; compact is an improved optional memory mode. Offline compact checkpoints
were deferred because the existing dense prepared format remains compatible and
startup compaction already preserves every FP16 stored value.

**487 final native checks passed**, including 66 ownership audits derived from
Steel lane coordinates, two/six/model head counts, zero norms, partial/full row
tiles, exact V outputs, dummy-buffer identity routes, and dense/compact attention
at both retained query sizes. The rejected variants separately passed 535 checks
before removal. All selected-profile and compact probabilities matched the
baseline on the 64 synthetic inputs. Full-model batches 1–7 also matched their
Round 1 controls exactly, and selected the baseline geometry as intended.
Natural-image and M4 validation were not
available in this pass. Raw samples, capabilities and final source hashes are
appended under `round2` in the existing [evidence JSON](wd-tagger-native-kernel-review.json).
