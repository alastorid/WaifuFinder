> Current policy: the capability-aware registry below supersedes Round 2’s exact-name
> automatic-selection restriction. Historical measurements remain scoped to base M3.

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


## Capability-aware profile registry — Round 3

The exact base-M3 gate is now a measurement identity in
[`OptimizationProfiles.swift`](../Runtime/Sources/WDTaggerMLX/OptimizationProfiles.swift),
not the foundation of kernel eligibility. Portable fused kernels remain the
foundation. The registry stores matching rules, provenance, kernel revision,
batch constraints, the fixed native SwinV2 workload, four independent stage
configurations, bias storage and evidence. The exact measured entry preserves
Stage 3 QKV/FC2 64×64×32, other GEMMs 32×64×32, query-16/key-32, dense bias and
standard NormRoute. The 1.5–1.9% claim and exact probabilities still describe only
the documented base-M3 synthetic-corpus measurements.

Metal discovery queries Apple7/Apple9 support and actual maximum threadgroup
memory/dimensions. Apple's [feature table](https://developer.apple.com/metal/capabilities/)
places SIMD matrix and reduction operations in Apple7;
[Apple9](https://developer.apple.com/documentation/metal/mtlgpufamily/apple9) is a
shared feature contract, including multiple chip generations, not a marketing
version number. No GPU family is converted to an invented M-series product list.

| Hardware/workload evidence | Automatic selection |
|---|---|
| Exact measured base M3, batch 8, standard fused graph | Round 2 measured profile |
| Unmeasured device reporting Apple9 compatibility, batch 8, standard fused graph | Same M3-derived configuration, explicitly inherited/unmeasured |
| Compatible M1/M2, smaller batches, or no modern feature contract | Conservative architecture-derived native baseline |
| No more specific applicable entry | Resource-validated universal native baseline |
| Missing required matrix/reduction contract or insufficient baseline resources | Clear unsupported-native diagnostic |

M3 variants and M4 or later devices are eligible through reported compatibility,
not names. Future M5/M6/M7 or unfamiliar names require no source update if they
report existing compatible features. Compatibility is checked, and performance is
not inferred from greater resource limits. New exact measured entries supersede
inheritance by registry priority. No additional hardware was available in this
pass, so no other GPU acquired a measured performance claim.

Explicit operation/stage tiles win over explicit global configuration, which wins
over measured, compatible inherited, architecture-derived and universal entries.
The existing global option also pins query-32 unless attention is explicitly
specified. Per-operation arrays survive a global override. Unspecified optional
norm/bias settings use profile choices; explicit values remain authoritative.
Current automatic profiles keep dense bias/standard NormRoute. Stage 1 reduction
is absent; patch projection and classifier retain their existing library fallback.
Invalid explicit geometry/resources produce descriptive errors. Inapplicable
automatic candidates fall through with rejection reasons exposed in diagnostics.
Checks include the real K/N shapes, QKV segment-safe effective geometry, padded
shared memory and group width. Partial M tiles remain safe: Stage 4 batch-eight
has 1,568 rows, including a 32-row remainder for BM=64.

Resolved typed plans are cached by batch before graph tracing; compiled closures
capture them per input shape/dtype. There is no per-block registry evaluation,
per-call logging, startup benchmark, persistent calibration or added dependency.
Diagnostics expose GPU, feature/resource snapshot, effective/base provenance,
profile/evidence/revision, overrides, workload and actual per-stage choices.
Both native and Core ML comparison reports include this information. Existing
nonthrowing prediction/comparison interfaces retain descriptive precondition
failures for invalid requests; the constructor/resolver expose throwing diagnostics.
The model remains intended for serial worker use, as before.

An additional universal change shares Stage 3's identical directional routes:
17 nonidentity block maps become two immutable maps, eliminating
`15 * 784 * 4 = 47,040` payload bytes. Fused QKV, register-based NormRoute,
identity elimination, direct input/window layout, fused GELU and bias/mask indexing
optimizations remain universal. No new throughput claim is attached to route
sharing or resolver overhead removal. Postnorm ordering and FP16/FP32 boundaries,
448/64 MiB budgets, prepared weights and WDT4 protocol are preserved.

The full-row GEMM/LayerNorm ownership/resource limitations, patch K=48 tradeoff,
compact-bias memory/performance distinction and rejected whole-head/larger Flash
trials recorded above still apply. Static compatibility checks cannot measure
physical register spills or certify pipeline-specific residency/optimality.
Registering future measurements means adding one revision/workload-scoped profile
entry with evidence; no inference logic or chip-name switch changes are required.

The [Metal README](../Runtime/Sources/WDTaggerMLX/Metal/README.md) now documents the
registry architecture, generation handling, precedence, diagnostics, extension
procedure and universal changes beside the preserved Round 1/2 decision record.

### Registry validation

Both release products build successfully. **528 checks passed**: the existing
487 ownership/kernel checks plus 41 pure profile-policy checks. Policy checks
cover older-generation conservative selection, unknown compatible devices,
misleading names, exact-profile priority, revision/shape scope, all partial batches,
resource fallback, authoritative overrides (including overrides that make a
candidate valid), malformed entries, index bounds and serializable diagnostics.
The compiled M3 batch-eight output remains byte-identical to the Round 2 selected
profile across all 695,104 probabilities from 64 synthetic inputs. Batches 1–7
match their retained same-batch controls exactly; compact-bias batch-eight output
also matches all 64 retained inputs exactly. Worker readiness and the 86,888-score
quantization check pass. Additional GPU generations remain unmeasured; synthetic
capability tests verify policy, not physical hardware performance. No interactive
UI testing was needed. Results and final source hashes are appended under `round3`
in the existing evidence JSON, retaining the previous rounds unchanged.

## Round 4 — barrier and scheduling sweep, 2026-10-23

Round 4 attacked the two structural findings of a fresh subagent-assisted review:
Flash32's K/V threadgroup aliasing, and the fact that the selected production
profile never reaches the relaxed-synchronization/full-tile-load paths (they
were reachable only through 32×64×32-only aliases). The session baseline on the
same base M3 was 14.73 fps prediction-only with the automatic measured profile,
64 synthetic RGBA inputs, batch eight; controls in each measurement block stayed
within ±0.1 fps except where thermal drift is noted.

### Retained changes

- **Flash32 stages K and V in separate threadgroup buffers.** The single aliased
  buffer forced four threadgroup barriers per 32-key block (reuse protection,
  publication, K-read-done before V overwrite, V publication). Separate buffers
  need two. Loaded values, MMA order, and the FP32 online-softmax order are
  unchanged: all 695,104 probabilities are byte-identical to the Round 3 binary
  across every configuration tested this round. The isolated attention
  microbenchmark improved from 69.08 ms to 66.09 ms of per-prediction dispatch
  time (−4.3%), concentrated in unshifted stage 1 (−8.1%) and stage 3 (−26.7%,
  offset by a shifted regression); attention is roughly 12% of prediction time,
  so the end-to-end effect is within session noise (paired runs: 14.61/14.65 old
  versus 14.66/14.64 new). Retained because the kernel is strictly faster and
  simpler to reason about; threadgroup memory rises from 2,561 to 4,866 bytes,
  far below the 32 KiB limit. The kernel revision becomes
  `swinv2-native-metal32-fused-qkv-v3`; registry entries pick it up automatically
  and the measured M3 profile was re-confirmed on the new revision (14.66 fps,
  byte-identical).
- **Cache mutations in `WDNativeTagger` are lock-guarded.** Production access is
  serially confined (actor-owned backend → worker subprocess stdin loop; no
  concurrent caller exists today), but `configurations`, `compiled`, and the lazy
  `compactParameters` path mutated shared dictionaries unlocked. An uncontended
  `NSLock` now guards all three for insurance against call-site drift, at
  nanosecond cost against ~68 ms predictions. No throughput claim.
- **Three tile geometries joined the vocabulary** and pass all checks:
  `relaxed64x64x32` and `aligned-relaxed64x64x32` (relaxed synchronization, and
  for the alias also full-tile loads, at 64×64×32 — previously unreachable in
  combination), and `64x64x16` at WM=2/WN=2, the configuration current upstream
  MLX ships as its medium-device default for these fp16 shapes (the existing
  `occupancy` case is a different 256-thread kernel). None is selected by any
  profile: measured neutral on this M3 (see below). They remain available
  building blocks for future per-device profiles.

### Measured and rejected

| Experiment | Result (vs adjacent controls 14.6–14.7 fps) | Verdict |
|---|---|---|
| Stage 3 QKV `relaxed64x64x32` | 14.63 | neutral; relaxed sync at 64×64 geometry buys nothing here |
| Stage 3 FC2 `relaxed64x64x32` | 14.67 | neutral |
| Stage 3 QKV `64x64x16` (WM2/WN2) | 14.70 | within noise |
| Stage 3 FC2 `64x64x16` | 14.65 | within noise |
| Stage 3 FC2 `64x64x64` | 14.42 (−1.6%) | rejected: halved K-loop iterations do not pay for 18.4 KiB threadgroup tiles |
| Stage 1+2 QKV `64x64x32` | 14.28 (−2.5%) | rejected: larger M-tiles hurt the small-C stages |
| `MLX_MAX_MB_PER_BUFFER=128` | 13.30 (−9%) | rejected |
| `MLX_MAX_MB_PER_BUFFER=256` | 12.72 (−13%) | rejected |
| `MLX_MAX_OPS_PER_BUFFER=512` | 13.96 (−1%) | rejected |
| `MLX_MAX_MB_PER_BUFFER=128` + ops 512 | 12.08 (−14%) | rejected |

All tile-variant outputs were byte-identical to the baseline probabilities; the
commit-threshold runs used the unchanged default kernels. Together with the M4
measurement that 16 MiB commits cost 5.9%, the default commit thresholds are now
bracketed from both sides: smaller and larger are both worse, and this direction
is closed.

### Findings and limits

- The M3 tile landscape is at a measured local optimum. Every geometry adjacent
  to the Round 2 selection — including the two never-before-compilable
  combinations — is neutral or slower. Further tile search on this device class
  has no expected value without new hardware evidence.
- Per-op timing recorded by earlier M4 passes attributes ~80–85% of batch time
  to the dense GEMM kernels; attention is ~12% (66–69 ms of ~540 ms per batch-8
  prediction in dispatch-time microbenchmarks). Flash32 K/V-in-registers (zero
  barriers) was assessed and skipped: with barriers already shown nearly free by
  the split-buffer result, its remaining upside is bounded well below session
  noise, against nontrivial uncoalesced-load risk.
- The fused-QKV cosine epilogue was audited against a lane-exact model: its one
  remaining barrier plus 512-byte partials is the minimum for cross-SIMD-head
  ownership; any MMA-based re-reduction would change the mandated FP32 order.
  Confirmed already optimal, not re-tried.
- The classifier head (`head.fc`, M=8, K=1024, N=10861 prime) stays on the MLX
  library path; it is a single ~22 MB memory-bound read, ≈0.05% of batch time.
  A `gemv_wide`-style kernel cannot pay for itself here.
- Concurrency audit: `WDTaggerWorker` is `@unchecked Sendable` with unlocked
  pipe I/O, safe only because the actor backend is its sole caller; the worker
  subprocess loop is single-threaded. `MLXArray` is not `Sendable` in
  mlx-swift 0.31.2, so full `Sendable` conformance for `WDNativeTagger` is not
  available without `@unchecked` plus the lock now in place.

**633 native checks passed** (528 prior + 105 from the three new tile geometries,
including lane-ownership audits for every new effective QKV shape at two/six/model
head counts). Full-model output remains byte-identical on all 64 inputs for every
retained change. Raw samples are appended under `round4` in the
[evidence JSON](wd-tagger-native-kernel-review.json).

### Round 4 addendum — measured base-M4 profile

A Mac mini with a base Apple M4 (10-core GPU, 16 GB, macOS 26.4.1) was profiled
over SSH against the same pinned MLX 0.31.1 metallib, the Round 4 kernel revision
`fused-qkv-v3`, the same 64 synthetic RGBA inputs and identical weights. All 633
native checks pass on that device, and **every M4 configuration measured in this
pass produced probabilities byte-identical to the base-M3 reference run** —
cross-machine determinism for this graph holds exactly.

The device previously selected the M3-derived inheritance as an unmeasured
candidate. A cooled, interleaved three-way comparison (three rounds each,
thermal-gated pauses between runs, batch eight) produced:

| Configuration | Round 1 | Round 2 | Round 3 | Median |
|---|---:|---:|---:|---:|
| Inherited M3-derived (Stage 3 64×64×32, query-16) | 24.054 | 24.024 | 23.962 | 24.024 |
| **Stage 3 QKV/FC2 64×64×16 (WM=2/WN=2), query-16** | **24.076** | **24.111** | **24.075** | **24.076** |
| Universal 32×64×32, query-32 | 24.009 | 24.000 | 24.048 | 24.009 |

The 64×64×16 geometry won all six pairwise comparisons (median +0.22% over the
inherited configuration, +0.28% over universal) — a small but consistent,
paired result, and notably a *different* Stage 3 geometry from base M3, where
64×64×16 measured neutral. Single-run sweeps on the same device also rejected
relaxed 64×64×32, global 64×64×32, FC2 64×64×64, and query-32 attention.

`measured-m4-b8-round4` is now registered: exact `Apple M4` device identity,
batch eight, standard fused graph, Stage 3 QKV/FC2 64×64×16, other GEMMs
32×64×32, query-16 attention, dense bias. Automatic selection on the profiled
Mac mini was re-verified to resolve to this entry with byte-identical output.
The gain is deliberately recorded as ~0.2–0.3%, not extrapolated to M4 Pro/Max
or M5; those remain unmeasured and continue to inherit via family-9
compatibility. Raw paired samples are under `round4.m4` in the
[evidence JSON](wd-tagger-native-kernel-review.json).
