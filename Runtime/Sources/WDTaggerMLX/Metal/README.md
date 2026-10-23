# WDTagger Metal kernels

[Current memory/input results](../../../../docs/WDTagger-native-memory.md).

**[Performance matrix and optimization guide](../../../../docs/WDTagger-native-tuning.md)**
(main matrix also appears near the top of the repository README).

These kernels are dispatched directly by Swift through MLX's native Metal API.
Inference, weight preparation, correctness checks and timing require no Python.

- `Flash32.metal`: fixed 196-token, 32-lane tiled attention with FP32 online
  softmax. The selected query tile is 16 rows for measured M3 or compatible
  inherited batch-eight execution and 32 rows for the conservative native profile. Compact per-head positional bias and per-window region IDs preserve
  shifted-window masking without an expanded score/mask tensor. The output is
  token-major, ready for the projection GEMM.
- `QKVCosine.metal`: one SIMD group per token/head combines FP32 cosine
  normalization, learned Q scaling, Q/K/V splitting and layout conversion.
- `QKVProject.metal`: fuses projection, cosine normalization and packing with
  geometry-specialized native matrix tiles. The baseline 32×64×32 tile keeps
  FP16 projected values in registers; 512 bytes of FP32 partial reductions cross
  SIMD groups. V bypasses normalization. `QKVCosine.metal` remains the unfused
  comparison path. See the [kernel review](../../../../docs/WDTagger-native-kernel-review.md)
  for supported geometries and the explicit unfused library reference.
- `NormRoute.metal`: two-pass FP32 LayerNorm, affine transform, residual addition
  and the next block's spatial permutation in one dispatch. Each thread retains
  its source values across both statistics passes and the epilogue.
- `NormSIMD.metal`: optional register-based normalization trial; not selected
  because its apparent speed gain did not survive confirmation.
- `PatchMerge.metal`: routes directly between persistent window layouts while
  forming timm's x00, x10, x01, x11 patch concatenation.
- `PatchInput.metal`: accepts direct RGBA UInt8 or reference RGB Float32 input,
  combining input normalization and 4×4 patch extraction,
  writing directly in the first stage's window order.
- `MLPGelu.metal`: tiled FP16-operand GEMM with FP32 accumulation for aligned
  QKV, projection, MLP and reduction matrices. Bias and optional tanh-GELU run
  in the epilogue; FC1 needs no pre-activation buffer or activation dispatch.

The two flattened Steel headers and the attention body derive from Apple's
MIT-licensed MLX 0.32.4 headers. Copyright notices remain in the headers;
`MLX-LICENSE.txt` contains the upstream license. The specialized native runtime
uses the project's existing MLX Swift 0.31.2 dependency. Metal source is bundled
as resources and compiled by Metal, independent of any Python installation.

This is the selected app backend on Apple Silicon. The worker uses fused QKV,
batch eight and a 448 MiB soft allocation budget and 64 MiB cache. See the linked measurements for
the Core ML comparison, memory tradeoff and numerical limitations.

## Optimization policy and review record

The [review and evidence](../../../../docs/WDTagger-native-kernel-review.md#round-2-decisions)
records the measurements, numerical checks and rejected trials. Device capability
queries reported **Apple M3**, **32,768 threadgroup bytes**, and a **1,024-thread group-width
limit**. Physical register spills and resident SIMD-group counts were not measured.

### Selected profile

| Default execution | Stage 3 QKV / FC2 | Other GEMMs | Attention Q×K tile |
|---|---|---|---|
| Base Apple M3, batch 8, native fused QKV and standard normalization | 64×64×32 | 32×64×32 | 16×32, 64 threads |
| Unmeasured devices reporting Apple family 9 compatibility, batch 8, native fused QKV and standard normalization | 64×64×32 (inherited) | 32×64×32 | 16×32, 64 threads |
| Other compatible devices, smaller batches, or comparison implementations | 32×64×32 | 32×64×32 | 32×32, 128 threads |

The M3 combination improved adjacent full-model prediction throughput by
**1.5–1.9%** in three comparisons; all output probabilities matched the control
exactly on 64 synthetic inputs. This is not a whole-app throughput measurement.
The exact-name gate now identifies **measurement provenance only**. Compatible
family-9 devices provisionally inherit this configuration, with no performance
claim for M3 variants, M4 or subsequent chips. See the registry policy below.
Existing 448 MiB scheduling and 64 MiB cache defaults remain.

### Capability-aware registry and precedence

[OptimizationProfiles.swift](../OptimizationProfiles.swift) separates device discovery,
profile data, resource validation and the resolved four-stage configuration.
[Tagger.swift](../Tagger.swift) caches plans by batch, and compiled closures by input
shape and dtype. Each closure captures its complete plan before graph tracing or
Metal specialization; blocks read that plan without registry lookups. There is no
startup calibration, shader-probing benchmark or per-inference logging.

Selection follows this precedence:

1. Explicit operation/stage tiles override every lower layer.
2. Explicit global GEMM configuration replaces all remaining tiles and pins the
   legacy query-32 scheduling; explicit attention scheduling then overrides it.
3. An exact measured device/workload entry wins over compatible entries.
4. A compatible Apple family 9 entry inherits the M3 batch-eight candidate.
5. Apple SIMD matrix compatibility selects the conservative architecture profile.
6. The universal native entry is the last compatible fallback.

An explicit global tile never erases operation overrides. Existing four-entry
operation arrays remain supported; reduction's Stage 1 entry has no operation to
execute. Optional normalization and bias arguments distinguish unspecified values
from explicit false. Profiles support four independent attention schedules and
normalization choices, plus a bias storage choice. Current automatic entries all
use standard NormRoute and dense bias. Patch-merge/final normalization retain MLX
LayerNorm. Comparison variants preserve the existing nonthrowing API and report
invalid tuning as a descriptive precondition failure; initialization and the
public resolver throw descriptive errors.

Discovery reads the actual device name, Apple7/Apple9 feature contracts and all
three maximum threadgroup dimensions plus threadgroup memory. Apple7 establishes
SIMD matrix/reduction compatibility for the retained FP16/FP32 kernels; Apple9 is
an inheritance criterion, **not a one-to-one marketing generation identifier**.
See Apple's [feature tables](https://developer.apple.com/metal/capabilities/) and
[Apple9 documentation](https://developer.apple.com/documentation/metal/mtlgpufamily/apple9).
M1/M2 families receive the full fused native baseline with 32×64×32 GEMMs and
query-32 attention; valid larger tiles alone are insufficient performance evidence.
M3-and-newer devices reporting Apple9, including previously unseen names, are
eligible for the inherited batch-eight candidate after resource checks. An unknown
name without that feature contract gets the conservative optimized native profile;
no M5/M6/M7 name parsing or invented product/capability table is needed. If future
hardware stops reporting the required feature contracts, compatibility cannot be
assumed: initialization gives a clear unsupported-native error.

Each candidate is scoped to the kernel implementation revision, fixed 448×448
SwinV2 shapes, batch constraints, and standard fused implementation where required.
Resolution validates actual stage K/N divisibility, effective fused-QKV geometry,
thread count, and padded shared-memory bytes (including cosine partials). M tails
remain safe and do not require padding the input batch. Row alignment is reported;
aligned aliases retain their existing shape-guarded full-tile paths. Inapplicable
automatic candidates descend through the registry. Invalid explicit choices throw
instead of silently choosing another tile. The patch projection (K=48) and
classifier (N=10861) retain their existing library geometry fallback.

To register new measurements, add a small `WDGPUOptimizationProfile` entry with an
exact matching rule, supported batches, revision, four stage configurations,
provenance and an evidence reference. Exact entries sort before inherited entries,
regardless of declaration position; ties retain registry order. No inference branch
changes. Changing a kernel revision requires revalidating or retiring its evidence.
The exported `resolvedConfiguration(batch:).diagnostics` and validation reports show
the actual GPU/capabilities, base and effective provenance, overrides, selection
reason, batch, stage rows/channels, tile choices, alignment, attention and norm/bias
choices. Measured, inherited, architecture-derived, universal and user-override
statuses are distinct. `--check-profiles` exercises policy without GPU execution.

### Universal structural optimizations

Fused QKV/cosine/packing, register-retained NormRoute values, identity-route
elimination, direct RGBA/window packing, fused FC1 GELU, direct compact-bias indexing,
and hoisted shifted-query regions remain available on every compatible GPU.
Alternating Stage 3 blocks now also share two immutable directional route arrays,
removing 47,040 bytes of duplicate payload. Configuration capture eliminates
per-block policy evaluation. Neither change claims a measured throughput gain.
The 448 MiB allocation and 64 MiB cache budgets, checkpoint format, WDT4 worker
protocol, FP16 rounding boundaries and FP32 reduction order remain unchanged.

The Round 2 exact base-M3 entry remains the only measured tuning profile. All
other modern selections are provisional. Register occupancy and pipeline-specific
resource limits still depend on Metal compilation; static resource validation does
not measure physical spills or establish the fastest geometry. No earlier rejected
kernel experiment was reintroduced.

`--gemm-tile 32x64x32 --attention-query-rows 32` pins the original geometry.

Dispatch and effective geometry are in [Kernels.swift](../Kernels.swift). Lane
ownership follows `BaseMMAFrag.get_coord` and `BlockMMA` in [MLPGelu.h](MLPGelu.h),
with independent coverage audits in the [native checks](../../WaifuWDTaggerValidation/main.swift).

### Decision record

| Concern | Finding and decision |
|---|---|
| Duplicate QKV configurations | `GEMMTile.effectiveQKVTile` canonicalizes `aligned` to `baseline` and `alignedRelaxed` to `relaxed`: their QKV template arguments and dispatches are identical. These aliases remain meaningful for MLPGelu's safe/full-tile load/store choice. QKV timing sweeps omit duplicate aliases. |
| `.library` backend | Now really executes MLX addMM followed by QKVCosine, even through `projectQKV`. It explicitly pays for `[M,3C]` and a packing dispatch. Select a native `.qkv` override with library GEMMs elsewhere for an intentional mixed configuration. There is no hidden native fallback. |
| Generalized QKV coverage | `qkvGeometry` reports effective dimensions. BN must divide C, not merely be ≤ C; non-model head counts such as six now fall back to a divisor-safe width. GPU reference checks include two and six heads. Independent enumeration of Steel's 8×8 lane coordinates proves one writer per tile element and no mixed Q/K/V segment. |
| Cross-SIMD normalization | A WM=4/WN=1 trial owned complete heads and removed norm shared memory/barriers, but changed operand-fragment residency and was slower in its full-model trial. It was removed. The selected WM=2/WN=2 mapping preserves the original FP32 reduction order. Q/K/V already occupy separate GEMM tiles within one dispatch; three separate dispatches would not remove a remaining projected tensor. |
| Exact batch-eight geometry | The claim that every stage is divisible by 64 is false: Stage 4 is 1,568 = 24×64+32. Its selected BM=32 is aligned. Shape flags and existing safe paths handle smaller batches; no batch-size contract changed. |
| Large tiles / physical occupancy | 64×128 retains 64 FP32 accumulator components plus 64 FP16 projected values per thread and uses 17 KiB shared memory, versus 16+16 components and 8 KiB for baseline. It did not justify selection. Those static costs do not prove spills or physical residency. |
| Flash32 scheduling | 16 query rows keep the same eight rows per SIMD group, BK=32 and per-row softmax reduction order; they pad to 208 rather than 224 query rows. A 64-row query trial padded to 256 and was slower. BK=64 reduced softmax iterations but padded keys to 256, enlarged score/register/shared-memory state, changed reduction order, and was generally slower. Both 64-row trials were removed. Stage 4 sometimes favored query-32 in isolation; the selected query-16 combination won full-model comparisons, without claiming a universal per-stage optimum. |
| Bias and masks | Compact keys now use direct `key + 13*(key/14)` arithmetic. Shared-table construction, its publication barrier, and a trial dependent constant-table read are gone. Shifted queries fetch their region once outside the K loop. Compact/dense results remain exact for both query sizes. Dense bias remains the compatibility default; compact remains a measured memory option, without changing the prepared format. |
| NormRoute and routing | `IDENTITY_ROUTE` removes route fetches and route index arithmetic from both normalization implementations. The model passes a shared one-element placeholder for identity operations. Unused `Stage.first` maps, full identity maps, and identity transition maps were eliminated: 200,704 bytes of route payload replaced by a four-byte placeholder. The two-pass FP32 algorithm, rounding boundaries and residual order remain; NormSIMD stays opt-in. |
| Projection/FC2 plus LayerNorm | The current output channels span threadgroups. Whole-row ownership at C=512 with the current BK=32 padded layout needs 43,520 shared bytes before statistics, exceeding this M3's 32 KiB limit. BK=16 can fit, but a 32×512 tile at 128 threads needs 128 FP32 accumulator components per thread, before retaining rounded values and reduction state. No efficient replacement was established. Adding residual before LayerNorm is mathematically wrong for this postnorm model. |
| Patch projection, packing and buffer lifetime | A dedicated K=48/N=128 projection+norm kernel is feasible; full-row synchronization is not impossible here. It requires different K tiling from the current native GEMM, and no measured bottleneck justified replacing the library path. Existing FP16 input rounding and direct window order remain. Weight loads already vectorize contiguous K; offline repacking adds checkpoint/layout complexity without an established benefit. MLX submission retains buffers until GPU completion; extra evaluation barriers would sacrifice graph scheduling. |

### Validation and limits

The native suite covers alias dispatch, two/six/model head counts, complete and
partial row tiles, zero norms, exact V packing, identity/nonidentity routing, and
both query sizes with dense/compact bias and shifted/unshifted attention.
No precision approximation, MPS execution framework, third-party dependency,
prepared-weight format change, or interactive UI test was introduced.
