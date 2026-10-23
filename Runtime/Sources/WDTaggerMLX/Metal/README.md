# WDTagger Metal kernels

[Current memory/input results](../../../../../docs/WDTagger-native-memory.md).

**[Performance matrix and optimization guide](../../../../../docs/WDTagger-native-tuning.md)**
(main matrix also appears near the top of the repository README).

These kernels are dispatched directly by Swift through MLX's native Metal API.
Inference, weight preparation, correctness checks and timing require no Python.

- `Flash32.metal`: fixed 196-token, 32-lane tiled attention with FP32 online
  softmax. The selected query tile is 16 rows for measured M3 batch-eight execution
  and 32 rows otherwise. Compact per-head positional bias and per-window region IDs preserve
  shifted-window masking without an expanded score/mask tensor. The output is
  token-major, ready for the projection GEMM.
- `QKVCosine.metal`: one SIMD group per token/head combines FP32 cosine
  normalization, learned Q scaling, Q/K/V splitting and layout conversion.
- `QKVProject.metal`: fuses projection, cosine normalization and packing with
  geometry-specialized native matrix tiles. The baseline 32×64×32 tile keeps
  FP16 projected values in registers; 512 bytes of FP32 partial reductions cross
  SIMD groups. V bypasses normalization. `QKVCosine.metal` remains the unfused
  comparison path. See the [kernel review](../../../../../docs/WDTagger-native-kernel-review.md)
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

## Optimization decisions — Rounds 1 and 2

The [review and evidence](../../../../../docs/WDTagger-native-kernel-review.md#round-2-decisions)
records the measurements, numerical checks and rejected trials. Device capability
queries reported **Apple M3**, **32,768 threadgroup bytes**, and a **1,024-thread group-width
limit**. Physical register spills and resident SIMD-group counts were not measured.

### Selected profile

| Default execution | Stage 3 QKV / FC2 | Other GEMMs | Attention Q×K tile |
|---|---|---|---|
| Base Apple M3, batch 8, native fused QKV and standard normalization | 64×64×32 | 32×64×32 | 16×32, 64 threads |
| Other devices, smaller batches, or comparison implementations | 32×64×32 | 32×64×32 | 32×32, 128 threads |

The M3 combination improved adjacent full-model prediction throughput by
**1.5–1.9%** in three comparisons; all output probabilities matched the control
exactly on 64 synthetic inputs. This is not a whole-app throughput measurement.
M3 Pro/Max and M4 keep the baseline until measured; the device-name gate is exact.
Existing 448 MiB scheduling and 64 MiB cache defaults remain.

[Tagger.swift](../Tagger.swift)'s `selectedTile` chooses by operation, stage and
batch at graph-trace time. An explicit `gemmTile` disables automatic selection;
`gemmOverrides` supplies four stage entries per overridden operation (`qkv`,
`projection`, `fc1`, `fc2`, `reduction`). Reduction has no Stage 1 entry in the
execution graph, so its first override entry is ignored. `attentionQueryRows` overrides attention independently. The
validation tool reports the effective choices, not just the requested label.
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
