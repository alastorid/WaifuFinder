# Native temporal workers for WaifuFinder

A Swift/MLX worker, derived from the xocialize image port and extended against MLX-Gen
0.38.0's temporal reference. SeedVR2 receives the actual degraded PNG dimensions and writes
temporal results at that size. The app then applies a separate framewise Core ML x2 model to
restore ROI dimensions. Production inference and video IO do not invoke Python, PyAV or
ffmpeg; AVFoundation continues to own source extraction and export in the app.

## MLX 0.31.x implementation notes

- `MLX.compile` closures trace lazy operations. Do not call `eval()` inside one, and do not
  rely on eager evaluation fences surviving compiled replays. Keep explicit evaluations
  between long-running stages when they bound intermediate lifetimes.
- The array form of `compile` supports multiple inputs and outputs:
  `compile { arrays in ... } -> ([MLXArray]) -> [MLXArray]`.
- `MLXFast.metalKernel` takes the Metal function body; MLX generates the signature and
  names output buffers `out`. Template arguments are limited to Bool, Int and DType.
  Pass runtime sizes explicitly or bake other constants into the source.
- `threads_per_threadgroup` in generated kernels is not a reliable way to recover a
  requested size. Pass tile dimensions as constants, and treat `grid` as total threads.

Build/configure from the repository root:

```sh
./tools/setup-native-uncover.sh
./build.sh
```

Setup reuses the configured 3B q4 model, or downloads the pinned prepared weights when
none are configured. MLX Swift is pinned to 0.31.2, whose core is 0.31.1. When the Metal
compiler is absent, the build obtains only the matching precompiled Metal library from
the official checksum-verified artifact. No wheel is installed and no interpreter is
needed. `Package.resolved` pins transitive dependencies. Source provenance is in NOTICE.md
and UPSTREAM_REVISION.

The native additions include:

- 5D video latents/masks and global seeded noise with consistent chunk/tile slices.
- Causal convolution memories and VAE temporal encode/decode slicing, including initial
  temporal-upsample trimming. Context remains multi-frame throughout inference.
- 29-frame chunks, eight context frames and source frame ownership. The last chunk starts
  on the four-frame latent stride; padded context outputs never appear in export.
- 256-pixel spatial model tiles with 64-pixel overlaps, weighted normalized reconstruction
  and identical tile layouts across chunks. Spatial tiling changes model context and may
  change details relative to whole-region inference.
- Direct q4 checkpoint module construction and inference-only placeholders, avoiding
  random training weights and unnecessary re-quantization on startup.
- Explicit graph evaluation boundaries, a capped reusable buffer cache and window-sized rotary
  grids to reduce intermediate lifetime and oversized allocations.
- Native PNG IO, bicubic conditioning, odd-size normalization, and no lossy intermediate
  video encode/decode.
- A 6 GB default physical-memory budget, preflight estimate, 20 ms footprint sampling,
  macOS warning/critical memory-pressure cancellation and one active worker lock. The MLX
  allocator setting is advisory. The OS can still swap other processes; zero system swap
  is not guaranteed.

Worker protocol:

```sh
.native-runtime/waifu-seedvr2 --job DIRECTORY --model DIRECTORY \
  --embedding .native-runtime/pos_emb.safetensors --memory-budget-gb 6
.native-runtime/waifu-seedvr2 --self-test
```

The manifest and prepared frames are identical to the app's existing `UncoverPrepared`
protocol. The worker writes same-size temporal PNGs under `restored/`; the app writes final
exact-ROI-size images under `reconstructed/` after learned spatial enlargement.
`report.json` records the backend, versions, geometry, frame count, chunk/tile parameters,
runtime and separate MLX/RSS/physical-memory measurements. Native failure never silently
switches to the Python reference. The app kills the worker's process group on cancellation
and keeps its previous successful result.

Validation uses the native planning/tile self-test, the existing Swift test harness,
explicit real model integration tests, and side-by-side comparisons with preserved Python
proofs. See `docs/Uncover.md` and `docs/native-uncover-validation.json` for results and limits.

## FlashVSR 1.1

The package also builds `waifu-flashvsr`, a separate native Tiny Streaming model with
official BF16 weights, causal projection, 30-block DiT, LCSA Metal attention, Tiny decoder
and ADAIN. Only native PNG IO, conditioning and memory monitoring are shared with SeedVR2;
the model computations remain separate. SeedVR2 remains the app default. Both executables
share one worker lock and are bundled alongside their matching `mlx.metallib`.

```sh
./tools/setup-native-flashvsr.sh
.native-runtime/waifu-flashvsr --self-test
.native-runtime/waifu-flashvsr --job DIRECTORY --model DIRECTORY --memory-budget-gb 6
```

FlashVSR supports 4× only and refuses padded regions beyond its conservative memory
estimate. It preserves whole-region LCSA rather than introducing spatial DiT tiles.
Offline checkpoint conversion and the official-reference oracle may use Python;
production inference never does. See `docs/FlashVSR.md`, `docs/flashvsr-validation.json`
and FLASHVSR-NOTICE.md for provenance, graph details, reproducible checks and limits.
Normal setup downloads the pinned, checksum-verified converted Hugging Face release
without Python or original checkpoint downloads. `--convert-official` remains an explicit
advanced option for reproducing the conversion from the authors' checkpoints.

## Fused Apple Silicon bundles

Normal setup now downloads packed projection bundles directly. 3B merges SwiGLU
gate/value inputs; both variants merge VAE QKV (including biases). Packed q4/group64
values are preserved without dequantization or re-quantization. 7B retains its GELU
architecture and last-layer checkpoint parameters, normalized pixel RoPE and no output Ada. Matching F16 compute avoids mixed
F16/BF16 convolution promotion; normalization and rotary math remain F32.

`./tools/setup-native-uncover.sh --7b` selects the optional 7B bundle. Streaming is
automatic for 7B, 8 GB machines and single-pass jobs, and can be requested with
`--stream-weights`. Only the active VAE half and transformer block are loaded.
Repeated 3B passes on larger machines keep weights resident to avoid reloading. The process budget is min(6 GB, half installed RAM).
`report.json` includes stage times and fusion/loading flags without a local model path.

```sh
.native-runtime/waifu-seedvr2 --benchmark-fusion
.native-runtime/waifu-seedvr2 --export-fused --model SOURCE --output NEW_DIRECTORY
```

Export is native and bounded by one block/VAE shard. The manifest is written last;
a failed export directory can be removed and recreated. Existing destinations and
already-fused inputs are refused. See `docs/seedvr2-fusion-validation.json` for real
video measurements, precision comparisons and the 8 GB hardware validation limit.

The 3B bundle removes 198 byte-identical shared attention arrays (325 MB) through
validated aliases. Shared video/text QKV, output and MLP projections execute jointly
when every required alias is present. `--no-shared-fusion` is a benchmark control.
Alias-only and shared-batch proofs preserved all five PNGs byte-for-byte.

Uncover's model selector offers optimized SeedVR2 3B and optimized SeedVR2 7B.
Native setup registers each size independently; installing 7B preserves the 3B
registration and vice versa. Both can remain installed and switch directly in the
app. The app rejects unfused checkpoint/Python substitutions. Size changes reuse
prepared inputs and preserve separate cached reconstruction results.

VAE residual/output normalization uses native Metal kernels that read channel-last
F16 activations directly, calculate hierarchical F32 Welford statistics, then fuse
affine scaling, F16 conversion and SiLU. This avoids the full F32 activation copy
and grouped-layout rearrangement. The F16 boundary and temporal normalization axes
are preserved; attention's per-frame normalization remains separate. Both optimized
model sizes use this runtime path with the existing published weights.

```sh
.native-runtime/waifu-seedvr2 --benchmark-vae-fusion
```

This checks numerical parity (including zero variance and large offsets) and reports
component timings. `--no-vae-norm-fusion` selects the original normalization path
for controlled end-to-end comparisons. See
`docs/seedvr2-vae-fusion-validation.json` for complete-job timing and memory results.

The VAE also prepares causal padding and compact history in one native Metal pass,
retaining NDHWC layout between slices. Unit-stride, unpadded 1×1×1 convolutions use
matrix multiplication. Residual blocks compile convolution bias and skip addition
into one epilogue while preserving both F16 rounding boundaries. General 3D
convolution and temporal slicing remain unchanged, and no new weights are needed.

```sh
.native-runtime/waifu-seedvr2 --benchmark-vae-convolution
```

The proof exercises noncontiguous/batched inputs, temporal strides, zero-padding
fallbacks and initialized/active causal history. Benchmark controls are
`--no-vae-causal-fusion`, `--no-vae-pointwise-conv` and `--no-vae-residual-fusion`.
`docs/seedvr2-causal-fusion-validation.json` retains all final paired timing/memory
reports and verifies reconstructed PNG parity.

The latest timing record remains provisional: the fanless Air's longer 3B run
regressed, including its unchanged transformer stage. A historical four-pair 3B
VAE-worker comparison on the 16 GB M4 Mac mini, run on `adae4c5` before the
reduced-ROI changes in `5aa454b`, found a 2.4% median time reduction (3/4 pairs
faster) and 1.495→1.415 GB median worker peaks across five 128→256 frames, with exact
PNG byte parity. It does not measure the updated app's learned spatial restoration
stage. Long-clip 3B/7B timing and end-to-end reduced-ROI validation remain pending.
The benchmark tool recorded that worker comparison in alternating order with PNG byte
parity and a shareable metric summary. Its current output handling needs adaptation for
the reduced-ROI worker's `restored/` directory; it also does not time the app-side
spatial stage. See the root README for the historical command and input preparation.
The updated runtime remains compatible with existing optimized model downloads.
