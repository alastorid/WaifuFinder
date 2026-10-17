# Native FlashVSR 1.1 Tiny Streaming

Uncover Waifu offers **SeedVR2** (the default) and **FlashVSR 1.1** in the same editor.
FlashVSR keeps its output fixed at **4× the source frame dimensions** and accepts the
selected source downsampling factor independently. The model is tuned for 4× super-resolution;
other input reductions change the effective input-to-output ratio. Preview
and export use the 4× output canvas, including an enlarged original background outside
the selected ROI. The preview offers original-pixel, fit-width and fit-height modes while
keeping pan and zoom available.
Prepared ROI frames are shared between engines. Successful reconstructions have separate
cache identities including algorithm, model revision, precision, attention settings,
model location and manifest modification time. Up to four results can be switched without
inference. Region, interval, zoom, comparison, feather and blend settings remain intact.
The B label names the selected engine; an incompatible result cannot be exported.

## Installation

The packaged app requires Apple Silicon and macOS 15 or later. Open
**WaifuFinder → Settings… → Models** and download **FlashVSR 1.1 Tiny**, or use
**Choose Existing…** to register a converted folder. The app handles downloading,
SHA-256 verification, activation, cancellation, and retry without Terminal or Python.
Verified files from an interrupted multi-file download are reused on retry; the current
incomplete transfer restarts. Configuration is written only after verification succeeds.
Model status, disk usage, and locations are visible in the library. Downloads default to
`~/Library/Application Support/Waifu/Models` and can be redirected to another folder.
App-owned models can be moved to Trash; external model folders can be disconnected.

Production inference uses Swift/MLX Metal; it does not invoke Python, PyTorch, CUDA or ffmpeg.
For developers building from source, Swift command-line tools and these scripts remain available:

```sh
./tools/setup-native-flashvsr.sh
./tools/setup-native-flashvsr.sh /absolute/path/to/converted/flashvsr
./build.sh
```

Normal setup downloads [alastorid/FlashVSR-v1.1-Tiny-Swift-MLX-BF16](https://huggingface.co/alastorid/FlashVSR-v1.1-Tiny-Swift-MLX-BF16).
`tools/flashvsr-release.json` pins its commit and manifest checksum. A native Swift
downloader validates the manifest, checks every weight and supporting document, reuses
correct existing files, resumes partial downloads and verifies replacements before
installation. It fetches neither the FP32 originals nor a Python interpreter.

For reproducing the conversion, `--convert-official` (legacy alias `--download-official`)
explicitly downloads the authors' 6.44 GB checkpoints. Only this advanced path needs
Python with torch and safetensors; `FLASHVSR_CONVERTER_PYTHON` selects its interpreter.
Keep space for source and converted weights if choosing that path. After successful
conversion and checksum verification, the original
`.native-runtime/flashvsr-official/` checkpoints are unnecessary for inference and may
be removed to reclaim 6.44 GB. Keep `.native-runtime/flashvsr/`, which is the configured
runtime model. Originals are useful only for reconversion or inspecting source weights;
setup does not delete them automatically.
The app's **Manage Models…** action opens Settings directly to the Models tab.
Weights are not downloaded when selecting an algorithm. Configuration is independent:
`~/Library/Application Support/Waifu/Uncover/flashvsr.json`. SeedVR2 configuration and
its default selection are preserved.

## Authoritative source and conversion

The published model card is maintained in `FlashVSR-model-card.md`. Distribution includes
the Apache license, notices, full validation record and attribution/modification metadata
embedded in every converted Safetensors file. Adding that metadata preserved all tensor
payloads byte for byte. The card identifies the custom Swift layout and matching runtime;
the package is a community conversion of the authors' model.

Reference: [OpenImagingLab/FlashVSR](https://github.com/OpenImagingLab/FlashVSR), revision
`cf910c61a60733e610e9c6e8b607f80c3a6c202b`. Checkpoints:
[JunhaoZhuang/FlashVSR-v1.1](https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1), revision
`27561b186ded3402d7c975f4fd722e2885b6135f`. The native modules follow the official
`wan_video_dit.py`, `examples/WanVSR/utils/utils.py`, Tiny decoder and streaming pipeline, rather
than substituting an image upscaler or the SeedVR2 decoder.

| Source | SHA-256 | Parameters |
|---|---|---|
| `diffusion_pytorch_model_streaming_dmd.safetensors` | `bd28180edcf3446c028e32fc6b731a80bf7e4da2ab4caac3186b9499964d37be` | 825 |
| `LQ_proj_in.ckpt` | `d6d011cdaaba6a52645086caa08fa04124e746f6ca568140a24007591142bfd2` | 8 |
| `TCDecoder.ckpt` | `e224bdcf2f52745cbf4d393ff5374c2ba09e90285d5d19062d2bf63b915b6161` | 66 |
| Official fixed `posi_prompt.pth` | `4601107a11e4e11a936a6b79df579e54dbc99872132bf542151f0ffd65b4b1ef` | [1,512,4096] |

`tools/convert-flashvsr.py` verifies every source checksum, expected name, shape and
finite value. Linear weights retain [out,in]; Conv3D [out,in,t,h,w] becomes
[out,t,h,w,in]; Conv2D [out,in,h,w] becomes [out,h,w,in]. Weights become BF16, with no
quantization or random initialization. The version-1 manifest describes every tensor
and hashes all 34 files. Loading repeats checksum, coverage, shape and dtype validation.
DiT weights are sharded by block to avoid conversion/loading duplicates. Wan VAE is
unused by Tiny and is not downloaded. Apache source/model notices ship in the bundle.

## Computation and streaming

The low-quality input is resized to the requested **4× source-ROI output dimensions**,
normalized to [-1,1], and padded by edge replication to output multiples of 128. The
resampling ratio therefore follows the selected source downsampling factor. The model is
tuned for 4× super-resolution, so results at other effective input-to-output ratios are
not guaranteed to match its preferred quality. ImageIO
preserves straight RGB, including translucent crop edges, matching official PIL RGB input.
The model receives only degraded ROI pixels. Its generated ROI is kept at exact 4× output
dimensions, including for odd source sizes, then placed into the 4× full-frame composite.

The causal projection pixel-unshuffles by 16, applies two causal temporal-stride-2
convolutions, normalization and SiLU, and projects to 1536 channels. Its two-frame
convolution memories carry between chunks. The initial warm prefix is three repeated
frames; only the first DiT block receives the projected low-quality features.

The diffusion transformer has 30 blocks, 12 heads of width 128, 16 latent channels,
1×2×2 patching, trained time/text modulation and rotary axes 44/42/42. One-step diffusion
uses the official timestep and update. Each block retains three temporal KV windows;
new chunks attend against the current window and preceding cached windows. Prompt KV
is fixed within a loaded model. Temporal positions advance globally.

LCSA groups 2×8×8 tokens per block. Coarse pooled Q/K scores use the official locality
eligibility and global top-k threshold **per head and temporal window**, including strict
threshold ties. The normal-slide locality range is 11. Geometry metadata is keyed by
height, width and range, so transposed aspect ratios do not reuse stale masks. The Metal
kernel streams allowed blocks with FP32 online softmax and avoids dense token score
matrices and CPU Q/K/V transfers. A small dense **masked** reference is kept only for
kernel self-tests. BF16 and FP32 kernels are checked over initial/cached windows and
both aspect orientations.

The actual Tiny Conditional Decoder uses its trained conditional inputs, causal memory
blocks, temporal 1/2/2 expansion and three spatial expansions. The first three decoded
warmup frames are trimmed. Official per-frame ADAIN color correction runs with FP32
statistics. Sensitive activations and normalizations also accumulate in FP32 and return
BF16 to match reference rounding.

The first chunk consumes 25 conditioning frames and emits 21; subsequent chunks consume
and emit eight frames, with four-frame lookahead. A short/final clip repeats its last
frame for required context and emits exactly the source frame count. Every new clip
resets projection, self-attention and decoder memories; fixed prompt state can be reused.
No whole-video pixel tensor is retained. The production worker exits after each job,
releasing its model; repeated loaded-model inference is exercised by an explicit benchmark
option, rather than retaining both engines in the app.

## Memory decisions

The existing native SeedVR2 worker provided direct checkpoint loading, explicit graph
boundaries, bounded temporal work, native image IO, process isolation and physical-memory
monitoring. Those facilities are reused; SeedVR2's model math, q4 weights and spatial tile
behavior remain unchanged. The measured SeedVR2 comparison is recorded separately in
`native-uncover-validation.json`; its savings are not assumed for FlashVSR.

FlashVSR uses BF16 rather than unvalidated lower-bit quantization. Weight storage is
already approximately half the official FP32 size. Per-block evaluation and allocator
cache clearing limit intermediate lifetime. MLX cache is capped at 64 MB. A shared native
worker lock prevents simultaneous SeedVR2/FlashVSR residency. Cancellation terminates the
worker group; a failed replacement keeps the preceding successful result.

Default physical-memory budget is 6 GB, with 20 ms sampling and macOS pressure-event
termination. A conservative preflight estimate is 3.6 GB + 12,000 bytes per padded output
pixel. Larger regions are refused explicitly. FlashVSR has **no spatial DiT tiling**:
splitting its region would change trained LCSA selections and KV context. Budget sampling
and MLX's advisory allocator setting cannot guarantee instantaneous caps or zero system
swap from other applications. Physical footprint, RSS and MLX allocations are separate
measurements of overlapping unified memory and must not be added.

## Validation and reproducible commands

Measured results and tolerances are in `flashvsr-validation.json`. The official full
BF16 graph ran with CPU projection/DiT and a PyTorch MPS Tiny decoder. CUDA sparse dispatch
was replaced by mathematically equivalent masked SDPA; the official model definitions,
weights and streaming state were retained. This verifies the port against that reference,
**not a measured CUDA run**. The offline oracle is never part of production inference.

For offline reference validation, set `FLASHVSR_CONVERTER_PYTHON` to the absolute interpreter path in an external conversion environment. Normal installation and app inference use the native runtime.

```sh
.native-runtime/waifu-flashvsr --self-test
WAIFU_TEST_SEEDVR2=1 WAIFU_TEST_ALGORITHM=flashvsr \
  WAIFU_TEST_VIDEO=/absolute/path/to/video.mp4 \
  WAIFU_TEST_SECONDS=1 WAIFU_TEST_ROI_WIDTH=128 WAIFU_TEST_ROI_HEIGHT=128 \
  WAIFU_PROOF_OUTPUT=/tmp/flashvsr-proof ./run-tests.sh --only 'Uncover real' --strict
.native-runtime/waifu-flashvsr --job /tmp/flashvsr-proof \
  --model .native-runtime/flashvsr --capture-fixture /tmp/flashvsr-fixture
"$FLASHVSR_CONVERTER_PYTHON" tools/validate-flashvsr-reference.py \
  --reference /path/to/pinned/FlashVSR --model .native-runtime/flashvsr \
  --job /tmp/flashvsr-proof --fixture /tmp/flashvsr-fixture
# Verify captured block selection against the official functions without model loading:
"$FLASHVSR_CONVERTER_PYTHON" tools/validate-flashvsr-reference.py \
  --reference /path/to/pinned/FlashVSR --model .native-runtime/flashvsr \
  --job /tmp/flashvsr-proof --fixture /tmp/flashvsr-fixture --mask-only
# Reuse one loaded model, reset all clip state, verify exact repeated outputs:
.native-runtime/waifu-flashvsr --job /tmp/flashvsr-proof \
  --model .native-runtime/flashvsr --benchmark-repeats 2 \
  --capture-fixture /tmp/flashvsr-repeat
./run-tests.sh --strict
./build.sh
```

Real integration checks exact frame counts, ROI geometry, rational presentation times,
matching source audio and export decode. Fixtures cover a short clip and a 30-frame
moving clip across three streaming chunks, plus odd geometry. Numeric output tolerance
was set to MAE <0.025 in [-1,1] and correlation >0.99 per visible frame. Inspect the
recorded maxima as well: a passing aggregate metric does not imply pixelwise identity.
Kernel parity, algorithm cache switching, cancellation and unsupported-scale preservation
have independent tests. Scene-cut quality, HDR, non-square pixels, long clips and large
whole-frame FlashVSR inference are not validated. Choose intervals within a shot.
