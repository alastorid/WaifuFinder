# Uncover Waifu

An isolated editing mode for a focused video moment: **select → degrade → reconstruct →
compare → export**. This document records the design and current limits. Historical model
validation is in [`native-uncover-validation.json`](native-uncover-validation.json). The
current actual-low-resolution SeedVR2 plus learned Core ML x2 path still needs real
inference and quality validation; the older recorded SeedVR2 runs predate that path.

## Processing contract

```text
Source video → consecutive native frames → fixed ROI → one antialiased downsample
                                                     ├→ Lanczos baseline → A
                                                     ├→ SeedVR2/native restoration → B
                                                     └→ FlashVSR native 4× ROI output → 4× frame composite
A/B → shared comparison and preview scale modes → export
```

At **1× Original (No Downsampling)**, preparation keeps the original ROI crop
directly for input and baseline. SeedVR2 restores it at that size. Any
model-required face-crop resizing is separate from this optional source reduction.

For a 640×480 ROI, 1× retains 640×480; the 2×/4×/8×/16×/32×/64× inputs are 320×240,
160×120, 80×60, 40×30, 20×15 and 10×8. For odd sizes, each degraded dimension is
`max(1, ceil(original / factor))`; this rounding appears in the geometry readout. SeedVR2
and other native-size backends return the ROI at source dimensions. FlashVSR instead
returns a 4× ROI and exports a full frame at 4× source dimensions; its input downsampling
factor is independent and the model is tuned for 4× output. The viewer switches between
original-pixel size, fit width and fit height, while pan and zoom remain available.

A is the degraded input enlarged by Lanczos. B is that same input reconstructed by
SeedVR2 using consecutive frames jointly. Original is an optional untouched reference.
For factors greater than 1×, high-resolution pixels inside the selection never enter inference, including any future
spatial context: context must undergo the same degradation. The original remains available
for the background, comparison and diagnostic measurements.

**PGTFormer** uses the original pretrained network through native Core ML and
is validated end to end. Its required 512-pixel face resizing is independent of
the optional source downsampling factor. Download it from its pinned release in
Settings → Models, or run `tools/setup-native-pgtformer.sh`; Python is used only
for offline checkpoint conversion and reference validation. The
[native face-model audit](Face-restoration-native-audit.md)
records the Swift implementations examined before choosing faithful components.

PGTFormer is redistributed under a **non-commercial** licence: upstream permits
redistribution for non-commercial use only, and commercial use requires permission
from Xidian University or the contributors. The hosted bundle carries that licence
and attribution verbatim. RealESRGAN, by contrast, is BSD 3-Clause.

The optimized PGTFormer package retains the original FP32 encoder/codebook
and uses an FP16 decoder with native fused normalization and center-frame
pruning after the last temporal operator. Compiled models are cached by package
fingerprint and OS version in `~/Library/Caches/Waifu/PGTFormerCompiled`; source
packages remain integrity-checked on every job. Hashing releases each read buffer,
and ROI decoding is reused. The interface and true 1× source preparation are unchanged.
On this 16 GB M3, paired inference improved 11.3%; the eight-frame video job's
peak footprint fell from 4.82 GB to 2.93 GB. The verified eight-frame product run
fell from 28.70 to 24.26 seconds; compiled-model loading fell to 0.15 seconds.
Full-run timing varies with system load.
[Measurements and ANE experiments](pgtformer-optimization-benchmark.json) include
all recorded runs. FP32 ANE configuration fell back to CPU; FP16 ANE loading
exceeded the 6 GB budget.

The worker now permits GPU execution for **both** stages, with low-precision GPU
accumulation disabled. Previously the encoder was forced onto the CPU, leaving
the GPU waiting between decoder predictions. The package format identifier
`cpu-codebook-gpu-decoder-v1` is retained for compatibility with installed weights;
the execution policy and result-cache identity are now `gpu-encoder-decoder-v2`.
The worker's `--encoder-units cpu` option reproduces the old placement for paired
benchmarks; normal app jobs use GPU. Reports record the selected units and stage
times. GPU eligibility does not promise constant utilization during image I/O.

On the same 16 GB M3, two alternating eight-frame CPU/GPU pairs averaged 26.94
seconds with the old placement and 17.39 seconds with both stages GPU-enabled
(35.5% less time). Worker CPU time fell from 27.22 to 2.27 CPU-seconds (91.7%),
and peak footprint averaged 2.88 versus 2.19 GB. All eight real-video windows pass
against the pinned original network: maximum error 0.01538, mean below 0.000150,
and preprocessing error zero. The existing 0.02/0.002 acceptance gates are unchanged.
The native product test also passes for 1×/2× preview, export with audio, timestamps
and cancellation. See [GPU placement measurements](pgtformer-gpu-benchmark.json)
for individual trials, CPU accounting, compute plans and the previous-output
comparison that prompted the independent original-network check.

The offline converter now also defaults to native fused attention for the nine
FP32 code-prediction transformer layers. Q/K positional embeddings, unmodified V,
head scaling and output projections are preserved; unused attention weights are
no longer materialized. Two alternating eight-frame trials measured 13.2% less
encoder time, 6.3% less total inference time and 5.7% less peak memory beyond the
GPU-placement change. All output PNGs matched the unfused GPU model byte for byte,
and all eight original-network reference windows passed. Encoder GroupNorm fusion
also passed accuracy but did not improve encoder time, so remains opt-in through
`--fuse-encoder-norm`. `--no-fuse-code-attention` reproduces the prior encoder.
See [fusion measurements and graph counts](pgtformer-fusion-benchmark.json).
The fused package is published in the pinned PGTFormer Hugging Face release.
Settings → Models downloads that version, and older unfused packages prompt for
the optimized download. PGTFormer upgrades install alongside the previous model
and activate only after verification; repeating setup reuses the verified current
release. Existing exports are independent of model installation.

[Original-model video validation](pgtformer-optimized-video-validation.json)
and [1× validation](pgtformer-optimized-1x-validation.json)
check all eight temporal windows and both 1× boundary windows and exact preprocessing against the pinned
original checkpoint. Maximum pixel error is 0.01660 and mean error is below
0.000219, within the existing 0.02/0.002 gates. Offline conversion enables these
optimizations by default; `--decoder-precision float32 --no-fuse-decoder-norm
--no-center-tail` reproduces the previous decoder conversion.

**RealESRGAN x2plus** now uses a validated fused Core ML package. Pixel-unshuffle
and the first convolution become an equivalent stride-2 convolution; all 23
original RRDB blocks remain. Validated 256/384/522 input shapes retain reflected
context, input tensors are reused, and single tiles bypass the blend accumulator.
PNG inputs use straight RGB so translucent Lanczos edges do not darken inference.
On this M3, Core ML selects ANE for all model operations; hardware power counters
confirmed ANE execution. The measured native 96×96 crop path improved from
0.175 to 0.050 seconds per image, about 3.5× faster. Standalone process footprint
remains approximately 35–38 MB; an overall memory reduction is not claimed.
The eight-frame 2× product job completes in approximately 0.8 seconds with warm
backend caches; initial Core ML compilation can add approximately six seconds.
[Benchmarks and original-model checks](realesrgan-optimization-benchmark.json)
include odd partial tiles, repeatability, GPU/ANE comparison, rejected context
settings, and full video validation. [Actual video checkpoint comparison](realesrgan-original-video-validation.json)
passed on all eight restored PNGs. True 1× source pixels, native preview, export
with audio/PTS and cancellation passed the product tests.
`tools/setup-native-realesrgan.sh` installs the hosted optimized package;
`--convert-official` prepares and validates one offline. Python remains limited
to checkpoint conversion and reference validation. Settings can also import the
validated compiled package. Legacy fixed-size model folders remain supported.

SVFR was dropped on October 12, 2026 because its processing cost did not meet the
one-minute requirement. Its app integration, tools, weights and caches were removed.

Reconstruction can invent plausible details. It does not guarantee recovery of information
erased by downsampling or masking. Visual inspection remains the primary evaluation.

## Integration and ownership

Keep scoring, clustering, ranking, arrangement, analysis sampling and still export intact.
The macOS 15+, Swift 6, direct `swiftc` build and custom test harness remain the foundation.
No Python runtime is linked into the application. The app remains a direct `swiftc` build;
the isolated native model worker is a separate pinned Swift package.

`WaifuSession` owns entering/leaving the mode and integrating the active video.
`UncoverSession` owns preparation, inference, results, comparison and clip export.
Pure geometry, image processing and temporal IO belong outside SwiftUI in `Uncover/`.
`TemporalVideoUpscaler` is the replaceable boundary between orchestration and inference.
The production adapter uses an isolated Swift/MLX process with disk-backed PNG inputs
and outputs, extending the xocialize Swift port with the Python reference video semantics.
The initial Python/MLX-Gen adapter remains a legacy reference, with no automatic fallback
from a failed native run. Native inference uses already-prepared 4-bit weights. Setup downloads the
[self-contained Swift/MLX bundle](https://huggingface.co/alastorid/SeedVR2-3B-Swift-MLX-4bit),
including the text embedding, with pinned checksums and no Python conversion. The weights
remain the unchanged [lpalbou / AbstractFramework conversion](https://huggingface.co/AbstractFramework/seedvr2-3b-4bit)
of [ByteDance Seed’s model](https://huggingface.co/ByteDance-Seed/SeedVR2-3B).

Opening from the native-resolution preview preserves the video and focused timestamp,
pauses the existing preview, and removes the still-preview keyboard monitor. The editing
mode owns transport keys exclusively. Closing restores browsing without changing grid
settings or selections. Video replacement cancels the editing job and discards its state.

## Selection and frame preparation

- Draw a rectangle, drag to move, and resize using corner/edge handles.
- While editing, the selected 1× through 64× reduction appears inside the ROI with its input
  dimensions. It uses the preparation pipeline's Lanczos downsample and enlarges those
  reduced pixels without smoothing. Moving/resizing the region or changing scale updates
  the cached preview; pan/zoom reuse it. Comparison mode removes this overlay, and Original
  shows the untouched source. This preview does not run inference.
- Store normalized top-left coordinates in the **oriented video content**, not the view.
  Aspect-fit letterboxing, zoom/pan, point scaling and Retina scale must not affect pixels.
- Quantize left/top down and right/bottom up to integer source pixels, clamp at boundaries,
  and show the resulting snapped rectangle. Preferred transforms must be applied exactly
  once, including normalization of translated/rotated extents.
- The first release uses a fixed ROI; motion tracking is deferred.
- Decode the original stream sequentially at native resolution. Never use 1-FPS analysis
  samples, nominal-FPS arithmetic or repeated still seeks as temporal inference inputs.
- Preserve ordinal frame identity and rational presentation timestamps. Requested intervals
  are half-open; source frames whose PTS is in `[start, end)` are visible output frames.
- Use the backend's temporal overlap, trim context/padding, and verify output counts and
  dimensions. Avoid crossing scene cuts where practical; do not mix context from unrelated
  shots or disguise misalignment with arbitrary temporal blending.

The initial interval defaults to about one second around the focused moment and is visible
and editable. Offer 1/2/5 seconds and a custom interval within the current bounded limit.
Preparation holds one full frame at a time and writes original frames and degraded/baseline
ROI images to temporary storage. Cached originals avoid repeated source seeks during
comparison/export; only degraded images enter inference. Inference holds bounded temporal
chunks, not a full source video array.
Keep frame records on disk for later streaming if limits are expanded. Current preparation
is capped at five seconds and 300 source frames per job.

## Backend and scaling

The adapter must expose availability, accepted geometry, minimum temporal frames, validated
scale factors, memory restrictions, meaningful progress, errors and cancellation. Model
loading alone does not count as successful integration. No test backend can be presented
as SeedVR2, and no interpolation fallback can be labelled a reconstruction.

Current upstream references:

- [MLX-Gen video upscaling](https://github.com/lpalbou/mlx-gen/blob/main/docs/upscaling.md)
- [SeedVR2 native Swift experiment](https://github.com/xocialize/seedvr2-mlx-swift)
- [SeedVR2 implementation](https://github.com/FurkanGozukara/SeedVR2)

The current backend is native Swift with **mlx-swift 0.31.2 / MLX core 0.31.1**, a
matching precompiled Metal kernel library, and the existing 3B q4 model revision
`9b2f6c3c1b1b4548c66f302ec5a4cd3cd9937eb5`. No Python interpreter is launched for inference,
video IO, setup or building. The build extracts only Metal kernels from the official,
checksum-verified MLX artifact; it does not install a Python wheel.

The Swift image port required real video additions: multi-frame latent noise and masks,
causal 3D-convolution memories, VAE encode/decode slicing, first-frame temporal upsampling
trim, 29-frame/8-context chunk planning and shared global noise slices. Transformer blocks
and convolution outputs are evaluated promptly to release intermediate graphs. Existing
quantized matrices are installed directly; random training weights are never quantized
just to be discarded. Spatial attention uses grids sized to the actual window.

Selections can span the source. Native inference uses model tiles up to 256×256 with
64-pixel overlaps; each tile processes consecutive frames jointly. Overlapping outputs
are weighted and normalized, with the same global noise coordinates and tile layout
across chunks. This is an approximation to whole-region inference and can change local
context; inspect seams and moving objects. Native PNG IO avoids the old intermediate
lossy MP4 round trip. Exact odd ROI dimensions are restored with Lanczos after removing
padding.

The default physical-memory budget is 6 GB. A sampled physical-footprint monitor,
macOS memory-pressure events, an estimate before loading and a single-worker lock reduce
swap pressure. The MLX memory-limit setting is only a guideline. A job that exceeds its
budget stops and preserves the editor's previous success; system-wide zero swap cannot
be guaranteed. Intervals are still limited to five seconds/300 frames.

The Python reference is pinned to MLX-Gen 0.38.0 and preserves the user's explicit
unsafe-memory preference. That preference does not disable the native worker's budget.

Start with the 3B 4-bit model and a real 2× proof: five consecutive frames from a real video,
256×256 ROI degraded to 128×128, reconstructed to 256×256. Record model/revision, backend
version, actual input/output geometry and frame counts, temporal configuration, duration
and memory when available. Subsecond tests prove the route; longer moving clips are needed
to assess temporal quality.

Upstream uses output dimensions divisible by 16. Pad the **degraded** input with replicated
edges to a compatible size before inference; remove that explicit padding afterwards and
normalize the unpadded result to exact ROI geometry. Never crop selected source detail to
make a model size fit.

Use whole-window inference for short windows and the production 29-frame/8-frame overlap
profile for longer ones. Overlap supplies context; it is not output crossfading.

SeedVR2 receives pixels at the selected degraded dimensions, bicubic enlarges the
conditioning frames, and runs native temporal inference at the source ROI dimensions.
The same SeedVR2 video model generates spatial detail and temporal consistency. FlashVSR
resizes degraded pixels to its fixed 4× source-ROI output size and composites that result
into a 4× full-frame video. Its downsampling factor remains independently selectable.
RealESRGAN is a separate, framewise Core ML algorithm. Its x2plus image model has
no temporal context; repeated stages may flicker. Download that option
independently in Settings → Models.

The control offers **1×, 2×, 4×, 8×, 16×, 32× and 64×** input reduction. SeedVR2
produces the selected ROI's exact original dimensions. These are
restoration targets, not enlargement of the whole video beyond its original size. FlashVSR
produces a 4× ROI and 4× full-frame output. A 16
megapixel output-ROI cap refuses selected regions whose final spatial buffers would be
unreasonable. A whole 1080p frame at 32× is still a 1080p output: SeedVR2 receives about
60×34 pixels and runs at 1920×1080 output geometry. The cap applies
to the selected output region itself, so a whole 8K frame exceeds it.
Factors above 8× are available but unvalidated and increasingly synthesize detail. RealESRGAN is
downloaded from our pinned
[optimized Core ML release](https://huggingface.co/alastorid/RealESRGAN-x2plus-Optimized-CoreML)
with pinned SHA-256s and compiled locally; no Python is used. Its model weights are
from [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) under BSD-3-Clause, and the
conversion uses the [BasicSR](https://github.com/XPixelGroup/BasicSR) reference
architecture under Apache-2.0. Both licences ship with the release. Earlier builds
downloaded the pre-converted package from
[real-esrgan-coreml](https://github.com/hanxiao/real-esrgan-coreml) instead; it is
still accepted when imported as an external folder, but it is no longer offered.

The original SeedVR2 2×/4×/8× learned-video-scale proofs used full-size SeedVR2 inference.
The current native 1× through 64× path needs real quality and memory validation on the Mac mini.
Never infer video support or quality from
image-only demonstrations.

## Three cached stages

1. **Prepare:** source identity, interval, snapped ROI, degradation and degraded frames.
2. **Reconstruct:** prepared input, backend/model identity, seed, temporal settings and
   relevant inference parameters → reconstructed ROI frames.
3. **Display:** source background + cached A/B ROI + mask + comparison viewport.

Only changes affecting stages 1/2 invalidate inference: source (including modification),
timestamps, ROI, degradation, model or inference settings. Blend, feather, comparison mode,
wipe, Original, zoom, pan and scrubbing reuse the successful result. Keep the previous
successful job if a replacement fails/cancels; label it with its original settings and
do not export it as the new request. Results are published only after validation succeeds.

## Compositing and comparison

`output = original × (1 - mask) + reconstructed × mask`.

The mask is zero outside the ROI, one in the interior at full strength, and transitions
smoothly inward at its edges. Edge Feather defaults to about eight native pixels, clamped
for small selections. Blend Strength defaults to 100% and is independent of feather.
Full-strength interiors contain only A/B pixels; ground-truth pixels must not hide model
errors. Both alternatives use exactly the same geometry and mask.

The video dominates a restrained native layout, with selection, interval, 2× through 64×,
feather, strength, Reconstruct/Cancel and Export controls. Show actual stages:
Preparing frames → Reconstructing → Compositing → Ready. Counts or percentages require
real backend evidence; indeterminate progress is preferable to invented percentages.

After success, open comparison automatically. Support side by side, a draggable wipe and
instant A/B toggle, plus Original reference. Use **one playback clock and frame identity**
to render both sides. Seeking, play/pause, frame stepping and interval looping stay aligned;
two unrelated players do not satisfy this. Share zoom and pan, preserve them across modes,
and initially frame the ROI with enough surrounding content to inspect seams. Cache a
bounded number of decoded preview images rather than the whole interval in RAM.

Optional diagnostics compare both results against the known original ROI interior,
excluding feathered pixels: measured PSNR, SSIM, processing time and temporal stability.
These are measurements, not absolute perceptual rankings; generative results may look
better while scoring worse. Never invent metrics.

## Export and failure handling

Begin with the processed interval, encoded as playable MP4/MOV at the original oriented
frame dimensions. Preserve source order and PTS relative to the interval's first visible
frame and retain the corresponding audio segment when present. Frames outside the ROI
remain unchanged before encoding; re-encoding does not preserve compressed bytes.

Avoid unnecessary color conversions; preserve or correctly transform orientation, pixel
aspect and color metadata. Reject unsupported HDR/non-square formats explicitly rather
than silently converting them. The compositor should later support whole-video export
with reconstruction limited to chosen intervals.

Write to a sibling temporary output, finish and verify, then publish atomically. Never
overwrite the source. Cancellation terminates inference descendants or cancels export,
releases buffers, removes incomplete outputs, preserves the last success and restores
usable controls. Guard asynchronous completion with job identity so stale work cannot
publish after rapid edits or video replacement.

## Verification and acceptance

Use the existing harness, with a deterministic temporal test backend for pipeline tests
and an explicitly enabled real SeedVR2 integration test. Cover geometry (letterboxing,
orientation, edges, odd sizes, padding), known-value compositing (unchanged outside pixels,
full/zero strength, smooth feather, identical A/B masks), consecutive frame order/PTS,
context trimming, chunk boundaries, bounded memory, invalidation and cancellation.

Exercise missing models, unsupported scales, memory refusal, process failure, corrupt
output, inference/export cancellation, video replacement, rapid selection changes,
missing audio and nonstandard frame rates. Run `./run-tests.sh` and `./build.sh` without
new warnings. Inspect a real textured/moving clip through selection, reconstruction,
synchronized comparison, boundary inspection, export and reopened video/audio. Repeat
for each enabled scale; a 2× proof says nothing about 4×/8× success.

The feature is complete only when the native selection, genuine temporal inference, exact
ROI restoration, compositing, synchronized comparison/transport/viewport, Original,
interval export, cancellation and error handling all work and have been tested on real
video, with existing WaifuFinder functionality intact.

## Source navigation and processing display

Before reconstruction or while editing, a Source seek bar and previous/next buttons
move by actual source frames, preserving the ROI and updating interval start. Container
edit mappings are applied to media timestamps, including variable-rate and B-frame
timing. Held arrows wait for each read; fresh presses and scrubbing supersede pending
reads. The source and processed-result transports are separate. Intervals containing
fewer than five frames disable Reconstruct with a visible explanation.

The algorithm selector uses native small segmented buttons: SeedVR2 / FlashVSR 1.1.
While inference or export runs, the canvas is replaced by progress. Source seeking,
comparison and playback are disabled until the job finishes; Cancel remains available.
