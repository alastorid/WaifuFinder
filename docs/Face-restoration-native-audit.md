# Native PGTFormer implementation audit

Audit date: 2026-10-11. Local hardware: Apple M3, 16 GB unified memory.
The requested M4 benchmarks cannot be substituted with measurements from this machine.

Current status (October 12, 2026): PGTFormer is installed and validated end to end
through native Core ML. SVFR was dropped because its processing cost did not meet
the one-minute requirement; its implementation, downloads and configuration were
removed. The architectural review below records how faithful PGTFormer components
were selected. No substitute network is offered under PGTFormer's name.

## Sources examined

All five repositories were cloned and their implementations inspected locally.
Revisions are fixed so that architectural comparisons can be reproduced.

| Implementation | Inspected revision | License of implementation |
|---|---|---|
| [InferKit](https://github.com/belisoful/InferKit) | `f4bdb7bf83802e321b44678e376c944c2e5de9a3` | MIT |
| [RestoreFormer Swift](https://github.com/xocialize/mlx-restoreformer-swift) | `c0a0106e118b4578889d22524af7c702b73fac7c` | Apache 2.0 |
| [GFPGAN Swift](https://github.com/xocialize/mlx-gfpgan-swift) | `2913dfad00206b86894db9d4b4f8a2c7f61967ce` | Apache 2.0 with third-party exceptions in its LICENSE |
| [SeedVR2 Swift](https://github.com/xocialize/seedvr2-mlx-swift) | `1dfa161ba9522bdf1295015fd0b00209a4e30093` | MIT port; model license separate |
| [LTX Swift](https://github.com/VincentGourbin/ltx-video-swift-mlx) | `0b7e48567fdc686478cb16463496c24f2eda5d1f` | MIT port; model license separate |

PGTFormer's own source LICENSE permits non-commercial use with redistribution
conditions; it is not an unrestricted MIT license. The offline converted model
is not redistributed by this change.
Licenses of model weights and inherited operators must be reviewed separately
before redistributing complete models. Repository claims about speed or parity
are not local validation results.

## PGTFormer: what matches

InferKit's `InferKitMLX/Sources/InferKitMLX/NFKMLXCodeFormer.swift` contains
usable residual normalization, attention and AdaIN implementations. Its
`NFKCFTransformerLayer` uses pre-LayerNorm, adds positional features to Q and K
only, splits a packed QKV projection, applies attention and a residual, then
uses another pre-LayerNorm and an exact GELU MLP. This matches PGTFormer's
original `TransformerSALayer`, with dimensions changed to 512, eight heads and
MLP width 1024. The original nine learned layers must all be loaded; the
complete CodeFormer model cannot replace PGTFormer.

The new `FaceRestorationMLX/PGTCodeFormer.swift` adapts these operators. It
requires all twelve official tensors with exact shapes and FP32 dtype, rejects
unexpected tensors, and provides no random initialization fallback. Its AdaIN
uses unbiased variance over the spatial positions with epsilon inside the
square root, matching the original code. MIT attribution is retained in
`Runtime/INFERKIT-LICENSE` and `THIRD_PARTY_LICENSES.md`.

InferKit's `NFKMLXBiSeNet.swift` offers compatible ResNet18 and context-path
primitives, but its complete network is different: PGTFormer's condition net
removes the spatial path, uses the ResNet feat8 path, and concatenates three
19-channel parsing heads into 57 channels at 32×32. PGTFormer uses bilinear
`align_corners=true` for the resized heads. These topology and interpolation
choices must be preserved, including evaluation BatchNorm running statistics.

InferKit's `NFKMLXVideoSR.swift` BasicVSR implements SPyNet optical flow,
bidirectional feature propagation, warping and pixel shuffle. Those are useful
examples of native video orchestration, **not matching PGTFormer layers**.
PGTFormer uses three-frame shifted-window attention and learned temporal
fusion. Introducing BasicVSR flow warping would change the network. BasicVSR's
retained full-clip feature arrays also do not meet the application's bounded
worker memory policy.

RestoreFormer Swift's `Sources/RestoreFormerMLXCore/RestoreFormer.swift`
provides useful checkpoint verification, NHWC layout handling, GroupNorm
(epsilon 1e-6), asymmetric right/bottom padding before stride-two convolution,
and nearest upsampling. Its facial codebook is selected by nearest Euclidean
distance. PGTFormer instead predicts 1024 logits per position and selects an
argmax, then gathers from a 1025×512 table that includes the padding row.
Neither RestoreFormer's 1024×256 table nor its nearest-neighbor decision can be
substituted. Its decoder cross-attention (encoder-derived Q, decoder K/V) also
does not match PGTFormer's global SALayer or its Swin window layers.

RestoreFormer's `oracle/convert.py` and strict `verify: .all` parameter updates
are good loading patterns. Conv2D weights change from OIHW to OHWI; rank-four
buffers should not be transposed indiscriminately. The new attention component
retains the packed linear weights in their original orientation.

`WinogradFreeConv2d.swift` and `Tests/GPULaneTests.swift` demonstrate that
convolution algorithm choice can affect numerical parity. Their exact
convolution route and GPU/CPU comparisons merit testing for the full PGT port;
compilation or successful tensor loading is insufficient. Their published
hardware measurements are not M3 or M4 measurements for this application.

GFPGAN's `Sources/GFPGANMLXCore/Ops.swift` implements explicit bilinear sampling
with separate `align_corners` semantics. This is a suitable reference for
PGTFormer's parsing interpolation.
`Sources/MLXGFPGAN/FaceAlign.swift` is useful for affine sampling and paste-back,
but its detection and landmark conventions need checking against each model's exact
alignment. It should not introduce mandatory automatic detection into the
existing manual-ROI workflow. GFPGAN's StyleGAN2 decoder and SFT network are
not PGTFormer. GFPGAN's FP16/BF16 observations do not establish a safe PGTFormer
precision policy. `StoreWeights.swift` shows how to reuse caches before
fetching a second copy; the application's existing model cache remains usable.

## PGTFormer: original network and export findings

The current production split uses an unchanged FP32 encoder/codebook and
an optimized FP16 GPU decoder. Native normalization replaces expanded GroupNorm
reductions, and the final per-frame output tail processes only the center frame
once every temporal operation has completed. The Torch graph remains exactly
equal before precision conversion. Original-model numerical gates and actual
video preprocessing pass; measured memory, timing, precision differences and
rejected ANE configurations are recorded in
[pgtformer-optimization-benchmark.json](pgtformer-optimization-benchmark.json).
Both stages now allow GPU execution; the encoder's previous CPU-only placement
was a substantial bottleneck. The installed split-package identifier is unchanged.
The GPU placement follow-up is recorded in
[pgtformer-gpu-benchmark.json](pgtformer-gpu-benchmark.json).
Nine code-prediction attention blocks now export as native SDPA, preserving
positional Q/K and the original V. The native comparison remains within the
original gates and output PNGs match the previous GPU model exactly. Encoder
GroupNorm fusion passed accuracy but showed no useful timing gain and is not
enabled by default. See [fusion measurements](pgtformer-fusion-benchmark.json).
The export investigations below describe earlier monolithic experiments.

Original source: `kepengxu/PGTFormer` revision
`e78a26cd463182f4a5aa96b8681110da8615d44d` (checkout records the remote).
Checkpoint: `kepeng/pgtformer-base`, revision
`bc0b7e3ff10cfa0f2f335f5256919de20fe2b29d`, SHA-256
`37f9a42b9e35e0ccb89016ca6cea6728a7bf783e70b99e886ac52bb37c6855f3`.
The complete pretrained state loads strictly into the original network.

Input is three RGB frames at 512×512 in [0,1]; inference restores the center
frame with repeated boundary frames. The encoder/decoder use temporal Swin
windows, 32×32 latent resolution, facial parsing positions, nine global
transformer layers, argmax code selection, AdaIN and four learned temporal SFT
fusion blocks. Face-network resizing is mandatory model preprocessing, not
optional source downsampling.

`tools/convert-pgtformer.py` is an offline Core ML alternative experiment. It
runs the original eager model first and writes input/output goldens. The first
naive trace had identical code logits and encoder features but restored-pixel
mean absolute error about 0.129 and maximum error 1 after clamping. CPU-only
and CPU/GPU Core ML execution both reproduced the trace error. It was therefore
an export failure, not evidence that a different model should be used.

Layer taps isolated the error: upstream converts tensor dimensions to string
keys for decoder fusion. PyTorch tracing produces `"tensor(32)"` instead of
`"32"`; all four fusion blocks are silently skipped. A fixed-shape adapter
resolves only these keys as Python integers. All neural operations and weights
remain original. The tool now compares the trace to the eager output before
conversion and rejects a mismatch. The corrected trace matches the eager output exactly on the recorded fixture.
Native Swift/Core ML execution passes both a distinct random three-frame fixture
and three frames from upstream SVFR's `assert/lq/lq1.mp4` face example. On the
face fixture, CPU/GPU maximum error is 0.00001663 and mean error 0.000000389.
Warm GPU inference takes approximately 2.1–2.2 seconds per restored center frame
on this M3; CPU-only takes approximately six seconds. Results and timings are
in `pgtformer-native-validation.json` and `pgtformer-face-validation.json`.
These validate the complete converted network on two fixtures, not the app's
crop preprocessing, end-to-end export, identity quality or a complete MLX port.
The converted package remains an offline alternative experiment.

## Native reference checks

Offline goldens from PGTFormer's original first pretrained transformer layer at 37 and 3072 tokens and original AdaIN at 3×512×32×32 were checked through `waifu-face-validation` on MLX Swift and Metal. All three retained PGTFormer checks passed; see `face-components-validation.json`. The converter rejects missing, extra and wrong-shape weights.

The complete native Core ML port is validated separately against original
pretrained inference; see `pgtformer-split-native-validation.json`,
`pgtformer-video-validation.json` and `pgtformer-downsampled-video-validation.json`.

## 1× implementation

The existing picker now includes **1× Original (No Downsampling)**. At this
factor the frame extractor directly keeps the original ROI crop for both
restoration input and baseline; it does not resize it. This is independent of
any future model-required 512×512 crop preprocessing. Original frame size,
ROI placement, audio and frame timing continue through the existing pipeline.

SeedVR2 performs zero spatial-upscale stages at 1×, does not load the spatial
Core ML model, and rejects any worker output with changed ROI dimensions.
FlashVSR accepts 1× source input without downsampling while keeping its tuned 4× output.
An odd 65×47 ROI test compares extracted pixels to the original crop and verifies
preview/export dimensions. Real native 3B and 7B runs also pass 1× inference and
export checks (six frames, 64×48 ROI in a 128×96 video; combined initial test 11.74 s).
A repeat with the spatial model configuration removed also passed (7.82 s
combined); `uncover-1x-native-validation.json` records both native reports.
These are small integration fixtures, not representative quality or performance
benchmarks for larger face crops or M4 hardware.

## Reproducing the offline validation

Use a dedicated environment and `tools/face-conversion-requirements.txt` (Python 3.13, torch 2.7.0, torchvision 0.22.0,
NumPy 2.2.6, coremltools 9.0, huggingface-hub 0.25.2, transformers 4.48.3,
diffusers 0.30.3, timm 1.0.30, safetensors and einops). The existing legacy
Uncover Python environment was restored to its previous principal dependency
versions after this experiment. No app operation launches these tools.

1. Clone the original model sources and check out the pinned revisions above.
2. Retrieve the original checkpoint/config files; retain their source and hashes.
3. Run `tools/convert-pgtformer.py` with `--upstream`, `--config`, `--weights`
   and `--output`. It checks the pinned checkpoint hash, produces the original
   reference, guards traced-graph parity and then validates Core ML output.
   For the actual face fixture, extract the first three frames of upstream
   `assert/lq/lq1.mp4` to PNGs and pass them in temporal order with
   `--input-frames FIRST.png SECOND.png THIRD.png --reference-only` to a separate
   output directory. This generates `reference.safetensors` for native comparison
   against the same previously converted package.
4. Build `waifu-face-validation` in the native Swift package. Place the matching
   MLX Metal library beside the executable, as the existing build setup does.
   Run it with the fixture directory and a report JSON path. The `--coreml-pgt`
   mode accepts a converted package, safetensors reference fixture and report
   path for complete graph execution. It benchmarks CPU-only and CPU/GPU and
   requires original-output parity on every run.
5. Run `./run-tests.sh --strict --only Uncover`; opt in to actual native 1× proof
   with `WAIFU_TEST_NATIVE_1X=1`. This proof explicitly removes the spatial model
   configuration, so it cannot accidentally depend on learned enlargement.

The pinned upstream code, model downloads, conversion environment and large
validation fixtures are generated under ignored `.build/model-conversion/` paths.
Reports and production conversion tools are tracked; model payloads are not added
to the application bundle.
