# TruFor for Photos

WaifuFinder's Photos workspace scores still photos with GRIP-UNINA's
[TruFor](https://github.com/grip-unina/TruFor) (Guillaro, Cozzolino, Sud, Dufour,
Verdoliva — CVPR 2023). The score comes from TruFor's trained detection head over
confidence-weighted localization statistics — never from mask area, heuristics or
synthetic numbers. This document records the conversion, the reference validation,
the measured performance and the licence posture. It is written to be checked.

## Licence posture

The upstream licence permits **informational and nonprofit use only**, requires
retaining all original notices, and requires crediting the original authors
whenever results obtained with the software are made public. WaifuFinder honors
all three:

- The full GRIP-UNINA licence travels with the weights (`LICENSE.txt` in the
  package, `Runtime/TRUFOR-LICENSE.txt` in the repository) and the CMX components
  keep their MIT notice (`LICENSE_CMX.txt`).
- The package `NOTICE` credits the authors and the paper, and states the
  redistribution terms. The Settings row repeats the restriction in plain words.
- TruFor is offered only in the non-commercial build, decided at compile time by
  `WaifuDistribution` — the same gate as PGTFormer, so a commercial build never
  ships the offer.
- The model card on the Hub declares `grip-unina-trufor-nonprofit`; the
  weights are not relicensed, and WaifuFinder's own integration code remains MIT.
- On 2026-10-25 a TruFor collaborator confirmed in
  [grip-unina/TruFor#36](https://github.com/grip-unina/TruFor/issues/36) that the
  pretrained weights are covered by the same licence as the code, which restricts use
  to nonprofit purposes, and that all redistributions retain the original licence and
  remain subject to its terms, including the nonprofit-use restriction — the terms the
  published package travels with.

## Conversion

`tools/convert-trufor.py` extracts the 927 inference tensors from the
checksum-verified official `TruFor.pth` archive into a safetensors file, with no
Torch, NumPy, ONNX or OpenCV runtime anywhere in the shipped path. The converted
package records the source checkpoint SHA-256
(`ac1d90e329a72e0d66e8665e123a19e94bfae3209c3ef8a4f9ca3b91578c7844`), the upstream
revision, the conversion version and the exact preprocessing policy: native
oriented RGB, `uint8 / 256`, no resize and no padding — the upstream convention,
including its division by 256 rather than 255. The worker refuses images that
would need silent resizing; geometry outside the validated envelope is an error.

## Native runtime

`Runtime/Sources/TruForMLX` implements the full architecture in Swift and MLX:
the Noiseprint++ DnCNN body, the CMX Mit-B2 encoder with FRM rectification and
FFM fusion, both decode heads, and the official detection head. Input conversion
uses a fused Metal kernel. Large images run the noise body in 1024-px tiles with
a 17-px halo, which reproduces the exact receptive field without materializing
the whole intermediate. The worker exposes three paths over one identical
forward: batch scores (up to four images when their combined area fits the
1M-pixel batch envelope), single-image evidence, and a warm `--serve` protocol.

## Measured optimization

The runtime is tuned by the same method as the WD Tagger kernel review: phase
profiling, paired interleaved runs, and parity gates before anything ships.

Adopted, each measured:

- **Per-shape compiled forward.** The serve path compiles the whole graph per
  input geometry with the noise body outside the compiled graph, so its
  memory-bounding evaluations survive. This removed most of the per-request
  graph-construction overhead (1.79 s → 1.46 s at 1024×768 on a quiet M3).
- **FP16 weights and activations** with FP32 islands — every LayerNorm, every
  BN including the DnCNN BN rows, and the whole detection head. FP16 remains an explicit worker benchmark option; production uses FP32,
  which passes the strict validation gate.
  Measured against the official FP32 reference: detection score within 0.002,
  anomaly and reliability maps within 0.012, Noiseprint++ relative RMS 0.087,
  and unchanged masks at the 0.5 threshold. The strict path re-passes the
  published gates (score ≤ 1e-5) after the refactor.
- **Decoder kernels pre-sliced** into four contiguous per-part weights at load,
  removing a strided weight read from every decode pass.
- **Noise tile 1024** (from 512), worth several percent at 1024×768; per-tile
  evaluations keep large-image memory bounded.

Measured and rejected: a custom direct 3×3 conv Metal kernel for the DnCNN
(92–96 ms per layer versus 26 ms for MLX's implicit-GEMM FP16 — weight reuse
does not amortize at 64 channels); joint batching beyond the 1M-pixel envelope
(1.36 s per image versus 1.30 s single); attention query-chunk changes
(neutral, ratio 0.992). MLX's conv is already within about 1.5× of this
device's compute bound for these shapes, so the body is not memory-bound and
the removed kernel was not competitive.

## Reference validation

`tools/compare-trufor-activations.py` compares native activations against the
official reference on odd, native-resolution geometry (516×644). Both the single
path and the tiled path pass every gate: relative RMS ≤ 5e-5 across all four
fusion stages, score absolute error ≤ 1e-5, and anomaly-mask IoU = 1.0. The same
comparison on the independently built M4 host passes with identical error
magnitudes, so the GPU path is deterministic across devices. The published
package carries the full report as `validation.json`, and the app re-verifies it
against pinned hashes at install time; a folder that has not passed official
score and spatial validation is refused.

## Measured performance

Warm worker, 1024×768, complete inference including decode, forward and result
serialization; medians of paired interleaved runs. Absolute clocks vary with
foreground load on a shared Mac, so the paired ratios are the reliable claim.

| Configuration | Median | Ratio |
| --- | ---: | ---: |
| M3 FP16 experiment (compiled, tile 1024) | 1.458 s | 0.919 |
| M3 FP32 production path | 1.586 s | 1.000 |
| M4 FP16 experiment (independently built) | 1.021 s | 0.922 |
| M4 FP32 production path | 1.108 s | 1.000 |

The FP16/FP32 ratio reproduces across both devices (0.919 / 0.922). On a quiet
M3 earlier the same day the configured path measured 1.30 s against the
pre-optimization 1.79 s (about 1.38×); on the M4 the configured path measures
1.02 s against the pre-optimization 1.43 s (about 1.40×). The M4 FP32 outputs
match the M3 FP32 outputs at the same error magnitudes, so cross-device
determinism holds for the strict path. Evidence mode is within 3% of the score
path. Images whose combined area exceeds the batch envelope
predict per image; small-image batches run as one fused forward. These
measurements do not establish a globally fastest configuration and do not
guarantee a busy machine's timing.

## Distribution and verification

The validated package is published at
[`alastorid/TruFor-Swift-MLX`](https://huggingface.co/alastorid/TruFor-Swift-MLX)
under an immutable `releases/trufor-native-v1-<identity>` path, pinned by
`tools/trufor-release.json`. `./build.sh` stages the descriptor into the app, the
installer downloads and hash-verifies every file, `TruForConfiguration.validate`
re-checks the manifest, the licence and the validation report, and the native
worker validates the weights before activation. A clean download through the real
installer returned bit-identical detection scores to the locally converted
package, and the encrypted-loopback test compares local and remote evidence on
the published weights (`WAIFU_TEST_TRUFOR_INSTALL=1 ./run-tests.sh --only "TruFor integration"`).

## Review corrections (2026-10-25)

Production now uses FP32 for both scalar scores and evidence. The previous FP16
default did not pass the published strict spatial/score gates (especially the
Noiseprint residual). The app pins worker precision to FP32, cache identities
include the execution revision, and remote capabilities require
`trufor-fp32-v2`, so old workers and old precision results cannot silently mix.
The immutable weights package and its artifact revision remain unchanged.
FP16 timing figures above describe an experiment, not the current default.
Compiled graphs retain at most four photo geometries.

Noninteractive regression tests now execute the actual Metal outline renderer
on portrait and landscape maps, compare every pixel against clamped morphology,
and reopen the scalar cache through a separate reader. This caught invalid Metal
casts and colliding buffer bindings that prevented overlays from rendering,
a width-only coordinate clamp,
and scalar cache records with zero analysis width that the shared cache reader
rejected on reopen. Source changes also cancel forensic jobs immediately; changes
to the current album snapshot cancel stale work before refresh.
