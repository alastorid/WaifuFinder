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
- The model card on the Hub declares `grip-unina-trufor-non-commercial`; the
  weights are not relicensed, and WaifuFinder's own integration code remains MIT.

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
uses a fused Metal kernel. Large images run the noise body in 512-px tiles with a
17-px halo, which reproduces the exact receptive field without materializing the
whole intermediate. Attention chunks queries into fixed rows so memory stays
bounded on any input size. The worker exposes three paths over one identical
forward: batch scores (up to four images when their combined area fits the
validated batch envelope), single-image evidence, and a warm `--serve` protocol.

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
serialization; medians of three runs. Cold start adds about 0.19 s of model load.

| Host | Single image | Batch of four (per-image fallback) |
| --- | --- | --- |
| M3 | 1.79 s | 7.19 s ≈ 4 × 1.8 s |
| M4 (`thefreebao.local`, independently built) | 1.43 s | 5.72 s |

Images whose combined area exceeds the batch envelope predict per image rather
than trading memory for batching; small-image batches run as one fused forward.
These measurements do not establish a globally fastest configuration and do not
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
