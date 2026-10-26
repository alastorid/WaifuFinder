#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="$ROOT/.native-runtime/trufor-dncnn-coreml-fp16"
DEST="${1:-$ROOT/.native-runtime/trufor-coreml-fp16-hf}"

[ -d "$SOURCE/model.mlpackage" ] || { echo 'Missing the validated TruFor FP16 Core ML package.' >&2; exit 1; }
[ ! -e "$DEST" ] || { echo "Refusing to overwrite existing staging directory: $DEST" >&2; exit 1; }
mkdir -p "$DEST"
cp -R "$SOURCE/model.mlpackage" "$DEST/model.mlpackage"
jq '.compiledModel = "Not bundled; compile model.mlpackage locally on the target Mac"' \
    "$SOURCE/conversion.json" > "$DEST/conversion.json"
cp "$ROOT/.native-runtime/trufor-publish-v1/LICENSE.txt" "$DEST/LICENSE.txt"
cp "$ROOT/.native-runtime/trufor-publish-v1/LICENSE_CMX.txt" "$DEST/LICENSE_CMX.txt"
cp "$ROOT/.native-runtime/trufor-publish-v1/NOTICE" "$DEST/NOTICE"

cat > "$DEST/README.md" <<'EOF'
---
license: other
license_name: grip-unina-trufor-nonprofit
library_name: coreml
tags:
  - coreml
  - apple-silicon
  - image-forensics
  - image-forgery-detection
  - fp16
---

# TruFor DnCNN Core ML FP16

This repository contains an FP16-compute Core ML conversion of only TruFor's
17-convolution DnCNN Noiseprint++ subnetwork. It is not a complete TruFor model.
The full CMX encoder, localization/confidence heads, and trained detection head
are not in this package. WaifuFinder runs those remaining stages with its native
Swift/MLX FP32 implementation.

The graph uses FP16 weights and internal activations where supported by Core ML;
its public RGB and validity-mask inputs and noiseprint output are FP32. The fixed
input shape is `[1, 3, 546, 546]` for RGB plus `[1, 1, 546, 546]` for the validity
mask. Inputs use opaque sRGB RGB divided by 256. A 17-pixel halo surrounds each
512-pixel output tile. The portable artifact is `model.mlpackage`; compile it
locally on the target Mac before use. The inference runtime is Apple Core ML and
native Swift/Objective-C; no Python is in the inference path.

## M4 measurements

Measurements use the same verified weights and 2000×2000 fixture on an Apple M4
running macOS 26.4.1. The FP16 DnCNN tile was compared with the native MLX FP32
reference over the central 512×512 pixels: maximum absolute error `0.2437461019`
and relative RMS error `0.5971685071`. The FP32 Core ML counterpart measured
`0.0004192144` maximum absolute error and `0.0007005027` relative RMS error.

In the full 4 MP TruFor pipeline, with this Core ML FP16 DnCNN output and the
remaining stages in native FP32, the result score was `0.0847633481`; the all
native FP32 score was `0.0779616311` (a `+8.72%` shift for this image). Two warm
repeats returned the same score. One full prediction took `12.61 s`; repeated
worker timings were `15.25 s` then `12.24 s`. The measured process peak was
`1.063 GB`, below the 2 GB worker budget. These are single-image engineering
measurements, not a labeled detection-accuracy evaluation.

The score and noiseprint differences are intentional FP16 quantization effects.
Evaluate against the intended workflow before interpreting scores as calibrated
probabilities. See `validation.json` for machine-readable measurements and
`conversion.json` for source checkpoint and converter hashes.

## Source and license

TruFor is from [GRIP-UNINA](https://github.com/grip-unina/TruFor), CVPR 2023:
Fabrizio Guillaro, Davide Cozzolino, Avneesh Sud, Nicholas Dufour, and Luisa
Verdoliva, “TruFor: Leveraging All-Round Clues for Trustworthy Image Forgery
Detection and Localization.” The pretrained weights are covered by the same
license as the code. A GRIP-UNINA collaborator confirmed in
[issue #36](https://github.com/grip-unina/TruFor/issues/36) that redistributions
must retain the original license and remain limited to nonprofit use.

**Use, reproduction, modification, and redistribution are restricted to
informational and nonprofit purposes.** Preserve all original notices and credit
the original authors whenever publishing results. The GRIP-UNINA license is in
`LICENSE.txt`; the CMX MIT notice is in `LICENSE_CMX.txt`. This FP16 conversion
does not relicense the source weights. WaifuFinder's separate integration code is
not included in this artifact.
EOF

cat > "$DEST/validation.json" <<'EOF'
{
  "device": "Apple M4",
  "macOS": "26.4.1",
  "sourceWeightsSHA256": "3b2c7146303e984c43b51f26e38d093c8b2ebba9a13e41bb029a98aa5b29783c",
  "submodel": "TruFor DnCNN Noiseprint++ 17-convolution body only",
  "coreMLPrecision": "FP16 internal weights and activations; FP32 interface",
  "tile": {
    "inputShapeRGB": [1, 3, 546, 546],
    "inputShapeValidity": [1, 1, 546, 546],
    "outputCompared": [512, 512],
    "haloPixels": 17,
    "fp16NoiseprintMaxAbsoluteError": 0.24374610185623169,
    "fp16NoiseprintRelativeRMS": 0.5971685071177092,
    "fp32CoreMLMaxAbsoluteError": 0.0004192143678665161,
    "fp32CoreMLRelativeRMS": 0.0007005027332581356
  },
  "fullPipeline": {
    "image": "2000x2000 maximum.png",
    "nativeFP32Score": 0.07796163111925125,
    "coreMLFP16DnCNNScore": 0.08476334810256958,
    "scoreRelativeChangePercent": 8.724442,
    "coreMLFP16WarmWorkerProfileMilliseconds": [15247.0, 12239.4],
    "coreMLFP16OneShotPredictionMilliseconds": 12612.7,
    "peakPhysicalBytes": 1062667584,
    "budgetBytes": 2000000000,
    "repeatedCoreMLFP16Scores": [0.08476334810256958, 0.08476334810256958],
    "evaluationNote": "One image; no labeled accuracy or calibration claim. Remaining TruFor stages used native MLX FP32."
  }
}
EOF

{
    printf '{\n  "format": "trufor-coreml-fp16-v1",\n  "fileSHA256": {\n'
    first=1
    while IFS= read -r file; do
        relative="${file#"$DEST"/}"
        digest="$(shasum -a 256 "$file" | awk '{print $1}')"
        if [ "$first" -eq 0 ]; then printf ',\n'; fi
        printf '    "%s": "%s"' "$relative" "$digest"
        first=0
    done < <(find "$DEST/model.mlpackage" -type f -print | sort)
    printf '\n  }\n}\n'
} > "$DEST/manifest.json"

echo "Staged Core ML FP16 model card and package: $DEST"
