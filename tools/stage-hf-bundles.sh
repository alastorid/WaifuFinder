#!/bin/bash
# Stage the hosted PGTFormer and RealESRGAN Core ML bundles.
#
# Both bundles are self-contained: the Core ML .mlpackage sources, a manifest
# that pins every file, the upstream licence, attribution and the validation
# record produced by the offline conversion. WaifuFinder downloads and verifies
# them like SeedVR2 and FlashVSR, so nothing needs Python or Terminal at
# install time.
#
# This script only assembles files and writes manifests. Uploading is a separate
# step (tools/upload-hf-bundles.sh) because the release revision is only known
# after the first push.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${1:-$ROOT/models/published}"
PGT_SOURCE="${PGTFORMER_NATIVE_MODEL:-$ROOT/.native-runtime/pgtformer}"
REAL_SOURCE="$ROOT/.native-runtime/realesrgan-optimized"

[ -d "$PGT_SOURCE/PGTEncoder.mlpackage" ] || {
    echo "Missing $PGT_SOURCE/PGTEncoder.mlpackage. Run tools/setup-native-pgtformer.sh first." >&2
    exit 1
}
[ -d "$REAL_SOURCE/RealESRGANValidated.mlpackage" ] || {
    echo "Missing $REAL_SOURCE/RealESRGANValidated.mlpackage. Run tools/setup-native-realesrgan.sh --convert-official first." >&2
    exit 1
}

sha() { shasum -a 256 "$1" | awk '{print $1}'; }

# manifest DIR OUTPUT -- pin the model members only. Licences, attribution, the
# card and the validation record are support files: those are pinned by the
# release descriptor instead, exactly as SeedVR2 and FlashVSR do, so the two sets
# stay disjoint and every file is verified exactly once.
manifest() {
    python3 - "$1" "$2" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, output = Path(sys.argv[1]), Path(sys.argv[2])
files = {
    str(path.relative_to(root)): hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
    for path in sorted(root.rglob("*"))
    if path.is_file() and ".mlpackage/" in str(path.relative_to(root))
}
if not files:
    raise SystemExit(f"{root} contains no model files to pin")
output.write_text(json.dumps({"format": 1, "fileSHA256": files}, indent=2) + "\n")
print(f"{output.name}: {len(files)} pinned model files")
PY
}

# ---------------------------------------------------------------------------
# PGTFormer-Swift-CoreML
# ---------------------------------------------------------------------------
PGT="$STAGE/PGTFormer-Swift-CoreML"
python3 "$ROOT/tools/stage-pgtformer.py" --source "$PGT_SOURCE" --stage "$STAGE"

# ---------------------------------------------------------------------------
# RealESRGAN-x2plus-Optimized-CoreML
# ---------------------------------------------------------------------------
REAL="$STAGE/RealESRGAN-x2plus-Optimized-CoreML"
rm -rf "$REAL"
mkdir -p "$REAL"
cp -R "$REAL_SOURCE/RealESRGANValidated.mlpackage" "$REAL/"
cp "$ROOT/Runtime/REALESRGAN-LICENSE" "$REAL/LICENSE"
cp "$REAL_SOURCE/BasicSR-LICENSE" "$REAL/BasicSR-LICENSE"
cp "$ROOT/docs/realesrgan-optimization-benchmark.json" "$REAL/validation.json"

cat > "$REAL/NOTICE" <<'NOTICE'
Real-ESRGAN x2plus, optimized Core ML package for WaifuFinder
Model weights copyright (c) 2021 Xintao Wang. BSD 3-Clause; see LICENSE.

Original weights: https://github.com/xinntao/Real-ESRGAN (RealESRGAN_x2plus.pth)
Reference architecture: https://github.com/XPixelGroup/BasicSR (revision 8d56e3a);
the architecture source and its Apache-2.0 licence are included as
BasicSR-LICENSE.

WaifuFinder converts these weights to Core ML offline. The conversion fuses the
network's first convolution into the following upsampling stage, expresses the
network's input and output shapes as enumerated Core ML shapes, and records the
native validation result in the model's metadata. The weights themselves are
unchanged: manifest.json pins the SHA-256 of the original checkpoint and of the
architecture definition.

This package replaces the pre-converted RealESRGAN_x2plus_522_fp16 model from
hanxiao/real-esrgan-coreml that earlier builds downloaded. WaifuFinder's version
adds validated 256 and 384 pixel inputs, so small tiles spend far less work than
padding to 522, and it is the only variant with a recorded original-model
parity result. See validation.json.

Inference uses only native Swift and Core ML. Conversion used coremltools and
PyTorch offline.
NOTICE

cat > "$REAL/README.md" <<'README'
# Real-ESRGAN x2plus — optimized Core ML

Self-contained framewise 2× upscaler for
[WaifuFinder](https://github.com/alastorid/WaifuFinder). WaifuFinder downloads
these files directly: no Python, no source checkpoint, no conversion step.

## Contents

| Path | What |
| --- | --- |
| `RealESRGANValidated.mlpackage` | Converted, original-model-validated model |
| `manifest.json` | Architecture, original fingerprints, SHA-256 of every file |
| `validation.json` | Original-model parity, tile and product checks |
| `LICENSE` | Real-ESRGAN weights, BSD 3-Clause |
| `BasicSR-LICENSE` | Reference architecture, Apache-2.0 |
| `NOTICE` | Attribution and conversion summary |

The app compiles the package once at install time, on the machine that will run
it, because compiled Core ML is tied to the local compiler and macOS version.

## What "optimized" means

The network is the unmodified Real-ESRGAN x2plus. The conversion:

- fuses the first convolution into the following stage;
- expresses 256, 384 and 522 pixel inputs as enumerated Core ML shapes, so a
  small tile is upscaled at its own size instead of being padded to 522;
- records the measured parity against the original PyTorch network in the model
  metadata, and the app refuses to use a package that is not that validated one.

`validation.json` records the measurements: maximum absolute error against the
original network stays at or below 0.0069 across every supported input size,
against a 0.01 acceptance bound.

## Provenance

Weights: [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN),
`RealESRGAN_x2plus.pth`. Architecture reference:
[XPixelGroup/BasicSR](https://github.com/XPixelGroup/BasicSR) revision
`8d56e3a`.

## Licence

BSD 3-Clause (weights). See `LICENSE`, `BasicSR-LICENSE` and `NOTICE`.
README

manifest "$REAL" "$REAL/manifest.json"

python3 - "$REAL_SOURCE/manifest.json" "$REAL/manifest.json" <<'PY'
import json, sys
from pathlib import Path
source = json.loads(Path(sys.argv[1]).read_text())
target_path = Path(sys.argv[2])
target = json.loads(target_path.read_text())
for key in ("algorithm", "weightsSHA256", "architectureSHA256", "basicsrRevision",
            "blocks", "features", "growth", "scale", "firstConvFusion", "inputSizes",
            "adaptiveContextMinimum", "contextValidation", "computeUnits",
            "nativeValidation", "compiledSourceSHA256"):
    if key in source:
        target[key] = source[key]
target["validated"] = True
# The app identifies its own package by these three metadata fields; see
# NativeSpatialUpscaler. The staged files have to be the validated ones.
for name, digest in source["compiledSourceSHA256"].items():
    if target["fileSHA256"].get(f"RealESRGANValidated.mlpackage/{name}") != digest:
        raise SystemExit("staged package does not match the validated fingerprint: " + name)
target_path.write_text(json.dumps(target, indent=2) + "\n")
PY

# ---------------------------------------------------------------------------
# No .gitattributes is staged. Hugging Face creates every repository with one
# that already routes .bin and .mlmodel through LFS, and a second copy would
# collide with it on the first commit.
# ---------------------------------------------------------------------------

echo "Staged:"
for REPO in "$PGT" "$REAL"; do
    printf '  %-42s %s\n' "$(basename "$REPO")" "$(du -sh "$REPO" | awk '{print $1}')"
done
echo
echo "Upload with: tools/upload-hf-bundles.py"
