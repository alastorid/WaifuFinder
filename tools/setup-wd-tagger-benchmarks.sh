#!/bin/bash
# Recreate benchmark dependencies without reading another Mac's app settings.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [ "${1:-}" = --help ] || [ "${1:-}" = -h ]; then
    echo 'Usage: ./tools/setup-wd-tagger-benchmarks.sh'
    echo 'Generate synthetic inputs, download pinned WD Tagger models, and compile benchmark tools.'
    exit 0
fi
[ "$#" -eq 0 ] || { echo 'Unexpected argument' >&2; exit 1; }
python3 tools/generate-wd-tagger-inputs.py --force
python3 tools/restore-model-bundles.py wd-tagger wd-tagger-coreml
./tools/build-native-uncover.sh
./tools/build-native-wdtagger.sh
NATIVE="$ROOT/.build/wd-native"
COREML="$ROOT/.build/wd-optimization/coreml"
mkdir -p "$NATIVE" "$COREML" "$NATIVE/memory-pass"
# Model files stay in their verified bundle directory; these links are created
# relative to this clone and are never checked into Git.
ln -sfn ../../models/published/WD-SwinV2-Tagger-v3-Swift-MLX/model.safetensors "$NATIVE/model.safetensors"
cp models/published/WD-SwinV2-Tagger-v3-CoreML/wd-tagger-labels.csv "$COREML/wd-tagger-labels.csv"
xcrun swift tools/compile-coreml-model.swift \
    models/published/WD-SwinV2-Tagger-v3-CoreML/WDTagger.mlpackage "$COREML/model.mlmodelc"
./tools/build-wd-tagger-pipeline-benchmarks.sh
xcrun swiftc -O -swift-version 6 -parse-as-library \
    Sources/Waifu/Models/WDTaggerBackend.swift Sources/Waifu/Models/WDTaggerWorker.swift \
    Sources/Waifu/Engine/TaggerBackend.swift Sources/Waifu/Model/TagVocabulary.swift \
    Sources/Waifu/Model/TagProbabilityCodec.swift \
    Sources/Waifu/Video/ThumbnailRegenerator.swift Sources/Waifu/Video/ThumbnailStore.swift \
    Sources/Waifu/Vision/FramePictureArea.swift tools/wd-tagger-shape-benchmark.swift \
    -o .build/wd-optimization/shape-benchmark-v4 -framework AppKit -framework CoreML
ln -sfn ../../wd-optimization/shape-benchmark-v4 "$NATIVE/memory-pass/coreml-stream"
ln -sfn ../../wd-optimization/coreml/model.mlmodelc "$NATIVE/memory-pass/coreml-model.mlmodelc"
./tools/generate-wd-tagger-baselines.sh
echo 'WD Tagger benchmark dependencies restored. Node.js and macmon are needed for cooled comparison scripts.'
