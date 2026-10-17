#!/bin/bash
set -euo pipefail
WD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WD_ROOT"
WD_OUTPUT="$WD_ROOT/.build/wd-native/pipeline-tools"
mkdir -p "$WD_OUTPUT"
WD_TARGET="$(uname -m)-apple-macosx15.0"

xcrun swiftc -O -swift-version 6 -parse-as-library -target "$WD_TARGET" \
    tools/wd-tagger-worker-benchmark.swift -o "$WD_OUTPUT/worker-benchmark"

xcrun swiftc -O -swift-version 6 -parse-as-library -target "$WD_TARGET" \
    Sources/Waifu/Models/WDTaggerBackend.swift \
    Sources/Waifu/Models/WDTaggerWorker.swift \
    Sources/Waifu/Engine/TaggerBackend.swift \
    Sources/Waifu/Engine/TaggingBatchPipeline.swift \
    Sources/Waifu/Model/TagVocabulary.swift \
    Sources/Waifu/Model/TagProbabilityCodec.swift \
    Sources/Waifu/Video/ThumbnailRegenerator.swift \
    Sources/Waifu/Video/ThumbnailStore.swift \
    Sources/Waifu/Vision/FramePictureArea.swift \
    tools/wd-tagger-app-benchmark.swift \
    -o "$WD_OUTPUT/app-benchmark" \
    -framework AppKit -framework AVFoundation -framework CoreImage

echo "Native benchmark tools: $WD_OUTPUT"
