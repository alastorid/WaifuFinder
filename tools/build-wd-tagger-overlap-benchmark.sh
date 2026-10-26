#!/bin/bash
set -euo pipefail
WD_OVERLAP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WD_OVERLAP_OUTPUT="${1:-$WD_OVERLAP_ROOT/.build/wd-overlap/benchmark}"
mkdir -p "$(dirname "$WD_OVERLAP_OUTPUT")"
xcrun swiftc -O -swift-version 6 -parse-as-library \
    "${WD_BENCH_BACKEND_SOURCE:-$WD_OVERLAP_ROOT/Sources/Waifu/Models/WDTaggerBackend.swift}" \
    "$WD_OVERLAP_ROOT/Sources/Waifu/Models/WDTaggerWorker.swift" \
    "$WD_OVERLAP_ROOT/Sources/Waifu/Engine/TaggerBackend.swift" \
    "$WD_OVERLAP_ROOT/Sources/Waifu/Engine/TaggingBatchPipeline.swift" \
    "$WD_OVERLAP_ROOT/Sources/Waifu/Model/TagVocabulary.swift" \
    "$WD_OVERLAP_ROOT/Sources/Waifu/Model/TagProbabilityCodec.swift" \
    "$WD_OVERLAP_ROOT/tools/wd-tagger-overlap-benchmark.swift" \
    -o "$WD_OVERLAP_OUTPUT" -framework AppKit -framework CoreImage -framework CoreML
