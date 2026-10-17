#!/bin/bash
# Native compute only; this benchmark never opens or controls the user's UI.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$(mktemp -d "${TMPDIR:-/tmp}/waifu-visual-benchmark.XXXXXX")"
trap 'rm -rf "$BUILD"' EXIT
SDK="$(xcrun --sdk macosx --show-sdk-path)"
SWIFTC="$(xcrun --find swiftc)"
"$SWIFTC" -O -swift-version 6 -sdk "$SDK" -target "$(uname -m)-apple-macosx15.0" \
    "$ROOT/Sources/Waifu/Grouping/CuOptTSP.swift" \
    "$ROOT/Sources/Waifu/Grouping/CuOptMetalTSP.swift" \
    "$ROOT/Sources/Waifu/Grouping/VisualFeatureVectors.swift" \
    "$ROOT/tools/benchmark-visual-sort.swift" \
    -framework Foundation -framework Vision -framework Metal -framework Accelerate \
    -o "$BUILD/benchmark"
"$BUILD/benchmark"
