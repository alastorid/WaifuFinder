#!/bin/bash
# Background native compute only. Deliberately capped at 10,000 candidates.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$(mktemp -d "${TMPDIR:-/tmp}/waifu-vector-benchmark.XXXXXX")"
trap 'rm -rf "$BUILD"' EXIT
SDK="$(xcrun --sdk macosx --show-sdk-path)"
SWIFTC="$(xcrun --find swiftc)"
SOURCES=(
    "$ROOT/Sources/Waifu/Grouping/CuOptTSP.swift"
    "$ROOT/Sources/Waifu/Grouping/CuOptMetalTSP.swift"
    "$ROOT/Sources/Waifu/Grouping/VisualFeatureVectors.swift"
    "$ROOT/Sources/Waifu/Vision/FeaturePrintCodec.swift"
    "$ROOT/Sources/Waifu/VectorSearch/"*.swift
)
FLAGS=(-D WAIFU_VECTOR_BENCHMARK)
if [ "${1:-}" = "--orb" ]; then
    shift
    SOURCES=()
    while IFS= read -r file; do SOURCES+=("$file"); done < <(rg --files "$ROOT/Sources" -g '*.swift' -g '!App/**' | sort | sed '\|/App/|d')
    FLAGS=(-D WAIFU_ORB_BENCHMARK)
fi
"$SWIFTC" -O -swift-version 6 -sdk "$SDK" -target "$(uname -m)-apple-macosx15.0" \
    "${FLAGS[@]}" "${SOURCES[@]}" "$ROOT/tools/benchmark-vector-search.swift" \
    -framework Foundation -framework Vision -framework Metal -framework Accelerate -lcompression \
    -framework CoreGraphics -framework AppKit -framework AVFoundation -framework AVKit \
    -framework CoreMedia -framework Photos -framework ImageIO -framework CoreImage \
    -framework Network -framework UniformTypeIdentifiers -o "$BUILD/benchmark"
"$BUILD/benchmark" "$@"
