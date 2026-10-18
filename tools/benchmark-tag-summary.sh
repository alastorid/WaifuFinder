#!/bin/bash
# Background compute only; no windows, PhotoKit access or interactive UI actions.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$(mktemp -d "${TMPDIR:-/tmp}/waifu-tag-summary-benchmark.XXXXXX")"
trap 'rm -rf "$BUILD"' EXIT
SDK="$(xcrun --sdk macosx --show-sdk-path)"
SWIFTC="$(xcrun --find swiftc)"
SOURCES=()
while IFS= read -r SOURCE; do SOURCES+=("$SOURCE"); done < <(find "$ROOT/Sources" -name '*.swift' ! -path '*/App/*' | sort)
"$SWIFTC" -O -swift-version 6 -sdk "$SDK" -target "$(uname -m)-apple-macosx15.0" \
    "${SOURCES[@]}" "$ROOT/tools/benchmark-tag-summary.swift" \
    -framework Network -framework AppKit -framework AVFoundation -framework AVKit \
    -framework CoreMedia -framework Vision -framework Metal -framework Accelerate \
    -framework Photos -framework CoreImage -framework UniformTypeIdentifiers \
    -o "$BUILD/benchmark"
"$BUILD/benchmark"
