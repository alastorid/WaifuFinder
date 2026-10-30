#!/bin/bash
set -euo pipefail
WD_JPEG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WD_JPEG_SNAPSHOT="$(mktemp -d "${TMPDIR:-/tmp}/waifu-jpeg-build.XXXXXX")"
trap 'rm -rf "$WD_JPEG_SNAPSHOT"' EXIT
WD_JPEG_OUTPUT="$WD_JPEG_SNAPSHOT/benchmark"
# Other chats can edit the shared checkout during this background build.
ditto "$WD_JPEG_ROOT/Sources" "$WD_JPEG_SNAPSHOT/Sources"
cp "$WD_JPEG_ROOT/tools/wd-tagger-remote-jpeg-benchmark.swift" "$WD_JPEG_SNAPSHOT/benchmark.swift"
WD_JPEG_SOURCES=()
while IFS= read -r file; do WD_JPEG_SOURCES+=("$file"); done < <(find "$WD_JPEG_SNAPSHOT/Sources" -name '*.swift' ! -path '*/App/*' | sort)
xcrun swiftc -O -swift-version 6 -parse-as-library \
    -sdk "$(xcrun --sdk macosx --show-sdk-path)" -target "$(uname -m)-apple-macosx15.0" \
    "${WD_JPEG_SOURCES[@]}" "$WD_JPEG_SNAPSHOT/benchmark.swift" -o "$WD_JPEG_OUTPUT" \
    -framework Network -framework AppKit -framework AVFoundation -framework AVKit \
    -framework CoreMedia -framework Vision -framework Metal -framework Accelerate \
    -framework Photos -framework ImageIO -framework CoreImage -framework UniformTypeIdentifiers
"$WD_JPEG_OUTPUT" "$@"
