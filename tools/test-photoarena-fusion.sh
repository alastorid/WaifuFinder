#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/photoarena-fusion-test.XXXXXX")"
trap 'rm -rf "$TEST_BUILD"' EXIT
"$ROOT/tools/build-native-photoarena.sh"
cmake -S "$ROOT/Runtime/PhotoArena" -B "$ROOT/.native-runtime/photoarena-ninja" -DPHOTOARENA_TEST_OUTPUT="$TEST_BUILD"
cmake --build "$ROOT/.native-runtime/photoarena-ninja" --target photoarena-fusion-tests photoarena-sampling-tests photoarena-order-tests --parallel 4
"$TEST_BUILD/photoarena-fusion-tests"
"$TEST_BUILD/photoarena-sampling-tests" "$ROOT/models/photoarena/model.gguf"
xcrun swift -sdk "$(xcrun --sdk macosx --show-sdk-path)" "$ROOT/tools/fixtures-photoarena-roi.swift" "$ROOT/Tests/Fixtures/RealESRGAN/nasa-collins-896x1134.jpg" "$TEST_BUILD"
sips --cropToHeightWidth 336 448 "$TEST_BUILD/watermarked.png" --out "$TEST_BUILD/asymmetric.png" >/dev/null
"$TEST_BUILD/photoarena-order-tests" "$ROOT/models/photoarena/model.gguf" "$ROOT/models/photoarena/mmproj.gguf" "$TEST_BUILD/clean.png" "$TEST_BUILD/asymmetric.png"
"$TEST_BUILD/photoarena-order-tests" "$ROOT/models/photoarena/model.gguf" "$ROOT/models/photoarena/mmproj.gguf" "$ROOT/Tests/Fixtures/RealESRGAN/nasa-collins-896x1134.jpg" "$TEST_BUILD/clean.png"
