#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/tools/native-toolchain/prepare.sh"
mkdir -p "$ROOT/.native-runtime"
rm -f .native-runtime/waifu-svfr .native-runtime/waifu-svfr.new
SWIFT_JOBS="${WAIFU_SWIFT_JOBS:-3}"
case "$SWIFT_JOBS" in
    ''|*[!0-9]*) echo "native build: invalid WAIFU_SWIFT_JOBS '$SWIFT_JOBS'" >&2; exit 1 ;;
esac
if [ "$SWIFT_JOBS" -lt 1 ]; then
    echo "native build: WAIFU_SWIFT_JOBS must be at least 1" >&2
    exit 1
fi
swift build --package-path Runtime -c release --jobs "$SWIFT_JOBS"
cp Runtime/.build/release/waifu-seedvr2 .native-runtime/waifu-seedvr2.new
codesign --force --sign - .native-runtime/waifu-seedvr2.new
mv -f .native-runtime/waifu-seedvr2.new .native-runtime/waifu-seedvr2
cp Runtime/.build/release/waifu-flashvsr .native-runtime/waifu-flashvsr.new
codesign --force --sign - .native-runtime/waifu-flashvsr.new
mv -f .native-runtime/waifu-flashvsr.new .native-runtime/waifu-flashvsr
cp Runtime/.build/release/waifu-pgtformer .native-runtime/waifu-pgtformer.new
codesign --force --sign - .native-runtime/waifu-pgtformer.new
mv -f .native-runtime/waifu-pgtformer.new .native-runtime/waifu-pgtformer
cp Runtime/.build/release/waifu-wdtagger .native-runtime/waifu-wdtagger.new
codesign --force --sign - .native-runtime/waifu-wdtagger.new
mv -f .native-runtime/waifu-wdtagger.new .native-runtime/waifu-wdtagger
ditto Runtime/.build/release/WaifuRuntime_WDTaggerMLX.bundle .native-runtime/WaifuRuntime_WDTaggerMLX.bundle
# Official MLX 0.31.1 Metal kernels, matching the core bundled by mlx-swift 0.31.2.
# Extract kernels only: no Python interpreter, wheel installation, or Python runtime.
# These are build artifacts, not model weights, so they live in their own Hub
# repository rather than in a model release.
EXPECTED_KERNEL=928bdaf4d28683442b31bf11760f4dec600fe705e2a6367aaa3de0c5801d2066
KERNEL_BASE="https://huggingface.co/alastorid/WaifuRuntime-Metal/resolve/6194c8819b66aafe41bca885ae3b607f5ded9e15"
if [ ! -f .native-runtime/mlx.metallib ]; then
    curl -L --fail --retry 2 --output .native-runtime/mlx.metallib.new "$KERNEL_BASE/mlx.metallib"
    ACTUAL="$(shasum -a 256 .native-runtime/mlx.metallib.new | cut -d ' ' -f 1)"
    [ "$ACTUAL" = "$EXPECTED_KERNEL" ] || { echo 'MLX kernel checksum mismatch' >&2; exit 1; }
    mv .native-runtime/mlx.metallib.new .native-runtime/mlx.metallib
fi
ACTUAL_KERNEL="$(shasum -a 256 .native-runtime/mlx.metallib | cut -d ' ' -f 1)"
[ "$ACTUAL_KERNEL" = "$EXPECTED_KERNEL" ] || { echo 'MLX kernel checksum mismatch' >&2; exit 1; }
if [ ! -f .native-runtime/pos_emb.safetensors ]; then
    xcrun swift tools/download-seedvr2.swift --release tools/seedvr2-release.json \
      --destination "$ROOT/.native-runtime/model" --only-embedding
    cp .native-runtime/model/pos_emb.safetensors .native-runtime/pos_emb.safetensors
fi
EXPECTED_EMBEDDING=d2e06cadb10f1f149b5aa781185b250eb60f0e64327f12efd17886cecc9c2984
ACTUAL_EMBEDDING="$(shasum -a 256 .native-runtime/pos_emb.safetensors | cut -d ' ' -f 1)"
[ "$ACTUAL_EMBEDDING" = "$EXPECTED_EMBEDDING" ] || { echo 'SeedVR2 embedding checksum mismatch' >&2; exit 1; }
# GPU parity checks are expensive; repeat them when their exact inputs change,
# and only record success after both workers pass.
SELF_TEST_KEY="$({
    shasum -a 256 "$ROOT/tools/build-native-uncover.sh" \
        .native-runtime/waifu-seedvr2 .native-runtime/waifu-flashvsr
    printf '%s\n' "$ACTUAL_KERNEL" "$ACTUAL_EMBEDDING" "$(sw_vers -buildVersion)"
} | shasum -a 256 | cut -d ' ' -f 1)"
if [ "$(cat .native-runtime/self-test-key 2>/dev/null || true)" != "$SELF_TEST_KEY" ]; then
    .native-runtime/waifu-seedvr2 --self-test
    .native-runtime/waifu-flashvsr --self-test
    printf '%s\n' "$SELF_TEST_KEY" > .native-runtime/self-test-key
else
    echo 'Native worker self-tests unchanged; reusing successful checks.'
fi
