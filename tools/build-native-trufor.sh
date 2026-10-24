#!/bin/bash
set -euo pipefail
TRUFOR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$TRUFOR_ROOT/tools/native-toolchain/prepare.sh"
TRUFOR_LIBRARY="$TRUFOR_ROOT/.native-runtime/mlx.metallib"
[ -f "$TRUFOR_LIBRARY" ] || { echo 'Missing the pinned native MLX Metal library.' >&2; exit 1; }
[ "$(shasum -a 256 "$TRUFOR_LIBRARY" | cut -d ' ' -f 1)" = 928bdaf4d28683442b31bf11760f4dec600fe705e2a6367aaa3de0c5801d2066 ] || { echo 'MLX library checksum mismatch' >&2; exit 1; }
swift build --package-path "$TRUFOR_ROOT/Runtime" -c release --product waifu-trufor --jobs "${WAIFU_SWIFT_JOBS:-3}"
TRUFOR_BIN="$(swift build --package-path "$TRUFOR_ROOT/Runtime" -c release --show-bin-path)"
if [ ! -e "$TRUFOR_BIN/mlx.metallib" ]; then ln -s "$TRUFOR_LIBRARY" "$TRUFOR_BIN/mlx.metallib"; fi
cp -p "$TRUFOR_BIN/waifu-trufor" "$TRUFOR_ROOT/.native-runtime/waifu-trufor"
if [ "$#" -gt 0 ]; then exec "$TRUFOR_BIN/waifu-trufor" "$@"; fi
echo "Native TRUFOR worker: $TRUFOR_BIN/waifu-trufor"
