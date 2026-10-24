#!/bin/bash
set -euo pipefail
SAFIRE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SAFIRE_ROOT/tools/native-toolchain/prepare.sh"
SAFIRE_LIBRARY="$SAFIRE_ROOT/.native-runtime/mlx.metallib"
[ -f "$SAFIRE_LIBRARY" ] || { echo 'Missing the pinned native MLX Metal library.' >&2; exit 1; }
[ "$(shasum -a 256 "$SAFIRE_LIBRARY" | cut -d ' ' -f 1)" = 928bdaf4d28683442b31bf11760f4dec600fe705e2a6367aaa3de0c5801d2066 ] || { echo 'MLX library checksum mismatch' >&2; exit 1; }
swift build --package-path "$SAFIRE_ROOT/Runtime" -c release --product waifu-safire --jobs "${WAIFU_SWIFT_JOBS:-3}"
SAFIRE_BIN="$(swift build --package-path "$SAFIRE_ROOT/Runtime" -c release --show-bin-path)"
if [ ! -e "$SAFIRE_BIN/mlx.metallib" ]; then ln -s "$SAFIRE_LIBRARY" "$SAFIRE_BIN/mlx.metallib"; fi
cp -p "$SAFIRE_BIN/waifu-safire" "$SAFIRE_ROOT/.native-runtime/waifu-safire"
if [ "$#" -gt 0 ]; then exec "$SAFIRE_BIN/waifu-safire" "$@"; fi
echo "Native SAFIRE worker: $SAFIRE_BIN/waifu-safire"
