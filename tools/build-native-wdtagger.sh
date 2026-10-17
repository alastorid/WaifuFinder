#!/bin/bash
set -euo pipefail
WD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$WD_ROOT/tools/native-toolchain/prepare.sh"
WD_JOBS="${WAIFU_SWIFT_JOBS:-3}"
case "$WD_JOBS" in ''|*[!0-9]*|0) echo 'Invalid WAIFU_SWIFT_JOBS' >&2; exit 1;; esac
WD_LIBRARY="${WAIFU_MLX_METALLIB:-$WD_ROOT/.native-runtime/mlx.metallib}"
[ -f "$WD_LIBRARY" ] || { echo 'Supply the native MLX 0.31.1 metallib using WAIFU_MLX_METALLIB.' >&2; exit 1; }
WD_EXPECTED=928bdaf4d28683442b31bf11760f4dec600fe705e2a6367aaa3de0c5801d2066
WD_ACTUAL="$(shasum -a 256 "$WD_LIBRARY" | cut -d ' ' -f 1)"
[ "$WD_ACTUAL" = "$WD_EXPECTED" ] || { echo 'Native MLX metallib version/checksum mismatch' >&2; exit 1; }
swift build --package-path "$WD_ROOT/Runtime" -c release --product waifu-wdtagger-validation --jobs "$WD_JOBS"
WD_BIN_DIR="$(swift build --package-path "$WD_ROOT/Runtime" -c release --show-bin-path)"
if [ ! -e "$WD_BIN_DIR/mlx.metallib" ]; then ln -s "$WD_LIBRARY" "$WD_BIN_DIR/mlx.metallib"; fi
if [ "$#" -gt 0 ]; then exec "$WD_BIN_DIR/waifu-wdtagger-validation" "$@"; fi
echo "Native WDTagger ready: $WD_BIN_DIR/waifu-wdtagger-validation"
