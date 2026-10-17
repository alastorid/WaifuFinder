#!/bin/bash
# Bootstrap a fresh clone. Model installs and benchmarks are separate options.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
BENCHMARKS=0
case "${1:-}" in
    '') ;;
    --benchmarks) BENCHMARKS=1 ;;
    --help|-h)
        echo 'Usage: ./tools/setup-development.sh [--benchmarks]'
        echo 'Build the macOS app; optionally generate synthetic WD Tagger inputs and prepare benchmark models/tools.'
        exit 0 ;;
    *) echo 'Usage: ./tools/setup-development.sh [--benchmarks]' >&2; exit 1 ;;
esac
[ "$#" -le 1 ] || { echo 'Unexpected extra arguments' >&2; exit 1; }
[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || {
    echo 'Development setup requires an Apple Silicon Mac with macOS 15 or later.' >&2; exit 1;
}
[ "$(sw_vers -productVersion | cut -d . -f 1)" -ge 15 ] || { echo 'macOS 15 or later is required.' >&2; exit 1; }
xcrun --find swiftc >/dev/null
xcrun --sdk macosx --show-sdk-path >/dev/null
./build.sh
if [ "$BENCHMARKS" -eq 1 ]; then ./tools/setup-wd-tagger-benchmarks.sh; fi
echo 'Development app ready: dist/WaifuFinder.app'
echo 'Open the app and use Settings > Models to install inference models.'
