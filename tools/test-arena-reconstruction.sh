#!/bin/bash
# Noninteractive, optimized production reconstruction tests; executables are temporary.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export WAIFU_TEST_OPTIMIZED=1
exec "$ROOT/run-tests.sh" --only 'PhotoArena reconstruction' "$@"
