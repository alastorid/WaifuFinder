#!/bin/bash
# Compatibility entry point: inference uses the native Swift/MLX runtime.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/tools/setup-native-uncover.sh" "$@"
