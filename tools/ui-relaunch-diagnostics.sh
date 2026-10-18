#!/bin/bash
# Interactive launch only: restart the built app with grid diagnostics enabled.
# The user chooses albums manually after the script returns control.
# Run only through ./run-ui-tests.sh tools/ui-relaunch-diagnostics.sh
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
APP="$ROOT/dist/WaifuFinder.app"
[ -d "$APP" ] || { echo "Build the app first." >&2; exit 2; }
osascript -e 'tell application id "com.alastorid.waifufinder" to quit' >/dev/null 2>&1 || true
sleep 1
open -n "$APP" --args --waifu-trace=grid,analysis,loading
echo "Diagnostic app launched. Choose the albums manually; no mouse or keyboard automation is running."
