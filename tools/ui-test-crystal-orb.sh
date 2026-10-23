#!/bin/bash
# Interactive smoke check. Invoke ONLY through ./run-ui-tests.sh.
# Restarts the built app, imports a generated video through its file picker,
# pastes an Apple-generated frame, visits all comparison modes, closes Orb,
# checks grid selection, then checks the Photos toolbar and paste activation.
# Restores clipboard contents. Saves AX trees and screenshots under .build/.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
TASK_OUT="$TASK_ROOT/.build/ui-crystal-orb/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$TASK_OUT"
xcrun swiftc -swift-version 6 -parse-as-library -target "$(uname -m)-apple-macosx15.0" \
    "$TASK_ROOT/Tests/NativeMediaFixture.swift" "$TASK_ROOT/tools/ui-crystal-orb-helper.swift" \
    -framework AppKit -framework ApplicationServices -framework AVFoundation -framework ImageIO \
    -o "$TASK_OUT/helper"
osascript -e 'tell application id "com.alastorid.waifufinder" to quit' >/dev/null 2>&1 || true
open -n "$TASK_ROOT/dist/WaifuFinder.app" --args --waifu-trace=grid,preview,focus
"$TASK_OUT/helper" "$TASK_OUT" > "$TASK_OUT/report.txt" 2>&1
cat "$TASK_OUT/report.txt"
echo "UI check artifacts: $TASK_OUT"
