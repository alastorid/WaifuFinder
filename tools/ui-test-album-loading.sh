#!/bin/bash
# Reviewable interactive check: restart the built app, open only the named
# albums, require actual ready thumbnail cells, and save window screenshots
# plus accessibility/diagnostic logs under .build/ui-album-loading/.
# Does not change filters, favorites, album membership, or iCloud settings.
# Run ONLY through ./run-ui-tests.sh tools/ui-test-album-loading.sh ALBUM...
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
APP="$ROOT/dist/WaifuFinder.app"
OUT="$ROOT/.build/ui-album-loading/$(date +%Y%m%d-%H%M%S)"
BUNDLE="com.alastorid.waifufinder"
[ "$#" -gt 0 ] || { echo "Specify the album names to check." >&2; exit 2; }
mkdir -p "$OUT"
SDK="$(xcrun --sdk macosx --show-sdk-path)"
"$(xcrun --find swiftc)" -swift-version 6 -sdk "$SDK" \
    "$ROOT/tools/ui-album-loading-helper.swift" -o "$OUT/helper" \
    -framework AppKit -framework ApplicationServices
osascript -e "tell application id \"$BUNDLE\" to quit" >/dev/null 2>&1 || true
sleep 1
open -n "$APP" --args --waifu-trace=grid,analysis,loading
ARGS=(--bundle "$BUNDLE" --out "$OUT")
for ALBUM in "$@"; do ARGS+=(--album "$ALBUM"); done
STATUS=0
"$OUT/helper" "${ARGS[@]}" > "$OUT/report.txt" 2>&1 || STATUS=$?
cat "$OUT/report.txt"
if [ -f "$OUT/app-pid.txt" ]; then
    APP_PID="$(cat "$OUT/app-pid.txt")"
    TRACE="${TMPDIR:-/tmp/}waifu-trace-$APP_PID.log"
    if [ -f "$TRACE" ]; then cp "$TRACE" "$OUT/app-trace.log"; fi
fi
echo "Album check results: $OUT"
exit "$STATUS"
