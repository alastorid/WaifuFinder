#!/bin/bash
#
# Interactive UI check for the Sort and Filter east egg.
#
# Opens the built app, enters the Photos workspace, opens an album, filters it down
# so the "shown" fraction is smaller than its total, then rests the pointer on that
# fraction and photographs the window: first while it merely holds still, then while
# the wave of light runs to the switched-off cloud in the title bar.
#
# It drives the mouse and reads the user's screen, so run it only through
# ../run-ui-tests.sh, which asks for approval before this script starts.
#
#   ../run-ui-tests.sh tools/ui-test-chiaroscuro-link.sh
#
# Screenshots and accessibility dumps land in .build/ui-chiaroscuro/.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
APP="$ROOT/dist/WaifuFinder.app"
OUT="$ROOT/.build/ui-chiaroscuro"
BUNDLE_ID="com.alastorid.waifufinder"
HELPER_SOURCE="$ROOT/tools/ui-chiaroscuro-helper.swift"

if [ ! -d "$APP" ]; then
    echo "ui-test: $APP does not exist; run ./build.sh first." >&2
    exit 2
fi

mkdir -p "$OUT"
rm -f "$OUT"/*.png "$OUT"/ax-*.txt 2>/dev/null || true

echo "==> Compiling the UI helper"
SDK="$(xcrun --sdk macosx --show-sdk-path)"
SWIFTC="$(xcrun -find swiftc)"
"$SWIFTC" -swift-version 6 -sdk "$SDK" -target "$(uname -m)-apple-macosx15.0" \
    "$HELPER_SOURCE" -o "$OUT/helper" \
    -framework AppKit -framework ApplicationServices

# The egg only reaches a cloud that is switched off, so the test makes sure of that
# rather than inheriting whatever the last session left behind.
echo "==> Setting $BUNDLE_ID photosDownloadFromICloud to off"
defaults write "$BUNDLE_ID" photosDownloadFromICloud -bool false

echo "==> Quitting any running copy"
osascript -e "tell application id \"$BUNDLE_ID\" to quit" >/dev/null 2>&1 || true
sleep 1

echo "==> Launching $APP"
open -n "$APP"

STATUS=0
"$OUT/helper" --bundle "$BUNDLE_ID" --out "$OUT" || STATUS=$?

echo "==> Quitting the app"
osascript -e "tell application id \"$BUNDLE_ID\" to quit" >/dev/null 2>&1 || true

echo "==> Screenshots and accessibility dumps in $OUT"
exit "$STATUS"