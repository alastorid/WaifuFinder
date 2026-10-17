#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP="${WAIFU_APP_PATH:-$ROOT/dist/WaifuFinder.app}"
VERSION="${WAIFU_VERSION:-}"
OUTPUT_DIR="${WAIFU_OUTPUT_DIR:-$ROOT/dist}"
ARCH="${WAIFU_ARCH:-$(uname -m)}"

if [ -z "$VERSION" ] || ! printf '%s' "$VERSION" | grep -qE '^[0-9]+(\.[0-9]+){0,2}$'; then
    echo "package-dmg: set WAIFU_VERSION to a numeric version such as 1.2.3" >&2
    exit 1
fi
if [ ! -d "$APP" ]; then
    echo "package-dmg: app bundle not found: $APP" >&2
    exit 1
fi
if [ "$(uname -s)" != "Darwin" ]; then
    echo "package-dmg: disk images can only be created on macOS" >&2
    exit 1
fi
if [ "$ARCH" != "arm64" ]; then
    echo "package-dmg: this release image is for Apple silicon; got '$ARCH'" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
DMG="$OUTPUT_DIR/WaifuFinder-${VERSION}-macos-arm64.dmg"
STAGING="$(mktemp -d "${TMPDIR:-/tmp}/waifufinder-dmg.XXXXXX")"
MOUNT="$STAGING/mount"
ATTACHED=0
cleanup() {
    if [ "$ATTACHED" = 1 ]; then
        if ! hdiutil detach "$MOUNT" -quiet; then
            echo "package-dmg: could not detach $MOUNT; keeping temporary files for recovery" >&2
            return
        fi
    fi
    rm -rf "$STAGING"
}
trap cleanup EXIT

CONTENTS="$STAGING/contents"
mkdir -p "$CONTENTS/.background" "$MOUNT"
ditto "$APP" "$CONTENTS/WaifuFinder.app"
cp "$ROOT/docs/DMG-Read-Me.html" "$CONTENTS/Start Here.html"
ln -s /Applications "$CONTENTS/Applications"
xcrun swift "$ROOT/tools/make-dmg-background.swift" "$CONTENTS/.background/install.tiff"

# Write layout metadata on the mounted filesystem so the background alias remains
# valid after compression/remounting. No Finder, AppleScript, or desktop is needed.
hdiutil create \
    -volname "WaifuFinder ${VERSION}" \
    -srcfolder "$CONTENTS" \
    -ov -fs HFS+ -format UDRW \
    "$STAGING/layout.dmg"
hdiutil attach "$STAGING/layout.dmg" -mountpoint "$MOUNT" -nobrowse -noautoopen -quiet
ATTACHED=1
python3 "$ROOT/tools/configure-dmg-layout.py" "$MOUNT"
hdiutil detach "$MOUNT" -quiet
ATTACHED=0
hdiutil convert "$STAGING/layout.dmg" \
    -format UDZO -imagekey zlib-level=9 \
    -o "$STAGING/release.dmg"
hdiutil verify "$STAGING/release.dmg"
# Preserve an existing release until the replacement is complete and verified.
mv -f "$STAGING/release.dmg" "$DMG"
echo "Created $DMG"
