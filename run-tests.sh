#!/bin/bash
#
# Builds and runs the Waifu regression suite.
#
# SwiftPM does not work on this machine — `swift package dump-package` fails to link
# `libPackageDescription` even for a three-line manifest — so there is no `swift test`
# and no XCTest target. Instead the production sources and the tests are compiled
# together by `swiftc` into one throwaway executable in a temporary directory.
#
# The production sources are compiled *unmodified*: every case in Tests/ calls the
# real functions, so a failure means the real code misbehaved. Only
# `Sources/Waifu/App/WaifuFinderApp.swift` is excluded, because it owns `@main` and Tests/
# owns the entry point.
#
# Nothing here needs a network connection, the sample video, or root, and nothing
# touches the real cache at ~/Library/Application Support/Waifu or dist/. The only
# thing it writes into the tree is nothing — everything stays in the temporary
# directory. Video/audio fixtures use Apple system frameworks only.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$(mktemp -d "${TMPDIR:-/tmp}/waifu-tests.XXXXXX")"
trap 'rm -rf "$BUILD"' EXIT

DEPLOYMENT_TARGET="15.0"
ARCH="${WAIFU_ARCH:-$(uname -m)}"
case "$ARCH" in
    arm64|x86_64) ;;
    *)
        echo "run-tests: unsupported architecture '$ARCH' — set WAIFU_ARCH to arm64 or x86_64" >&2
        exit 1
        ;;
esac
TARGET="${ARCH}-apple-macosx${DEPLOYMENT_TARGET}"

# Resolve the compiler through `xcrun` and take its `-sdk` from the same developer
# directory, exactly as build.sh does. A compiler shadowed on PATH (Homebrew,
# Swift.org) paired with the system SDK fails with "this SDK is not supported by the
# compiler", usually alongside a `redefinition of module 'SwiftBridging'`.
SDK="$(xcrun --sdk macosx --show-sdk-path)"
SWIFTC="$(xcrun --find swiftc)"

# Every production source except the `@main` scaffold. UI/ is included: the drop
# acceptance rule and the session wiring are compiled into the same binary the tests
# run in, so a change to them is caught here rather than only in a launched app.
SOURCES=()
while IFS= read -r file; do
    SOURCES+=("$file")
done < <(find "$ROOT/Sources" -name '*.swift' ! -path "*/App/*" | sort)

TESTS=()
while IFS= read -r file; do
    TESTS+=("$file")
done < <(find "$ROOT/Tests" -name '*.swift' | sort)

if [ "${#SOURCES[@]}" -eq 0 ]; then
    echo "run-tests: no production sources found under $ROOT/Sources" >&2
    exit 1
fi
if [ "${#TESTS[@]}" -eq 0 ]; then
    echo "run-tests: no test sources found under $ROOT/Tests" >&2
    exit 1
fi

echo "==> Compiling ${#SOURCES[@]} production + ${#TESTS[@]} test sources for $TARGET"
"$SWIFTC" \
    -Onone \
    -swift-version 6 \
    -sdk "$SDK" \
    -target "$TARGET" \
    "${SOURCES[@]}" \
    "${TESTS[@]}" \
    -o "$BUILD/waifu-tests" \
    -framework Network \
    -framework AppKit \
    -framework AVFoundation \
    -framework AVKit \
    -framework CoreMedia \
    -framework Vision \
    -framework Metal \
    -framework Accelerate \
    -framework Photos \
    -framework ImageIO \
    -framework CoreImage \
    -framework UniformTypeIdentifiers

echo "==> Running"
TEST_EXECUTABLE="$BUILD/waifu-tests"
if [ "${WAIFU_TEST_PGT_INSTALL:-0}" = "1" ] || [ "${WAIFU_TEST_TAGGER_INSTALL:-0}" = "1" ] || [ "${WAIFU_TEST_SAFIRE_INSTALL:-0}" = "1" ] || [ "${WAIFU_TEST_TRUFOR_INSTALL:-0}" = "1" ]; then
    # Opt-in release/setup QA uses the real installer and the built app catalog.
    TEST_APP="$BUILD/ModelSetupTests.app"
    mkdir -p "$TEST_APP/Contents/MacOS" "$TEST_APP/Contents/Resources/Runtime"
    mv "$TEST_EXECUTABLE" "$TEST_APP/Contents/MacOS/waifu-tests"
    cp -R "$ROOT/dist/WaifuFinder.app/Contents/Resources/ModelReleases" "$TEST_APP/Contents/Resources/"
    cp "$ROOT/dist/WaifuFinder.app/Contents/Resources/Runtime/waifu-pgtformer" "$TEST_APP/Contents/Resources/Runtime/"
    cp "$ROOT/dist/WaifuFinder.app/Contents/Resources/Runtime/waifu-wdtagger" "$TEST_APP/Contents/Resources/Runtime/"
    if [ "${WAIFU_TEST_SAFIRE_INSTALL:-0}" = "1" ]; then
        cp "$ROOT/dist/WaifuFinder.app/Contents/Resources/Runtime/waifu-safire" "$TEST_APP/Contents/Resources/Runtime/"
    fi
    if [ "${WAIFU_TEST_TRUFOR_INSTALL:-0}" = "1" ]; then
        cp "$ROOT/dist/WaifuFinder.app/Contents/Resources/Runtime/waifu-trufor" "$TEST_APP/Contents/Resources/Runtime/"
    fi
    cp "$ROOT/dist/WaifuFinder.app/Contents/Resources/Runtime/mlx.metallib" "$TEST_APP/Contents/Resources/Runtime/"
    ditto "$ROOT/dist/WaifuFinder.app/Contents/Resources/Runtime/WaifuRuntime_WDTaggerMLX.bundle" "$TEST_APP/Contents/Resources/Runtime/WaifuRuntime_WDTaggerMLX.bundle"
    cat > "$TEST_APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict><key>CFBundleIdentifier</key><string>local.waifu.model-setup-tests</string><key>CFBundleExecutable</key><string>waifu-tests</string><key>CFBundlePackageType</key><string>APPL</string></dict></plist>
PLIST
    TEST_EXECUTABLE="$TEST_APP/Contents/MacOS/waifu-tests"
fi
# Log noise goes to a file rather than into the report; it is only shown if a case
# fails, so a red run is diagnosable and a green one is quiet.
STDERR_LOG="$BUILD/stderr.log"
set +e
"$TEST_EXECUTABLE" "$@" 2>"$STDERR_LOG"
STATUS=$?
set -e

if [ -s "$STDERR_LOG" ]; then
    echo
    echo "----- stderr from the test process -----"
    cat "$STDERR_LOG"
    echo "--------------------------------------"
fi

exit "$STATUS"
