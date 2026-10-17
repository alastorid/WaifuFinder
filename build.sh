#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

APP="$ROOT/dist/WaifuFinder.app"
BINARY="$APP/Contents/MacOS/WaifuFinder"
DEPLOYMENT_TARGET="15.0"
CONFIGURATION="${WAIFU_BUILD_CONFIGURATION:-release}"
case "$CONFIGURATION" in
    release|debug) ;;
    *) echo "build: WAIFU_BUILD_CONFIGURATION must be release or debug" >&2; exit 1 ;;
esac
BUILD_STARTED=$SECONDS

ARCH="${WAIFU_ARCH:-$(uname -m)}"
case "$ARCH" in
    arm64|x86_64) ;;
    *) echo "build: unsupported architecture '$ARCH'" >&2; exit 1 ;;
esac
TARGET="${ARCH}-apple-macosx${DEPLOYMENT_TARGET}"

VERSION="${WAIFU_VERSION:-0.1.0}"
if ! printf '%s' "$VERSION" | grep -qE '^[0-9]+(\.[0-9]+){0,2}$'; then
    echo "build: invalid WAIFU_VERSION '$VERSION'" >&2; exit 1
fi

if ! SWIFTC="$(xcrun --find swiftc 2>/dev/null)" \
   || ! SWIFT="$(xcrun --find swift 2>/dev/null)" \
   || ! SDK="$(xcrun --sdk macosx --show-sdk-path 2>/dev/null)"; then
    echo "build: no usable Swift toolchain" >&2; exit 1
fi

BUILD_LOG="$(mktemp -t waifu-build)"
trap 'rm -f "$BUILD_LOG"' EXIT

diagnose_swift_failure() {
    grep -qE "SDK is not supported by the compiler|redefinition of module 'SwiftBridging'" "$1" || return 0
    cat >&2 <<DIAG
build: Swift toolchain and macOS SDK mismatch.
  compiler: $SWIFTC
  SDK:      $SDK
DIAG
    exit 1
}

SOURCES=()
while IFS= read -r SOURCE; do SOURCES+=("$SOURCE"); done < <(find "$ROOT/Sources" -name '*.swift' -type f | sort)
if [ "${#SOURCES[@]}" -eq 0 ]; then echo "build: no Swift sources under Sources/" >&2; exit 1; fi
CACHE="$ROOT/.build/app/$TARGET/$CONFIGURATION-${WAIFU_COMMERCIAL_DISTRIBUTION:-0}"
mkdir -p "$CACHE"
# Include build logic, compiler and SDK identity as well as source contents.
# Cache the unsigned executable separately: bundling/signing must not mutate it.
SETTINGS_KEY="$({
    shasum -a 256 "$ROOT/build.sh"
    printf '%s\n' "$SWIFTC" "$SDK" "$TARGET" "$CONFIGURATION" "${WAIFU_COMMERCIAL_DISTRIBUTION:-0}"
    "$SWIFTC" --version
    xcrun --sdk macosx --show-sdk-version
    shasum -a 256 "$SDK/SDKSettings.json"
} | shasum -a 256 | cut -d ' ' -f 1)"
SOURCE_KEY="$({ printf '%s\n' "$SETTINGS_KEY"; shasum -a 256 "${SOURCES[@]}"; } | shasum -a 256 | cut -d ' ' -f 1)"

echo "Building WaifuFinder for $TARGET ($VERSION, $CONFIGURATION)..."
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
# Remove resources left by the abandoned SVFR integration in existing bundles.
rm -f "$APP/Contents/Resources/Runtime/waifu-svfr" \
    "$APP/Contents/Resources/NativeVideoRuntime/waifu-svfr" \
    "$APP/Contents/Resources/NativeSeedVR2/waifu-svfr" \
    "$APP/Contents/Helpers/waifu-svfr" "$APP/Contents/Resources/OpenCV-Preprocessing-NOTICE.txt"

if [ "$ARCH" = arm64 ]; then
    "$ROOT/tools/build-native-uncover.sh"
    NATIVE="$APP/Contents/Resources/Runtime"
    PREVIOUS_NATIVE="$APP/Contents/Resources/NativeVideoRuntime"
    LEGACY_NATIVE="$APP/Contents/Resources/NativeSeedVR2"
    if [ ! -e "$NATIVE" ]; then
        if [ -d "$PREVIOUS_NATIVE" ]; then
            mv "$PREVIOUS_NATIVE" "$NATIVE"
        elif [ -d "$LEGACY_NATIVE" ]; then
            mv "$LEGACY_NATIVE" "$NATIVE"
        fi
    fi
    if [ -d "$PREVIOUS_NATIVE" ]; then rm -rf "$PREVIOUS_NATIVE"; fi
    if [ -d "$LEGACY_NATIVE" ]; then rm -rf "$LEGACY_NATIVE"; fi
    rm -f "$APP/Contents/Resources/NativeVideoRuntime-NOTICE.md" \
        "$APP/Contents/Resources/NativeSeedVR2-NOTICE.md"
    mkdir -p "$NATIVE"
    # Migrate artifacts from the earlier layout; Metal data is a resource, not nested code.
    for ARTIFACT in waifu-seedvr2 mlx.metallib; do
        if [ -f "$APP/Contents/Helpers/$ARTIFACT" ]; then
            mv -f "$APP/Contents/Helpers/$ARTIFACT" "$NATIVE/$ARTIFACT"
        fi
    done
    if [ -d "$APP/Contents/Helpers" ]; then rmdir "$APP/Contents/Helpers"; fi
    cp "$ROOT/.native-runtime/waifu-seedvr2" "$NATIVE/waifu-seedvr2"
    cp "$ROOT/.native-runtime/mlx.metallib" "$NATIVE/mlx.metallib"
    cp "$ROOT/.native-runtime/waifu-flashvsr" "$NATIVE/waifu-flashvsr"
    cp "$ROOT/.native-runtime/waifu-pgtformer" "$NATIVE/waifu-pgtformer"
    cp "$ROOT/.native-runtime/waifu-wdtagger" "$NATIVE/waifu-wdtagger"
    ditto "$ROOT/.native-runtime/WaifuRuntime_WDTaggerMLX.bundle" "$NATIVE/WaifuRuntime_WDTaggerMLX.bundle"
    cp "$ROOT/.native-runtime/pos_emb.safetensors" "$APP/Contents/Resources/pos_emb.safetensors"
    # Keep the full license texts for every third-party package linked into the
    # native workers together in the app. The previous bundle omitted licenses
    # for transitive MLX components, Swift Numerics and InferKit.
    THIRD_PARTY="$APP/Contents/Resources/ThirdPartyLicenses"
    rm -f "$APP/Contents/Resources/Runtime-NOTICE.md" \
        "$APP/Contents/Resources/FlashVSR-NOTICE.md" \
        "$APP/Contents/Resources/FlashVSR-LICENSE" \
        "$APP/Contents/Resources/RealESRGAN-NOTICE.md" \
        "$APP/Contents/Resources/RealESRGAN-LICENSE" \
        "$APP/Contents/Resources/MLX-Swift-LICENSE" \
        "$APP/Contents/Resources/MLX-LICENSE"
    rm -rf "$THIRD_PARTY"
    mkdir -p "$THIRD_PARTY"
    cp "$ROOT/Runtime/APP-THIRD-PARTY-NOTICES.md" "$THIRD_PARTY/README.md"
    cp "$ROOT/LICENSE" "$THIRD_PARTY/WaifuFinder-LICENSE"
    cp "$ROOT/docs/Licensing.md" "$THIRD_PARTY/WaifuFinder-Licensing.md"
    cp "$ROOT/Runtime/THIRD_PARTY_LICENSES.md" "$THIRD_PARTY/THIRD_PARTY_LICENSES.md"
    cp "$ROOT/Runtime/INFERKIT-LICENSE" "$THIRD_PARTY/INFERKIT-LICENSE"
    cp "$ROOT/Runtime/NOTICE.md" "$THIRD_PARTY/Runtime-NOTICE.md"
    cp "$ROOT/Runtime/FLASHVSR-NOTICE.md" "$THIRD_PARTY/FlashVSR-NOTICE.md"
    cp "$ROOT/Runtime/FLASHVSR-LICENSE" "$THIRD_PARTY/FlashVSR-LICENSE"
    cp "$ROOT/Runtime/REALESRGAN-NOTICE.md" "$THIRD_PARTY/RealESRGAN-NOTICE.md"
    cp "$ROOT/Runtime/REALESRGAN-LICENSE" "$THIRD_PARTY/RealESRGAN-LICENSE"
    cp "$ROOT/Runtime/FLASHVSR-LICENSE" "$THIRD_PARTY/Apache-2.0-LICENSE"
    MLX="$ROOT/Runtime/.build/checkouts/mlx-swift"
    cp "$MLX/LICENSE" "$THIRD_PARTY/MLX-Swift-LICENSE"
    cp "$MLX/Source/Cmlx/mlx/LICENSE" "$THIRD_PARTY/MLX-LICENSE"
    cp "$MLX/Source/Cmlx/mlx-c/LICENSE" "$THIRD_PARTY/MLX-C-LICENSE"
    cp "$MLX/Source/Cmlx/json/LICENSE.MIT" "$THIRD_PARTY/Nlohmann-JSON-LICENSE"
    cp "$MLX/Source/Cmlx/fmt/LICENSE" "$THIRD_PARTY/fmt-LICENSE"
    cp "$MLX/Source/Cmlx/metal-cpp/LICENSE.txt" "$THIRD_PARTY/Metal-CPP-LICENSE"
    cp "$ROOT/Runtime/.build/checkouts/swift-numerics/LICENSE.txt" "$THIRD_PARTY/Swift-Numerics-LICENSE"
    codesign --force --sign - "$NATIVE/waifu-seedvr2"
    codesign --force --sign - "$NATIVE/waifu-flashvsr"
    codesign --force --sign - "$NATIVE/waifu-pgtformer"
    codesign --force --sign - "$NATIVE/waifu-wdtagger"
fi

# The native Metal TSP port is present on every architecture, including builds
# that use its CPU fallback. Preserve the upstream license and modification notice.
mkdir -p "$APP/Contents/Resources/ThirdPartyLicenses"
cp "$ROOT/ThirdParty/cuopt/LICENSE" "$APP/Contents/Resources/ThirdPartyLicenses/cuOpt-LICENSE"
cp "$ROOT/ThirdParty/cuopt/NOTICE" "$APP/Contents/Resources/ThirdPartyLicenses/cuOpt-NOTICE"

# Distribution mode. PGTFormer's upstream licence permits non-commercial use only, so it
# cannot be offered from a build meant for commercial use; WaifuFinder's own code is MIT.
# WAIFU_COMMERCIAL_DISTRIBUTION=1 compiles the offer out and omits its release descriptor,
# leaving the permissive model runtimes and Apple's native media frameworks. The default
# build offers PGTFormer and is therefore non-commercial. See
# Sources/Waifu/Model/WaifuDistribution.swift and docs/Licensing.md.
RELEASES=(seedvr2 seedvr2-7b flashvsr realesrgan wd-tagger)
SWIFT_FLAGS=()
if [ "${WAIFU_COMMERCIAL_DISTRIBUTION:-0}" = "1" ]; then
    SWIFT_FLAGS+=(-D WAIFU_COMMERCIAL_DISTRIBUTION)
    echo "Building a COMMERCIAL distribution: PGTFormer is excluded."
else
    RELEASES+=(pgtformer)
fi

# Remove FFmpeg artifacts left by an older build so the app bundle contains no LGPL media runtime.
rm -f "$APP/Contents/Frameworks/"libavformat*.dylib "$APP/Contents/Frameworks/"libavcodec*.dylib "$APP/Contents/Frameworks/"libavutil*.dylib "$APP/Contents/Frameworks/"libswscale*.dylib
rm -rf "$APP/Contents/Resources/ThirdPartyLicenses/FFmpeg"

# Release retains whole-module optimization. Debug uses Swift's dependency
# graph and persistent per-file outputs for faster edit/build cycles.
COMPILE_STARTED=$SECONDS
if [ -f "$CACHE/WaifuFinder" ] && [ "$(cat "$CACHE/source-key" 2>/dev/null || true)" = "$SOURCE_KEY" ]; then
    echo "App sources unchanged; using cached $CONFIGURATION executable."
else
    OPTIMIZATION=(-O -whole-module-optimization)
    if [ "$CONFIGURATION" = debug ]; then
        OBJECTS="$CACHE/objects-$SETTINGS_KEY"
        mkdir -p "$OBJECTS"
        json_path() {
            local VALUE="$1"
            VALUE="${VALUE//\\/\\\\}"
            VALUE="${VALUE//\"/\\\"}"
            printf '"%s"' "$VALUE"
        }
        {
            printf '{"": {"swift-dependencies": '
            json_path "$OBJECTS/master.swiftdeps"
            printf '}'
            for SOURCE in "${SOURCES[@]}"; do
                OBJECT="$OBJECTS/${SOURCE#"$ROOT/Sources/"}"
                mkdir -p "$(dirname "$OBJECT")"
                printf ',\n'; json_path "$SOURCE"; printf ': {"object": '
                json_path "$OBJECT.o"; printf ', "swift-dependencies": '
                json_path "$OBJECT.swiftdeps"; printf '}'
            done
            printf '}\n'
        } > "$CACHE/output-map.json.new"
        if ! cmp -s "$CACHE/output-map.json.new" "$CACHE/output-map.json"; then
            mv "$CACHE/output-map.json.new" "$CACHE/output-map.json"
        else
            rm "$CACHE/output-map.json.new"
        fi
        OPTIMIZATION=(-Onone -incremental -enable-batch-mode -j "${WAIFU_SWIFT_JOBS:-3}" -output-file-map "$CACHE/output-map.json")
    fi
    # Invalidate before compiling so an interrupted/failed build cannot be a hit.
    rm -f "$CACHE/source-key"
"$SWIFTC" \
    -target "$TARGET" \
    -sdk "$SDK" \
    -parse-as-library \
    -swift-version 6 \
    -warnings-as-errors \
    "${OPTIMIZATION[@]}" \
    -g \
    -o "$CACHE/WaifuFinder" \
    "${SWIFT_FLAGS[@]+"${SWIFT_FLAGS[@]}"}" \
    "${SOURCES[@]}" \
    -framework Network \
    -framework AppKit \
    -framework AVFoundation \
    -framework AVKit \
    -framework CoreMedia \
    -framework Vision \
    -framework Metal \
    -framework Accelerate \
    -framework Photos \
    -framework CoreImage \
    -framework CoreML \
    -framework UniformTypeIdentifiers \
    2>"$BUILD_LOG" || { diagnose_swift_failure "$BUILD_LOG"; cat "$BUILD_LOG"; exit 1; }

    printf '%s\n' "$SOURCE_KEY" > "$CACHE/source-key"
fi
cp "$CACHE/WaifuFinder" "$BINARY"
echo "App compile: $((SECONDS - COMPILE_STARTED))s"

# Pinned releases are part of the app's native model manager. A descriptor whose
# revision is still the placeholder means the bundle was never published, and
# shipping it would give every user a download that cannot succeed.
for RELEASE in "${RELEASES[@]}"; do
    if [ ! -f "$ROOT/tools/$RELEASE-release.json" ]; then
        echo "build: tools/$RELEASE-release.json is missing." >&2
        echo "      Publish the bundle, then rerun tools/upload-hf-bundles.py to pin it." >&2
        exit 1
    fi
    if grep -q '"revision": "0\{40\}"' "$ROOT/tools/$RELEASE-release.json"; then
        echo "build: tools/$RELEASE-release.json has no published revision." >&2
        echo "      Run tools/stage-hf-bundles.sh && tools/upload-hf-bundles.py first." >&2
        exit 1
    fi
done
# Clear first: switching from the default build to a commercial one must not leave the
# PGTFormer descriptor behind in an existing dist/ tree, or the commercial bundle would
# still carry the pointer it is meant not to ship.
rm -rf "$APP/Contents/Resources/ModelReleases"
mkdir -p "$APP/Contents/Resources/ModelReleases"
DESCRIPTORS=()
for RELEASE in "${RELEASES[@]}"; do DESCRIPTORS+=("$ROOT/tools/$RELEASE-release.json"); done
cp "${DESCRIPTORS[@]}" "$APP/Contents/Resources/ModelReleases/"

# Bundle
rm -rf "$APP/Contents/Resources/FirefoxExtension"
cp -R "$ROOT/FirefoxExtension" "$APP/Contents/Resources/FirefoxExtension"
rm -f "$APP/Contents/Resources/WaifuFinderFirefox.xpi"
( cd "$ROOT/FirefoxExtension" && /usr/bin/zip -q "$APP/Contents/Resources/WaifuFinderFirefox.xpi" manifest.json background.js popup.html popup.js )

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key><string>en</string>
    <key>CFBundleExecutable</key><string>WaifuFinder</string>
    <key>CFBundleIdentifier</key><string>com.alastorid.waifufinder</string>
    <key>NSHumanReadableCopyright</key><string>Video loading and decoding use Apple's AVFoundation media frameworks.</string>
    <key>CFBundleName</key><string>WaifuFinder</string>
    <key>CFBundleDisplayName</key><string>WaifuFinder</string>
    <key>CFBundleShortVersionString</key><string>$VERSION</string>
    <key>CFBundleVersion</key><string>$VERSION</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleURLTypes</key><array><dict><key>CFBundleURLName</key><string>Media URL</string><key>CFBundleURLSchemes</key><array><string>waifufinder</string></array></dict></array>
    <key>CFBundleSignature</key><string>????</string>
    <key>LSMinimumSystemVersion</key><string>15.0</string>
    <key>NSLocalNetworkUsageDescription</key><string>Connect to another Mac for remote tagging and reconstruction, or share this Mac’s installed models.</string>
    <key>NSPhotoLibraryUsageDescription</key><string>Choose a Photos album to analyze its photos directly in WaifuFinder.</string>
    <key>NSAppleEventsUsageDescription</key><string>WaifuFinder can show a selected photo in Photos when you choose Show in Photos from its thumbnail menu.</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>CFBundleIconFile</key><string>AppIcon</string>
</dict>
</plist>
PLIST

echo "APPL????" > "$APP/Contents/PkgInfo"
# Python is reserved for offline conversion and reference validation.
rm -f "$APP/Contents/Resources/uncover_seedvr2.py"

# The icon is drawn from code, not copied from a checked-in blob: see
# tools/make-icon.swift. A failure here is not fatal to the build — an app without
# an icon still runs, it just shows the generic macOS one — but it is worth saying
# out loud rather than silently shipping a bundle whose Info.plist names an icon
# that is not there.
ICON="$APP/Contents/Resources/AppIcon.icns"
ICON_CACHE="$ROOT/.build/app/icon-$SETTINGS_KEY"
ICON_KEY="$(shasum -a 256 "$ROOT/tools/make-icon.swift" | cut -d ' ' -f 1)"
mkdir -p "$ICON_CACHE"
if [ ! -f "$ICON_CACHE/$ICON_KEY.icns" ]; then
    echo "Drawing the icon"
    if ! "$SWIFT" -sdk "$SDK" "$ROOT/tools/make-icon.swift" "$ICON_CACHE/$ICON_KEY.new.icns" 2>&1 | tee "$BUILD_LOG"; then
        rm -f "$ICON_CACHE/$ICON_KEY.new.icns" "$ICON"
        echo "build: could not render the icon; it will be the generic macOS one" >&2
    elif [ -f "$ICON_CACHE/$ICON_KEY.new.icns" ]; then
        mv "$ICON_CACHE/$ICON_KEY.new.icns" "$ICON_CACHE/$ICON_KEY.icns"
    fi
fi
if [ -f "$ICON_CACHE/$ICON_KEY.icns" ]; then cp "$ICON_CACHE/$ICON_KEY.icns" "$ICON"; fi

codesign --force --sign - "$APP"
codesign --verify --deep --strict "$APP"

echo "Build complete in $((SECONDS - BUILD_STARTED))s: $APP"
