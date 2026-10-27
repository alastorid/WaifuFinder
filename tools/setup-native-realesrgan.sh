#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="$ROOT/.native-runtime"
CONFIG_DIR="$HOME/Library/Application Support/Waifu/Uncover"
VARIANT=x2plus
if [ "${1:-}" = --variant ]; then
    VARIANT="${2:-}"
    case "$VARIANT" in x2plus|x4plus) ;; *) echo 'Choose x2plus or x4plus' >&2; exit 2 ;; esac
    shift 2
fi
MODEL="$RUNTIME/realesrgan-optimized"
RELEASE="$ROOT/tools/realesrgan-release.json"
CONFIG_NAME=realesrgan-model.txt
SOURCE_NAME=realesrgan-reference
CHECKPOINT_RELEASE=v0.2.1
if [ "$VARIANT" = x4plus ]; then
    MODEL="$RUNTIME/realesrgan-x4plus-optimized"
    RELEASE="$ROOT/tools/realesrgan-x4plus-release.json"
    CONFIG_NAME=realesrgan-x4plus-model.txt
    SOURCE_NAME=realesrgan-x4plus-reference
    CHECKPOINT_RELEASE=v0.1.0
fi

case "${1:-}" in
    '')
        if [ ! -f "$RELEASE" ]; then echo 'This variant has no published release yet. Use --convert-official with an audited converter environment.' >&2; exit 1; fi
        echo "Installing the hosted optimized RealESRGAN $VARIANT Core ML package (about 33 MB)."
        xcrun swift "$ROOT/tools/download-coreml-release.swift" --release "$RELEASE" \
            --destination "$MODEL" --algorithm "realesrgan-$VARIANT"
        ;;
    --convert-official)
        # Advanced reproducibility path only. Normal setup never needs these originals.
        CONVERTER="${REALESRGAN_CONVERTER_PYTHON:-}"
        [ -x "$CONVERTER" ] || { echo 'Set REALESRGAN_CONVERTER_PYTHON to an external conversion environment.' >&2; exit 1; }
        SOURCE="$RUNTIME/$SOURCE_NAME"
        OUTPUT="$MODEL"
        REVISION=8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a
        mkdir -p "$SOURCE" "$OUTPUT"
        if [ ! -f "$SOURCE/RealESRGAN_${VARIANT}.pth" ]; then
            curl --fail --location --retry 3 https://github.com/xinntao/Real-ESRGAN/releases/download/$CHECKPOINT_RELEASE/RealESRGAN_${VARIANT}.pth --output "$SOURCE/RealESRGAN_${VARIANT}.pth"
        fi
        if [ ! -f "$SOURCE/rrdbnet_arch.py" ]; then
            curl --fail --location --retry 3 "https://raw.githubusercontent.com/XPixelGroup/BasicSR/$REVISION/basicsr/archs/rrdbnet_arch.py" --output "$SOURCE/rrdbnet_arch.py"
        fi
        if [ ! -f "$SOURCE/LICENSE" ]; then
            curl --fail --location --retry 3 "https://raw.githubusercontent.com/XPixelGroup/BasicSR/$REVISION/LICENSE.txt" --output "$SOURCE/LICENSE"
        fi
        printf '{"sha":"%s"}\n' "$REVISION" > "$SOURCE/basicsr-revision.json"
        cd "$ROOT"
        TILE_OPTIONS=()
        if [ "$VARIANT" = x4plus ]; then TILE_OPTIONS=(--x4-tile-size 512); fi
        "$CONVERTER" tools/convert-realesrgan-optimized.py --variant "$VARIANT" --source "$SOURCE" --output "$OUTPUT" "${TILE_OPTIONS[@]}"
        xcrun swift tools/compile-coreml-model.swift "$OUTPUT/RealESRGAN.mlpackage" "$OUTPUT/RealESRGAN.mlmodelc"
        xcrun swiftc -O -parse-as-library Sources/Waifu/Uncover/NativeSpatialUpscaler.swift Runtime/Sources/WaifuPGTFormer/ComputeProfiling.swift tools/realesrgan-benchmark.swift -o "$OUTPUT/benchmark"
        SCALE=2; if [ "$VARIANT" = x4plus ]; then SCALE=4; fi
        "$OUTPUT/benchmark" --scale "$SCALE" --model "$OUTPUT/RealESRGAN.mlmodelc" --fixtures "$OUTPUT/fixtures" --units ane --repeats 2 --compute-plan --report "$OUTPUT/native-validation.json"
        "$CONVERTER" tools/seal-realesrgan-optimized.py "$OUTPUT" --validation "$OUTPUT/native-validation.json" --units ane
        xcrun swift tools/compile-coreml-model.swift "$OUTPUT/RealESRGANValidated.mlpackage" "$OUTPUT/RealESRGANValidated.mlmodelc"
        cp "$SOURCE/LICENSE" "$OUTPUT/BasicSR-LICENSE"
        MODEL="$OUTPUT"
        ;;
    --*) echo 'Usage: setup-native-realesrgan.sh [--variant x2plus|x4plus] [--convert-official]' >&2; exit 1 ;;
esac

# Compiled Core ML is tied to the local compiler and macOS version, so it is built
# here rather than downloaded. The app does the same at install time.
COMPILED="$RUNTIME/RealESRGAN_${VARIANT}_optimized.mlmodelc"
xcrun swift "$ROOT/tools/compile-coreml-model.swift" "$MODEL/RealESRGANValidated.mlpackage" "$COMPILED"

mkdir -p "$CONFIG_DIR"
printf '%s\n' "$COMPILED" > "$CONFIG_DIR/$CONFIG_NAME"
echo "RealESRGAN configured: $COMPILED"
