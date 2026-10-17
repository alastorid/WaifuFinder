#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="$ROOT/.native-runtime"
CONFIG_DIR="$HOME/Library/Application Support/Waifu/Uncover"
MODEL="$RUNTIME/realesrgan-optimized"

case "${1:-}" in
    '')
        echo 'Installing the hosted optimized RealESRGAN x2plus Core ML package (about 33 MB).'
        xcrun swift "$ROOT/tools/download-coreml-release.swift" --release "$ROOT/tools/realesrgan-release.json" \
            --destination "$MODEL" --algorithm realesrgan-x2plus
        ;;
    --convert-official)
        # Advanced reproducibility path only. Normal setup never needs these originals.
        CONVERTER="${REALESRGAN_CONVERTER_PYTHON:-}"
        [ -x "$CONVERTER" ] || { echo 'Set REALESRGAN_CONVERTER_PYTHON to an external conversion environment.' >&2; exit 1; }
        SOURCE="$RUNTIME/realesrgan-reference"
        OUTPUT="$RUNTIME/realesrgan-optimized"
        REVISION=8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a
        mkdir -p "$SOURCE" "$OUTPUT"
        if [ ! -f "$SOURCE/RealESRGAN_x2plus.pth" ]; then
            curl --fail --location --retry 3 https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth --output "$SOURCE/RealESRGAN_x2plus.pth"
        fi
        if [ ! -f "$SOURCE/rrdbnet_arch.py" ]; then
            curl --fail --location --retry 3 "https://raw.githubusercontent.com/XPixelGroup/BasicSR/$REVISION/basicsr/archs/rrdbnet_arch.py" --output "$SOURCE/rrdbnet_arch.py"
        fi
        if [ ! -f "$SOURCE/LICENSE" ]; then
            curl --fail --location --retry 3 "https://raw.githubusercontent.com/XPixelGroup/BasicSR/$REVISION/LICENSE.txt" --output "$SOURCE/LICENSE"
        fi
        printf '{"sha":"%s"}\n' "$REVISION" > "$SOURCE/basicsr-revision.json"
        cd "$ROOT"
        "$CONVERTER" tools/convert-realesrgan-optimized.py --source "$SOURCE" --output "$OUTPUT"
        xcrun swift tools/compile-coreml-model.swift "$OUTPUT/RealESRGAN.mlpackage" "$OUTPUT/RealESRGAN.mlmodelc"
        xcrun swiftc -O -parse-as-library Sources/Waifu/Uncover/NativeSpatialUpscaler.swift Runtime/Sources/WaifuPGTFormer/ComputeProfiling.swift tools/realesrgan-benchmark.swift -o "$OUTPUT/benchmark"
        "$OUTPUT/benchmark" --model "$OUTPUT/RealESRGAN.mlmodelc" --fixtures "$OUTPUT/fixtures" --units all --repeats 2 --compute-plan --report "$OUTPUT/native-validation.json"
        "$CONVERTER" tools/seal-realesrgan-optimized.py "$OUTPUT" --validation "$OUTPUT/native-validation.json" --units all
        xcrun swift tools/compile-coreml-model.swift "$OUTPUT/RealESRGANValidated.mlpackage" "$OUTPUT/RealESRGANValidated.mlmodelc"
        cp "$SOURCE/LICENSE" "$OUTPUT/BasicSR-LICENSE"
        MODEL="$OUTPUT"
        ;;
    --*) echo 'Usage: setup-native-realesrgan.sh [--convert-official]' >&2; exit 1 ;;
esac

# Compiled Core ML is tied to the local compiler and macOS version, so it is built
# here rather than downloaded. The app does the same at install time.
COMPILED="$RUNTIME/RealESRGAN_x2plus_optimized.mlmodelc"
xcrun swift "$ROOT/tools/compile-coreml-model.swift" "$MODEL/RealESRGANValidated.mlpackage" "$COMPILED"

mkdir -p "$CONFIG_DIR"
printf '%s\n' "$COMPILED" > "$CONFIG_DIR/realesrgan-model.txt"
echo "RealESRGAN configured: $COMPILED"