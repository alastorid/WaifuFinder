#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODEL="${PGTFORMER_NATIVE_MODEL:-$ROOT/.native-runtime/pgtformer}"
case "${1:-}" in
    '')
        echo 'Installing the hosted PGTFormer Core ML packages (about 381 MB). Existing verified files are reused.'
        xcrun swift tools/download-coreml-release.swift --release tools/pgtformer-release.json \
            --destination "$MODEL" --algorithm pgtformer
        ;;
    --convert-official)
        # Advanced reproducibility path only. Normal setup never needs these originals.
        CONVERTER="${PGTFORMER_CONVERTER_PYTHON:-}"
        [ -x "$CONVERTER" ] || { echo 'Set PGTFORMER_CONVERTER_PYTHON to an external conversion environment.' >&2; exit 1; }
        SOURCE="$ROOT/.build/model-conversion/pgtformer/face-upstream"
        mkdir -p "$SOURCE"
        if [ ! -d "$SOURCE/PGTFormer/.git" ]; then git clone https://github.com/kepengxu/PGTFormer "$SOURCE/PGTFormer"; fi
        git -C "$SOURCE/PGTFormer" checkout e78a26cd463182f4a5aa96b8681110da8615d44d
        for PAIR in 'model.safetensors:pgt-model.safetensors' 'config.json:pgt-config.json'; do
            FILE="${PAIR%%:*}"; LOCAL="${PAIR##*:}"
            if [ ! -f "$SOURCE/$LOCAL" ]; then
                curl -L --fail --retry 2 --output "$SOURCE/$LOCAL.part" "https://huggingface.co/kepeng/pgtformer-base/resolve/bc0b7e3ff10cfa0f2f335f5256919de20fe2b29d/$FILE"
                mv -f "$SOURCE/$LOCAL.part" "$SOURCE/$LOCAL"
            fi
        done
        MODEL="$ROOT/.native-runtime/pgtformer"
        "$CONVERTER" tools/convert-pgtformer.py --upstream "$SOURCE/PGTFormer" --config "$SOURCE/pgt-config.json" --weights "$SOURCE/pgt-model.safetensors" --output "$MODEL" --reference-only
        tools/build-native-uncover.sh
        "$CONVERTER" tools/convert-pgtformer-split.py --upstream "$SOURCE/PGTFormer" --config "$SOURCE/pgt-config.json" --weights "$SOURCE/pgt-model.safetensors" --output "$MODEL"
        cp "$SOURCE/PGTFormer/LICENSE" "$MODEL/PGTFormer-LICENSE"
        ;;
    --*) echo 'Usage: setup-native-pgtformer.sh [CONVERTED_DIRECTORY | --convert-official]' >&2; exit 1 ;;
    *) MODEL="$1" ;;
esac
[ -f "$MODEL/manifest.json" ] || {
    echo 'PGTFormer is missing. Run setup without arguments, or provide a converted model directory.' >&2
    exit 1
}
tools/build-native-uncover.sh
.native-runtime/waifu-pgtformer --configure --model "$MODEL"