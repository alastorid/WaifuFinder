#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODEL="${FLASHVSR_NATIVE_MODEL:-$ROOT/.native-runtime/flashvsr}"
case "${1:-}" in
    ''|--download-converted)
        echo 'Installing converted FlashVSR 1.1 Tiny BF16 weights (about 3.51 GB). Existing verified files are reused.'
        xcrun swift tools/download-flashvsr.swift --release tools/flashvsr-release.json --destination "$MODEL"
        ;;
    --download-official|--convert-official)
        # Advanced reproducibility path only. Normal setup never needs these originals.
        CONVERTER="${FLASHVSR_CONVERTER_PYTHON:-}"
        [ -x "$CONVERTER" ] || { echo 'Set FLASHVSR_CONVERTER_PYTHON to an external conversion environment with torch and safetensors.' >&2; exit 1; }
        echo 'Downloading official FlashVSR 1.1 Tiny checkpoints (6.44 GB), then converting offline.'
        SOURCE="$ROOT/.native-runtime/flashvsr-official"
        mkdir -p "$SOURCE"
        for FILE in diffusion_pytorch_model_streaming_dmd.safetensors LQ_proj_in.ckpt TCDecoder.ckpt; do
            if [ ! -f "$SOURCE/$FILE" ]; then
                curl -L --fail --retry 2 --output "$SOURCE/$FILE.part" \
                  "https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1/resolve/27561b186ded3402d7c975f4fd722e2885b6135f/$FILE"
                mv -f "$SOURCE/$FILE.part" "$SOURCE/$FILE"
            fi
        done
        if [ ! -f "$SOURCE/posi_prompt.pth" ]; then
            curl -L --fail --retry 2 --output "$SOURCE/posi_prompt.pth" \
              https://raw.githubusercontent.com/OpenImagingLab/FlashVSR/cf910c61a60733e610e9c6e8b607f80c3a6c202b/examples/WanVSR/prompt_tensor/posi_prompt.pth
        fi
        "$CONVERTER" tools/convert-flashvsr.py "$SOURCE" "$MODEL" --prompt "$SOURCE/posi_prompt.pth"
        ;;
    --*) echo 'Usage: setup-native-flashvsr.sh [CONVERTED_DIRECTORY | --convert-official]' >&2; exit 1 ;;
    *) MODEL="$1" ;;
esac
[ -f "$MODEL/manifest.json" ] || { echo 'Converted weights are missing. Run setup without arguments or provide a converted model directory.' >&2; exit 1; }
tools/build-native-uncover.sh
.native-runtime/waifu-flashvsr --configure --model "$MODEL"
