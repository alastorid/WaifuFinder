#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$HOME/Library/Application Support/Waifu/Uncover/backend.json"
RELEASE="$ROOT/tools/seedvr2-release.json"
DEFAULT_MODEL="$ROOT/.native-runtime/model-optimized"
RUNTIME="$ROOT/.native-runtime"
if [ "${1:-}" = "--7b" ]; then
    RELEASE="$ROOT/tools/seedvr2-7b-release.json"
    DEFAULT_MODEL="$ROOT/.native-runtime/model-7b-optimized"
    shift
fi
MODEL="${1:-$DEFAULT_MODEL}"
mkdir -p "$MODEL"
MODEL="$(cd "$MODEL" && pwd)"
REUSE="$MODEL"
if [ -f "$CONFIG" ]; then REUSE="$(plutil -extract model raw -o - "$CONFIG")"; fi
xcrun swift "$ROOT/tools/download-seedvr2.swift" --release "$RELEASE" \
    --destination "$MODEL" --reuse-directory "$REUSE"
mkdir -p "$ROOT/.native-runtime"
if [ "$MODEL/pos_emb.safetensors" != "$ROOT/.native-runtime/pos_emb.safetensors" ]; then
    cp "$MODEL/pos_emb.safetensors" "$ROOT/.native-runtime/pos_emb.safetensors"
fi
"$ROOT/tools/build-native-uncover.sh"
"$ROOT/.native-runtime/waifu-seedvr2" --configure --model "$MODEL" \
    --embedding "$MODEL/pos_emb.safetensors"
