#!/bin/bash
# Produce local comparison outputs from synthetic inputs and the pinned models.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python3 tools/generate-wd-tagger-inputs.py --verify-only
NATIVE="$ROOT/.build/wd-native"
COREML="$ROOT/.build/wd-optimization/coreml"
VALIDATOR="$ROOT/Runtime/.build/release/waifu-wdtagger-validation"
mkdir -p "$NATIVE/tuning/fused-memory" "$NATIVE/tuning/coreml"
"$VALIDATOR" --weights "$NATIVE/model.safetensors" \
    --inputs "$NATIVE/inputs-rgba.bin" --rgba-input --stream-input \
    --frames 64 --batch 8 --repeats 1 --memory-mb 768 --cache-mb 128 \
    --report "$NATIVE/synthetic-fused.json" \
    --probabilities "$NATIVE/tuning/fused-memory/fused8-768.bin"
"$VALIDATOR" --weights "$NATIVE/model.safetensors" \
    --inputs .build/wd-optimization/inputs.bin --stream-input --unfused-qkv \
    --frames 64 --batch 8 --repeats 1 --memory-mb 1536 --cache-mb 128 \
    --report "$NATIVE/synthetic-unfused.json" --probabilities "$NATIVE/native8-probabilities.bin"
.build/wd-optimization/shape-benchmark-v4 --model "$COREML/model.mlmodelc" \
    --labels "$COREML/wd-tagger-labels.csv" --tensor-input .build/wd-optimization/inputs.bin \
    --stream-input --repeats 1 --report .build/wd-optimization/synthetic-coreml.json \
    --probability-output .build/wd-optimization/batch8-probabilities.bin
cp .build/wd-optimization/batch8-probabilities.bin "$NATIVE/tuning/coreml/coreml8.bin"
echo 'Synthetic comparison outputs generated locally. These warmup runs are not controlled performance measurements.'
