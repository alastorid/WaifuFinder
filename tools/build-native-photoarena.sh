#!/bin/bash
set -euo pipefail
ARENA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARENA_REV=7071ba378afa96dcd69c41e1ec2b4912364dbee4
ARENA_SOURCE="$ARENA_ROOT/.native-runtime/photoarena-llama"
if [ ! -d "$ARENA_SOURCE/.git" ]; then
    git clone https://github.com/ggml-org/llama.cpp.git "$ARENA_SOURCE"
fi
[ "$(git -C "$ARENA_SOURCE" rev-parse HEAD)" = "$ARENA_REV" ] || git -C "$ARENA_SOURCE" checkout "$ARENA_REV"
# Apply our small, reviewed patch only to the pinned private runtime checkout.
for patch in "$ARENA_ROOT"/Runtime/PhotoArena/patches/*.patch; do
    if git -C "$ARENA_SOURCE" apply --reverse --check "$patch" 2>/dev/null; then
        continue
    fi
    git -C "$ARENA_SOURCE" apply --check "$patch"
    git -C "$ARENA_SOURCE" apply "$patch"
done
cmake -S "$ARENA_ROOT/Runtime/PhotoArena" -B "$ARENA_ROOT/.native-runtime/photoarena-ninja" -G Ninja -DCMAKE_C_COMPILER="$(xcrun --find clang)" -DCMAKE_CXX_COMPILER="$(xcrun --find clang++)" -DCMAKE_OBJC_COMPILER="$(xcrun --find clang)" -DCMAKE_OBJCXX_COMPILER="$(xcrun --find clang++)" -DCMAKE_OSX_SYSROOT="$(xcrun --sdk macosx --show-sdk-path)" -DCMAKE_BUILD_TYPE=Release -DPHOTOARENA_LLAMA_SOURCE="$ARENA_SOURCE" -DPHOTOARENA_TEST_OUTPUT=
cmake --build "$ARENA_ROOT/.native-runtime/photoarena-ninja" --target waifu-photoarena --parallel 4
cp "$ARENA_ROOT/.native-runtime/photoarena-ninja/waifu-photoarena" "$ARENA_ROOT/.native-runtime/waifu-photoarena"
echo "Native PhotoArena worker ready"
