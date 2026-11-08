#!/bin/bash
# Reconstruct an actual pre-optimization executable from immutable source.
# Offline source/build preparation only; no inference or performance measurement.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REFERENCE_COMMIT="${PHOTOARENA_REFERENCE_COMMIT:-6a8de440b27413d8dac276d35ac30058541fbb02}"
REFERENCE="$(mktemp -d "${TMPDIR:-/tmp}/photoarena-opt-reference.XXXXXX")"
KEEP=0
trap 'if [ "$KEEP" = 0 ]; then rm -rf "$REFERENCE"; fi' EXIT
mkdir -p "$REFERENCE/runtime" "$REFERENCE/llama"
git -C "$ROOT" show "$REFERENCE_COMMIT:Runtime/PhotoArena/main.mm" > "$REFERENCE/main.mm"
git -C "$ROOT" show "$REFERENCE_COMMIT:Sources/Waifu/PhotoArena/ArenaJudge.swift" > "$REFERENCE/ArenaJudge.swift"
git -C "$ROOT" show "$REFERENCE_COMMIT:Runtime/PhotoArena/CMakeLists.txt" > "$REFERENCE/runtime/CMakeLists.txt"
cp "$REFERENCE/main.mm" "$REFERENCE/runtime/main.mm"
git -C "$ROOT/.native-runtime/photoarena-llama" archive 7071ba378afa96dcd69c41e1ec2b4912364dbee4 | /usr/bin/tar -xf - -C "$REFERENCE/llama"
cmake -S "$REFERENCE/runtime" -B "$REFERENCE/reference-build" -G Ninja \
    -DCMAKE_C_COMPILER="$(xcrun --find clang)" -DCMAKE_CXX_COMPILER="$(xcrun --find clang++)" \
    -DCMAKE_OBJC_COMPILER="$(xcrun --find clang)" -DCMAKE_OBJCXX_COMPILER="$(xcrun --find clang++)" \
    -DCMAKE_OSX_SYSROOT="$(xcrun --sdk macosx --show-sdk-path)" -DCMAKE_BUILD_TYPE=Release \
    -DPHOTOARENA_LLAMA_SOURCE="$REFERENCE/llama"
cmake --build "$REFERENCE/reference-build" --target waifu-photoarena --parallel 4
cp "$REFERENCE/reference-build/waifu-photoarena" "$REFERENCE/original-worker"
python3 - "$ROOT" "$REFERENCE" "$REFERENCE_COMMIT" <<'PY'
import sys,pathlib,hashlib,json
root=pathlib.Path(sys.argv[1]);r=pathlib.Path(sys.argv[2])
value={'directory':str(r),'commit':sys.argv[3],'rebuiltFromImmutableSource':True,
 'workerSHA256':hashlib.sha256((r/'original-worker').read_bytes()).hexdigest(),
 'unrelatedUIHash':hashlib.sha256((root/'Sources/Waifu/UI/FrameGridView.swift').read_bytes()).hexdigest()}
p=root/'.build/reports/photoarena-optimization-reference.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2)+'\n')
PY
# Standalone build artifacts are temporary; only the comparison binary and small
# source snapshots survive until the final session's cleanup.
rm -rf "$REFERENCE/reference-build" "$REFERENCE/llama" "$REFERENCE/runtime"
KEEP=1
echo "Archived an actual reference executable in temporary storage."
