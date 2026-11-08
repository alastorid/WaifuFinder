#!/bin/bash
# Preparation/correctness only. Does not execute model inference or measure speed.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if ! python3 -c 'import json,pathlib; r=json.load(open(".build/reports/photoarena-optimization-reference.json")); assert (pathlib.Path(r["directory"])/"original-worker").is_file()' 2>/dev/null; then
    "$ROOT/tools/archive-photoarena-reference.sh"
fi
REFERENCE="$(python3 -c 'import json;print(json.load(open(".build/reports/photoarena-optimization-reference.json"))["directory"])')"
SDK="$(xcrun --sdk macosx --show-sdk-path)"
xcrun clang++ -O2 -dynamiclib -fobjc-arc -std=c++17 -isysroot "$SDK" "$ROOT/tools/photoarena-memory-observer.mm" -framework Foundation -framework Metal -o "$REFERENCE/memory-observer.dylib"
python3 - "$REFERENCE" <<'PY'
import sys,pathlib
r=pathlib.Path(sys.argv[1])
for name,path in [('original',r/'ArenaJudge.swift'),('optimized',pathlib.Path('Sources/Waifu/PhotoArena/ArenaJudge.swift'))]:
 s=path.read_text(); start=s.index('enum ArenaJudgeConfiguration {');end=s.index('\n}\n',start)+3
 s=s[:start]+'''enum ArenaJudgeConfiguration {
    static let revision = "minicpm-v4.7-f16-16x-llama-7071ba3-v1"
    static var directory: URL? { URL(fileURLWithPath: ProcessInfo.processInfo.environment["ARENA_MODEL"]!) }
    static var executable: URL? { URL(fileURLWithPath: ProcessInfo.processInfo.environment["ARENA_WORKER"]!) }
}
'''+s[end:]
 anchor='        let result = try JSONDecoder().decode(ArenaJudgeResult.self, from: reply)'
 assert anchor in s
 s=s.replace(anchor,'        ArenaBenchmarkTelemetry.save(reply)\n'+anchor)
 (r/(name+'-judge.swift')).write_text(s)
PY
for variant in original optimized; do
    xcrun swiftc -O -swift-version 6 -sdk "$SDK" -target arm64-apple-macosx15.0 \
        "$ROOT/Sources/Waifu/PhotoArena/PhotoArena.swift" "$REFERENCE/$variant-judge.swift" \
        "$ROOT/tools/photoarena-optimization-client.swift" -o "$REFERENCE/$variant-client"
done
cp "$ROOT/.native-runtime/waifu-photoarena" "$REFERENCE/optimized-worker"
xcrun swift -sdk "$SDK" "$ROOT/tools/fixtures-photoarena-roi.swift" "$ROOT/Tests/Fixtures/RealESRGAN/nasa-collins-896x1134.jpg" "$REFERENCE"
xcrun swift -sdk "$SDK" "$ROOT/tools/fixtures-photoarena.swift" "$ROOT/Tests/Fixtures/RealESRGAN/nasa-collins-896x1134.jpg" "$REFERENCE/overview"
echo "Prepared actual reference/optimized clients, unchanged reference worker, observer and native fixtures in temporary storage. No inference executed."
