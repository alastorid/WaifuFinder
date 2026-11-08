#!/bin/bash
# Noninteractive, optimized reconstruction/fixture benchmark, without app launch.
# Compiles production geometry, forensics, reconstruction and ranking unchanged.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$(mktemp -d "${TMPDIR:-/tmp}/arena-reconstruction.XXXXXX")"
trap 'rm -rf "$BUILD"' EXIT
{
    echo 'import Foundation'
    sed -n '/^struct ArenaSourceRank/,/^actor ArenaRegionalJudge/p' "$ROOT/Sources/Waifu/PhotoArena/ArenaRegionalJudge.swift" | sed '$d'
    sed -n '/^enum ArenaRanking/,$p' "$ROOT/Sources/Waifu/PhotoArena/ArenaRegionalJudge.swift"
} > "$BUILD/Ranking.swift"
{
    echo 'import Foundation'
    sed -n '/^enum UncoverError/,/^\/\/\//p' "$ROOT/Sources/Waifu/Model/UncoverTypes.swift" | sed '$d'
} > "$BUILD/Errors.swift"
cat > "$BUILD/Runner.swift" <<'SWIFT'
import Foundation
@main struct ArenaReconstructionTestRunner {
    static func main() async {
        registerArenaReconstructionTests()
        let results = await Registry.shared.run(only: nil, strict: true)
        for result in results where !result.messages.isEmpty {
            print("\(result.name): \(result.messages.joined(separator: "\n"))")
        }
        if results.contains(where: { if case .failed = $0.outcome { return true }; return false }) { exit(1) }
    }
}
SWIFT
SDK="$(xcrun --sdk macosx --show-sdk-path)"
"$(xcrun --find swiftc)" -O -swift-version 6 -sdk "$SDK" \
    -target "$(uname -m)-apple-macosx15.0" \
    "$ROOT/Sources/Waifu/PhotoArena/PhotoArena.swift" \
    "$ROOT/Sources/Waifu/PhotoArena/ArenaAnalysisTypes.swift" \
    "$ROOT/Sources/Waifu/PhotoArena/ArenaForensics.swift" \
    "$ROOT/Sources/Waifu/PhotoArena/ArenaReconstruction.swift" \
    "$ROOT/Sources/Waifu/Uncover/RealESRGANJobResources.swift" \
    "$ROOT/Tests/TestHarness.swift" "$ROOT/Tests/ArenaReconstructionTests.swift" \
    "$BUILD/Ranking.swift" "$BUILD/Errors.swift" "$BUILD/Runner.swift" \
    -framework Vision -framework CoreImage -framework ImageIO -o "$BUILD/arena-tests"
"$BUILD/arena-tests"
