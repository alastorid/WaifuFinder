#!/bin/bash
# Invoke ONLY through ./run-ui-tests.sh. Opens a separate instance of the built
# app; opens Photos and the smallest accessible nonempty album (at most 50 items),
# enters Image Lineage, inspects accessibility, tests and CANCELS the deletion
# confirmation, returns to ranking, and closes ONLY the test app instance.
# Never confirms a deletion or modifies the Photos library.
set -euo pipefail
LINEAGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
LINEAGE_OUT="$LINEAGE_ROOT/.build/ui-image-lineage/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LINEAGE_OUT"
xcrun swiftc -swift-version 6 -parse-as-library -target "$(uname -m)-apple-macosx15.0" \
    "$LINEAGE_ROOT/tools/ui-image-lineage-helper.swift" -framework AppKit -framework ApplicationServices \
    -o "$LINEAGE_OUT/helper"
"$LINEAGE_OUT/helper" "$LINEAGE_ROOT/dist/WaifuFinder.app" "$LINEAGE_OUT" > "$LINEAGE_OUT/report.txt" 2>&1
cat "$LINEAGE_OUT/report.txt"
echo "Image Lineage UI artifacts: $LINEAGE_OUT"
