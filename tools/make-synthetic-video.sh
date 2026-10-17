#!/bin/bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/waifu-native-fixture.XXXXXX")"
trap 'rm -rf "$TASK_BUILD"' EXIT
xcrun swiftc -O -swift-version 6 -parse-as-library -target "$(uname -m)-apple-macosx15.0" \
  "$TASK_ROOT/Tests/NativeMediaFixture.swift" "$TASK_ROOT/tools/make-synthetic-video.swift" \
  -framework AVFoundation -framework CoreGraphics -framework CoreMedia -framework CoreVideo \
  -o "$TASK_BUILD/generate"
"$TASK_BUILD/generate" "$@"
