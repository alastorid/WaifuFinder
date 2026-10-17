#!/bin/bash
# Source this before native SwiftPM builds. The workaround stays clone-local.
NATIVE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NATIVE_MANIFEST="$(dirname "$(xcrun --find swiftc)")/../lib/swift/pm/ManifestAPI"
NATIVE_LOCAL="$NATIVE_ROOT/.native-toolchain/ManifestAPI"
mkdir -p "$NATIVE_LOCAL"
# Preserve timestamps: SwiftPM treats a rewritten ManifestAPI dylib as a
# toolchain change and recompiles otherwise unchanged native workers.
cp -pR "$NATIVE_MANIFEST/." "$NATIVE_LOCAL/"
# Some CLT releases ship an outdated private interface beside the public one.
for NATIVE_ARCH in arm64 x86_64; do
    NATIVE_PUBLIC="$NATIVE_LOCAL/PackageDescription.swiftmodule/$NATIVE_ARCH-apple-macos.swiftinterface"
    NATIVE_PRIVATE="$NATIVE_LOCAL/PackageDescription.swiftmodule/$NATIVE_ARCH-apple-macos.private.swiftinterface"
    if [ -f "$NATIVE_PUBLIC" ]; then cp -p "$NATIVE_PUBLIC" "$NATIVE_PRIVATE"; fi
done
export SWIFT_EXEC="$NATIVE_ROOT/tools/native-toolchain/swiftc"
