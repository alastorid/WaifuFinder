#!/usr/bin/env python3
"""Headless build-cache regression checks; no app launch or real compiler needed."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


REPO = Path(__file__).resolve().parent.parent


with tempfile.TemporaryDirectory(prefix="waifu build cache ") as temporary:
    root = Path(temporary)
    for directory in ("Sources", "tools", "FirefoxExtension", "bin", "sdk"):
        (root / directory).mkdir()
    shutil.copy2(REPO / "build.sh", root / "build.sh")
    (root / "Sources/main.swift").write_text("// first source\n")
    (root / "tools/make-icon.swift").write_text("// first icon\n")
    (root / "sdk/SDKSettings.json").write_text("{}")
    for release in ("seedvr2", "seedvr2-7b", "flashvsr", "realesrgan", "wd-tagger", "pgtformer"):
        (root / f"tools/{release}-release.json").write_text('{"revision":"published"}')
    for name in ("manifest.json", "background.js", "popup.html", "popup.js"):
        (root / "FirefoxExtension" / name).write_text("fixture")

    def executable(name, source):
        path = root / "bin" / name
        path.write_text("#!/bin/bash\nset -eu\n" + source)
        path.chmod(0o755)

    executable("xcrun", '''
if [ "$1" = --find ]; then printf '%s/bin/%s\n' "$FIXTURE" "$2"
elif [ "$3" = --show-sdk-path ]; then printf '%s/sdk\n' "$FIXTURE"
else echo 15.0; fi
''')
    executable("swiftc", '''
if [ "$1" = --version ]; then echo 'fixture compiler 1'; exit; fi
echo compile >> "$FIXTURE/events"
if [ "${FAIL_COMPILE:-0}" = 1 ]; then exit 1; fi
while [ "$#" -gt 0 ]; do
    if [ "$1" = -o ]; then shift; printf binary > "$1"; fi
    shift
done
''')
    executable("swift", '''
echo icon >> "$FIXTURE/events"
printf icon > "${@: -1}"
''')
    executable("codesign", "exit 0\n")
    env = dict(os.environ, PATH=f"{root}/bin:{os.environ['PATH']}", FIXTURE=str(root),
               WAIFU_ARCH="x86_64", WAIFU_BUILD_CONFIGURATION="release",
               WAIFU_COMMERCIAL_DISTRIBUTION="0", WAIFU_VERSION="0.1.0")

    def build(success=True, **overrides):
        result = subprocess.run(["/bin/bash", str(root / "build.sh")], env=env | overrides,
                                text=True, capture_output=True)
        assert (result.returncode == 0) == success, result.stdout + result.stderr
        return result.stdout

    def counts():
        events = (root / "events").read_text().splitlines()
        return events.count("compile"), events.count("icon")

    build()
    assert counts() == (1, 1)
    assert "using cached release" in build()
    assert counts() == (1, 1), "unchanged inputs must reuse executable and icon"
    shutil.rmtree(root / "dist")
    build(WAIFU_VERSION="0.2.0")
    assert counts() == (1, 1), "recreating the bundle/version must reuse compilation"
    assert "0.2.0" in (root / "dist/WaifuFinder.app/Contents/Info.plist").read_text()

    source = root / "Sources/main.swift"
    before = source.stat()
    source.write_text("// edited source\n")
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    build()
    assert counts() == (2, 1), "source contents must invalidate even with preserved mtime"
    (root / "Sources/extra.swift").write_text("// extra\n")
    build()
    (root / "Sources/extra.swift").unlink()
    build()
    assert counts() == (4, 1), "source additions and deletions must invalidate"

    source.write_text("// compiler error\n")
    build(success=False, FAIL_COMPILE="1")
    build()
    assert counts() == (6, 1), "failed compilation must not be cached"
    build(WAIFU_COMMERCIAL_DISTRIBUTION="1")
    assert counts() == (7, 2), "distribution modes need independent compiler caches"
    assert not (root / "dist/WaifuFinder.app/Contents/Resources/ModelReleases/pgtformer-release.json").exists()

    (root / "tools/make-icon.swift").write_text("// edited icon\n")
    build()
    assert counts() == (7, 3), "icon-only edits must not recompile the app"
    (root / "sdk/SDKSettings.json").write_text('{"version":2}')
    build()
    assert counts() == (8, 4), "SDK changes must invalidate the cache"
    build(WAIFU_BUILD_CONFIGURATION="debug")
    assert counts() == (9, 5)
    mapping = json.loads(next((root / ".build/app").glob("*/debug-0/output-map.json")).read_text())
    assert str(source) in mapping, "debug output map must support paths containing spaces"
    assert "using cached debug" in build(WAIFU_BUILD_CONFIGURATION="debug")
    assert counts() == (9, 5)
    build(success=False, WAIFU_BUILD_CONFIGURATION="invalid")
    print("Build-cache checks passed: reuse, input changes, failure recovery, modes, SDK, paths with spaces.")

    if "--real-compiler" in sys.argv:
        # Compile a tiny isolated executable, without launching it or touching
        # the user's app sources. Leave the codesign stub in this fixture.
        (root / "bin/xcrun").unlink()
        shutil.copy2(REPO / "tools/make-icon.swift", root / "tools/make-icon.swift")
        source.write_text("@main struct App { static func main() { print(value()) } }\n")
        helper = root / "Sources/Helper.swift"
        helper.write_text("func value() -> Int { 1 }\n")
        (root / "Sources/Unrelated.swift").write_text("func unrelated() -> Int { 42 }\n")
        build(WAIFU_BUILD_CONFIGURATION="debug")
        object_root = next((root / ".build/app").glob("*/debug-0/objects-*/Helper.swift.o")).parent
        # There is only one real compiler object directory: the stub did not
        # generate objects. Dependency tracking should leave unrelated code alone.
        helper_time = (object_root / "Helper.swift.o").stat().st_mtime_ns
        unrelated_time = (object_root / "Unrelated.swift.o").stat().st_mtime_ns
        helper.write_text("func value() -> Int { 2 }\n")
        build(WAIFU_BUILD_CONFIGURATION="debug")
        assert (object_root / "Helper.swift.o").stat().st_mtime_ns != helper_time
        assert (object_root / "Unrelated.swift.o").stat().st_mtime_ns == unrelated_time
        assert "using cached debug" in build(WAIFU_BUILD_CONFIGURATION="debug")
        print("Real Swift compiler checks passed: changed code rebuilt; unrelated object reused.")
