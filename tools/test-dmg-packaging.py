#!/usr/bin/env python3
"""Headless DMG regression: no stdin, denied UI tools, remounted layout validation.

Uses a tiny synthetic app bundle; never launches the app or controls the desktop.
Run on macOS with Xcode Command Line Tools: python3 tools/test-dmg-packaging.py
"""
import os
from pathlib import Path
import plistlib
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent


def run(command, **kwargs):
    return subprocess.run(command, check=True, stdin=subprocess.DEVNULL, timeout=180, **kwargs)


def main():
    with tempfile.TemporaryDirectory(prefix='waifu dmg test ') as directory:
        temp = Path(directory)
        app = temp / 'Test App.app'
        (app / 'Contents' / 'MacOS').mkdir(parents=True)
        (app / 'Contents' / 'Info.plist').write_bytes(plistlib.dumps({
            'CFBundleName': 'WaifuFinder',
            'CFBundleIdentifier': 'com.alastorid.waifufinder.packaging-test',
            'CFBundleExecutable': 'WaifuFinder',
            'CFBundlePackageType': 'APPL',
        }))
        executable = app / 'Contents' / 'MacOS' / 'WaifuFinder'
        executable.write_text('#!/bin/sh\nexit 0\n')
        executable.chmod(0o755)
        denied = temp / 'denied-ui-tools'
        denied.mkdir()
        for name in ('osascript', 'open'):
            script = denied / name
            script.write_text('#!/bin/sh\necho "Unexpected UI command" >&2\nexit 99\n')
            script.chmod(0o755)
        # Guard against absolute UI calls and reintroducing the approval launcher.
        for name in ('package-dmg.sh', 'configure-dmg-layout.py'):
            source = (ROOT / 'tools' / name).read_text()
            for forbidden in ('osascript', 'run-ui-tests.sh', 'configure-dmg-window'):
                assert forbidden not in source, f'Interactive dependency in {name}: {forbidden}'
        output = temp / 'output images'
        env = dict(os.environ, WAIFU_APP_PATH=str(app), WAIFU_VERSION='0.0.42',
                   WAIFU_OUTPUT_DIR=str(output), PATH=str(denied) + os.pathsep + os.environ['PATH'])
        run([str(ROOT / 'tools' / 'package-dmg.sh')], env=env)
        image = output / 'WaifuFinder-0.0.42-macos-arm64.dmg'
        mount = temp / 'remounted elsewhere'
        mount.mkdir()
        run(['hdiutil', 'attach', str(image), '-mountpoint', str(mount),
             '-nobrowse', '-noautoopen', '-readonly', '-quiet'])
        try:
            run([sys.executable, str(ROOT / 'tools' / 'configure-dmg-layout.py'), '--verify', str(mount)])
            assert (mount / 'Start Here.html').read_bytes() == (ROOT / 'docs' / 'DMG-Read-Me.html').read_bytes()
            assert (mount / 'WaifuFinder.app' / 'Contents' / 'MacOS' / 'WaifuFinder').read_bytes() == executable.read_bytes()
            assert not (mount / 'vendor').exists(), 'Build dependencies must not ship in the DMG'
            sys.path.insert(0, str(ROOT / 'tools' / 'vendor' / 'dmg_layout'))
            from ds_store import DSStore
            from ds_store.store import codecs
            codecs.pop(b'pBBk', None)
            alias_file = temp / 'background.alias'
            with DSStore.open(str(mount / '.DS_Store'), 'r') as store:
                alias_file.write_bytes(store['.']['icvp']['backgroundImageAlias'])
                # Check the label footprint against a conservative content viewport,
                # independently of the production layout constants. Reserve 80 px
                # for window chrome and 84 px below each icon center for its label.
                import re
                bounds = [int(value) for value in re.findall(r'\d+', store['.']['bwsp']['WindowBounds'])]
                width, height = bounds[2:]
                for name in ('WaifuFinder.app', 'Applications', 'Start Here.html'):
                    x, y = store[name]['Iloc']
                    assert 80 <= x <= width - 80, f'{name} label exceeds window width'
                    assert 44 <= y and y + 84 <= height - 80, f'{name} label requires scrolling'
                view = store['.']['icvp']
                assert view['scrollPositionX'] == 0 and view['scrollPositionY'] == 0
            run(['xcrun', 'swift', str(ROOT / 'tools' / 'test-dmg-background-alias.swift'),
                 str(alias_file), str(mount / '.background' / 'install.tiff')])
            bookmark_file = temp / 'background.bookmark'
            with DSStore.open(str(mount / '.DS_Store'), 'r') as store:
                bookmark_file.write_bytes(store['.']['pBBk'][1])
            run(['xcrun', 'swift', str(ROOT / 'tools' / 'test-dmg-background-alias.swift'),
                 str(bookmark_file), str(mount / '.background' / 'install.tiff')])
        finally:
            run(['hdiutil', 'detach', str(mount), '-quiet'])
        print('PASS: unattended packaging, paths with spaces, new version, and layout after remount.')


if __name__ == '__main__':
    main()
