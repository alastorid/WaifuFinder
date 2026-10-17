#!/usr/bin/env python3
"""Write/verify Finder metadata directly, without launching or controlling Finder.

The image must already be mounted: aliases must refer to its filesystem, rather
than the staging folder. Dependencies are vendored so release packaging is offline.
"""
import argparse
import os
from pathlib import Path
import sys
import subprocess
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent / 'vendor' / 'dmg_layout'))
from ds_store import DSStore
from mac_alias import Alias
from ds_store.store import codecs

# Keep native bookmark bytes intact; the legacy Python bookmark codec loses
# information needed for modern macOS resolution.
codecs.pop(b'pBBk', None)

# Match the three card centers in make-dmg-background.swift. Keep all icons on
# one row, with Start Here directly beneath its heading, inside the 640 × 420 default window; the 800 × 600 canvas adds color only.
POSITIONS = {
    'WaifuFinder.app': (120, 205),
    'Applications': (320, 205),
    'Start Here.html': (520, 205),
}
BOUNDS = '{{140, 100}, {640, 420}}'


def verify_layout(root):
    with DSStore.open(str(root / '.DS_Store'), 'r') as store:
        if store['.']['icvl'] != (b'type', b'icnv'):
            raise ValueError('Disk image must open in icon view')
        window = store['.']['bwsp']
        if window['WindowBounds'] != BOUNDS or window['ShowToolbar'] or window['ShowSidebar']:
            raise ValueError('Unexpected disk image window settings')
        view = store['.']['icvp']
        if view['backgroundType'] != 2 or view['iconSize'] != 88:
            raise ValueError('Missing background or incorrect icon size')
        bookmark = store['.']['pBBk']
        if bookmark[0] != b'blob' or not bookmark[1].startswith(b'book'):
            raise ValueError('Missing native background bookmark')
        alias = Alias.from_bytes(view['backgroundImageAlias'])
        background = root / '.background' / 'install.tiff'
        if alias.target.posix_path != '/.background/install.tiff':
            raise ValueError('Background alias must point inside the image')
        if alias.target.cnid != background.stat().st_ino:
            raise ValueError('Background alias refers to a different file')
        if alias.target.folder_cnid != background.parent.stat().st_ino:
            raise ValueError('Background alias refers to a different directory')
        for name, position in POSITIONS.items():
            if store[name]['Iloc'] != position:
                raise ValueError(f'Incorrect icon position: {name}')
    if not (root / 'Applications').is_symlink() or os.readlink(root / 'Applications') != '/Applications':
        raise ValueError('Missing Applications shortcut')
    print('Verified DMG background, icon positions, window settings, and Applications shortcut.')


def write_layout(root):
    background = root / '.background' / 'install.tiff'
    alias = Alias.for_file(str(background))
    with tempfile.TemporaryDirectory(prefix='waifu-bookmark-') as directory:
        output = Path(directory) / 'background.bookmark'
        subprocess.run(['xcrun', 'swift', str(Path(__file__).with_name('make-dmg-bookmark.swift')),
                        str(background), str(output)], check=True, stdin=subprocess.DEVNULL)
        bookmark = output.read_bytes()
    with DSStore.open(str(root / '.DS_Store'), 'w+') as store:
        store['.']['vSrn'] = ('long', 1)
        store['.']['icvl'] = ('type', b'icnv')
        store['.']['bwsp'] = {
            'WindowBounds': BOUNDS,
            'ShowStatusBar': False,
            'ShowTabView': False,
            'ShowToolbar': False,
            'ShowPathbar': False,
            'ShowSidebar': False,
            'ContainerShowSidebar': False,
            'PreviewPaneVisibility': False,
            'SidebarWidth': 0,
        }
        # Modern Finder uses a bookmark; keep the alias for older macOS versions.
        store['.']['pBBk'] = ('blob', bookmark)
        store['.']['icvp'] = {
            'viewOptionsVersion': 1,
            'backgroundType': 2,
            'backgroundColorRed': 1.0,
            'backgroundColorGreen': 1.0,
            'backgroundColorBlue': 1.0,
            'backgroundImageAlias': alias.to_bytes(),
            'gridOffsetX': 0.0,
            'gridOffsetY': 0.0,
            'gridSpacing': 100.0,
            'arrangeBy': 'none',
            'showIconPreview': False,
            'showItemInfo': False,
            'labelOnBottom': True,
            'textSize': 13.0,
            'iconSize': 88.0,
            'scrollPositionX': 0.0,
            'scrollPositionY': 0.0,
        }
        for name, position in POSITIONS.items():
            store[name]['Iloc'] = position
    verify_layout(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true', help='Verify an existing layout without changing it')
    parser.add_argument('mount', type=Path)
    args = parser.parse_args()
    root = args.mount.resolve(strict=True)
    for name in (*POSITIONS, '.background/install.tiff'):
        if not (root / name).exists():
            parser.error(f'Missing image content: {name}')
    if args.verify:
        verify_layout(root)
    else:
        write_layout(root)
