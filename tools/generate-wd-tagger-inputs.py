#!/usr/bin/env python3
"""Generate synthetic WD Tagger inputs using only Python's standard library.

All pixels come from arithmetic patterns or a fixed integer noise function.
No images, videos, archives, model files or local media are read.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile

ROOT = Path(__file__).resolve().parent.parent
SIDE = 448
FRAMES = 64
VERSION = "synthetic-patterns-v1"
FLOATS = tuple(struct.pack("<f", value) for value in range(256))


def frame(index):
    red, green, blue = bytearray(), bytearray(), bytearray()
    kind = index % 8
    for y in range(SIDE):
        for x in range(SIDE):
            if kind == 0:
                r, g, b = index * 17, index * 37, index * 71
            elif kind == 1:
                r, g, b = x * 255 // (SIDE - 1), y * 255 // (SIDE - 1), index * 13
            elif kind == 2:
                r = 255 * ((x // 16 + y // 16) % 2)
                g, b = 255 - r, index * 23
            elif kind == 3:
                r, g, b = (x // 7) * 31, (y // 11) * 47, (x + y + index) * 3
            elif kind == 4:
                r, g, b = 255 * (x >= SIDE // 2), 255 * (y >= SIDE // 2), 127
            elif kind == 5:
                radius = (x - SIDE // 2) ** 2 + (y - SIDE // 2) ** 2
                r, g, b = radius // 32, radius // 71, radius // 113
            elif kind == 6:
                value = ((x + SIDE * y + index * SIDE * SIDE) * 1664525 + 1013904223) & 0xFFFFFFFF
                value ^= value >> 13
                r, g, b = value, value >> 8, value >> 16
            else:
                r = 255 * ((x - index * 3) % 83 == 0)
                g = 255 * ((y + index * 5) % 61 == 0)
                b = 255 * ((x + y) % 47 == 0)
            red.append(r & 255)
            green.append(g & 255)
            blue.append(b & 255)
    rgba = bytearray(b"\xff") * (SIDE * SIDE * 4)
    for channel, plane in enumerate((red, green, blue)):
        rgba[channel::4] = plane
    return b"".join(FLOATS[value] for plane in (red, green, blue) for value in plane), rgba


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def generate(output_root, force=False, verify_only=False):
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    names = ("wd-optimization/inputs.bin", "wd-native/inputs-rgba.bin")
    with tempfile.TemporaryDirectory(prefix="synthetic-inputs-", dir=output_root) as temporary:
        staged = [Path(temporary) / "rgb.bin", Path(temporary) / "rgba.bin"]
        with staged[0].open("wb") as rgb_file, staged[1].open("wb") as rgba_file:
            for index in range(FRAMES):
                rgb, rgba = frame(index)
                rgb_file.write(rgb)
                rgba_file.write(rgba)
        metadata = {
            "generator": VERSION, "naturalFrames": 0, "syntheticFrames": FRAMES,
            "rgbShape": [FRAMES, 3, SIDE, SIDE], "rgbType": "little-endian Float32, 0...255",
            "rgbaShape": [FRAMES, SIDE, SIDE, 4], "rgbaType": "UInt8",
            "files": {name: {"sha256": digest(path), "bytes": path.stat().st_size}
                      for name, path in zip(names, staged)},
        }
        for name, path in zip(names, staged):
            target = output_root / name
            if not target.resolve().is_relative_to(output_root) or target.is_symlink():
                raise ValueError(f"Output path escapes build directory or is a symlink: {target}")
            matches = target.is_file() and digest(target) == metadata["files"][name]["sha256"]
            if verify_only and not matches:
                raise ValueError(f"Missing or different synthetic input: {target}")
            if target.exists() and not matches and not force:
                raise ValueError(f"Different input exists: {target}; use --force to replace generated inputs")
        if not verify_only:
            for name, path in zip(names, staged):
                target = output_root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(path, target)
            (output_root / "wd-optimization/synthetic-inputs.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"{'Verified' if verify_only else 'Generated'} {FRAMES} synthetic inputs; no local media used")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / ".build")
    parser.add_argument("--force", action="store_true", help="Replace different generated inputs")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    generate(args.output_root, args.force, args.verify_only)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(f"Synthetic input generation: {error}")
