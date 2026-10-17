#!/usr/bin/env python3
"""Restore published model bundles from the repository's pinned releases.

Uses only Python 3.9+ and curl. Every file is SHA256-verified; model conversion
and the original training checkpoints are unnecessary for development setup.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
BUNDLES = {
    "wd-tagger": "WD-SwinV2-Tagger-v3-Swift-MLX",
    "wd-tagger-coreml": "WD-SwinV2-Tagger-v3-CoreML",
    "seedvr2": "SeedVR2-3B-Swift-MLX-4bit",
    "seedvr2-7b": "SeedVR2-7B-Swift-MLX-4bit",
    "flashvsr": "FlashVSR-v1.1-Tiny-Swift-MLX-BF16",
    "pgtformer": "PGTFormer-Swift-CoreML",
    "realesrgan": "RealESRGAN-x2plus-Optimized-CoreML",
}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def member_path(directory, name, expected):
    if (not name or "\\" in name or PurePosixPath(name).is_absolute()
            or any(part in ("", ".", "..") for part in name.split("/"))
            or not re.fullmatch(r"[0-9a-f]{64}", expected)):
        raise ValueError(f"Invalid bundle filename/checksum: {name}")
    target = directory / name
    if not target.resolve().is_relative_to(directory.resolve()) or target.is_symlink():
        raise ValueError(f"Bundle member escapes destination or is a symlink: {name}")
    return target


def fetch(directory, name, expected, base, verify_only=False):
    target = member_path(directory, name, expected)
    if target.is_file() and digest(target) == expected:
        print(f"Verified {name}", flush=True)
        return target
    if verify_only:
        raise ValueError(f"Missing or incompatible file: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = member_path(directory, name + ".part", expected)
    print(f"Downloading {name}", flush=True)
    if not partial.is_file() or digest(partial) != expected:
        subprocess.run([
            "curl", "--fail", "--location", "--retry", "2", "--connect-timeout", "30",
            "--continue-at", "-", "--output", str(partial), base + "/" + quote(name, safe="/"),
        ], check=True)
    if digest(partial) != expected:
        partial.unlink()
        raise ValueError(f"Checksum mismatch: {name}; rerun to retry the download")
    os.replace(partial, target)
    return target


def restore_bundle(name, destination, verify_only=False, metadata_only=False):
    release = json.loads((ROOT / "tools" / (name + "-release.json")).read_text())
    if (not re.fullmatch(r"[0-9a-f]{40}", release["revision"])
            or not re.fullmatch(r"[\w.-]+/[\w.-]+", release["repository"])):
        raise ValueError(f"Invalid pinned release: {name}")
    prefix = release.get("pathPrefix", "")
    if prefix and not re.fullmatch(r"[\w-]+", prefix):
        raise ValueError("Invalid release path prefix")
    base = f"https://huggingface.co/{release['repository']}/resolve/{release['revision']}"
    if prefix:
        base += "/" + prefix
    directory = destination / BUNDLES[name]
    if directory.is_symlink():
        raise ValueError(f"Use a real bundle directory instead of a machine-specific link: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    print(f"\n{name}: {release['repository']} @ {release['revision']}", flush=True)
    manifest_path = fetch(directory, "manifest.json", release["manifestSHA256"], base, verify_only)
    manifest = json.loads(manifest_path.read_text())
    members = manifest.get("fileSHA256", {})
    if not members:
        raise ValueError(f"Empty bundle manifest: {name}")
    support = release["supportFilesSHA256"]
    if "manifest.json" in members or set(members) & set(support):
        raise ValueError("Manifest members and support files overlap")
    # Validate every name even when only inspecting remote metadata.
    for filename, expected in {**members, **support}.items():
        member_path(directory, filename, expected)
    if not metadata_only:
        for filename, expected in sorted(members.items()):
            fetch(directory, filename, expected, base, verify_only)
    for filename, expected in sorted(support.items()):
        fetch(directory, filename, expected, base, verify_only)
    print(f"{name}: {'metadata' if metadata_only else 'complete bundle'} verified", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", metavar="MODEL", help="Choose: " + ", ".join(BUNDLES))
    parser.add_argument("--all", action="store_true", help="Restore every bundle, including multi-GB video models")
    parser.add_argument("--destination", type=Path, default=ROOT / "models/published")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--verify-only", action="store_true", help="Check existing files without downloading")
    modes.add_argument("--check-remote", action="store_true", help="Fetch and verify release manifests/support files in temporary storage; weights are not fetched")
    args = parser.parse_args()
    names = list(BUNDLES) if args.all else list(dict.fromkeys(args.models))
    if not names or (args.all and args.models) or any(name not in BUNDLES for name in names):
        parser.error("Supply one or more model names, or --all")
    if args.check_remote:
        with tempfile.TemporaryDirectory(prefix="waifu-release-check-") as temporary:
            for name in names:
                restore_bundle(name, Path(temporary), metadata_only=True)
    else:
        for name in names:
            restore_bundle(name, args.destination.resolve(), args.verify_only)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        sys.exit(f"Model restore: {error}")
