#!/usr/bin/env python3
"""Publish the WaifuFinder WDTagger, PGTFormer and RealESRGAN bundles to Hugging Face.

Publishing is a maintainer task: it needs a Hugging Face token with write
access to the alastorid organisation, from ``HF_TOKEN`` or a previous
``huggingface-cli login``. Users of the app never run this — they download the
resulting pinned releases from Settings → Models.

Stage the native WDTagger with tools/stage-native-wdtagger-release.mjs, the Core ML
WDTagger with tools/stage-wd-tagger.py and the restoration bundles with
tools/stage-hf-bundles.sh. This uploads them, writes the model cards, reads back
the revision the Hub created, and updates their release descriptors so
./build.sh pins exactly what was published.

PGTFormer is redistributed under a NON-COMMERCIAL licence. Rather than silently
republishing it under the organisation, the repo is created private and the run
aborts unless WAIFU_HF_ALLOW_NONCOMMERCIAL=1 acknowledges the restriction; make
it public afterwards only if your distribution is non-commercial. RealESRGAN is
BSD 3-Clause and is published publicly.
"""
from __future__ import annotations

import argparse, hashlib, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BUNDLES = [
    {
        "directory": "SAFIRE-Swift-MLX", "release": "safire-release.json",
        "support": ["LICENSE", "APACHE-LICENSE", "KMEANS-LICENSE", "OPENCV-LICENSE", "NOTICE", "README.md", "validation.json", "profiles.json"],
        "tags": ["safetensors", "mlx", "metal", "apple-silicon", "waifufinder", "image-forensics", "image-segmentation"],
        "library_name": "mlx", "license": "cc-by-4.0", "license_name": None, "license_note": None,
        "private": False, "acknowledgement": None,
    },
    {
        "directory": "TruFor-Swift-MLX", "release": "trufor-release.json",
        "support": ["LICENSE.txt", "LICENSE_CMX.txt", "NOTICE", "README.md", "validation.json"],
        "tags": ["safetensors", "mlx", "metal", "apple-silicon", "waifufinder", "image-forensics", "image-forgery-detection"],
        # GRIP-UNINA's TruFor licence permits informational and nonprofit use only,
        # and requires crediting the original authors whenever results are made
        # public. SPDX has no identifier for it, so the card declares "other" plus a
        # slug; the full terms travel in LICENSE.txt, LICENSE_CMX.txt and NOTICE,
        # not in card metadata.
        "library_name": "mlx", "license": "other",
        "license_name": "grip-unina-trufor-non-commercial",
        "license_note": "Informational and nonprofit use only; retain all notices and credit GRIP-UNINA whenever results are made public.",
        "private": False, "acknowledgement": None,
    },
    {
        "directory": "WD-SwinV2-Tagger-v3-CoreML",
        "release": "wd-tagger-coreml-release.json",
        "support": ["LICENSE", "NOTICE", "README.md", "validation.json"],
        "tags": ["coreml", "apple-silicon", "waifufinder", "image-tagging"],
        "license": "apache-2.0", "license_name": None, "license_note": None,
        "private": False, "acknowledgement": None,
    },
    {
        "directory": "WD-SwinV2-Tagger-v3-Swift-MLX",
        "release": "wd-tagger-release.json",
        "support": ["LICENSE", "NOTICE", "README.md", "validation.json"],
        "tags": ["safetensors", "mlx", "metal", "apple-silicon", "waifufinder", "image-tagging"],
        "library_name": "mlx",
        "license": "apache-2.0", "license_name": None, "license_note": None,
        "private": False, "acknowledgement": None,
    },
    {
        "directory": "PGTFormer-Swift-CoreML",
        "release": "pgtformer-release.json",
        "support": ["LICENSE", "NOTICE", "README.md", "validation.json"],
        "tags": ["coreml", "coremltools", "apple-silicon", "waifufinder", "pgtformer", "face-restoration"],
        # SPDX has no identifier for a BSD-style licence carrying a non-commercial
        # restriction, so the card declares "other" plus a slug. Hugging Face
        # validates license_name against /^[a-z0-9-.]+$/, so it must be lowercase.
        # The full terms travel in LICENSE and NOTICE, not in card metadata.
        "license": "other",
        "license_name": "pgtformer-bsd-style-non-commercial",
        "license_note": None,
        "private": True,
        "acknowledgement": "WAIFU_HF_ALLOW_NONCOMMERCIAL",
    },
    {
        "directory": "RealESRGAN-x2plus-Optimized-CoreML",
        "release": "realesrgan-release.json",
        "support": ["LICENSE", "NOTICE", "README.md", "validation.json", "BasicSR-LICENSE"],
        "tags": ["coreml", "coremltools", "apple-silicon", "waifufinder", "real-esrgan", "image-super-resolution"],
        "license": "bsd-3-clause",
        "license_name": None,
        "license_note": None,
        "private": False,
        "acknowledgement": None,
    },
]


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        value = hashlib.sha256()
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
        return value.hexdigest()


def card(bundle, directory: Path, repository: str) -> str:
    manifest = json.loads((directory / "manifest.json").read_text())
    front = {
        "license": bundle["license"],
        "library_name": bundle.get("library_name", "coreml"),
        "tags": bundle["tags"],
    }
    if bundle["license_name"]:
        front["license_name"] = bundle["license_name"]
    if bundle["license_note"]:
        front["license_note"] = bundle["license_note"]
    lines = ["---"]
    for key, value in front.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            lines.extend(f"  - {item}" for item in value)
        else:
            lines.append(f"{key}: {value}")
    lines.extend([
        "---",
        "",
        f"Published by WaifuFinder. `manifest.json` pins {len(manifest['fileSHA256'])} model files by SHA-256;",
        "the app verifies every one of them, plus the licence, attribution and validation files,",
        "before installation.",
        "",
    ])
    return "\n".join(lines) + (directory / "README.md").read_text()


def acknowledge(bundle, repository: str) -> None:
    """Refuse to publish a non-commercial bundle unless a person says so."""
    if not bundle["acknowledgement"]:
        return
    if os.environ.get(bundle["acknowledgement"]) == "1":
        return
    raise SystemExit(
        f"refusing to publish {repository}.\n\n"
        "PGTFormer is redistributed under a NON-COMMERCIAL licence: redistribution is\n"
        "permitted for non-commercial use, and commercial use requires permission from\n"
        "Xidian University or the contributors. It is created private, but making it\n"
        "public is your call, so the acknowledgement must be explicit:\n\n"
        f"    {bundle['acknowledgement']}=1 {sys.argv[0]}\n"
    )


def remote_sha(repository: str) -> str | None:
    """The revision currently on the Hub, or None if the repo is missing."""
    result = subprocess.run(
        ["git", "ls-remote", f"git@hf.co:{repository}", "refs/heads/main"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.split()[0] if result.stdout.strip() else None


def create_repo(api, bundle, repository: str) -> None:
    """Create the repo over the API. Needs a token; SSH cannot create repos."""
    if api is not None:
        api.create_repo(repository, repo_type="model", private=bundle["private"], exist_ok=True)
        print(f"created {repository}{' (private)' if bundle['private'] else ''}")
        return
    visibility = "private" if bundle["private"] else "public"
    raise SystemExit(
        f"{repository} does not exist yet, and creating a repository needs a Hugging Face\n"
        "write token — SSH can push but not create. Either:\n\n"
        f"  1. create it once at https://huggingface.co/new-model as '{visibility}', then\n"
        "     rerun this script; or\n"
        "  2. set HF_TOKEN to a write token for the organisation and rerun.\n"
    )


def run_git(checkout: Path, *arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(checkout), *arguments], capture_output=True, text=True)


def push_over_ssh(staging: Path, repository: str, readme: str) -> str:
    """Publish by pushing over SSH, so uploading needs no token."""
    with tempfile.TemporaryDirectory() as work:
        checkout = Path(work) / "bundle"
        shutil.copytree(staging, checkout)
        (checkout / "README.md").write_text(readme)
        run_git(checkout, "init", "-q", "-b", "main", ".")
        # The encoder weights are ~250 MB. Hugging Face rejects a plain git blob over
        # 100 MB, so the LFS filters must be active or the push fails outright.
        # --local keeps this out of the developer's global git configuration.
        filters = run_git(checkout, "lfs", "install", "--local")
        if filters.returncode != 0:
            raise SystemExit(f"git-lfs is required to publish: {filters.stderr.strip()}")
        run_git(checkout, "remote", "add", "hf", f"git@hf.co:{repository}")

        # A new Hugging Face repository already has a commit holding its LFS
        # .gitattributes. Build on top of it rather than orphaning that history,
        # so the push is a fast-forward and nothing already published is rewritten.
        fetched = run_git(checkout, "fetch", "-q", "hf", "main")
        if fetched.returncode == 0:
            run_git(checkout, "reset", "-q", "FETCH_HEAD")
            # FETCH_HEAD tracks a .gitattributes the staged bundle does not carry.
            # Restore it so the LFS routing is not recorded as a deletion.
            run_git(checkout, "checkout", "HEAD", "--", ".gitattributes")

        # The Hub's default patterns cover weights (.safetensors, .ckpt, .onnx ...)
        # but not a compiled Metal kernel library. Without an explicit pattern a
        # .metallib is committed as a plain blob and the push is rejected outright
        # for exceeding 10 MiB. Only touch .gitattributes when the bundle actually
        # carries one: rewriting it otherwise would record a spurious change and
        # move the repository revision the descriptor pins.
        if any(path.suffix == ".metallib" for path in staging.iterdir()):
            attributes = checkout / ".gitattributes"
            existing = attributes.read_text() if attributes.exists() else ""
            if "*.metallib" not in existing:
                with attributes.open("a") as handle:
                    handle.write("*.metallib filter=lfs diff=lfs merge=lfs -text\n")

        run_git(checkout, "add", "-A")
        commit = run_git(checkout, "commit", "-q", "-m", f"Publish WaifuFinder {staging.name} bundle")
        # Re-publishing identical content is a no-op, not a failure.
        if commit.returncode != 0 and "nothing to commit" not in commit.stdout + commit.stderr:
            raise SystemExit(f"git commit failed: {commit.stderr.strip() or commit.stdout.strip()}")

        # A package left as a plain git blob would be a corrupt release: the bytes at
        # download time would not match the hashes manifest.json pins.
        pointers = run_git(checkout, "lfs", "ls-files", "-n").stdout.split()
        if not pointers:
            raise SystemExit("no files were converted to LFS; refusing to publish weights as plain git blobs")
        push = run_git(checkout, "push", "--progress", "hf", "main:main")
        if push.returncode != 0:
            raise SystemExit(f"git push failed: {push.stderr.strip() or push.stdout.strip()}")
    revision = remote_sha(repository)
    if not revision:
        raise SystemExit(f"pushed {repository} but could not read back its revision")
    return revision


def publish(api, bundle, directory: Path, organization: str, stage: Path) -> tuple[str, str]:
    repository = f"{organization}/{bundle['directory']}"
    acknowledge(bundle, repository)
    staging = stage / bundle["directory"]
    # Compose the card here but let push_over_ssh write it into the throwaway
    # checkout. Writing it over the staged README would nest a card inside the card
    # on every re-publish, so the bytes on the Hub — and the revision the descriptor
    # pins — would change each time for no reason.
    readme = card(bundle, staging, repository)
    if remote_sha(repository) is None:
        create_repo(api, bundle, repository)
    print(f"publishing {repository}")
    revision = push_over_ssh(staging, repository, readme)
    return revision, readme


def write_release(bundle, directory: Path, repository: str, revision: str, readme: str) -> None:
    manifest = directory / "manifest.json"
    # The manifest pins model files and the descriptor pins support files. If they
    # overlapped, every support file would be hashed and verified twice.
    pinned = set(json.loads(manifest.read_text())["fileSHA256"])
    overlap = pinned & set(bundle["support"])
    if overlap:
        raise SystemExit(f"{repository}: manifest and support files overlap: {sorted(overlap)}")
    document = {
        "repository": repository,
        "revision": revision,
        "manifestSHA256": digest(manifest),
        "supportFilesSHA256": {
            # The published README is the composed card, not the staged body.
            name: hashlib.sha256(readme.encode()).hexdigest() if name == "README.md"
            else digest(directory / name)
            for name in bundle["support"]
        },
    }
    output = ROOT / "tools" / bundle["release"]
    output.write_text(json.dumps(document, indent=2) + "\n")
    print(f"{output.name}: {repository} @ {revision[:12]} "
          f"({len(pinned)} model + {len(document['supportFilesSHA256'])} support files)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", type=Path, default=ROOT / "models" / "published",
                        help="directory holding the staged bundles (default: models/published)")
    parser.add_argument("--organization", default=os.environ.get("WAIFU_HF_ORG", "alastorid"))
    parser.add_argument("--only", action="append", default=None,
                        help="publish only these bundle directories (repeatable)")
    parser.add_argument("--dry-run", action="store_true", help="print the cards without uploading")
    arguments = parser.parse_args()

    if arguments.dry_run:
        for bundle in BUNDLES:
            directory = arguments.stage / bundle["directory"]
            if not directory.is_dir():
                continue
            print(f"--- {bundle['directory']} ---")
            print(card(bundle, directory, f"{arguments.organization}/{bundle['directory']}")[:600])
            print("...")
        return

    # Uploading uses SSH, so a token is only needed to create a repository that does
    # not exist yet. Everything else works with the registered key alone.
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = None
    if token:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        print(f"publishing as {api.whoami()['name']} to {arguments.organization}")
    else:
        print(f"no HF_TOKEN: pushing over SSH as the registered key, to {arguments.organization}")
    for bundle in BUNDLES:
        if arguments.only and bundle["directory"] not in arguments.only:
            continue
        directory = arguments.stage / bundle["directory"]
        if not directory.is_dir():
            sys.exit(f"{directory} is missing; run tools/stage-hf-bundles.sh first")
        acknowledge(bundle, f"{arguments.organization}/{bundle['directory']}")
        revision, readme = publish(api, bundle, directory, arguments.organization, arguments.stage)
        write_release(bundle, directory, f"{arguments.organization}/{bundle['directory']}", revision, readme)
    print("\nPublished. ./build.sh now bundles the pinned descriptors.")


if __name__ == "__main__":
    main()