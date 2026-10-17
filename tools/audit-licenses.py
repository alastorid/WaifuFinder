#!/usr/bin/env python3
"""Repository guard for prohibited media tools and GPL/LGPL package artifacts.

This supplements the dependency-by-dependency license inventory; it does not
replace review of direct/transitive licenses before adding a dependency.
Uses only the Python standard library and never executes inspected software.
"""
import os
import re
from email.parser import Parser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
errors = []
files_checked = 0
metadata_checked = 0
license_checked = 0
prohibited_binary = re.compile(r"^(?:ffmpeg|ffprobe)(?:\.exe)?$|^lib(?:avcodec|avformat|avfilter|avdevice|avutil|swscale|swresample|postproc|x264|x265|gfortran|quadmath|gomp)(?:[.\d-]|$)", re.I)
copyleft = re.compile(r"\b(?:A?GPL|LGPL)(?:[-\s]?v?\d|\b)|GNU (?:Lesser |Affero )?General Public License", re.I)
invocation = re.compile(r"(?:runFFmpeg|requireFFmpeg)\s*\(|brew\s+install\s+ffmpeg|(?:^|[\s\"'])ffmpeg\s+-|[\"'](?:/[^\"']*/)?(?:ffmpeg|ffprobe)(?:\.exe)?[\"']|import\s+(?:ffmpeg|av)\b", re.I)

for directory, directories, names in os.walk(ROOT):
    directories[:] = [name for name in directories if name not in {".git", "node_modules", "models"}]
    for name in names:
        path = Path(directory) / name
        relative = path.relative_to(ROOT)
        files_checked += 1
        if prohibited_binary.search(name):
            errors.append(f"Prohibited executable/library artifact: {relative}")
        if name == "METADATA" and path.parent.name.endswith(".dist-info"):
            metadata_checked += 1
            metadata = Parser().parsestr(path.read_text(errors="replace"))
            expression = metadata.get("License-Expression", "")
            classifiers = " ".join(c for c in metadata.get_all("Classifier", []) if c.startswith("License ::"))
            declared = expression or classifiers or metadata.get("License", "")[:512]
            if copyleft.search(declared):
                errors.append(f"Prohibited package license: {relative}: {metadata.get('Name')} ({declared[:140]})")
        if name.upper().startswith(("LICENSE", "COPYING")) and path.is_file():
            license_checked += 1
            beginning = path.read_bytes()[:2048].decode(errors="replace")
            if re.match(r"\s*(?:GNU\s+)?(?:LESSER\s+|AFFERO\s+)?GENERAL PUBLIC LICENSE", beginning, re.I):
                errors.append(f"Prohibited component license: {relative}")
        source_root = relative.parts[0] in {"Sources", "Tests", "tools", ".github"} or relative.parts[:2] == ("Runtime", "Sources")
        if source_root and path != Path(__file__).resolve() and path.suffix in {".swift", ".py", ".sh", ".yml", ".js", ".mjs", ".cjs", ".h", ".cpp"}:
            text = path.read_text(errors="replace")
            for line_number, line in enumerate(text.splitlines(), 1):
                if "SPDX-License-Identifier:" in line and copyleft.search(line):
                    errors.append(f"Prohibited SPDX license: {relative}:{line_number}")
                if line.lstrip().startswith(("//", "#", "*")):
                    continue
                if invocation.search(line):
                    errors.append(f"Prohibited media-tool dependency: {relative}:{line_number}")

if errors:
    print("Dependency policy audit FAILED:")
    print("\n".join(errors))
    raise SystemExit(1)
print(f"Dependency policy audit passed: {files_checked} project-local files, {license_checked} license files, {metadata_checked} Python package records; no prohibited artifacts or active media-tool calls found.")
