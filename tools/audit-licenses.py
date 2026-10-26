#!/usr/bin/env python3
"""Repository guard for prohibited media tools and GPL/LGPL package artifacts.

This supplements the dependency-by-dependency license inventory; it does not
replace review of direct/transitive licenses before adding a dependency.
Uses only the Python standard library and never executes inspected software.
"""
import ast
import os
import re
from email.parser import Parser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
prohibited_binary = re.compile(r"^(?:ffmpeg|ffprobe)(?:\.exe)?$|^lib(?:avcodec|avformat|avfilter|avdevice|avutil|swscale|swresample|postproc|x264|x265|gfortran|quadmath|gomp)(?:[.\d-]|$)", re.I)
copyleft = re.compile(r"\b(?:A?GPL|LGPL)(?:[-\s]?v?\d|\b)|GNU (?:Lesser |Affero )?General Public License", re.I)
invocation = re.compile(r"(?:runFFmpeg|requireFFmpeg)\s*\(|brew\s+install\s+ffmpeg|(?:^|[\s\"'])ffmpeg\s+-|[\"'](?:/[^\"']*/)?(?:ffmpeg|ffprobe)(?:\.exe)?[\"']|import\s+(?:ffmpeg|av)\b", re.I)

def mask_denylist_literals(line):
    """Do not mistake strings being rejected for executable dependencies.

    Only mask literal tuple/list members in an any/all membership guard. Keep
    the expression being checked intact, so executable calls inside a guard
    are still audited. This also handles Python embedded in setup shell scripts.
    """
    stripped = line.strip()
    if not stripped.startswith("if ") or not stripped.endswith(":"):
        return line
    try:
        statement = ast.parse(stripped + "\n    pass").body[0]
    except SyntaxError:
        return line
    test = statement.test
    if not (isinstance(test, ast.Call) and isinstance(test.func, ast.Name)
            and test.func.id in {"any", "all"} and len(test.args) == 1
            and isinstance(test.args[0], ast.GeneratorExp)):
        return line
    expression = test.args[0]
    if not (isinstance(expression.elt, ast.Compare) and len(expression.elt.ops) == 1
            and isinstance(expression.elt.ops[0], (ast.In, ast.NotIn))):
        return line
    masked = bytearray(stripped.encode())
    for generator in expression.generators:
        literals = generator.iter
        if not (isinstance(generator.target, ast.Name)
                and isinstance(expression.elt.left, ast.Name)
                and generator.target.id == expression.elt.left.id
                and isinstance(literals, (ast.Tuple, ast.List))
                and all(isinstance(item, ast.Constant) and isinstance(item.value, str)
                        for item in literals.elts)):
            continue
        for item in literals.elts:
            masked[item.col_offset:item.end_col_offset] = b" " * (item.end_col_offset - item.col_offset)
    return masked.decode()


def audit(root):
    errors = []
    files_checked = 0
    metadata_checked = 0
    license_checked = 0
    for directory, directories, names in os.walk(root):
        directories[:] = [name for name in directories if name not in {".git", "node_modules", "models"}]
        for name in names:
            path = Path(directory) / name
            relative = path.relative_to(root)
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
                    if invocation.search(mask_denylist_literals(line)):
                        errors.append(f"Prohibited media-tool dependency: {relative}:{line_number}")
    return errors, files_checked, license_checked, metadata_checked


def main():
    errors, files_checked, license_checked, metadata_checked = audit(ROOT)
    if errors:
        print("Dependency policy audit FAILED:")
        print("\n".join(errors))
        return 1
    print(f"Dependency policy audit passed: {files_checked} project-local files, {license_checked} license files, {metadata_checked} Python package records; no prohibited artifacts or active media-tool calls found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
