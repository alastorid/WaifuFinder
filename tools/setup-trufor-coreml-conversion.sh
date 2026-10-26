#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${WAIFU_TRUFOR_CONVERTER_PYTHON:-python3.13}"
ENV_DIR="$ROOT/.build/trufor-coreml-venv"

command -v "$PYTHON_BIN" >/dev/null || {
    echo "A Homebrew or python.org Python 3.13 interpreter is required: $PYTHON_BIN" >&2
    exit 1
}
"$PYTHON_BIN" -m venv "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install --upgrade pip
# NumPy is built from its BSD-licensed source against Apple's Accelerate system
# framework. PyPI's binary NumPy wheel bundles LGPL libgfortran/libquadmath, so
# it is deliberately excluded. Core ML Tools is conversion-only (BSD-3-Clause);
# the inference worker imports only Apple's Swift Core ML framework.
"$ENV_DIR/bin/python" -m pip install \
    --no-binary=numpy \
    -Csetup-args=-Dblas=accelerate \
    -Csetup-args=-Dlapack=accelerate \
    "numpy==2.5.4" "coremltools==9.0" "torch==2.7.0"
"$ENV_DIR/bin/python" - <<'PY'
import pathlib, numpy, coremltools
import subprocess
for item in pathlib.Path(__import__("sys").prefix).rglob("*"):
    if not item.is_file():
        continue
    if any(token in item.name.lower() for token in ("gfortran", "quadmath", "ffmpeg", "avcodec", "avformat")):
        raise SystemExit("Refusing a conversion environment with a prohibited runtime: " + str(item))
    if item.suffix in (".so", ".dylib"):
        links = subprocess.run(["otool", "-L", str(item)], capture_output=True, text=True).stdout.lower()
        if any(token in links for token in ("libgfortran", "quadmath", "libopenblas", "libgpl", "libavcodec", "libavformat", "ffmpeg")):
            raise SystemExit("Refusing a conversion environment with an unapproved linked runtime: " + str(item))
print("TruFor converter ready:", "NumPy", numpy.__version__, "Core ML Tools", coremltools.__version__)
print("NumPy BLAS configuration:")
numpy.show_config()
PY
