#!/usr/bin/env python3
"""Convert the WD SwinV2 tagger to a WaifuFinder Core ML bundle.

The upstream project (``SmilingWolf/wd-swinv2-tagger-v3``) is a 98M-parameter
SwinV2-Base that scores 10,861 tags with independent sigmoid heads. This tool
turns it into the same kind of bundle the app already ships for its restoration
backends: a Core ML ``.mlpackage`` plus the label table, published to a release
repository and downloaded by ``ModelLibrary`` / ``ModelInstaller``.

## What is fused, and why

The reference client (``wd-tagger-photos-app/wd_daemon/server.py``) runs four
steps per image that have nothing to do with the network:

    longest side -> 1024 (LANCZOS), letterbox to square on WHITE, -> 448x448,
    normalise mean/std 0.5, then **reverse the channels to BGR**.

The last one is a Danbooru training-data quirk, not a typo: the model was
trained on BGR-ordered tensors despite its config declaring a generic
``[0.5, 0.5, 0.5]`` normalisation. Feed it RGB and every prediction is noise.

Two of those steps disappear entirely:

  * **BGR is folded into the PatchEmbed convolution.** Permuting the *input*
    axis of a conv weight is the same convolution over swapped channels, so the
    swap costs no op and no extra tensor. Done on the weight, outside the
    exported graph, so nothing is traced.
  * **mean/std normalisation stays in the graph** as two scalar ops
    (``x * 2/255 - 1``). It is 600K multiply-adds against a network that spends
    ~12.5 GFLOPs, so folding it into the conv would buy nothing and would risk
    fp16 underflow on small weights.

Geometry stays in Swift. The app already owns the decode and the thumbnail
crop, and CoreImage gives exact control over interpolation and the white
letterbox, which is where a CoreML baked-in resize would quietly drift.

## Native fused attention

All 24 SwinV2 blocks use native scaled-dot-product attention. Cosine Q/K
normalization, learned temperature, positional bias and shifted-window masks
remain unchanged. Window and head axes are folded into one head axis so queries
and masks remain rank four, avoiding the prior rank-five broadcast/view encoding
path. The exported ranks are checked before a package can be saved.

## Why the production batch is fixed

The reference daemon batches, and the app tags frames as they are analysed, so
the production batch axis is pinned to 8 rather than made flexible. One shape lets Core ML
pick its best kernel, and the tail of a run is padded with a duplicate frame
whose results are dropped. Use --batch-size to build isolated shapes for benchmarking;
changing the model shape also requires a compatible app backend.

Usage::

    tools/convert-wd-tagger.py [--output DIR] [--repo REPO] [--skip-validation]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch
import coremltools as ct
from safetensors.torch import load_file

DEFAULT_REPO = "SmilingWolf/wd-swinv2-tagger-v3"
LABEL_FILE = "selected_tags.csv"
WEIGHT_FILE = "model.safetensors"

# The shape the model is compiled for. See the module docstring: fixed on
# purpose, not an oversight.
BATCH = 8
HEIGHT = 448
WIDTH = 448
CHANNELS = 3

# Danbooru general-tag threshold the reference client uses (server.py:601).
GENERAL_THRESHOLD = 0.35

# Indices 0..3 of the 10,861 heads are the rating classes, in this order.
RATING_LABELS = ("general", "sensitive", "questionable", "explicit")

OPTIMIZATION_VERSION = "swinv2-sdpa-rank4-v1"

# How close the compiled model has to be to the torch reference before the
# bundle is allowed to call itself validated. fp16 rounding on a 98M-parameter
# network is not zero, so the tolerance is a real measurement, not a formality.
MAX_ABS_ERROR_TOLERANCE = 5e-3
MIN_TOP15_OVERLAP = 12


def log(message: str) -> None:
    print(f"[convert-wd-tagger] {message}", flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(repo: str, destination: Path) -> tuple[Path, Path]:
    """Fetch the two artifacts the conversion needs, into ``destination``."""
    from huggingface_hub import hf_hub_download

    destination.mkdir(parents=True, exist_ok=True)
    weights = destination / WEIGHT_FILE
    labels = destination / LABEL_FILE
    for name, target in ((WEIGHT_FILE, weights), (LABEL_FILE, labels)):
        if target.exists():
            log(f"reusing {target}")
            continue
        log(f"downloading {repo}/{name}")
        path = Path(hf_hub_download(repo, name, local_dir=str(destination)))
        if path != target:
            shutil.move(str(path), target)
    return weights, labels


def build_model(weights: Path, fused_attention: bool = True) -> torch.nn.Module:
    """Build the timm tagger and load the upstream weights."""
    import timm

    log("building timm SwinV2 tagger")
    model = timm.create_model("swinv2_base_window8_256", pretrained=False,
                              num_classes=10_861, img_size=448, window_size=14,
                              act_layer="gelu_tanh").eval()

    state = load_file(str(weights))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"weight load mismatch: {len(missing)} missing, {len(unexpected)} unexpected")
    # Fold the BGR channel swap into PatchEmbed's convolution. A permutation of
    # a conv's *input* channel axis is the identical convolution over swapped
    # input channels, so this is exact and adds no op. Performed under no_grad
    # with a contiguous copy so the slice is never captured by the exporter.
    projection = model.patch_embed.proj
    with torch.no_grad():
        swapped = projection.weight.detach()[:, [2, 1, 0], :, :].contiguous().clone()
    projection.weight = torch.nn.Parameter(swapped)
    del swapped
    if fused_attention:
        from wd_tagger_attention import install_sdpa
        install_sdpa(model)
        log("using 24 native SDPA blocks with rank-four queries/masks")

    log(f"loaded {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


class Tagger(torch.nn.Module):
    """0..255 RGB in, sigmoid probabilities out.

    Kept deliberately small: the exporter captures the normalisation as two
    scalar ops, which is what the module docstring argues for.
    """

    def __init__(self, core: torch.nn.Module) -> None:
        super().__init__()
        self.core = core

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * (2.0 / 255.0) - 1.0
        return torch.sigmoid(self.core(x))


def export(model: torch.nn.Module):
    """Export to EXIR, decomposing the ops Core ML cannot lower.

    ``run_decompositions`` is required: the exported graph carries ATEN/EDGE
    dialect ops the MIL frontend refuses. The ``alias`` nodes that
    functionalization emits around SwinV2's window reshapes are then stripped —
    they are no-ops wrapping ``view``, and the frontend has no alias op.
    """
    example = torch.zeros(BATCH, CHANNELS, HEIGHT, WIDTH)
    program = torch.export.export(model, (example,), strict=False)
    program = program.run_decompositions({})
    stripped = 0
    for node in list(program.graph.nodes):
        if node.op == "call_function" and "alias" in str(node.target):
            node.replace_all_uses_with(node.args[0])
            stripped += 1
    program.graph.eliminate_dead_code()
    log(f"exported ({stripped} alias nodes stripped)")
    return program


def convert(program, destination: Path, weights_sha: str) -> Path:
    """Lower the exported graph to a Core ML mlprogram."""
    log("converting to Core ML (fp16 mlprogram, macOS 15)")
    mlmodel = ct.convert(
        program,
        source="pytorch",
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        inputs=[ct.TensorType(name="input", shape=(BATCH, CHANNELS, HEIGHT, WIDTH), dtype=np.float32)],
        outputs=[ct.TensorType(name="output", dtype=np.float32)],
        skip_model_load=True,
    )

    if OPTIMIZATION_VERSION == "swinv2-sdpa-rank4-v1":
        from wd_tagger_attention import check_sdpa_spec
        check_sdpa_spec(mlmodel.get_spec())
        mlmodel.user_defined_metadata["waifu.sdpaRank4"] = "true"

    mlmodel.author = "WaifuFinder"
    mlmodel.license = "Apache-2.0"
    mlmodel.short_description = f"WD SwinV2-Base multi-label tagger, fp16, batch {BATCH}"
    mlmodel.user_defined_metadata.update({
        "waifu.optimizationVersion": OPTIMIZATION_VERSION,
        "waifu.validated": "false",
        "waifu.weightsSHA256": weights_sha,
        "waifu.labelCount": "10861",
        "waifu.generalThreshold": str(GENERAL_THRESHOLD),
        "waifu.batchSize": str(BATCH),
        "waifu.input": f"RGB float32 0..255, 448x448, batch {BATCH}; BGR and mean/std 0.5 applied by the graph",
        "waifu.ratingLabels": ",".join(RATING_LABELS),
        "com.github.apple.coremltools.source": "timm/swinv2-bgr-fused",
    })

    bundle = destination / "WDTagger.mlpackage"
    if bundle.exists():
        shutil.rmtree(bundle)
    mlmodel.save(str(bundle))
    return bundle


def topk_overlap(got: np.ndarray, expected: np.ndarray, k: int) -> int:
    a = set(np.argsort(got)[::-1][:k].tolist())
    b = set(np.argsort(expected)[::-1][:k].tolist())
    return len(a & b)


def validate(bundle: Path, weights: Path) -> dict:
    """Run the compiled model and compare it against the reference in torch.

    This is the step that makes the fusion above trustworthy: the same weights,
    the same graph, two runtimes. A tagger whose probabilities drifted would
    still *look* like it worked, so the check is on the top-k overlap and on the
    spread of the error, not on "it produced output".
    """
    spec = ct.models.MLModel(str(bundle), skip_model_load=True).get_spec()
    shape = [int(d) for d in spec.description.input[0].type.multiArrayType.shape]
    if shape != [BATCH, CHANNELS, HEIGHT, WIDTH]:
        raise SystemExit(f"compiled input shape is {shape}, expected {[BATCH, CHANNELS, HEIGHT, WIDTH]}")

    model = ct.models.MLModel(str(bundle), compute_units=ct.ComputeUnit.CPU_AND_GPU)

    rng = np.random.default_rng(20261013)
    images = rng.integers(0, 256, size=(BATCH, CHANNELS, HEIGHT, WIDTH),
                          dtype=np.uint8).astype(np.float32)

    torch.set_num_threads(4)
    # Restore the original unfused projection for an independent daemon reference.
    # Comparing two uses of the fused helper would conceal a broken channel swap.
    original = build_model(weights, fused_attention=False)
    original.load_state_dict(load_file(str(weights)), strict=True)
    reference = Tagger(original).eval()
    with torch.inference_mode():
        expected = reference(torch.from_numpy(images[:, [2, 1, 0], :, :].copy())).numpy()

    log("validating against the torch reference")
    start = time.time()
    produced = model.predict({"input": np.ascontiguousarray(images)})["output"]
    elapsed = time.time() - start

    got = np.asarray(produced, dtype=np.float32)
    if got.shape != expected.shape:
        raise SystemExit(f"output shape {got.shape} != reference {expected.shape}")

    delta = np.abs(got - expected)
    record = {
        "format": 1,
        "optimizationVersion": OPTIMIZATION_VERSION,
        "modelRepo": DEFAULT_REPO,
        "batchSize": BATCH,
        "inputShape": shape,
        "outputShape": [int(d) for d in spec.description.output[0].type.multiArrayType.shape],
        "precision": "fp16",
        "computeUnits": "cpuAndGPU",
        "weightsSHA256": sha256(weights),
        "validation": {
            "maxAbsoluteError": float(delta.max()),
            "meanAbsoluteError": float(delta.mean()),
            "top15Overlap": min(topk_overlap(a, b, 15) for a, b in zip(got, expected)),
            "top15Overlaps": [topk_overlap(a, b, 15) for a, b in zip(got, expected)],
            "threshold": GENERAL_THRESHOLD,
            "secondsForBatch": elapsed,
            "imagesPerSecond": BATCH / elapsed,
        },
    }
    v = record["validation"]
    log(f"max|err|={v['maxAbsoluteError']:.3e}  mean|err|={v['meanAbsoluteError']:.3e}  "
        f"top15={v['top15Overlap']}/15  {v['imagesPerSecond']:.1f} frames/s (cpuAndGPU)")

    if v["top15Overlap"] < MIN_TOP15_OVERLAP:
        raise SystemExit(f"top-15 tag overlap {v['top15Overlap']}/15 is below {MIN_TOP15_OVERLAP}; "
                         "the conversion is not faithful")
    if v["maxAbsoluteError"] > MAX_ABS_ERROR_TOLERANCE:
        raise SystemExit(f"max absolute error {v['maxAbsoluteError']:.3e} exceeds "
                         f"{MAX_ABS_ERROR_TOLERANCE:.0e}")
    return record


def write_labels(labels: Path, destination: Path) -> Path:
    """Copy the label table next to the bundle and report its shape.

    The app parses this at load time to build the autocomplete index, so its row
    order is load-bearing: the model's 10,861 output indices are positions in
    this file. Sorting it would silently renumber every tag.
    """
    names, categories = [], []
    with labels.open(newline="") as handle:
        for row in csv.DictReader(handle):
            names.append(row["name"])
            categories.append(row["category"])
    if len(names) != 10861:
        raise SystemExit(f"label table has {len(names)} rows, expected 10861")
    if tuple(names[:4]) != RATING_LABELS:
        raise SystemExit(f"label table does not begin with the rating classes: {names[:4]}")
    output = destination / "wd-tagger-labels.csv"
    shutil.copyfile(labels, output)
    log(f"labels: {len(names)} rows ({categories.count('0')} general, "
        f"{categories.count('4')} character, {categories.count('9')} rating)")
    return output


def mark_validated(bundle: Path) -> None:
    """Rewrite the metadata flag now that a comparison has passed.

    The package is a directory holding ``model.mlmodel`` and a weights file, and
    ``save_spec`` will not overwrite a non-empty one — so the spec is patched in
    place and the weights are left untouched.
    """
    model_file = bundle / "Data" / "com.apple.CoreML" / "model.mlmodel"
    spec = ct.models.utils.load_spec(str(model_file))
    spec.description.metadata.userDefined["waifu.validated"] = "true"
    ct.models.utils.save_spec(spec, str(model_file))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent.parent / ".build/model-conversion/wd-tagger")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--unfused-attention", action="store_true", help="Diagnostic baseline without SDPA")
    parser.add_argument("--batch-size", type=int, choices=[1, 2, 4, 8, 16, 32], default=8,
                        help="Fixed export shape; production uses 8")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    weights, labels = download(args.repo, args.output)

    global OPTIMIZATION_VERSION, BATCH
    BATCH = args.batch_size
    if args.unfused_attention:
        OPTIMIZATION_VERSION = "swinv2-bgr-fused-v1"
    model = build_model(weights, fused_attention=not args.unfused_attention)
    program = export(Tagger(model).eval())
    bundle = convert(program, args.output, sha256(weights))
    write_labels(labels, args.output)

    if not args.skip_validation:
        record = validate(bundle, weights)
        (args.output / "validation.json").write_text(json.dumps(record, indent=2) + "\n")
        mark_validated(bundle)

    log(f"bundle: {bundle}")
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
