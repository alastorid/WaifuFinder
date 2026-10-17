#!/usr/bin/env python3
"""Compare native predictions using exact Swift-prepared inputs and original weights."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import timm
import torch
from safetensors.torch import load_file


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(actual, expected, general):
    if len(actual) == 0:
        return None
    delta = np.abs(actual - expected)
    overlaps = [len(set(np.argsort(a)[-15:]) & set(np.argsort(b)[-15:])) for a, b in zip(actual, expected)]
    general_overlaps = [len(set(np.argsort(a[general])[-15:]) & set(np.argsort(b[general])[-15:]))
                        for a, b in zip(actual, expected)]
    return {
        "maximumAbsoluteError": float(delta.max()), "meanAbsoluteError": float(delta.mean()),
        "perImageMaximumAbsoluteError": delta.max(axis=1).tolist(),
        "top15Overlaps": overlaps, "minimumTop15Overlap": min(overlaps),
        "top15GeneralOverlaps": general_overlaps, "minimumTop15GeneralOverlap": min(general_overlaps),
        "thresholdDifferencesAllHeads": int(((actual > 0.35) != (expected > 0.35)).sum()),
        "thresholdDifferencesGeneralHeads": int(((actual[:, general] > 0.35) != (expected[:, general] > 0.35)).sum()),
        "ratingDifferences": int((actual[:, :4].argmax(axis=1) != expected[:, :4].argmax(axis=1)).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--natural-frame-count", type=int, default=0)
    args = parser.parse_args()
    directory = args.directory
    inputs = np.memmap(directory / "inputs.bin", dtype="<f4", mode="r").reshape(-1, 3, 448, 448)
    natural_count = args.natural_frame_count
    if not 0 <= natural_count <= len(inputs):
        parser.error("Natural frame count must be between zero and the input count")
    reference_path = directory / "reference.bin"
    reference_metadata = directory / "reference-metadata.json"
    identity = {"inputSHA256": sha256(directory / "inputs.bin"), "weightsSHA256": sha256(args.weights)}
    cached_identity = json.loads(reference_metadata.read_text()) if reference_metadata.exists() else None
    if not reference_path.exists() or cached_identity != identity:
        torch.set_num_threads(4)
        core = timm.create_model("swinv2_base_window8_256", pretrained=False, num_classes=10861,
                                 img_size=448, window_size=14, act_layer="gelu_tanh").eval()
        core.load_state_dict(load_file(str(args.weights)), strict=True)
        expected = []
        with torch.inference_mode():
            for offset in range(0, len(inputs), 4):
                # Independent daemon path: original projection weights, runtime BGR.
                tensor = torch.from_numpy(inputs[offset:offset+4, [2, 1, 0]].copy()) * (2 / 255) - 1
                expected.append(torch.sigmoid(core(tensor)).numpy())
        np.concatenate(expected).astype("<f4").tofile(reference_path)
        reference_metadata.write_text(json.dumps(identity, indent=2) + "\n")
    if args.reference_only:
        return
    expected = np.fromfile(reference_path, dtype="<f4").reshape(-1, 10861)
    with args.labels.open(newline="") as handle:
        general = np.array([i for i, row in enumerate(csv.DictReader(handle)) if row["category"] == "0"])
    baseline = np.fromfile(directory / "batch8-probabilities.bin", dtype="<f4").reshape(-1, 10861)
    report = {"frames": len(inputs), "naturalVideoFrames": natural_count, "syntheticFrames": len(inputs) - natural_count,
              **identity,
              "reference": "Original unfused SwinV2 FP32 weights with runtime BGR, exact native-preprocessed inputs",
              "note": "Synthetic numerical smoke inputs; not labeled accuracy data." if natural_count == 0
                      else "User-supplied mixed corpus; not labeled accuracy data.",
              "candidates": {}}
    for path in sorted(directory.glob("*-probabilities.bin")):
        actual = np.fromfile(path, dtype="<f4").reshape(-1, 10861)
        if actual.shape != expected.shape or not np.isfinite(actual).all() or (actual < 0).any() or (actual > 1).any():
            raise RuntimeError(f"Invalid probabilities in {path.name}")
        name = path.name.removesuffix("-probabilities.bin")
        report["candidates"][name] = {
            "againstReference": metrics(actual, expected, general),
            "naturalAgainstReference": metrics(actual[:natural_count], expected[:natural_count], general),
            "syntheticAgainstReference": metrics(actual[natural_count:], expected[natural_count:], general),
            "againstProduction": metrics(actual, baseline, general),
        }
    (directory / "accuracy.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, value in report["candidates"].items():
        result = value["againstReference"]
        print(name, "max/mean", result["maximumAbsoluteError"], result["meanAbsoluteError"],
              "top15", result["minimumTop15Overlap"], "threshold", result["thresholdDifferencesGeneralHeads"])


if __name__ == "__main__":
    main()
