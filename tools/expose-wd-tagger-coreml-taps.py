#!/usr/bin/env python3
"""Create a diagnostic Core ML package that returns selected MIL activations.

Use only with the conversion environment described in
tools/setup-trufor-coreml-conversion.sh. The distributed runtime remains Swift
and Core ML; this edits an offline copy of the checked-in ML Program.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import coremltools as ct
from coremltools.proto import FeatureTypes_pb2, MIL_pb2


DEFAULT_TAPS = (
    # First stage-1 block, after cosine attention, attention residual/norm,
    # and the complete MLP block, respectively.
    "scaled_dot_product_attention_cast_fp16",
    "layer_norm_1_cast_fp16",
    "add_1_cast_fp16",
    # End of stages 1, 2, 3 and 4.
    "add_4_cast_fp16",
    "add_9_cast_fp16",
    "add_54_cast_fp16",
    "add_58_cast_fp16",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Source WDTagger.mlpackage")
    parser.add_argument("--output", type=Path, required=True, help="New diagnostic .mlpackage path")
    parser.add_argument("--tap", action="append", default=[], help="MIL tensor name; repeat to add taps")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    taps = args.tap or list(DEFAULT_TAPS)
    if len(set(taps)) != len(taps):
        parser.error("Tap names must be unique")
    if not source.is_dir() or source.suffix != ".mlpackage":
        parser.error(f"Not an ML Program package: {source}")
    if output.exists():
        parser.error(f"Output already exists: {output}")

    model = ct.models.MLModel(str(source), skip_model_load=True)
    spec = model.get_spec()
    try:
        function = spec.mlProgram.functions["main"]
        block = next(iter(function.block_specializations.values()))
    except (KeyError, StopIteration) as error:
        raise SystemExit("Could not locate the main ML Program block") from error

    variables = {value.name: value for operation in block.operations for value in operation.outputs}
    missing = [name for name in taps if name not in variables]
    if missing:
        raise SystemExit(f"Unknown MIL tensor(s): {', '.join(missing)}")

    for name in taps:
        variable = variables[name]
        tensor = variable.type.tensorType
        if tensor.dataType != MIL_pb2.FLOAT16:
            raise SystemExit(f"{name} is not FP16; got MIL data type {tensor.dataType}")
        shape = []
        for dimension in tensor.dimensions:
            if not dimension.HasField("constant"):
                raise SystemExit(f"{name} has a non-fixed dimension")
            shape.append(dimension.constant.size)
        block.outputs.append(name)
        feature = spec.description.output.add()
        feature.name = name
        feature.type.multiArrayType.dataType = FeatureTypes_pb2.ArrayFeatureType.FLOAT16
        feature.type.multiArrayType.shape.extend(shape)
        print(f"Tap {name}: fp16 {shape}")

    output.mkdir(parents=True)
    shutil.copytree(source, output, dirs_exist_ok=True)
    model_file = output / "Data/com.apple.CoreML/model.mlmodel"
    ct.models.utils.save_spec(spec, str(model_file))

    # Validate the edited package's public outputs without loading the runtime.
    checked = ct.models.MLModel(str(output), skip_model_load=True).get_spec()
    output_names = [feature.name for feature in checked.description.output]
    expected = [feature.name for feature in spec.description.output]
    if output_names != expected:
        raise SystemExit(f"Saved outputs {output_names} do not match expected {expected}")
    print(f"Wrote diagnostic package: {output}")


if __name__ == "__main__":
    main()
