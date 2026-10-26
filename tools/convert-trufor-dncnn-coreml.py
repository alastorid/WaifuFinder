#!/usr/bin/env python3
"""Convert only TruFor's DnCNN noiseprint subgraph to a fixed-shape ML Program.

This is an offline conversion utility. The inference worker is Swift/Core ML and
does not import Python, PyTorch, or coremltools. Input weights are read directly
from the validated Safetensors package; no checkpoint pickle is executed.
"""
import argparse, hashlib, json, mmap, shutil, struct, subprocess, sys
from pathlib import Path

SOURCE_SHA = "ac1d90e329a72e0d66e8665e123a19e94bfae3209c3ef8a4f9ca3b91578c7844"
TILE = 512
HALO = 17
INPUT = TILE + HALO * 2


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_header(path):
    stream = path.open("rb")
    length = struct.unpack("<Q", stream.read(8))[0]
    if length <= 0 or length > 100_000_000:
        stream.close()
        raise ValueError("Invalid TruFor Safetensors header")
    metadata = json.loads(stream.read(length))
    return stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ), 8 + length, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Validated TruFor model directory")
    parser.add_argument("--output", type=Path, required=True, help="New, empty output directory")
    parser.add_argument("--precision", choices=("fp32", "fp16", "mixed"), default="fp32")
    parser.add_argument("--fp32-conv-interval", type=int, default=4,
                        help="In mixed mode, retain every Nth DnCNN convolution in FP32")
    parser.add_argument("--runtime-dir", type=Path, default=Path(__file__).resolve().parents[1] / "Runtime")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new directory")

    try:
        import numpy as np
        import coremltools as ct
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import types
    except ImportError as error:
        raise SystemExit("Install the audited offline conversion tools; runtime does not need them: " + str(error))

    architecture = json.loads((args.model / "architecture.json").read_text())
    manifest = json.loads((args.model / "manifest.json").read_text())
    weights_path = args.model / "model.safetensors"
    expected = manifest.get("fileSHA256", {}).get("model.safetensors")
    if architecture.get("sourceSHA256") != SOURCE_SHA or not expected or digest(weights_path) != expected:
        raise SystemExit("TruFor architecture or checkpoint hash is not the validated source")

    stream, mapped, data_start, tensors = read_header(weights_path)
    arrays = {}

    def tensor(name, shape=None, required=True):
        item = tensors.get(name)
        if not item and not required:
            return None
        if not item or item.get("dtype") != "F32":
            raise ValueError("Missing or unsupported TruFor tensor: " + name)
        if shape is not None and item["shape"] != shape:
            raise ValueError("Unexpected TruFor tensor shape: " + name)
        lo, hi = item["data_offsets"]
        value = np.frombuffer(mapped, dtype="<f4", count=(hi - lo) // 4, offset=data_start + lo)
        return value.reshape(item["shape"]).astype(np.float32, copy=True)

    def conv_weights(index):
        prefix = "dncnn." + str(index)
        weight = tensor(prefix + ".weight")
        # The original DnCNN disables convolution bias for every layer that
        # immediately feeds BatchNorm. The MLX reference treats absent bias as
        # zero; use the same exact convention before folding BatchNorm.
        bias = tensor(prefix + ".bias", required=False)
        if bias is None:
            bias = np.zeros(weight.shape[0], dtype=np.float32)
        if index not in (0, 47):
            bn = "dncnn." + str(index + 1) + "."
            gamma, beta = tensor(bn + "weight"), tensor(bn + "bias")
            mean, variance = tensor(bn + "running_mean"), tensor(bn + "running_var")
            scale = gamma / np.sqrt(variance + np.float32(1e-5))
            weight *= scale.reshape((-1, 1, 1, 1))
            bias = (bias - mean) * scale + beta
        return weight, bias

    indices = [0] + [2 + 3 * i for i in range(15)] + [47]
    for index in indices:
        arrays[index] = conv_weights(index)
    stream.close()
    mapped.close()

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, 3, INPUT, INPUT), dtype=types.fp32),
            mb.TensorSpec(shape=(1, 1, INPUT, INPUT), dtype=types.fp32),
        ],
        opset_version=ct.target.macOS15,
    )
    def noiseprint(rgb, valid):
        x = rgb
        for level, index in enumerate(indices):
            weight, bias = arrays[index]
            x = mb.conv(x=x, weight=weight, bias=bias, strides=(1, 1), pad_type="same",
                        dilations=(1, 1), groups=1, name="dncnn_conv_" + str(index))
            if level < 16:
                x = mb.relu(x=x, name="dncnn_relu_" + str(index))
                # Tiles touching an image border are padded to the same fixed
                # model shape. Zeroing invalid features after every activated
                # layer reproduces the original zero-padded convolution border.
                x = mb.mul(x=x, y=valid, name="dncnn_valid_" + str(index))
        return x

    fp32_conv_indices = []
    if args.precision == "fp16":
        precision = ct.precision.FLOAT16
    elif args.precision == "mixed":
        if args.fp32_conv_interval < 1:
            parser.error("--fp32-conv-interval must be positive")
        fp32_conv_indices = [index for level, index in enumerate(indices)
                             if level % args.fp32_conv_interval == 0 or level == len(indices) - 1]
        preserved = {"dncnn_conv_" + str(index) for index in fp32_conv_indices}
        precision = ct.transform.FP16ComputePrecision(
            op_selector=lambda op: not (op.op_type == "conv" and op.name in preserved)
        )
    else:
        precision = ct.precision.FLOAT32
    model = ct.convert(noiseprint, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                       compute_precision=precision)
    args.output.mkdir(parents=True)
    package = args.output / "model.mlpackage"
    model.save(str(package))
    compiled = args.output / "model.mlmodelc"
    subprocess.run(["xcrun", "coremlcompiler", "compile", str(package), str(args.output),
                    "--platform", "macOS", "--deployment-target", "15.0"], check=True)
    for name in ("TRUFOR-LICENSE.txt", "TRUFOR-CMX-LICENSE.txt"):
        shutil.copy2(args.runtime_dir / name, args.output / name)
    spec = model.get_spec()
    record = {
        "format": "trufor-dncnn-coreml-v1", "sourceSHA256": SOURCE_SHA,
        "weightsSHA256": expected, "modelSHA256": digest(args.output / "model.mlpackage" / "Data" / "com.apple.CoreML" / "model.mlmodel"),
        "compiledModel": str(compiled),
        "coremltoolsVersion": ct.__version__, "numpyVersion": np.__version__,
        "precision": args.precision, "tilePixels": TILE, "haloPixels": HALO,
        "fp32ConvIndices": fp32_conv_indices,
        "fixedInputShape": [1, 3, INPUT, INPUT], "maskInputShape": [1, 1, INPUT, INPUT],
        "inputNames": [value.name for value in spec.description.input],
        "outputNames": [value.name for value in spec.description.output],
        "computeUnitsNote": "Placement is determined and recorded separately with MLComputePlan for each policy.",
        "runtimeDependency": "Swift Core ML only; Python, PyTorch and coremltools are conversion-time tools.",
    }
    (args.output / "conversion.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
