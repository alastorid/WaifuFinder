#!/usr/bin/env python3
"""OFFLINE ONLY: strict official-weight PGTFormer -> fixed-shape Core ML.

Never invoked by the app. Supply the pinned upstream checkout, HF config and
safetensors. All neural operations remain upstream code. BasicSR's unused
training registry/logger imports are isolated from its incompatible installer.
"""
import argparse
import hashlib
import json
import logging
import inspect
import textwrap
from pathlib import Path
import subprocess
import sys
import types

UPSTREAM = "e78a26cd463182f4a5aa96b8681110da8615d44d"
WEIGHTS_REVISION = "bc0b7e3ff10cfa0f2f335f5256919de20fe2b29d"
WEIGHTS_SHA256 = "37f9a42b9e35e0ccb89016ca6cea6728a7bf783e70b99e886ac52bb37c6855f3"


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reference(upstream, config, weights):
    if file_sha256(weights) != WEIGHTS_SHA256:
        raise ValueError("Expected the pinned original PGTFormer checkpoint")
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM:
        raise ValueError(f"Expected upstream {UPSTREAM}, got {revision}")
    # No BasicSR neural operators are imported by these architecture files.
    class Registry:
        def register(self):
            return lambda cls: cls
    registry = Registry()
    for name in ["basicsr", "basicsr.archs", "basicsr.utils", "basicsr.utils.registry"]:
        module = types.ModuleType(name)
        module.ARCH_REGISTRY = registry
        module.get_root_logger = logging.getLogger
        sys.modules[name] = module
    sys.path.insert(0, str(upstream.resolve()))
    from archs.pgtformer_arch import PGTFormer
    from safetensors.torch import load_file
    settings = json.loads(config.read_text())
    settings.pop("type", None)
    model = PGTFormer(**settings)
    # Upstream overrides train() without returning self, so do not chain eval().
    model.eval()
    model.requires_grad_(False)
    model.load_state_dict(load_file(str(weights)), strict=True)
    return model


def fixed_shape_trace_adapter(model):
    # During tracing str(tensor.shape[-1]) becomes "tensor(32)", so upstream's
    # string-keyed decoder fusion branches are silently skipped. Resolve only
    # these dictionary keys to fixed Python integers; all neural math is intact.
    original = type(model).forward
    source = textwrap.dedent(inspect.getsource(original))
    for expression in ("str(feat.shape[-1])", "str(h.shape[-1])"):
        if expression not in source:
            raise ValueError("Upstream fusion-key source changed")
        dimension = expression[len("str("):-1]
        source = source.replace(expression, f"str(int({dimension}))")
    namespace = dict(original.__globals__)
    exec(compile(source, "<pgtformer-fixed-shape-fusion-keys>", "exec"), namespace)
    model.forward = types.MethodType(namespace["forward"], model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--input-frames", type=Path, nargs=3,
                        help="Offline face reference: three ordered RGB images resized bilinearly to 512x512")
    parser.add_argument("--resume-trace", action="store_true", help="Reuse this tool's saved trace and numerical fixture")
    args = parser.parse_args()
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(77)
    if args.resume_trace and args.input_frames:
        raise ValueError("--resume-trace reuses its recorded input; generate a separate face reference first")
    if args.resume_trace:
        manifest = json.loads((args.output / "manifest.json").read_text())
        if manifest["weightsSHA256"] != WEIGHTS_SHA256 or file_sha256(args.weights) != WEIGHTS_SHA256 or manifest["upstreamRevision"] != UPSTREAM:
            raise ValueError("Saved trace provenance mismatch")
        inputs = torch.from_numpy(np.load(args.output / "reference-input.npy"))
        reference = torch.from_numpy(np.load(args.output / "reference-output.npy"))
        traced = torch.jit.load(str(args.output / "reference-trace.pt"))
        convert(traced, inputs, reference, args.output, manifest)
        return
    model = load_reference(args.upstream, args.config, args.weights)
    class CenterFrame(torch.nn.Module):
        def __init__(self, network):
            super().__init__()
            self.network = network
        def forward(self, frames):
            return self.network(frames, w=1.0)[0][1:2].clamp(0, 1)
    wrapper = CenterFrame(model).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    # Distinct temporal inputs; RGB [0,1], not CodeFormer's [-1,1].
    inputs = torch.rand(3, 3, 512, 512)
    if args.input_frames:
        from PIL import Image
        inputs = torch.stack([torch.from_numpy(np.array(Image.open(path).convert("RGB")
            .resize((512, 512), Image.Resampling.BILINEAR))).permute(2, 0, 1).float() / 255
            for path in args.input_frames])
    np.save(args.output / "reference-input.npy", inputs.numpy())
    print("Running official pretrained reference", flush=True)
    with torch.no_grad():
        reference = wrapper(inputs)
    if not torch.isfinite(reference).all():
        raise ValueError("Reference produced nonfinite output")
    np.save(args.output / "reference-output.npy", reference.numpy())
    from safetensors.torch import save_file
    save_file({"frames": inputs.contiguous(), "restored": reference.contiguous()},
              str(args.output / "reference.safetensors"))
    manifest = {"format": 1, "algorithm": "pgtformer", "upstreamRevision": UPSTREAM,
                "weightsRevision": WEIGHTS_REVISION, "weightsSHA256": WEIGHTS_SHA256,
                "inputShape": [3, 3, 512, 512], "outputShape": [1, 3, 512, 512],
                "inputRange": [0, 1], "centerFrame": 1, "w": 1.0,
                "validated": False}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.reference_only:
        return
    print("Tracing official graph", flush=True)
    fixed_shape_trace_adapter(model)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, inputs, check_trace=False)
    traced.save(str(args.output / "reference-trace.pt"))
    convert(traced, inputs, reference, args.output, manifest)


def convert(traced, inputs, reference, output, manifest):
    manifest["validated"] = False
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    import numpy as np
    import torch
    with torch.no_grad():
        trace_output = traced(inputs)
    difference = (trace_output - reference).abs()
    trace_metrics = {"maximumAbsoluteError": float(difference.max()),
                     "meanAbsoluteError": float(difference.mean()),
                     "finite": bool(torch.isfinite(trace_output).all())}
    (output / "trace-validation.json").write_text(json.dumps(trace_metrics, indent=2) + "\n")
    if not trace_metrics["finite"] or trace_metrics["maximumAbsoluteError"] > 1e-5:
        raise ValueError(f"Traced graph differs from official eager reference: {trace_metrics}")
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op
    from coremltools.converters.mil.frontend.torch.ops import _get_inputs
    # Core ML Tools 9 lacks aten::chunk. Fixed-size checkpoint projection chunks
    # lower exactly to MIL split; no weight or network approximation is involved.
    @register_torch_op
    def chunk(context, node):
        x, count, axis = _get_inputs(context, node, expected=3)
        count, axis = int(count.val), int(axis.val)
        length = int(x.shape[axis])
        size = (length + count - 1) // count
        sizes = [min(size, length - start) for start in range(0, length, size)]
        result = mb.split(x=x, split_sizes=sizes, axis=axis, name=node.name)
        context.add(result, torch_name=node.name)
    print("Converting fixed-shape FP32 graph", flush=True)
    converted = ct.convert(traced, inputs=[ct.TensorType(name="frames", shape=inputs.shape)],
        outputs=[ct.TensorType(name="restored")], convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15, compute_precision=ct.precision.FLOAT32)
    converted.save(str(output / "PGTFormer.mlpackage"))
    print("Comparing converted output to official reference", flush=True)
    # Own the NumPy buffer: Core ML may release its input on a dispatch thread.
    # A Torch-owned NumPy view can crash when that thread destroys the tensor.
    prediction_inputs = {"frames": inputs.numpy().copy()}
    prediction = converted.predict(prediction_inputs)["restored"]
    np.save(output / "native-output.npy", prediction)
    difference = np.abs(prediction - reference.numpy())
    metrics = {"maximumAbsoluteError": float(difference.max()), "meanAbsoluteError": float(difference.mean()),
               "finite": bool(np.isfinite(prediction).all()), "shape": list(prediction.shape)}
    (output / "validation.json").write_text(json.dumps(metrics, indent=2) + "\n")
    # A conversion is not an installed/validated model until the real weights
    # execute and this numerical gate succeeds. Additional video QA is required.
    if not metrics["finite"] or metrics["shape"] != [1, 3, 512, 512] or metrics["maximumAbsoluteError"] > 0.02 or metrics["meanAbsoluteError"] > 0.002:
        raise ValueError(f"Core ML parity failed: {metrics}")
    manifest["validated"] = True
    manifest["packageSHA256"] = {str(path.relative_to(output / "PGTFormer.mlpackage")): file_sha256(path)
        for path in sorted((output / "PGTFormer.mlpackage").rglob("*")) if path.is_file()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
