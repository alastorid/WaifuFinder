#!/usr/bin/env python3
"""Offline conversion of the complete TruFor graph to a Core ML FP16 ML Program.

Python/PyTorch/Core ML Tools are conversion-only. The app and inference worker
load the resulting model with Apple's Swift Core ML API; no Python inference is
used. The official upstream source is supplied separately so its nonprofit
license and notices remain intact.
"""
import argparse
import gc
import hashlib
import json
import mmap
import os
import shutil
import struct
import subprocess
import sys
import types
from pathlib import Path

SOURCE_SHA = "ac1d90e329a72e0d66e8665e123a19e94bfae3209c3ef8a4f9ca3b91578c7844"
UPSTREAM_REVISION = "ae54475df6f41a491d7615100feb19263dec13f7"
SUPPORTED_TORCH = "2.7.0"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def safetensors(path, np):
    stream = path.open("rb")
    length = struct.unpack("<Q", stream.read(8))[0]
    if length <= 0 or length > 100_000_000:
        stream.close()
        raise ValueError("Invalid TruFor Safetensors header")
    header = json.loads(stream.read(length))
    mapped = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
    data_start = 8 + length
    result = {}
    for name, item in header.items():
        if name == "__metadata__":
            continue
        if item.get("dtype") != "F32":
            raise ValueError("Unexpected checkpoint dtype for " + name)
        lo, hi = item["data_offsets"]
        array = np.frombuffer(mapped, dtype="<f4", count=(hi - lo) // 4,
                              offset=data_start + lo).reshape(item["shape"]).copy()
        result[name] = array
    mapped.close()
    stream.close()
    return result


def install_timm_compat(torch):
    """Supply only the three eval-time helpers used by the licensed source."""
    layers = types.ModuleType("timm.models.layers")
    layers.DropPath = torch.nn.Identity
    layers.to_2tuple = lambda value: value if isinstance(value, tuple) else (value, value)
    layers.trunc_normal_ = torch.nn.init.trunc_normal_
    models = types.ModuleType("timm.models")
    models.layers = layers
    timm = types.ModuleType("timm")
    timm.models = models
    sys.modules.update({"timm": timm, "timm.models": models, "timm.models.layers": layers})


def coreml_sdpa_attention(torch, attention_type):
    """Replace equivalent QK-softmax-V math with Core ML's fused SDPA op."""
    functional = torch.nn.functional

    def forward(self, x, height, width):
        head_dim = self.dim // self.num_heads
        q = self.q(x).reshape(1, -1, self.num_heads, head_dim).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            source = x.permute(0, 2, 1).reshape(1, self.dim, height, width)
            source = self.sr(source).reshape(1, self.dim, -1).permute(0, 2, 1)
            source = self.norm(source)
        else:
            source = x
        kv = self.kv(source).reshape(1, -1, 2, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        attended = functional.scaled_dot_product_attention(q, kv[0], kv[1],
                                                            dropout_p=0.0, is_causal=False,
                                                            scale=self.scale)
        attended = attended.transpose(1, 2).reshape(1, -1, self.dim)
        return self.proj_drop(self.proj(attended))

    attention_type.forward = forward


def coreml_cross_attention(torch, cross_type):
    """Keep CMX's channel-space cross-attention equations, with static dims."""
    def forward(self, x1, x2):
        head_dim = self.dim // self.num_heads
        q1 = x1.reshape(1, -1, self.num_heads, head_dim).permute(0, 2, 1, 3)
        q2 = x2.reshape(1, -1, self.num_heads, head_dim).permute(0, 2, 1, 3)
        kv1 = self.kv1(x1).reshape(1, -1, 2, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        kv2 = self.kv2(x2).reshape(1, -1, 2, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        context1 = torch.softmax(torch.matmul(kv1[0].transpose(-2, -1), kv1[1]) * self.scale, dim=-2)
        context2 = torch.softmax(torch.matmul(kv2[0].transpose(-2, -1), kv2[1]) * self.scale, dim=-2)
        out1 = torch.matmul(q1, context2).transpose(1, 2).reshape(1, -1, self.dim)
        out2 = torch.matmul(q2, context1).transpose(1, 2).reshape(1, -1, self.dim)
        return out1, out2

    cross_type.forward = forward


def coreml_weighted_statistics(torch, layer_utils):
    """Rewrite log-sigmoid as the equivalent Core ML-supported softplus form."""
    functional = torch.nn.functional
    functional.logsigmoid = lambda value: -functional.softplus(-value)

    def weighted_statistics_pooling(values, log_weights=None):
        values = values.reshape(1, values.shape[1], -1)
        if log_weights is None:
            log_weights = torch.zeros_like(values[:, :1, :])
        else:
            log_weights = log_weights.reshape(1, 1, -1)
        log_weights = functional.log_softmax(log_weights, dim=-1)
        minimum = -torch.logsumexp(log_weights - values, dim=-1)
        maximum = torch.logsumexp(log_weights + values, dim=-1)
        weights = torch.exp(log_weights)
        average = torch.sum(weights * values, dim=-1)
        mean_square = torch.sum(weights * values * values, dim=-1)
        return torch.cat((minimum, maximum, average, mean_square), dim=1)

    layer_utils.weighted_statistics_pooling = weighted_statistics_pooling


def coreml_dynamic_bilinear_resize(ct, np):
    """Lower dynamic PyTorch bilinear target sizes to Core ML's dynamic resize op."""
    import importlib
    module = importlib.import_module(
        "coremltools.converters.mil.frontend.torch.ssa_passes.torch_upsample_to_core_upsample")
    from coremltools.converters.mil.mil import Builder as mb

    @module.block_context_manager
    def replace_in_block(block):
        for operation in list(block.operations):
            for nested in operation.blocks:
                replace_in_block(nested)
            if operation.op_type != "torch_upsample_bilinear":
                if operation.op_type in module.target_ops:
                    module._try_replace_with_core_upsample(operation)
                continue
            sampling = "ALIGN_CORNERS" if operation.align_corners.val else "UNALIGN_CORNERS"
            target_size = mb.stack(values=[operation.output_height, operation.output_width],
                                   axis=0,
                                   name=operation.name + "_target_size", before_op=operation)
            resized = mb.resize(x=operation.x, shape=target_size, resized_dims=np.uint32(2),
                                interpolation_mode="LINEAR", sampling_mode=sampling,
                                name=operation.name, before_op=operation)
            block.replace_uses_of_var_after_op(anchor_op=operation,
                                               old_var=operation.outputs[0], new_var=resized)
            block.remove_ops([operation])

    module._torch_upsample_to_core_upsample_block = replace_in_block


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Validated TruFor model directory")
    parser.add_argument("--output", type=Path, required=True, help="New, empty output directory")
    parser.add_argument("--upstream-dir", type=Path,
                        default=Path(".build/trufor-coreml-upstream-review/TruFor_train_test"),
                        help="Official TruFor_train_test checkout at the validated revision")
    parser.add_argument("--trace-side", type=int, default=512,
                        help="Small trace input; model input remains a bounded flexible shape")
    parser.add_argument("--max-side", type=int, default=4096,
                        help="Maximum supported input side in the Core ML model")
    parser.add_argument("--flexible", action="store_true",
                        help="Convert with independent bounded height and width dimensions")
    parser.add_argument("--fp32-score-island", action="store_true",
                        help="Keep global detection pooling and its small score head in FP32")
    parser.add_argument("--debug-score", action="store_true",
                        help="Expose internal detection statistics for native Core ML diagnostics")
    parser.add_argument("--skip-compile", action="store_true",
                        help="Write the Core ML package without invoking the local Core ML compiler")
    parser.add_argument("--dump-int-nodes", action="store_true",
                        help="Print traced integer-conversion nodes for converter diagnosis")
    parser.add_argument("--runtime-dir", type=Path, default=Path(__file__).resolve().parents[1] / "Runtime")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new directory")
    if args.trace_side < 128 or args.trace_side > args.max_side or args.max_side > 4096:
        parser.error("unsupported trace or maximum input side")

    import numpy as np
    import torch
    import coremltools as ct
    from coremltools.converters.mil.mil import types
    if torch.__version__.split("+")[0] != SUPPORTED_TORCH:
        raise SystemExit("Use the audited converter version torch==" + SUPPORTED_TORCH)

    architecture_path = args.model / "architecture.json"
    manifest_path = args.model / "manifest.json"
    weights_path = args.model / "model.safetensors"
    architecture = json.loads(architecture_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    expected = manifest.get("fileSHA256", {}).get("model.safetensors")
    if architecture.get("sourceSHA256") != SOURCE_SHA or not expected or digest(weights_path) != expected:
        raise SystemExit("TruFor architecture or checkpoint hash is not the validated source")
    if architecture.get("architecture") != "trufor-cmx-mit-b2":
        raise SystemExit("This converter supports only the validated TruFor CMX-MiT-B2 checkpoint")
    if subprocess.check_output(["git", "-C", str(args.upstream_dir), "rev-parse", "HEAD"],
                               text=True).strip() != UPSTREAM_REVISION:
        raise SystemExit("The official source checkout does not match the validated revision")

    install_timm_compat(torch)
    sys.path.insert(0, str(args.upstream_dir))
    from lib.models.cmx.builder_np_conf import EncoderDecoder
    from lib.models.cmx.encoders.dual_segformer import Attention
    from lib.models.cmx.net_utils import CrossAttention
    from lib.models.cmx import layer_utils
    coreml_sdpa_attention(torch, Attention)
    coreml_cross_attention(torch, CrossAttention)
    coreml_weighted_statistics(torch, layer_utils)
    coreml_dynamic_bilinear_resize(ct, np)

    class Config(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

    extra = Config(
        NP_OUT_CHANNELS=1,
        MODULES=["NP++", "backbone", "loc_head", "conf_head", "det_head"],
        FIX_MODULES=[], BACKBONE="mit_b2", DECODER="MLPDecoder",
        DECODER_EMBED_DIM=512, DETECTION="confpool", BN_EPS=1e-5,
        BN_MOMENTUM=0.1, PREPRC="imagenet", NP_WEIGHTS=None,
    )
    config = Config(MODEL=Config(EXTRA=extra, MODS=["RGB", "NP++"], PRETRAINED=None),
                    DATASET=Config(NUM_CLASSES=2))
    model = EncoderDecoder(config).eval()

    weights = safetensors(weights_path, np)
    expected_names = set(model.state_dict())
    provided_names = set(weights)
    missing = expected_names - provided_names
    extra_names = provided_names - expected_names
    if extra_names or any(not name.endswith("num_batches_tracked") for name in missing):
        raise SystemExit("Checkpoint and official source state dictionaries do not match")
    state = {name: torch.from_numpy(value) for name, value in weights.items()}
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise SystemExit("TruFor checkpoint load mismatch: missing=" + repr(sorted(result.missing_keys)) +
                         "; unexpected=" + repr(sorted(result.unexpected_keys)))
    del state, weights
    gc.collect()

    class CompleteTruFor(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, rgb):
            localization, confidence, detection, noiseprint = self.wrapped(rgb, save_np=True)
            score = torch.sigmoid(detection).reshape(1, 1)
            anomaly = torch.softmax(localization, dim=1)[:, 1:2]
            reliability = torch.sigmoid(confidence)
            noiseprint = noiseprint[:, 0:1]
            if args.debug_score:
                functional = torch.nn.functional
                confidence_stats = layer_utils.weighted_statistics_pooling(confidence)
                localization_stats = layer_utils.weighted_statistics_pooling(
                    localization[:, 1:2, :, :] - localization[:, 0:1, :, :],
                    functional.logsigmoid(confidence))
                return (score, anomaly, reliability, noiseprint, detection.reshape(1, 1),
                        confidence_stats, localization_stats)
            return score, anomaly, reliability, noiseprint

    complete = CompleteTruFor(model).eval()
    torch.set_num_threads(1 if args.trace_side >= 1024 else max(1, min(6, os.cpu_count() or 1)))
    trace_side = args.trace_side
    sample = torch.zeros((1, 3, trace_side, trace_side), dtype=torch.float32)
    with torch.inference_mode():
        traced = torch.jit.trace(complete, sample, strict=False, check_trace=False)
        traced = torch.jit.freeze(traced)
    if args.dump_int_nodes:
        for node in traced.graph.nodes():
            if node.kind() == "aten::Int":
                inputs = [(value.debugName(), str(value.type())) for value in node.inputs()]
                print("INT_NODE", node.scopeName(), inputs, str(node))
        return

    if args.flexible:
        height = ct.RangeDim(lower_bound=64, upper_bound=args.max_side, default=trace_side)
        width = ct.RangeDim(lower_bound=64, upper_bound=args.max_side, default=trace_side)
        input_shape = (1, 3, height, width)
    else:
        input_shape = (1, 3, trace_side, trace_side)
    inputs = [ct.TensorType(name="rgb", shape=input_shape, dtype=np.float32)]
    outputs = [ct.TensorType(name="score", dtype=np.float32),
               ct.TensorType(name="anomaly", dtype=np.float32),
               ct.TensorType(name="reliability", dtype=np.float32),
               ct.TensorType(name="noiseprint", dtype=np.float32)]
    if args.debug_score:
        outputs.extend([ct.TensorType(name="score_logit", dtype=np.float32),
                        ct.TensorType(name="confidence_stats", dtype=np.float32),
                        ct.TensorType(name="localization_stats", dtype=np.float32)])
    if args.fp32_score_island:
        debug_precision = os.environ.get("TRUFOR_COREML_DEBUG_PRECISION") == "1"

        def precision_decision(operation, preserve_fp16):
            if debug_precision:
                output = operation.outputs[0] if operation.outputs else None
                if output is not None and (len(output.shape) >= 3 or any(
                        token in operation.name for token in
                        ("log_weights", "weights", "minimum", "maximum", "average", "mean_square", "op_27"))):
                    print("PRECISION", operation.name, operation.op_type,
                          tuple(str(dimension) for dimension in output.shape),
                          "FP16" if preserve_fp16 else "FP32")
            return preserve_fp16

        def preserve_score_precision(operation):
            if not operation.outputs:
                return precision_decision(operation, True)
            output = operation.outputs[0]
            if output.dtype not in (types.fp16, types.fp32):
                return precision_decision(operation, True)
            # TruFor's confidence-weighted statistics reduce over every pixel.
            # Preserve the complete flattened per-pixel pooling branches in
            # FP32. Half precision can overflow in x*x and can flush normalized
            # weights to zero before a later FP32 reduction. Keep the compact
            # rank-two score and channel-gating layers in FP32 too.
            if len(output.shape) == 3:
                try:
                    dimensions = [int(dimension) for dimension in output.shape]
                    if dimensions[0] == 1 and dimensions[1] <= 2 and dimensions[2] >= 4096:
                        return precision_decision(operation, False)
                except (TypeError, ValueError):
                    pass
            if operation.op_type in {
                    "softmax", "log_softmax", "log", "exp", "reduce_log_sum_exp", "reduce_sum"}:
                value = operation.inputs.get("x")
                if value is not None and len(value.shape) == 3:
                    return precision_decision(operation, False)
            if len(output.shape) == 2 and operation.op_type in {
                    "linear", "matmul", "relu", "sigmoid", "add", "sub", "mul",
                    "maximum", "minimum", "concat"}:
                try:
                    if all(not getattr(dimension, "is_symbolic", False) for dimension in output.shape) \
                            and int(np.prod([int(dimension) for dimension in output.shape])) <= 512:
                        return precision_decision(operation, False)
                except (TypeError, ValueError):
                    pass
            return precision_decision(operation, True)
        compute_precision = ct.transform.FP16ComputePrecision(op_selector=preserve_score_precision)
        precision_note = "FP16 graph with FP32 flattened detection pooling and compact score/channel heads"
    else:
        compute_precision = ct.precision.FLOAT16
        precision_note = "FP16 for the complete neural graph; FP32 model inputs and outputs"
    converted = ct.convert(
        traced, convert_to="mlprogram", inputs=inputs, outputs=outputs,
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=compute_precision,
        # Loading a converted ML Program can trigger expensive local ANE AOT
        # compilation. The native worker performs runtime compilation and
        # validation on the target device instead.
        skip_model_load=True,
    )
    args.output.mkdir(parents=True)
    package = args.output / "TruForCoreMLFP16.mlpackage"
    converted.save(str(package))
    if not args.skip_compile:
        subprocess.run(["xcrun", "coremlcompiler", "compile", str(package), str(args.output),
                        "--platform", "macOS", "--deployment-target", "15.0"], check=True)
    for source, destination in (("TRUFOR-LICENSE.txt", "LICENSE.txt"),
                                ("TRUFOR-CMX-LICENSE.txt", "LICENSE_CMX.txt")):
        shutil.copy2(args.runtime_dir / source, args.output / destination)
    model_description = {
        "format": "trufor-full-coreml-fp16-v1",
        "sourceSHA256": SOURCE_SHA,
        "upstreamRevision": UPSTREAM_REVISION,
        "weightsSHA256": expected,
        "torchVersion": torch.__version__,
        "coremltoolsVersion": ct.__version__,
        "precision": precision_note,
        "traceSide": trace_side,
        "flexibleInputSides": [64, args.max_side] if args.flexible else None,
        "input": "RGB float32 NCHW in [0,1), matching native uint8 RGB divided by 256",
        "outputs": ["score", "anomaly", "reliability", "noiseprint"],
        "runtime": "Swift Core ML only; Python and PyTorch are offline conversion tools",
        "computeUnits": "Core ML selects CPU, GPU, and ANE placement at runtime",
    }
    (args.output / "conversion.json").write_text(json.dumps(model_description, indent=2) + "\n")
    print(json.dumps(model_description, indent=2))


if __name__ == "__main__":
    main()
