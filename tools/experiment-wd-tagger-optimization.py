#!/usr/bin/env python3
"""Build isolated WDTagger optimization candidates; never installs a model.

Use the native shape benchmark and an independent reference before selecting any
candidate. Weight compression changes predictions and needs corpus validation.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import types

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("wd_converter", ROOT / "tools/convert-wd-tagger.py")
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


def rsqrt_attention(self, x, mask=None):
    """Same cosine attention, expressing normalization as reciprocal * input."""
    windows, tokens, channels = x.shape
    qkv = F.linear(x, self.qkv.weight, self._sdpa_qkv_bias)
    qkv = qkv.reshape(windows, tokens, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q = q * torch.rsqrt((q * q).sum(dim=-1, keepdim=True).clamp_min(1e-24)) * self._sdpa_query_scale
    k = k * torch.rsqrt((k * k).sum(dim=-1, keepdim=True).clamp_min(1e-24))
    bias = self._sdpa_bias
    if mask is not None:
        count = mask.shape[0]
        q = q.reshape(-1, count * self.num_heads, tokens, channels // self.num_heads)
        k = k.reshape_as(q)
        v = v.reshape_as(q)
        bias = (bias.unsqueeze(1) + mask.unsqueeze(0).unsqueeze(2)).reshape(1, count * self.num_heads, tokens, tokens)
    result = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.0)
    result = result.reshape(windows, self.num_heads, tokens, -1).transpose(1, 2).reshape(windows, tokens, channels)
    return self.proj(result)


class FoldedInput(torch.nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core

    def forward(self, x):
        return torch.sigmoid(self.core(x))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=["rsqrt", "fold-input", "int8", "int4",
                                                           "gather-layout", "persistent-layout"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(20261015)
    if args.variant in ["int8", "int4"]:
        if args.source is None:
            parser.error("Compression requires --source PACKAGE")
        from coremltools.optimize.coreml import OptimizationConfig, OpLinearQuantizerConfig, linear_quantize_weights
        dtype = "int8" if args.variant == "int8" else "int4"
        options = {"dtype": dtype, "mode": "linear_symmetric", "weight_threshold": 2048,
                   "granularity": "per_channel" if dtype == "int8" else "per_block"}
        if dtype == "int4":
            options["block_size"] = 32
        configuration = OptimizationConfig(op_type_configs={
            "linear": OpLinearQuantizerConfig(**options),
            "conv": OpLinearQuantizerConfig(**options),
        })
        model = ct.models.MLModel(str(args.source), skip_model_load=True)
        result = linear_quantize_weights(model, configuration)
        result.user_defined_metadata["waifu.validated"] = "false"
        result.user_defined_metadata["waifu.experiment"] = args.variant
        result.save(str(args.output / "WDTagger.mlpackage"))
        return
    if args.weights is None:
        parser.error("Graph variants require --weights FILE")
    converter.BATCH = args.batch_size
    core = converter.build_model(args.weights)
    wrapper = converter.Tagger(core).eval()
    inputs = torch.rand(2, 3, 448, 448) * 255
    with torch.inference_mode():
        baseline = wrapper(inputs).numpy()
        if args.variant == "rsqrt":
            for module in core.modules():
                if hasattr(module, "_sdpa_bias"):
                    module.forward = types.MethodType(rsqrt_attention, module)
        elif args.variant == "fold-input":
            projection = core.patch_embed.proj
            # No padding: every pixel in the 4x4 patch is present, so the
            # constant offset folds exactly in real arithmetic.
            assert projection.padding == (0, 0)
            original = projection.weight.detach().clone()
            projection.weight.mul_(2 / 255)
            projection.bias.sub_(original.sum(dim=(1, 2, 3)))
            wrapper = FoldedInput(core).eval()
        else:
            from wd_tagger_layout import install_layout
            layout = install_layout(core, persistent=args.variant == "persistent-layout")
            (args.output / "layout.json").write_text(json.dumps(layout, indent=2) + "\n")
        candidate = wrapper(inputs).numpy()
    delta = np.abs(candidate - baseline)
    parity = {"maximumAbsoluteError": float(delta.max()), "meanAbsoluteError": float(delta.mean())}
    (args.output / "torch-parity.json").write_text(json.dumps(parity, indent=2) + "\n")
    if delta.max() > 1e-4:
        raise RuntimeError(f"Torch rewrite error: {parity}")
    bundle = converter.convert(converter.export(wrapper), args.output, converter.sha256(args.weights))
    model = ct.models.MLModel(str(bundle), skip_model_load=True)
    model.user_defined_metadata["waifu.experiment"] = args.variant
    model.save(str(args.output / "marked.mlpackage"))
    # Conversion already marks the candidate unvalidated. Keep its exported
    # package; the extra copy is unnecessary after updating the metadata file.
    import shutil
    shutil.rmtree(bundle)
    (args.output / "marked.mlpackage").rename(bundle)


if __name__ == "__main__":
    main()
