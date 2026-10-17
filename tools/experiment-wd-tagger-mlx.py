#!/usr/bin/env python3
"""Isolated SwinV2 MLX probe. Never installs or replaces the app's model.

Export with the converter's Torch environment, then benchmark with an isolated
MLX environment. Inputs are the native benchmark's retained FP32 RGB tensors.
Timings exclude file IO, preprocessing, model loading and compilation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


class PhysicalFootprint:
    """Darwin task_vm_info, sampled like the native benchmark (GPU included)."""

    def __init__(self):
        import ctypes
        import threading
        self.ctypes = ctypes
        self.library = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        self.port = ctypes.c_uint32.in_dll(self.library, "mach_task_self_").value
        self.peak = self.sample()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.watch, daemon=True)
        self.thread.start()

    def sample(self):
        c = self.ctypes
        buffer = (c.c_uint64 * 128)()
        count = c.c_uint32(c.sizeof(buffer) // 4)
        status = self.library.task_info(self.port, 22, c.byref(buffer), c.byref(count))
        if status != 0 or count.value < 38:
            raise RuntimeError("Darwin task_vm_info footprint query failed")
        # macOS SDK task_info.h: phys_footprint follows the 144-byte rev0 prefix.
        return buffer[18]

    def watch(self):
        while not self.stop.wait(0.01):
            self.peak = max(self.peak, self.sample())

    def finish(self):
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, self.sample())
        return self.peak


def export_weights(source: Path, destination: Path):
    import importlib.util
    import sys
    import torch._dynamo  # Initialize before timm's compiler decorators.
    from safetensors.torch import save_file

    directory = Path(__file__).resolve().parent
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location("converter", directory / "convert-wd-tagger.py")
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    core = converter.build_model(source)
    arrays = {key: value.contiguous() for key, value in core.state_dict().items()}
    for i, stage in enumerate(core.layers):
        for j, block in enumerate(stage.blocks):
            if block.attn_mask is not None:
                arrays[f"layers.{i}.blocks.{j}.attn_mask"] = block.attn_mask.contiguous()
    save_file(arrays, str(destination))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--export-from", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--probabilities", type=Path)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--precision", choices=["float32", "float16"], default="float16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--force-fused", action="store_true")
    parser.add_argument("--pad-head-dim", type=int, choices=[32, 64], default=32)
    args = parser.parse_args()
    if args.export_from:
        export_weights(args.export_from, args.weights)
        return
    if not args.inputs or not args.report:
        parser.error("Inference requires --inputs and --report")
    if min(args.batch, args.frames, args.repeats) < 1:
        parser.error("Batch, frames and repeats must be positive")

    import mlx.core as mx
    import numpy as np
    import platform
    import resource

    footprint = PhysicalFootprint()
    dtype = getattr(mx, args.precision)
    loaded = mx.load(args.weights)
    p = {k: v.astype(dtype) for k, v in loaded.items()
         if not any(s in k for s in ["cpb_mlp", "relative_"])
         and not k.endswith((".logit_scale", ".q_bias", ".v_bias"))}
    del loaded
    p["patch_embed.proj.weight"] = p["patch_embed.proj.weight"].transpose(0, 2, 3, 1)
    mx.eval(p)

    def norm(x, prefix):
        return mx.fast.layer_norm(x, p[prefix + ".weight"], p[prefix + ".bias"], 1e-5)

    def linear(x, prefix):
        value = x @ p[prefix + ".weight"].T
        return value + p[prefix + ".bias"] if prefix + ".bias" in p else value

    blocks = []
    for stage, (side, depth, heads) in enumerate(zip([112, 56, 28, 14], [2, 2, 18, 2], [4, 8, 16, 32])):
        stage_blocks = []
        for block in range(depth):
            prefix = f"layers.{stage}.blocks.{block}"
            attn = prefix + ".attn"
            shift = 7 if block % 2 and side > 14 else 0
            indices = np.arange(side * side).reshape(side, side)
            indices = np.roll(indices, (-shift, -shift), axis=(0, 1))
            indices = indices.reshape(side // 14, 14, side // 14, 14).transpose(0, 2, 1, 3).reshape(-1)
            order = mx.array(indices.astype(np.int32))
            inverse = mx.array(np.argsort(indices).astype(np.int32))
            windows = (side // 14) ** 2
            bias = p[attn + "._sdpa_bias"]
            if shift:
                bias = (bias[:, None] + p[prefix + ".attn_mask"][None, :, None]).reshape(1, windows * heads, 196, 196)

            def make_block(prefix, attn, order, inverse, windows, heads, bias, side):
                def run(x):
                    batch, _, _, channels = x.shape
                    y = mx.take(x.reshape(batch, -1, channels), order, axis=1).reshape(batch * windows, 196, channels)
                    qkv = y @ p[attn + ".qkv.weight"].T + p[attn + "._sdpa_qkv_bias"]
                    qkv = qkv.reshape(batch * windows, 196, 3, heads, channels // heads).transpose(2, 0, 3, 1, 4)
                    q, k, v = qkv[0], qkv[1], qkv[2]
                    # FP32 reductions avoid overflow and preserve the reference's
                    # max(norm, epsilon) cosine denominator, including zero input.
                    def normalize(a):
                        full = a.astype(mx.float32)
                        denominator = mx.maximum(mx.sqrt(mx.sum(full * full, axis=-1, keepdims=True)), 1e-12)
                        return (full / denominator).astype(dtype)
                    q = normalize(q) * p[attn + "._sdpa_query_scale"]
                    k = normalize(k)
                    if bias.shape[1] != heads:
                        q = q.reshape(batch, windows * heads, 196, channels // heads)
                        k = k.reshape(q.shape)
                        v = v.reshape(q.shape)
                    # MLX's full fused attention kernel requires at least 64
                    # lanes; zero padding preserves all 32-lane dot products.
                    if args.pad_head_dim == 64:
                        q = mx.pad(q, [(0, 0), (0, 0), (0, 0), (0, 32)])
                        k = mx.pad(k, [(0, 0), (0, 0), (0, 0), (0, 32)])
                        v = mx.pad(v, [(0, 0), (0, 0), (0, 0), (0, 32)])
                    y = mx.fast.scaled_dot_product_attention(q, k, v, scale=(channels // heads) ** -0.5,
                                                           mask=bias, force_fused=args.force_fused)
                    if args.pad_head_dim == 64:
                        y = y[..., :32]
                    y = y.reshape(batch * windows, heads, 196, channels // heads).transpose(0, 2, 1, 3)
                    y = linear(y.reshape(batch * windows, 196, channels), attn + ".proj")
                    y = mx.take(y.reshape(batch, -1, channels), inverse, axis=1).reshape(batch, side, side, channels)
                    x = x + norm(y, prefix + ".norm1")
                    y = linear(x, prefix + ".mlp.fc1")
                    y = 0.5 * y * (1 + mx.tanh(0.7978845608028654 * (y + 0.044715 * y * y * y)))
                    return x + norm(linear(y, prefix + ".mlp.fc2"), prefix + ".norm2")
                return mx.compile(run) if args.compile else run
            stage_blocks.append(make_block(prefix, attn, order, inverse, windows, heads, bias, side))
        blocks.append(stage_blocks)

    def predict(x):
        x = x.astype(dtype).transpose(0, 2, 3, 1) * (2 / 255) - 1
        x = mx.conv2d(x, p["patch_embed.proj.weight"], stride=4) + p["patch_embed.proj.bias"]
        x = norm(x, "patch_embed.norm")
        for stage, stage_blocks in enumerate(blocks):
            if stage:
                batch, h, w, channels = x.shape
                x = x.reshape(batch, h // 2, 2, w // 2, 2, channels).transpose(0, 1, 3, 4, 2, 5)
                x = x.reshape(batch, h // 2, w // 2, 4 * channels)
                x = norm(linear(x, f"layers.{stage}.downsample.reduction"), f"layers.{stage}.downsample.norm")
            for block in stage_blocks:
                x = block(x)
        x = mx.mean(norm(x, "norm"), axis=(1, 2))
        return mx.sigmoid(linear(x, "head.fc").astype(mx.float32))

    all_inputs = np.memmap(args.inputs, dtype=np.float32, mode="r").reshape(-1, 3, 448, 448)
    if args.frames > len(all_inputs):
        parser.error("Not enough retained input frames")
    # Only one batch is resident, as in the native Core ML benchmark. Transfers
    # are outside prediction timing, so compare prediction stage timings only.
    times, probabilities = [], []
    mx.reset_peak_memory()
    warmup_seconds = 0
    for repeat in range(-1, args.repeats):
        elapsed = 0
        for start in range(0, args.frames, args.batch):
            if repeat == -1 and start:
                break
            count = min(args.batch, args.frames - start)
            batch = np.array(all_inputs[start:start + count], copy=True)
            if count < args.batch:
                batch = np.concatenate([batch, np.repeat(batch[-1:], args.batch - count, axis=0)])
            x = mx.array(batch)
            mx.eval(x)
            tick = time.perf_counter()
            y = predict(x)
            mx.eval(y)
            elapsed += time.perf_counter() - tick
            if repeat == 0:
                probabilities.append(np.array(y)[:count])
            del x, y
        if repeat == -1:
            warmup_seconds = elapsed
        else:
            times.append(elapsed)
            print(f"sweep {repeat}: {args.frames / elapsed:.2f} prediction frames/s", flush=True)
    if args.probabilities:
        np.concatenate(probabilities).astype(np.float32).tofile(args.probabilities)
    report = {"mlxVersion": mx.__version__, "machine": platform.machine(),
              "precision": args.precision, "compiledBlocks": args.compile, "forceFusedSDPA": args.force_fused,
              "paddedHeadDimension": args.pad_head_dim,
              "batch": args.batch, "frames": args.frames, "repeats": args.repeats,
              "warmupSeconds": warmup_seconds, "predictionSweepSeconds": times,
              "predictionFramesPerSecond": args.frames / float(np.median(times)),
              "peakMLXBytes": mx.get_peak_memory(), "activeMLXBytes": mx.get_active_memory(),
              "cacheMLXBytes": mx.get_cache_memory(), "peakProcessRSSBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "peakPhysicalFootprintBytes": footprint.finish(),
              "timingNote": "Prediction only; file IO, input transfers, preprocessing and initial compilation excluded."
              }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
