#!/usr/bin/env python3
"""Offline official FlashVSR 1.1 conversion. Python is never used for app inference."""
import argparse
import hashlib
import json
from pathlib import Path

REVISION = "27561b186ded3402d7c975f4fd722e2885b6135f"
CONVERSION_METADATA = {
    "license": "apache-2.0",
    "source_model": "JunhaoZhuang/FlashVSR-v1.1",
    "source_revision": REVISION,
    "conversion": "Modified by Waifu: FP32 to BF16, convolution layout transposes, per-block shards and Safetensors prompt. No fine-tuning.",
}
SOURCES = {
    "diffusion_pytorch_model_streaming_dmd.safetensors": "bd28180edcf3446c028e32fc6b731a80bf7e4da2ab4caac3186b9499964d37be",
    "LQ_proj_in.ckpt": "d6d011cdaaba6a52645086caa08fa04124e746f6ca568140a24007591142bfd2",
    "TCDecoder.ckpt": "e224bdcf2f52745cbf4d393ff5374c2ba09e90285d5d19062d2bf63b915b6161",
}

def expected_shapes():
    dit, projection, decoder = {}, {}, {}
    def linear(dst, name, inputs, outputs, bias=True):
        dst[name + ".weight"] = [outputs, inputs]
        if bias: dst[name + ".bias"] = [outputs]
    for i in range(30):
        p = f"blocks.{i}."
        for attention in ("self_attn", "cross_attn"):
            for op in ("q", "k", "v", "o"): linear(dit, p + attention + "." + op, 1536, 1536)
            for norm in ("norm_q", "norm_k"): dit[p + attention + "." + norm + ".weight"] = [1536]
        linear(dit, p + "ffn.0", 1536, 8960)
        linear(dit, p + "ffn.2", 8960, 1536)
        dit[p + "modulation"] = [1, 6, 1536]
        dit[p + "norm3.weight"] = dit[p + "norm3.bias"] = [1536]
    dit["patch_embedding.weight"] = [1536, 16, 1, 2, 2]
    dit["patch_embedding.bias"] = [1536]
    for name, a, b in [("text_embedding.0",4096,1536), ("text_embedding.2",1536,1536),
                       ("time_embedding.0",256,1536), ("time_embedding.2",1536,1536),
                       ("time_projection.1",1536,9216), ("head.head",1536,64)]: linear(dit,name,a,b)
    dit["head.modulation"] = [1,2,1536]
    for name, a, b in [("conv1",768,2048), ("conv2",2048,3072)]:
        projection[name + ".weight"] = [b,a,4,3,3]
        projection[name + ".bias"] = [b]
    projection["norm1.gamma"] = [2048,1,1,1]
    projection["norm2.gamma"] = [3072,1,1,1]
    linear(projection, "linear_layers.0", 3072, 1536)
    def conv(name, a, b, kernel=3, bias=True):
        decoder[name + ".weight"] = [b,a,kernel,kernel]
        if bias: decoder[name + ".bias"] = [b]
    conv("decoder.1",784,512)
    conv("decoder.3",512,512,bias=False)
    for i, channels in [(5,512),(6,512),(7,512),(11,256),(12,256),(13,256),(17,128),(18,128),(19,128)]:
        for n, a in [(0,channels*2),(2,channels),(4,channels)]: conv(f"decoder.{i}.conv.{n}",a,channels)
    for i, a, b in [(9,512,512),(15,256,512),(21,128,256)]: conv(f"decoder.{i}.conv",a,b,kernel=1,bias=False)
    for i, a, b in [(10,512,256),(16,256,128),(22,128,128),(24,128,128)]: conv(f"decoder.{i}",a,b,bias=False)
    conv("decoder.26",128,3)
    return {"dit":dit, "projection":projection, "decoder":decoder}

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""): h.update(chunk)
    return h.hexdigest()

def main():
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--prompt", type=Path, required=True, help="Official repository's posi_prompt.pth")
    args = p.parse_args()
    for name, digest in SOURCES.items():
        if sha256(args.source / name) != digest: raise ValueError(f"Official checkpoint checksum mismatch: {name}")
    if sha256(args.prompt) != "4601107a11e4e11a936a6b79df579e54dbc99872132bf542151f0ffd65b4b1ef": raise ValueError("Official prompt checksum mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    index, schema = {}, expected_shapes()
    def emit(component, tensors):
        shapes = {k:list(v.shape) for k,v in tensors.items()}
        if shapes != schema[component]:
            missing, extra = schema[component].keys()-shapes.keys(), shapes.keys()-schema[component].keys()
            mismatched = {k:(schema[component][k],shapes[k]) for k in shapes.keys() & schema[component].keys() if shapes[k]!=schema[component][k]}
            raise ValueError(f"{component}: missing={missing}, unexpected={extra}, shapes={mismatched}")
        groups = {}
        for key in tensors:
            group = '.'.join(key.split('.')[:2]) if component == "dit" and key.startswith("blocks.") else "shared"
            groups.setdefault(group,[]).append(key)
        for group, keys in groups.items():
            arrays = {}
            for key in keys:
                value = tensors[key]
                if value.ndim == 5: value = value.permute(0,2,3,4,1)
                elif value.ndim == 4 and key.endswith("weight"): value = value.permute(0,2,3,1)
                value = value.to(torch.bfloat16).contiguous()
                if not torch.isfinite(value).all(): raise ValueError(f"Non-finite converted parameter: {key}")
                arrays[key] = value
            filename = f"{component}-{group}.safetensors"
            save_file(arrays, str(args.output / filename), metadata=CONVERSION_METADATA)
            for key,value in arrays.items(): index[f"{component}:{key}"] = {"file":filename,"shape":list(value.shape),"dtype":"bfloat16"}
            del arrays
    with safe_open(str(args.source / "diffusion_pytorch_model_streaming_dmd.safetensors"),framework="pt",device="cpu") as f:
        # mmap source tensors; convert one transformer block at a time.
        emit("dit", {k:f.get_tensor(k) for k in f.keys()})
    for component, name in [("projection","LQ_proj_in.ckpt"),("decoder","TCDecoder.ckpt")]:
        state = torch.load(args.source / name, map_location="cpu", weights_only=True, mmap=True)
        emit(component,state)
        del state
    prompt = torch.load(args.prompt,map_location="cpu",weights_only=True)
    if list(prompt.shape) != [1,512,4096] or not torch.isfinite(prompt).all(): raise ValueError("Invalid official prompt tensor")
    save_file({"context":prompt.to(torch.bfloat16).contiguous()}, str(args.output / "prompt.safetensors"), metadata=CONVERSION_METADATA)
    manifest = {"format":1,"algorithm":"FlashVSR 1.1 Tiny Streaming","revision":REVISION,
                "precision":"bfloat16","sourceSHA256":SOURCES,"promptSHA256":sha256(args.prompt),
                "weightMap":index,"fileSHA256":{f.name:sha256(f) for f in args.output.glob('*.safetensors')}}
    (args.output / "manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print("Official FlashVSR Tiny weights converted; complete parameter coverage verified.")

if __name__ == "__main__": main()
