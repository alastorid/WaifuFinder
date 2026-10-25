---
license: apache-2.0
base_model: JunhaoZhuang/FlashVSR-v1.1
library_name: mlx
pipeline_tag: video-to-video
tags:
  - video-super-resolution
  - mlx-swift
  - apple-silicon
  - bfloat16
  - flashvsr
---

# FlashVSR 1.1 Tiny — Swift/MLX BF16 conversion

A community conversion of the official **FlashVSR 1.1 Tiny Streaming** checkpoints for
[WaifuFinder's native Swift/MLX Metal worker](https://github.com/alastorid/WaifuFinder).
The original model was developed by Junhao Zhuang, Shi Guo, Xin Cai, Xiaohui Li,
Yihao Liu, Chun Yuan and Tianfan Xue. This conversion is not an official author release.

## What changed

All included `.safetensors` files are modified derivatives of the official checkpoints:
FP32 parameters were converted to BF16; convolution layouts were transposed for MLX;
the transformer was split into per-block shards; the fixed prompt was converted into
Safetensors. No fine-tuning or low-bit quantization was performed. `manifest.json`
records source revisions, source checksums, every parameter mapping and converted-file
checksums. Model parameters and architecture follow the authors' Tiny Streaming path.

This is a **custom Swift runtime format**. It is not a drop-in checkpoint for Python
MLX, Transformers, Diffusers or the author's CUDA scripts. Use the matching [WaifuFinder](https://github.com/alastorid/WaifuFinder) worker.
No Python interpreter or PyTorch is required for inference. The offline converter and
numerical oracle remain available in the [WaifuFinder](https://github.com/alastorid/WaifuFinder) source repository.

## Contents

Approximately 3.51 GB of BF16 parameters, 899 weight tensors and a fixed [1,512,4096]
prompt. The 34 Safetensors shards cover the 30-block streaming DiT, causal low-quality
projection, Tiny Conditional Decoder and prompt. The Full-model Wan VAE is not included.

## Use

On Apple Silicon with macOS 15 or newer, run setup from the current [WaifuFinder](https://github.com/alastorid/WaifuFinder) source
repository. It downloads the pinned converted release, verifies its checksums and
reuses files already installed. No original checkpoints or Python conversion are needed:

```sh
./tools/setup-native-flashvsr.sh
./build.sh
```

If you downloaded this repository manually, pass the model folder to
`./tools/setup-native-flashvsr.sh /absolute/path/to/downloaded/model` instead.

Open Uncover Waifu, choose **FlashVSR 1.1**, choose a source downsampling factor, select a
region and reconstruct. FlashVSR outputs at four times the source frame dimensions and is
tuned for 4× super-resolution; other downsampling factors change its effective input-to-
output ratio. Both native algorithms share the existing A/B comparison, compositor,
audio/video export and prepared-input cache. SeedVR2 remains the default.

The worker preserves locality-constrained sparse attention with a Metal sparse
online-softmax kernel, causal projection/decoder memories, streaming KV windows and
ADAIN color correction. It does not substitute unrestricted attention or independent
image inference.

## Validation and limitations

On a 16 GB Apple Silicon Mac, native five-frame and 30-frame outputs were compared
against the pinned official BF16 graph. Projection/DiT used CPU PyTorch, the decoder
used PyTorch MPS, and CUDA sparse dispatch was replaced with equivalent masked SDPA.
This is not a measured CUDA comparison. Every visible frame passed preset tolerances
of mean absolute error <0.025 in [-1,1] and correlation >0.99. Detailed intermediate
metrics and native reports are in [WaifuFinder](https://github.com/alastorid/WaifuFinder)'s `docs/flashvsr-validation.json`.

Measured peak physical footprint was about 3.8 GB for a 128×128 output ROI and 4.3 GB
for a 159×253 ROI padded to 256×256. These are small-region measurements, not full-frame
performance claims. The default 6 GB budget has a conservative geometry preflight,
20 ms footprint monitoring and OS pressure cancellation. It cannot guarantee zero
system swap. Whole-region attention is preserved; spatial DiT tiling is not used.

Only native 4× is validated. HDR, non-square pixels, long clips, scene-cut quality and
large whole-frame inference are unverified. Reconstruction can invent details.
Aggregate similarity does not imply pixelwise identity. Consult [WaifuFinder](https://github.com/alastorid/WaifuFinder)'s documentation
for supported geometry and interval limits.

## Provenance and license

- Official weights: [JunhaoZhuang/FlashVSR-v1.1](https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1),
  revision `27561b186ded3402d7c975f4fd722e2885b6135f`.
- Official implementation and prompt: [OpenImagingLab/FlashVSR](https://github.com/OpenImagingLab/FlashVSR),
  revision `cf910c61a60733e610e9c6e8b607f80c3a6c202b`.
- Native converter/runtime: [alastorid/WaifuFinder](https://github.com/alastorid/WaifuFinder).
- Research: [FlashVSR: Towards Real-Time Diffusion-Based Streaming Video Super-Resolution](https://arxiv.org/abs/2510.12747).

The official model is listed under Apache 2.0. Redistributed packages must include the
Apache license and preserve applicable attribution/notices. This repository's LICENSE
and NOTICE accompany the converted weights. The authors have not endorsed this conversion.
