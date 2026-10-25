---
license: apache-2.0
base_model:
- ByteDance-Seed/SeedVR2-3B
- ByteDance-Seed/SeedVR2-7B
- AbstractFramework/seedvr2-3b-4bit
- AbstractFramework/seedvr2-7b-4bit
library_name: mlx
tags:
- mlx
- mlx-swift
- apple-silicon
- seedvr2
- 4-bit
- video-super-resolution
---
# SeedVR2 — fused Swift/MLX video bundles

Ready-to-use Apple Silicon bundles for [WaifuFinder](https://github.com/alastorid/WaifuFinder).
The root contains 3B (~2.4 GB); `7b/` contains 7B (~5.14 GB). Both include the positive
text embedding. Setup downloads prepared weights directly, without Python, conversion,
training or further quantization.

## Optimizations

The native exporter joins 3B's packed SwiGLU gate/value projections and both models' VAE
query/key/value projections, including biases. Packed integers, scales and offsets are
reordered without dequantization or rounding. 198 byte-identical shared attention arrays
are represented by aliases, removing 325 MB of duplicate 3B weights. Shared video/text
attention and MLP projection calls are batched after verifying weight identity. Transformer attention already has packed
QKV. One-block transformer shards support bounded loading. The 7B architecture uses
GELU, so it receives VAE attention fusion and block sharding, not SwiGLU fusion.

The Swift runtime uses matching F16 VAE activations/weights and F16 3B transformer
activations. The 7B transformer retains F32 to avoid F16 overflow. Normalization and rotary
calculations remain F32. The runtime compiles the small SwiGLU elementwise chain and retains a capped
64 MB reusable buffer cache. Streamed inference loads only the active encoder or decoder
and one transformer block. Temporal context, causal state and overlapping tiles remain.
F16 computation introduces small output differences; this is not bit-exact F32 inference.

## Use

```sh
./tools/setup-native-uncover.sh       # 3B default
./tools/setup-native-uncover.sh --7b  # optional larger model
./build.sh
```

Setup verifies pinned hashes and resumes interrupted downloads. The worker uses a
physical-memory budget of the smaller of 6 GB and half installed RAM. Streaming is
selected automatically for 7B, 8 GB Macs and single-pass 3B jobs. Repeated 3B passes on
larger Macs keep weights resident to avoid reloading. `--stream-weights` also enables it manually.
The release targets [WaifuFinder](https://github.com/alastorid/WaifuFinder)'s native loader, not Diffusers or an unchanged image loader.
See `validation.json` for measured runtime, memory, pixel differences and test scope.
Measurements on a 16 GB M3 with a bounded process budget do not establish zero swap on
an actual 8 GB machine. Generated details and spatial tiling can differ from the source.

## Measured results

Three paired runs on a 16 GB Apple M3, five consecutive real-video frames, 128×128
input restored to a 256×256 ROI at 2×, seed 42. Time is the median; memory is the
highest sampled physical footprint across those runs.

| Variant | Seconds | Peak GB |
| --- | ---: | ---: |
| Original 3B, resident | 11.58 | 5.19 |
| Prepared 3B, resident | 9.85 | 4.03 |
| Prepared 3B, streamed single pass | 9.03 | 2.27 |
| Original 7B, streamed | 13.49 | 2.89 |
| Prepared 7B, streamed | 12.60 | 2.24 |

Each prepared path was faster in all three corresponding runs. The F16 runtime
changes produced a mean RGB pixel error around 0.13 on a 0–255 scale, with maximum
error 4. Alias-only and shared projection batching preserved output PNG bytes.
These measurements describe this fixture, not a speed guarantee for every clip.

## Authors and license

Original models: [ByteDance Seed](https://github.com/ByteDance-Seed/SeedVR),
[3B](https://huggingface.co/ByteDance-Seed/SeedVR2-3B) and
[7B](https://huggingface.co/ByteDance-Seed/SeedVR2-7B), Apache-2.0.
MLX q4/group64 conversion: [lpalbou / AbstractFramework](https://huggingface.co/lpalbou),
[3B](https://huggingface.co/AbstractFramework/seedvr2-3b-4bit) revision
`9b2f6c3c1b1b4548c66f302ec5a4cd3cd9937eb5`,
[7B](https://huggingface.co/AbstractFramework/seedvr2-7b-4bit) revision
`22f491ba03647864491520cb803916c21eaad303`.
Embedding: [mlx-community/SeedVR2-3B-mlx](https://huggingface.co/mlx-community/SeedVR2-3B-mlx)
revision `46f851fe23b2a9fbefee6419b94e4fdf07b92965`.
Implementation credit: [mflux](https://github.com/filipstrand/mflux),
[MLX-Gen](https://github.com/lpalbou/mlx-gen) and the
[Swift reference port](https://github.com/xocialize/seedvr2-mlx-swift).

Model weights and embedding retain Apache-2.0; `LICENSE`, `NOTICE` and upstream model
cards are included. [WaifuFinder](https://github.com/alastorid/WaifuFinder) adds packed projection fusion, sharding, checksums and
native video runtime optimizations. These are derivatives of the named authors' models,
not independently trained models. Runtime source retains its separate MIT/upstream notices.
