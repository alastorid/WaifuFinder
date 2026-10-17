# Final models and published bundles

The app installs and uses final models in `~/Library/Application Support/Waifu/Models/`:

| Model | Default folder | Published bundle in `models/published/` |
| --- | --- | --- |
| WD SwinV2 tagger, native fused QKV | `wd-tagger` | `WD-SwinV2-Tagger-v3-Swift-MLX` |
| WD SwinV2 Core ML comparison, rank-four SDPA | development benchmark only | `WD-SwinV2-Tagger-v3-CoreML` |
| SeedVR2 3B, fused MLX 4-bit | `seedvr2-3b` | `SeedVR2-3B-Swift-MLX-4bit` |
| SeedVR2 7B, fused MLX 4-bit | `seedvr2-7b` | `SeedVR2-7B-Swift-MLX-4bit` |
| FlashVSR 1.1 Tiny | `flashvsr` | `FlashVSR-v1.1-Tiny-Swift-MLX-BF16` |
| PGTFormer fused GPU Core ML | `pgtformer-<release revision>` | `PGTFormer-Swift-CoreML` |
| RealESRGAN x2plus optimized Core ML | `realesrgan` | `RealESRGAN-x2plus-Optimized-CoreML` |

Model files are ignored by Git and can be restored through Settings → Models using
the pinned published releases. To recreate portable development bundles without
old app settings or machine-specific links, run:

```sh
python3 tools/restore-model-bundles.py wd-tagger wd-tagger-coreml pgtformer realesrgan
# Include the multi-GB SeedVR2 and FlashVSR models:
python3 tools/restore-model-bundles.py --all
```

The downloader verifies every manifest member and support file. See
[clean-machine setup](../docs/Development-setup.md) for the app and benchmark bootstrap.

The app configuration points to the default installation folders. WDTagger also discovers the default folder automatically when its configuration file is absent or stale, including a user-selected model storage folder. It does not depend on repository development folders. The current native tagger release is revision `11d78056b0788a1f3aebe0e92bd0346ecbbdc459` of `alastorid/WD-SwinV2-Tagger-v3-Swift-MLX`. The Core ML comparison is pinned separately in `tools/wd-tagger-coreml-release.json`.

The production models all have specific speed or memory work, with different levels of evidence and different tradeoffs:

| Model | Production optimizations | Measured result |
| --- | --- | --- |
| WD SwinV2 tagger | Native Core ML with rank-four SDPA fused into all 24 attention blocks; CPU + GPU | 15.1 frames/s at 1080p batches on an M3, 687 MB peak process footprint. Top-15 parity passed on 24 synthetic frames; natural-photo quality was not established by that fixture. |
| SeedVR2 3B | 4-bit weights, packed MLP and VAE projections, fused Metal normalization/activation and causal-buffer preparation | On the tested 5-frame fixture, projection fusion cut streamed time by 22%; added VAE fusion cut 30-frame peak footprint by 22%. |
| SeedVR2 7B | 4-bit weights, packed VAE projections and fused Metal VAE kernels | Projection fusion cut time by 6.6% on the tested streamed fixture. Its GELU transformer does not use the 3B SwiGLU fusion. |
| FlashVSR | Streaming BF16 graph, sparse local temporal attention with online softmax, bounded window/cache state | The published path is numerically checked against the official model graph. These reports do not include a paired before/after speedup claim. |
| PGTFormer | GPU-enabled FP32 encoder with nine native fused attention blocks; FP16 GPU decoder with fused normalization | GPU placement reduced eight-frame time by 35.5% and CPU time by 91.7%. Attention fusion then reduced encoder time by 13.2%, total inference by 6.3% and peak memory by 5.7%; eight output PNGs matched the prior GPU model exactly. |
| RealESRGAN x2plus | FP16 Core ML, algebraically fused first convolution and adaptive context around ROI tiles | The tested M3 image fixture measured about 3.5× speedup. Overall memory reduction was not claimed. |

These are model-specific optimizations, not one uniform low-bit or fused-kernel recipe. The benchmarks are tied to their recorded M3 fixtures and do not guarantee the same gains on other clips or Macs; full methods and limits are in `docs/`.

Old `.native-runtime` model locations remain compatibility links for existing setup scripts. New staging and publishing default to `models/published/`. Intermediate tagger packages, original upstream weights and the temporary development archive have been removed. Native runtime executables remain available. Python conversion environments are external to the repository and are only needed to regenerate models.
