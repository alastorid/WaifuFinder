# Credits

WaifuFinder's native Mac integration builds on the work of these projects and
their contributors. Hashtags are acknowledgements, not claims of endorsement.
The project page presents the visual-sequence idea using entirely synthetic
drawings; no personal photos or video samples are used in that explanation.

## The visual sequence: NVIDIA source, native Mac execution

NVIDIA made the actual cuOpt routing source available under Apache 2.0. We adapted
the candidate search, ranked reductions and snapshot move execution in its
[`two_opt.cu`](https://github.com/NVIDIA/cuopt/blob/02bfe53163c3e8eba25827f548623a8ab8071f88/cpp/src/routing/local_search/two_opt.cu)
to Swift and Metal for an open photo sequence. The objective is the sum of visual
distances between consecutive shown photos; the last photo does not return to
the first. This remains an approximate search for a minimum-distance path.

WaifuFinder supplies the Apple Vision embedding integration, checked distance
convention, Metal distance kernels, native buffers, several starting routes,
CPU fallback, cancellation and mapping back to photo IDs. This credits the cuOpt
implementation without attributing the general 2-opt method itself to NVIDIA.
Apple Metal is a system framework, not an open-source component of cuOpt.

The measured M3 workload took **116.296 ms on CPU and 29.068 ms on Metal** at
1,024 items: **4.00× throughput, about 75% less solver time**. It used optimized
code, 768-dimensional synthetic vectors, four starting routes and 16 iterations,
including distance-matrix construction. It excludes image analysis and first
shader compilation. See [the audit and benchmark](CuOptMetal.md).

Large videos can contain 200,000+ frames, so efficient sequencing is valuable.
That scale has not been benchmarked. A dense Float32 matrix for 200,000 items
would require roughly **149 GiB**; the current app uses a bounded CPU fallback
above 16,384 shown images. The measured ratio must not be extrapolated to an
entire video pipeline or a 200,000-frame run. Grouping defaults off; visual
sequencing works independently, and grouping remains an optional control.

## Native runtime, source references and packaging

| Project / contributors | Contribution and license | Credit tags |
| --- | --- | --- |
| [NVIDIA cuOpt](https://github.com/NVIDIA/cuopt) | Routing search adapted for native Swift/Metal; Apache-2.0. [Pinned notice](../ThirdParty/cuopt/NOTICE). | #NVIDIA #cuOpt #TSP |
| [Apple MLX](https://github.com/ml-explore/mlx), [MLX Swift](https://github.com/ml-explore/mlx-swift), [MLX C](https://github.com/ml-explore/mlx-c) | Native machine-learning execution; MIT. | #MLX #MLXSwift #AppleSilicon |
| [ByteDance Seed / SeedVR2](https://github.com/ByteDance-Seed/SeedVR) | Video restoration model and weights; Apache-2.0. | #ByteDanceSeed #SeedVR2 |
| [xocialize/seedvr2-mlx-swift](https://github.com/xocialize/seedvr2-mlx-swift) | Starting point for the native SeedVR2 port; MIT. | #SeedVR2 #MLX |
| [lpalbou/mlx-gen](https://github.com/lpalbou/mlx-gen) and its contributors | Additional SeedVR2 implementation reference; MIT. | #MLXGen #SeedVR2 |
| [AbstractFramework](https://huggingface.co/AbstractFramework/seedvr2-3b-4bit), [MLX community](https://huggingface.co/mlx-community/SeedVR2-3B-mlx) | Prepared SeedVR2 weights and positional embedding; Apache-2.0. | #AbstractFramework #MLXCommunity |
| [OpenImagingLab/FlashVSR](https://github.com/OpenImagingLab/FlashVSR), [Junhao Zhuang](https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1) | Streaming restoration graph and published model; Apache-2.0. | #FlashVSR #OpenImagingLab |
| [Xintao Wang / Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | Image restoration weights; BSD-3-Clause. | #RealESRGAN |
| [XPixelGroup/BasicSR](https://github.com/XPixelGroup/BasicSR) | Restoration reference architecture; Apache-2.0. | #BasicSR |
| [SmilingWolf / WD SwinV2 Tagger v3](https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3) | Tag and rating model; Apache-2.0. | #SmilingWolf #WDTagger |
| [Belisoful/InferKit](https://github.com/belisoful/InferKit) | Adapted face-restoration formulas; MIT. | #Belisoful #InferKit |
| [Apple Swift Numerics](https://github.com/apple/swift-numerics) | MLX Swift transitive dependency; Apache-2.0. | #SwiftNumerics #Swift |
| [Niels Lohmann / JSON for Modern C++](https://github.com/nlohmann/json) | MLX C transitive dependency; MIT. | #nlohmannJSON |
| [fmt](https://github.com/fmtlib/fmt) | MLX transitive formatting dependency; MIT. | #fmt |
| [Apple Metal-cpp](https://developer.apple.com/metal/cpp/) | MLX's C++ Metal interface; Apache-2.0. | #MetalCpp #Metal |
| [ds_store](https://github.com/dmgbuild/ds_store), [mac_alias](https://github.com/dmgbuild/mac_alias) | Offline disk-image layout helpers; MIT. Not bundled in the app. | #ds_store #mac_alias #Python |

Apple's Vision, Accelerate, Metal, Core ML, AVFoundation, CoreMedia, Photos and
SwiftUI system frameworks make the native Mac workflow possible.

## Separately acknowledged model

Kepeng Xu, Li Xu, Gang He, Wenxin Yu and Yunsong Li created
[PGTFormer](https://github.com/kepengxu/PGTFormer), described in
[“Beyond Alignment: Blind Video Face Restoration via Parsing-Guided
Temporal-Coherent Transformer” (IJCAI 2024)](https://www.ijcai.org/proceedings/2024/165).
Thank you to the authors for this video face-restoration work.

PGTFormer is credited separately because its network terms restrict use to
non-commercial purposes, and the model weights have no declared licence. This
acknowledgement is not a claim that the network or weights are permissively licensed;
see [Licensing.md](Licensing.md).

#PGTFormer #KepengXu #LiXu #GangHe #WenxinYu #YunsongLi #IJCAI2024

## Offline conversion and validation tools

These are development tools and references, not application runtime dependencies.
Only distributions with approved direct and transitive licenses may be installed;
a permissive project's name does not approve every binary wheel or vendored tool.

| Project | Project license | Credit tags |
| --- | --- | --- |
| [PyTorch](https://github.com/pytorch/pytorch), [TorchVision](https://github.com/pytorch/vision) | BSD-3-Clause | #PyTorch #TorchVision |
| [NumPy](https://github.com/numpy/numpy) | BSD-3-Clause; binary builds must also satisfy the repository's dependency policy | #NumPy |
| [Apple Core ML Tools](https://github.com/apple/coremltools) | BSD-3-Clause | #CoreMLTools |
| [Hugging Face Hub](https://github.com/huggingface/huggingface_hub), [Transformers](https://github.com/huggingface/transformers), [Tokenizers](https://github.com/huggingface/tokenizers), [Diffusers](https://github.com/huggingface/diffusers), [Safetensors](https://github.com/huggingface/safetensors) | Apache-2.0 | #HuggingFace #Transformers #Diffusers #Safetensors |
| [timm / PyTorch Image Models](https://github.com/huggingface/pytorch-image-models) | Apache-2.0 | #timm |
| [Einops](https://github.com/arogozhnikov/einops) | MIT | #Einops |
| [Pillow](https://github.com/python-pillow/Pillow) | MIT-CMU / HPND | #Pillow |
| [Python](https://www.python.org/psf/license/) | PSF license | #Python |

Full component licenses and modification notices remain authoritative; hashtags
are additional recognition and do not replace them. No GPL/LGPL project is included
in the credit inventory or approved as a dependency.

#OpenSource #NVIDIA #cuOpt #Metal #AppleSilicon #Swift #MLX #SeedVR2 #FlashVSR
#RealESRGAN #BasicSR #WDTagger #InferKit #PyTorch #HuggingFace
