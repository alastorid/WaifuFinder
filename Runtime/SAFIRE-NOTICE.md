# SAFIRE native adaptation

SAFIRE: Segment Any Forged Image Region (AAAI 2025), by Myung-Joon Kwon,
Wonjun Lee, Seung-Hun Nam, Minji Son, and Changick Kim.
Official source: https://github.com/mjkwon2021/SAFIRE
Inspected revision: ec2af6af8c789fb6359493a3e352f89c6fd8107f.
Upstream announced Apache-2.0 for code and CC BY 4.0 for weights on 2026-08-05.
Official safire.pth SHA-256:
65bdf6880380162d57f77d71b3fd367c3f16aebca316f942a821da16e9225b12.

The Swift implementation adapts the architecture and algorithms from:
- networks/safire_model.py, safire_predictor_binary.py, safire_predictor_multi.py
- segment_anything/modeling/image_encoder_with_adaptor.py
- segment_anything/modeling/prompt_encoder.py, mask_decoder.py, transformer.py
- segment_anything/modeling/common.py

SAM components: Copyright (c) Meta Platforms, Inc. and affiliates,
Apache License, Version 2.0. The encoder/relative-position calculations derive
from ViTDet and MViT (Meta Platforms, Apache-2.0). Native adaptations retain
these upstream notices. Independently authored WaifuFinder source remains MIT.
Converted weight bytes remain CC BY 4.0, with attribution; not MIT.

Production runtime dependencies are the already pinned mlx-swift 0.31.2/core
0.31.1 (MIT), Swift Numerics (Apache-2.0 with Swift exception), and Apple system
frameworks. No new third-party package dependency is added. No training,
OpenCV, torchvision, ONNX, or Python inference dependencies are shipped.

Offline oracle only: PyTorch 2.8.0 macOS arm64 CPU wheel (BSD-3-Clause),
SHA-256 e9f071f5b52a9f6970dc8a919694b27a91ae9dc08898b2b988abbef5eddfd1ae.
Its bundled dylibs are libtorch*, libc10, libshm (BSD) and libomp
(Apache-2.0 with LLVM exception); otool confirms Apple frameworks, libc++,
libSystem, libc10 and libomp. Its combined license inventory includes licenses
for unshipped Linux/vendor build/test paths (ffnvcodec, cpr/test, Vulkan).
Those components are absent from the inspected wheel and are not installed,
linked or invoked. Required Python-only dependencies: typing_extensions 4.15.0
(PSF-2.0), filelock 3.19.1 (Unlicense), sympy 1.14.0 (BSD), mpmath 1.3.0 (BSD),
networkx 3.2.1 (BSD), Jinja2 3.1.6/MarkupSafe 3.0.3 (BSD), fsspec 2025.10.0 (BSD).
No optional extras, NumPy, SciPy, training libraries, or media binaries are used.
Oracle files reside in a temporary directory and never enter the app, builds,
CI, fixtures, or distribution.

Automatic cosine clustering follows kmeans_pytorch, Copyright (c) 2020
subhadarshi (MIT; full text in SAFIRE-KMEANS-LICENSE). The native port retains
raw-feature mean centers and uses deterministic farthest-first initialization.
Fused resampling follows the 11-bit uint8 INTER_LINEAR algorithm from OpenCV
resize.cpp (Intel/Willow Garage/Itseez, BSD-3-Clause; full source notice in
SAFIRE-OPENCV-LICENSE). No OpenCV binary or library is used.
