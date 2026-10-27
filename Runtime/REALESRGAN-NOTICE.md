# Real-ESRGAN Core ML spatial upscalers

Original model authors: **Xintao Wang, Liangbin Xie, Chao Dong and Ying Shan**,
authors of *Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure
Synthetic Data*. The RRDBNet reference architecture is by the BasicSR contributors.
WaifuFinder performs conversion, native optimization, validation and application
integration; it does not claim authorship of these networks or pretrained weights.

x4plus uses the official 23-block RealESRGAN_x4plus checkpoint from release v0.1.0,
SHA256 `4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1`.
Its FP16 ML Program preserves the native 4× network without the x2 input fusion.
See `docs/RealESRGAN-x4-integration.md` for the validation and release status.

WaifuFinder downloads its own pinned, original-model-validated Core ML conversion of
`RealESRGAN_x2plus` from
[alastorid/RealESRGAN-x2plus-Optimized-CoreML](https://huggingface.co/alastorid/RealESRGAN-x2plus-Optimized-CoreML).
Earlier builds instead downloaded the pre-converted `RealESRGAN_x2plus_522_fp16`
model from [hanxiao/real-esrgan-coreml](https://github.com/hanxiao/real-esrgan-coreml);
our version adds validated 256 and 384 pixel inputs and is the only variant with a
recorded original-model parity result. The model weights are from
[xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) and are distributed under
the BSD 3-Clause License; see `REALESRGAN-LICENSE`. The reference architecture is from
[XPixelGroup/BasicSR](https://github.com/XPixelGroup/BasicSR) under Apache-2.0, shipped
with the release as `BasicSR-LICENSE`.

Real-ESRGAN is a separate framewise Core ML restoration algorithm.
It does not add temporal consistency. Real-ESRGAN is a general-image restoration model,
not an anime-specific model.
