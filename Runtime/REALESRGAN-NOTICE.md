# Real-ESRGAN Core ML spatial upscaler

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

The model is used as a framewise 2× Core ML stage after temporal SeedVR2 restoration.
It does not add temporal consistency. Real-ESRGAN is a general-image restoration model,
not an anime-specific model.
