WD SwinV2 Tagger v3 — native Metal conversion for WaifuFinder.
Original model: SmilingWolf/wd-swinv2-tagger-v3, Apache-2.0.
https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3
Prepared natively from original FP32 weights: BGR channel permutation and input normalization folded into patch embedding; FP16 parameters; fixed learned attention bias and temperature.
Weight format: swinv2-native-metal32-v1. Runtime: swinv2-native-metal32-fused-qkv-v3.
The label CSV retains all original output heads in their original order.
Metal code and third-party notices are published in the WaifuFinder GitHub repository and shipped in the app.
