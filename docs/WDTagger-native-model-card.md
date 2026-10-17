# WD SwinV2 Tagger v3 — optimized native Metal

Apple Silicon model for [WaifuFinder](https://github.com/alastorid/WaifuFinder).
Apache-2.0 upstream weights from [SmilingWolf](https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3); see LICENSE and NOTICE.

The app bundles a persistent Swift/C++/Metal worker using MLX Swift 0.31.2, fused QKV projection/normalization/packing, native attention and fused GEMM epilogues. No Python runtime is used.

Input: RGB Float32 [8,3,448,448] in 0...255. Preserve the app's white letterbox and resizing; do not normalize or swap channels again. Pad tails with the last real frame. Output: Float32 sigmoid [8,10861]. Labels: 4 ratings, 8106 general, 2751 character heads.

Two controlled batch-eight comparisons on an M3 Air measured 7.9–8.1% faster model inference than Core ML. A separate 768 MiB soft-budget run measured 15.50 fps and 950 MB peak worker process footprint. These exclude host app/image-preparation overhead; memory is not a hard cap.

All 138 kernel checks passed. On 64 inputs the fused path added zero general-tag threshold/rating decision changes versus the prior native path, but probabilities differ slightly. This is numerical validation, not human-labeled quality evaluation. See validation.json and the repository's docs/WDTagger-native-tuning.md for detailed evidence and failed trials.

Every downloadable model/support file is SHA-256 pinned by the app's release descriptor and manifest. Kernel source bundles and mlx.metallib are packaged in the app, built from published source/dependency pins.
