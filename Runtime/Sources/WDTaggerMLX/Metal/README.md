# WDTagger Metal kernels

[Current memory/input results](../../../../../docs/WDTagger-native-memory.md).

**[Performance matrix and optimization guide](../../../../../docs/WDTagger-native-tuning.md)**
(main matrix also appears near the top of the repository README).

These kernels are dispatched directly by Swift through MLX's native Metal API.
Inference, weight preparation, correctness checks and timing require no Python.

- `Flash32.metal`: fixed 196-token, 32-lane tiled attention with FP32 online
  softmax. Compact per-head positional bias and per-window region IDs preserve
  shifted-window masking without an expanded score/mask tensor. The output is
  token-major, ready for the projection GEMM.
- `QKVCosine.metal`: one SIMD group per token/head combines FP32 cosine
  normalization, learned Q scaling, Q/K/V splitting and layout conversion.
- `QKVProject.metal`: the selected path fuses projection, cosine normalization
  and packing with the original 32×64 matrix tile. FP16 projected values stay in
  registers; 512 bytes of FP32 partial reductions cross SIMD groups. V bypasses
  normalization. `QKVCosine.metal` remains the unfused comparison path.
- `NormRoute.metal`: two-pass FP32 LayerNorm, affine transform, residual addition
  and the next block's spatial permutation in one dispatch.
- `NormSIMD.metal`: optional register-based normalization trial; not selected
  because its apparent speed gain did not survive confirmation.
- `PatchMerge.metal`: routes directly between persistent window layouts while
  forming timm's x00, x10, x01, x11 patch concatenation.
- `PatchInput.metal`: accepts direct RGBA UInt8 or reference RGB Float32 input,
  combining input normalization and 4×4 patch extraction,
  writing directly in the first stage's window order.
- `MLPGelu.metal`: tiled FP16-operand GEMM with FP32 accumulation for aligned
  QKV, projection, MLP and reduction matrices. Bias and optional tanh-GELU run
  in the epilogue; FC1 needs no pre-activation buffer or activation dispatch.

The two flattened Steel headers and the attention body derive from Apple's
MIT-licensed MLX 0.32.4 headers. Copyright notices remain in the headers;
`MLX-LICENSE.txt` contains the upstream license. The specialized native runtime
uses the project's existing MLX Swift 0.31.2 dependency. Metal source is bundled
as resources and compiled by Metal, independent of any Python installation.

This is the selected app backend on Apple Silicon. The worker uses fused QKV,
batch eight and a 448 MiB soft allocation budget and 64 MiB cache. See the linked measurements for
the Core ML comparison, memory tradeoff and numerical limitations.
