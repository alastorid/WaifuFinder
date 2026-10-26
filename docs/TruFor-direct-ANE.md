# TruFor direct-ANE experiment

**Status:** isolated research worker only. The app continues to use the validated MLX/Metal path. Direct ANE is not enabled as an application backend.

## What was ported

The experiment adapts the relevant MIT-licensed parts of [maderix/ANE](https://github.com/maderix/ANE): MIL generation, FP16 BLOBFILE weights, dynamic resolution of the private `_ANEInMemoryModel` bridge, and IOSurface-backed input/output. The upstream project itself warns that these Apple interfaces are private and unstable; its [MIT license](https://github.com/maderix/ANE/blob/main/LICENSE) is retained with the adaptation.

The native worker is `waifu-trufor-direct-ane-probe`. It reads only the DnCNN tensors from the verified Safetensors checkpoint in Swift, folds each inference BatchNorm into its convolution, packs all FP16 weights and biases into one BLOBFILE, and compiles one 17-convolution MIL graph. The 17-pixel halo is kept around each compared output tile. The 3-channel input and 1-channel result are padded to eight channels with zeros for the experiment. IOSurfaces carry FP16 data directly between the worker and the private runtime.

The worker has no Python inference dependency. The normal app path does not link this target or call the private bridge. The direct worker tests accuracy against the same TFM1 noiseprint produced by the existing MLX/Metal worker. On ordinary API errors or an accuracy-gate failure it runs the public Core ML FP32 model. `tools/run-trufor-direct-ane-probe.sh` also supervises the worker and launches the Core ML fallback if the private process exits abnormally or exceeds its 120-second default timeout.

## M4 same-input comparison

Measurements below use the same verified TruFor weights, opaque 2000×2000 PNG, RGB/256 preprocessing, and interior 546×546 tile at image origin (512, 512). Each result compares the same central 512×512 noiseprint crop against the MLX/Metal TFM1 reference. Sustained measurements use 20 predictions; the first is reported separately and excluded from warm percentiles. Physical peak is the worker's sampled task footprint.

| Backend | M4 execution evidence | Warm p50 / p95 | Peak footprint | Noiseprint max abs / relative RMS |
|---|---|---:|---:|---:|
| MLX / Metal FP32 | Native 17-layer DnCNN, same tile | 263.6 / 273.6 ms | 892 MB | 0 / 0 |
| Core ML FP32 `.all` | `MLComputePlan` preferred GPU for 49 available operations; 119 unavailable | 90.0 / 93.8 ms | 249 MB | 0.000419 / 0.000701 |
| Core ML FP16 `.all` | `MLComputePlan` preferred ANE for 52 available operations; 122 unavailable | 20.9 / 21.9 ms | 41 MB | 0.243746 / 0.597169 |
| Core ML mixed, 5/17 convolutions FP32 | Every fourth convolution retained FP32; plan preferred ANE for 43 and GPU for 15 available operations | 56.3 / 68.0 ms | 489 MB | 0.236733 / 0.569557 |
| Core ML mixed, 9/17 convolutions FP32 | Alternating FP32/FP16 convolutions; plan preferred ANE for 39 and GPU for 27 available operations | 90.9 / 106.9 ms | 808 MB | 0.266131 / 0.551013 |
| Direct ANE FP16 | Graph compiled and loaded, but evaluation failed with `ANEProgramProcessRequestDirect()` status `0x1d`, type `0x9` | No valid measurement | Not isolated | No output |

The `MLComputePlan` values are public API placement recommendations, not a low-level trace of runtime execution; Apple documents them as model performance analysis evidence ([Core ML performance analysis](https://developer.apple.com/documentation/coreml/analyzing-a-core-ml-model-s-performance-in-xcode)). Core ML FP16 is therefore fast on the plan's ANE placement, but fails the strict `0.001` max-absolute and relative-RMS gates. It is rejected. Core ML FP32 and MLX/Metal both pass; their timings are for the DnCNN tile, not a full 4 MP TruFor prediction.

The 5/17-FP32 mixed model is about 38% faster than Core ML FP32, but its relative RMS error remains 56.96%, close to all-FP16's 59.72%. The 9/17-FP32 model does not improve latency over FP32 and still has 55.10% relative RMS error. These continuous-output error metrics do not directly mean a classification accuracy percentage; this probe does not measure labeled manipulation-detection accuracy.

## Direct-ANE findings

The bridge itself works on this M4: a 64-channel FP16 3×3 convolution with a bias compiled once and evaluated, and a synthetic 17-layer 64-channel chain evaluated at 32×32. The same synthetic chain failed at 36×36 and 66×66. The real TruFor graph also compiled at a small tile but failed during evaluation. These results point to a per-activation or graph-execution constraint in this direct path; they do not establish an Apple hardware limit. A 17-layer 3×3 network needs a 35-pixel input even for one valid output pixel, so the working 32×32 smoke shape cannot produce an exact TruFor result. The private backend has no usable TruFor accuracy or throughput result and remains rejected.

The upstream INT8 example is a synthetic W8A8 chain: it packs int8 weights, uses `constexpr_affine_dequantize` for weights, and inserts activation quantize/dequantize operations between convolutions. I reviewed this path but did not port it. The repository has no calibrated, accuracy-validated INT8 TruFor weights, and direct FP16 execution already fails on the real network; quantization would add another unvalidated variable without producing a backend candidate. FP16 is the only direct-ANE precision measured here.

On the real-image attempt, the isolated worker caught that execution failure, ran the public Core ML FP32 fallback, and passed the same accuracy gate (`max abs 0.000419`, relative RMS `0.000701`). This is the fallback behavior configured for a private API error. The supervisor script covers crashes and hangs as well.

The failed direct attempt followed by its fallback peaked at 534 MB combined process footprint. This is a lifecycle measurement, not a direct-ANE-only peak; the direct backend produced no valid inference output to profile.

## Memory budget and full-image behavior

All successful M4 tile workers stayed below 0.9 GB process footprint. The existing full 4 MP MLX/Metal worker stayed near 1.03 GB; score-only strip streaming was effectively neutral at that size. A 24 MP full-image MLX attempt was stopped by the existing hard guard at approximately 1.706 GB, below 2 GB; reducing the DnCNN row chunk to 8192 did not reduce that peak. The guard was not relaxed.

## Reproducing the native probe

Build the worker with `tools/build-native-trufor-direct-ane-probe.sh`. Then run:

```sh
tools/run-trufor-direct-ane-probe.sh \
  .native-runtime/trufor-publish-v1 \
  .native-runtime/trufor-memory-validation/fixtures/maximum.png \
  .native-runtime/trufor-memory-validation/maximum-evidence.bin \
  .native-runtime/trufor-dncnn-coreml-fp32/model.mlpackage \
  20 32
```

The final argument selects the direct-ANE output tile side; `32`, `64`, `128`, `256`, and `512` are available for experiments. Core ML fallback always evaluates the same fixed interior 512×512 output crop. Set `WAIFU_TRUFOR_DIRECT_ANE_TIMEOUT_SECONDS` to change the supervisor timeout.

The direct bridge uses only Foundation, IOSurface, Objective-C runtime lookup, and Apple frameworks. No GPL/LGPL component or FFmpeg dependency was added. The Core ML converter remains an offline artifact-generation tool; Python is not part of either inference worker.
