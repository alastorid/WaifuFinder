# Native model status: WD Tagger, TruFor and SAFIRE

Status and measurements as of 2026-10-25. “Parity” below means agreement with a
reference implementation on the documented fixtures. It does not mean accuracy
on all real images. Speed and process-memory results are tied to the named Mac,
input and worker path; they are not whole-app or universal hardware guarantees.

## Version and comparison summary

| Model | Original model and current package/runtime | Reference parity | Speed and memory evidence |
|---|---|---|---|
| **WD Tagger** | Original: [SmilingWolf WD SwinV2 Tagger v3](https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3), model v2.0 / dataset v3. The installed model package is `wd-tagger-native-v2`, pinned to release `11d78056b0788a1f3aebe0e92bd0346ecbbdc459`; runtime: `swinv2-native-metal32-fused-qkv-v2`. It is the same trained checkpoint, converted from FP32 to FP16 parameters and prepared for native Metal inference; it was not retrained. | On the 64-image reference corpus, mean probability error was **0.00000446**, maximum error **0.01394**, with **2 general-tag threshold decisions** differing from the original FP32 reference and no rating argmax changes. Against the prior native path, there were no tag-threshold or rating changes. The later direct-RGBA input path separately matched the Float32 patch-preparation reference exactly. | On the M3 batch-8 comparison, current native inference measured **15.52–15.53 fps / 665.3 MB peak worker memory**; the historical Core ML CPU+GPU run measured **14.24–14.48 fps / 702.5 MB**, with ANE excluded. A later Core ML CPU+ANE run invoked ANE execution but measured **14.62 fps vs native 15.66 fps** and failed numerical agreement (697 non-rating-tag threshold flips); it is not an acceptable backend. Worker memory is not app-plus-worker memory. See the [native tuning and parity](WDTagger-native-tuning.md), [current input/memory comparison](WDTagger-native-memory.md), and [direct ANE/Core ML investigation](WDTagger-ANE.md). |
| **TruFor** | Original: official GRIP-UNINA TruFor checkpoint, source SHA-256 `ac1d90e3…`, upstream revision `ae54475d…`. The installed model package is pinned to `80cdfb30d990a115b309793851435b8907ecd648`; runtime revision `trufor-native-v1`, production execution capability `trufor-fp32-v2`. The full model is adapted to Swift/MLX with converted FP32 weights. | On the documented reference fixtures, score error was **3.8×10⁻⁶**, anomaly-map maximum error **4.15×10⁻⁵**, reliability-map maximum error **2.45×10⁻⁵**, and anomaly-mask IoU **1.0**. FP16 was not retained for production because it missed the strict parity gates. | Historical warm 1024×768 timings before the RAM-budget refactor: **1.586 s (M3)** and **1.108 s (M4)**. Earlier native FP32 timings were **1.79 s** and **1.43 s**, respectively. This shows lower latency than the earlier WaifuFinder native path in those runs; there is no matched upstream-PyTorch peak-memory comparison. The current runtime targets a 2 GB process budget with streamed FP32 scratch rows and the original 24 MP input envelope; see [RAM validation](TruFor-memory-validation.json). TruFor’s upstream license allows informational and nonprofit use only, with credit required for public results. See [TruFor validation and benchmarks](TruFor.md). |
| **SAFIRE** | Original: official SAFIRE checkpoint, source SHA-256 `65bdf688…`, upstream revision `ec2af6af…`. The installed model package remains the pinned FP32 checkpoint `6422600bd8a8b361ec4869bd597267b7553b5533`; runtime revision `safire-native-v2`. Production computes the two-way decoder transformer in FP16 while keeping the image encoder, prompt encoder and upscaling tail in FP32. Source-partition clustering uses deterministic farthest-first initialization instead of upstream randomized initialization, so cluster labels are not guaranteed to match for every image. | On the gradient fixture, the selected M3 mixed-precision path measured **99.71%** source-partition pixel agreement with the FP32 oracle. The remeasured M4 256/16 profile measured **99.7157%** partition agreement and **0.999293 mask IoU**; its minimum individual source-region IoU was **0.9874**. These are fixture results, not dataset-level forensic accuracy. | M3: **2.252 s FP32 vs 2.045 s mixed precision** at 256/4. M4: current 256/16 profile **1.479 s**, 3.2% faster than 256/8 in three interleaved rounds; peak worker footprint was about **1.623 GB** for chunks 4, 8 and 16. The strict FP32 fallback matches the saved M4 reference; see [SAFIRE architecture, validation and benchmarks](ImageLineage.md) and the detailed [M4 profile sweep](ImageLineage-native-benchmark.json). |

## What the evidence supports

- The current outputs are very close to their upstream references on the tested
  inputs. “More than 99% the same” is a fair shorthand for those parity fixtures,
  but it is not a single cross-model score and does not establish real-world
  model accuracy. The documented numerical differences and clustering caveat
  still apply.
- WD Tagger is measurably faster and uses less peak worker memory than the
  historical CPU+GPU-only Core ML comparison on the tested M3 setup. Core ML
  CPU+ANE invoked the Neural Engine, but its measured throughput was lower and
  its outputs failed parity; the `.all` cold-load compile did not finish.
  SAFIRE shows lower latency than its
  earlier native path, with unchanged measured worker memory. TruFor's earlier
  latency measurements predate its current RAM-budget refactor; the current
  streaming path trades time and disk traffic for memory. There is no matched
  upstream speed/RAM comparison for either forensic model.
- Release pins and the detailed methodology remain in
  [`tools/wd-tagger-release.json`](../tools/wd-tagger-release.json),
  [`tools/trufor-release.json`](../tools/trufor-release.json),
  [`tools/safire-release.json`](../tools/safire-release.json),
  [`docs/TruFor-native-validation.json`](TruFor-native-validation.json), and
  [`docs/ImageLineage-native-validation.json`](ImageLineage-native-validation.json).

Whether any model is downloaded and configured is per-user installation state;
the release descriptors above record the versions the app can install. On the
2026-10-25 development workspace, all three model pointers resolved to installed
packages; other users may not have downloaded them.
