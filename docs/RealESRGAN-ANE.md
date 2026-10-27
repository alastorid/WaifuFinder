# Real-ESRGAN: CPU + ANE optimization (2026-10-26)

Both official 23-RRDB networks retain 64 feature / 32 growth channels, their
original weights, native scales and FP16 execution. GPU execution is excluded at
every model load, independently of imported model metadata or hardware. Core
Graphics and software Core Image perform input/output processing. No interactive
UI tests or GPU comparisons were performed.

## Measured result

Apple M3 MacBook Air, Mac15,12, 16 GiB. Input: **896 × 1134, 1,016,064 pixels**,
[NASA public-domain portrait](../Tests/Fixtures/RealESRGAN/README.md). All rows use
CPU + ANE; timing includes CPU packing, inference, stitching and CGImage assembly,
but excludes model loading and PNG encoding. RAM is sampled **whole benchmark
process physical footprint**, not all Core ML/ANE service memory or system RAM.

| Model | Export/runtime | Median latency | Peak footprint | Quality versus starting FP16 |
|---|---|---:|---:|---|
| x2plus | Starting FP16 | 1.616 s | 163.50 MiB | Reference |
| x2plus | Optimized FP16 | 1.532 s | 154.27 MiB | 56.52 dB PSNR; 0.99896 SSIM |
| x4plus | Starting FP16 (remeasured) | 87.114 s | 170.80 MiB | Reference |
| x4plus | Optimized FP16 | 19.883 s | 153.89 MiB | 61.76 dB PSNR; 0.99964 SSIM |

x2 latency improved **5.2%**, with **5.6% less** process footprint. Against the
remeasured starting x4 baseline, the prior optimized FP16 measurement is **4.38×**
faster with **9.9% less** footprint. Starting x4 now has two warm-up-controlled
runs, replacing the initial single 92.928 s run; all prior raw measurements remain
in the JSON history. Optimized x4 has two earlier runs; final x2 rows have three.
Only the starting x4 and W8A8 pipeline were rerun in this follow-up. These are
limited M3 results, not universal optimality claims. M4 and reliable power/energy
measurements were unavailable.

Full measurements, raw tensor checks, placements and rejected candidates:
[realesrgan-ane-20261026.json](realesrgan-ane-20261026.json).

## Retained changes

- **x2:** preserve unshuffle/first-convolution fusion and adaptive 256/384/522
  context; fold each dense block's scale and identity into the centre of its
  existing final convolution. Fold outer RRDB scaling into the third transformed
  final convolution. This removes 69 adds and 23 remaining multiplications after
  Core ML's existing constant folding, without splitting convolutions or changing
  dense concatenations. The compute plan has 932 ANE operations, down from 1024.
- **x4:** preserve the original convolution/activation/upsampling graph. Add one
  measured **512px** shape, retain the **160px halo**, and use **192px** interior
  cores. Coalesce leading cores into a 352px span. The existing 384px / 64px-core
  fallback remains for other hardware or packages without tuning acceptance.
- **x4 output:** crop and quantize directly into the final owned RGB allocation.
  Transfer that allocation to CGDataProvider, eliminating the full RGB Data copy
  and temporary Float tiles. Release the previous prediction before allocating
  the next output. The first 512px candidate used 233.55 MiB and was rejected;
  these storage fixes were required before accepting the larger tile.
- Load each model once, retain bounded input-buffer reuse, keep independent
  selections/paths, and adapt small x4 images to validated smaller input shapes.
  Every choice uses `.cpuAndNeuralEngine`; CPU-only diagnostics are explicit.

Only the measured hardware identifier may enable the larger tile. Sealing with
full-image quality and matched benchmark reports requires lower median latency,
no more than 5% footprint growth and zero GPU placement. Otherwise the runtime
keeps its conservative 384px tile. Existing FP16 packages remain available.

## Rejected experiments and quality gates

Both W8A16 and true W8A8 graph exports were executed on M3 with CPU + ANE.
W8A8 uses three Apple-native packed views of the NASA photograph for calibration;
calibration itself is CPU-only. Weight quantization is symmetric INT8 per-channel.
Both configurations failed numerical quality for both models, so none is installed.
W8A8's worst synthetic-fixture PSNR/SSIM versus native FP16 was 21.33 dB / 0.81182
for x2 and 15.62 dB / 0.79523 for x4. This rejects these candidates, not future
quantization approaches or M4 support. The export API follows
[Apple's quantization workflow](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-api.html).

x4 residual-identity fusion passed small tensor checks but failed the full photo
(maximum difference **17/255**); x4 keeps its original graph. Algebraically
reparameterizing nearest-neighbour + convolution as a 4×4 stride-2 transposed
convolution passed CPU border/bias tests, but tensor timings were slower. It
remains experimental; no end-to-end speed benefit is claimed.

Each native tensor must match the original CPU FP32 checkpoint within maximum
0.01 and mean 0.001. Full-image FP16-vs-FP16 byte checks require at most five code
levels (allowing both outputs' FP16 rounding), mean error ≤0.002, PSNR ≥45 dB and
SSIM ≥0.995. Actual selected maxima were **4/255 x2** and **2/255 x4**. SSIM here
uses nonoverlapping 8×8 luma windows. Signed RGB bias, edge-gradient error, odd
borders, singleton images, dimensions, tiled rectangular reference, model
switching/cancellation and authenticated loopback are also checked. Visual
inspection is limited to the supplied photo; it is not a broad quality dataset.

## Reproduction and integration

Use the existing audited offline conversion environment (PyTorch 2.7 / Core ML
Tools 9), or review every dependency/license before creating another. No new
runtime dependency was added. The common loader/fusions live in
`tools/realesrgan_graph.py`; both export entry points use them.

```sh
# Set CONVERTER to an audited external conversion Python, SOURCE to the pinned
# official checkpoints plus rrdbnet_arch.py, and OUTPUT to a new directory.
"$CONVERTER" tools/experiment-realesrgan-ane.py --source "$SOURCE" \
  --variant x2plus --fusion identity --output "$OUTPUT/x2"
"$CONVERTER" tools/experiment-realesrgan-ane.py --source "$SOURCE" \
  --variant x4plus --fusion none --sizes 128 256 384 512 --tile-size 512 \
  --output "$OUTPUT/x4"
```

The production converter also supports `--residual-fusion` and `--x4-tile-size`;
its default is identity fusion for x2, original residuals for x4. Retain
`--residual-fusion none` for the starting export. Native benchmark defaults to
ANE and rejects `all`/`gpu`. Quantization is separate, through
`tools/quantize-realesrgan-ane.py`; its outputs are explicitly unvalidated.

Compile, validate every exported shape using the native benchmark, then seal
with `tools/seal-realesrgan-optimized.py --experimental`, supplying
`--validation`, `--image-validation`, `--baseline-benchmark` and
`--candidate-benchmark`. The same fixture and output geometry must be used in
both performance reports. Export alone never enables measured tile tuning.

Local accepted packages are staged independently under
`.native-runtime/realesrgan-ane-v2/x2plus` and `x4plus`; original installations
are preserved. The updated app is `.build/realesrgan-ane/WaifuFinder.app`.
Nothing has been published to the hosted releases.

The repository-wide license audit still flags pre-existing setuptools/vendor
LGPL autocommand in `.build/trufor-coreml-venv`. That environment was not invoked,
modified or bundled by this work. Conversion uses the separate previously
reviewed `.build/realesrgan-x4/venv`; app/model artifacts contain no new dependency.

Final verification: **511 passed, 0 failed**, 13 optional fixture checks skipped;
two offline algebra tests passed. Real native x2/x4, tiled reference, model
switching, cancellation and authenticated loopback passed. Release build and
strict deep code-signature verification passed. Scoped license audits passed for
the conversion environment, staged models and built app; the unrelated
repository-wide finding above remains.

The two saved model paths are activated for the updated build. Previous path
files are retained in `.native-runtime/realesrgan-ane-v2/previous-configuration`.
Use the updated app build to recognize the new x2 graph version. Original model
files and hosted releases were not overwritten.

## Requested x4 starting-FP16 remeasurement

The starting model was rerun on the same M3 / 1.016 MP photo with an explicit
small warm-up and **two full-photo passes per precision**. Both precisions used
the identical original 384px tensor / 64px core / 160px halo pipeline and copied
RGB output assembly. This isolates quantization from tiling/storage changes.
The rejected W8A8 package ran only through an explicit diagnostic allowance in
an isolated benchmark copy; no app source or package validation was changed.

| Precision, starting pipeline | Runs | Median | Peak process footprint |
|---|---|---:|---:|
| FP16 | 85.245 / 88.984 s | **87.114 s** | **170.80 MiB** |
| W8A8, rejected quality | 72.131 / 71.922 s | **72.027 s** | **171.69 MiB** |

W8A8 used **17.32% less time (1.209× throughput)**, with **no RAM gain**:
footprint was 0.89 MiB / 0.52% higher, effectively unchanged at this scope.
The current W8A8 candidate remains rejected because of its existing numerical
quality failures. No further quantization tuning or production activation was
performed. Optimized FP16's earlier 19.883 s / 153.89 MiB result uses different
512px tiling and is not the denominator for this quantization comparison.

The remeasured starting result supersedes the earlier 92.928 s / 166.33 MiB
single-run baseline from the initial comparison. Against the remeasured
baseline, the prior optimized FP16 measurement is **4.38× faster** and uses
**9.9% less** process footprint. Original raw measurements remain preserved.
See [full remeasurement evidence](realesrgan-x4-remeasurement-20261026.json).
