# RealESRGAN x4plus integration evidence

## Current scope

The user resumed full integration and publication, excluding further profiling.
Previously collected performance/placement evidence is preserved. New model runs
are correctness, transport and installation checks only. Interactive UI testing
remains prohibited. Existing concurrent tagging/TruFor edits are preserved.

## Stage checklist (2026-10-26)

- [x] 0: Audit and x2 native baseline (M3, 16 GB).
- [x] 1: Official x4 FP16 conversion and original-model validation.
- [x] 2: Compute-unit comparison, placement and optimization.
- [x] 3: Shared runtime and independent installation.
- [x] 4: Photos preview and stale-request protection.
- [x] 5: Actual model execution through authenticated remote transport.
- [x] 6: Regression, memory, tiling and release build.
- [x] 7: Attribution, verified publication and download/install.

Stages are marked complete only after their gates pass. Interactive UI testing is
prohibited by AGENTS.md; all verification here must be noninteractive.

## Architecture and implementation checklist

- Offline conversion: `tools/convert-realesrgan-optimized.py` loads the pinned
  BasicSR RRDBNet and official x2 checkpoint. x2 pixel-unshuffle/first-convolution
  fusion is scale-specific; x4 must skip it. Native benchmark and sealing are
  `tools/realesrgan-benchmark.swift` and `tools/seal-realesrgan-optimized.py`.
- Runtime: `NativeSpatialUpscaler` owns the Core ML model and reusable input
  tensors; x2 uses reflected tiles, 32px overlap and 256/384/522 shapes.
  `RealESRGANBackend` implements Uncover restoration. `RealESRGANJobResources`
  provides the native residency lock and sampled whole-process footprint limit.
- Installation: `ManagedModel`, `ModelInstaller`, and `ModelLibrary` provide
  Settings controls, references, integrity verification and configuration.
  Preserve `realesrgan-model.txt`, legacy `backend.json`, and x2 release pins.
- Preview: `FramePreviewView` has a fixed 350px right scroll panel containing
  Photo Info and evidence controls. `PreviewPhotoSource` supplies original or
  rendition pixels. `TruForPreviewModel` and `SafirePreviewModel` independently
  own evidence jobs. Preserve their original input and thumbnail arrangement.
  `ZoomableFrameImage` owns rendering/zoom geometry.
- Remote: `RemoteCapabilities`, `RemoteInferencePool`, `RemoteClient`,
  `WireProtocol`, `RemoteServerSession`, `RemoteServerJobs`, and
  `RemoteModelCatalog.local` are the real app/server transport and execution.
  New optional variant capabilities must decode absent on legacy servers.
- Release: stage/upload/restore scripts and pinned release descriptors drive
  app distribution. x4 needs its own descriptor/repository; x2 is immutable.
- Checks: `./run-tests.sh` compiles production and test sources together;
  `./build.sh` builds the app. Do not invoke `run-ui-tests.sh`.

## Stage 0 evidence

Existing model paths were dangling links and no x2 configuration was present.
The normal setup command failed creating its dangling destination. Downloaded
and hash-verified the existing pinned x2 release into
`.build/realesrgan-x2-baseline` using `download-coreml-release.swift`, without
altering those links or installed model configuration. Compiled with Apple's
Core ML compiler and ran the existing native benchmark on an Apple-generated
48×33 RGB fixture: four genuine predictions produced 96×66 outputs.
See `realesrgan-x2-baseline-20261026.json`. Warm image latencies:
43.05, 42.56, 42.76 ms; first prediction 61.26 ms. Cold backend load:
6.85 s. Sampled peak process physical footprint: 79,544,992 bytes.
This excludes some Core ML service memory; no total-memory reduction is claimed.

`./run-tests.sh --only Uncover`: 20 passed, 0 failed, 4 optional real-video/model
fixtures skipped. The standalone benchmark independently executes real x2.
Existing user changes (including NativeSpatialUpscaler/ModelInstaller/README)
were present before this task and must be preserved.

## Conversion dependency review

An existing conversion environment contains setuptools' LGPL autocommand vendor.
It was inspected without executing that environment. Conversion uses a separate
`--without-pip` environment with only the reviewed conversion dependency closure:
PyTorch 2.7.0 (BSD), Core ML Tools 9.0 (BSD), NumPy 2.5.4 (BSD/permissive),
SymPy/mpmath/NetworkX (BSD), Jinja2/MarkupSafe (BSD), filelock/attrs/cattrs/
PyYAML/pyaml (MIT), fsspec/protobuf (BSD), packaging (Apache/BSD),
typing_extensions (PSF), tqdm (MPL/MIT). NumPy and PyTorch CPU binaries link
Apple Accelerate; PyTorch includes LLVM libomp, not GNU OpenMP. PyTorch's
aggregate notices include unused upstream vcpkg codec ports and CPR test
licenses; these are not linked macOS inference/conversion components.
No setuptools, autocommand, torchvision, media tooling, or new app dependency
is used. Python is exclusively offline reference/conversion tooling.
The repository audit currently reports the pre-existing LGPL vendor in the
other environment; do not remove another task's environment to hide this.

## Stage 1 evidence

Official checkpoint SHA256:
`4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1`.
BasicSR revision `8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a`, architecture
SHA256 `6aa8eb3f77e918b90eeff4ebfaf8745abe0c40eb8fcc03abbc1928693d70ab58`.
Real-ESRGAN v0.1.0 tag commit: `4c05b20cb30aded73ed3731848a9fbbf70c7c0ac`.
BSD licence fetched separately at `a4abfb2979a7bbff3f69f58f58ae324608821e27`
(the original release tag did not yet contain a licence file); its SHA256 matches
the existing bundled Real-ESRGAN licence.

Converted the official 23-block, 64-feature, growth-32 x4 RRDBNet to an ML Program
with FP16 weights, computation, inputs and outputs. No quantization, x2 fusion,
pixel-unshuffle, or substituted network. Native Core ML compiled and executed
30 checks (five image geometries × three shapes × two repeats), each producing
exactly 4× width and height. Maximum absolute FP32-reference error: 0.006221;
worst mean absolute error: 0.000348. Predeclared gates: 0.01 maximum, 0.001 mean.
See the native validation reports. Numerical cases are deterministic synthetic
RGB; real-photo quality remains a later product gate.

Reproduce using an audited conversion environment:

```sh
$CONVERTER tools/convert-realesrgan-optimized.py --variant x4plus --source SOURCE --output OUTPUT
xcrun swift tools/compile-coreml-model.swift OUTPUT/RealESRGAN.mlpackage OUTPUT/RealESRGAN.mlmodelc
# Compile benchmark exactly as setup-native-realesrgan.sh does, then:
BENCHMARK --model OUTPUT/RealESRGAN.mlmodelc --fixtures OUTPUT/fixtures --scale 4 --units all --repeats 2 --compute-plan --report OUTPUT/native-all.json
```

## Stage 2 measurements (M3)

| Configuration | 128 median ms | 256 median ms | 384 median ms | Peak process footprint bytes |
|---|---:|---:|---:|---:|
| all | 46.11 | 187.75 | 430.14 | 79,938,184 |
| CPU+ANE | 46.81 | 187.79 | 432.98 | 74,678,896 |
| CPU+GPU | 227.81 | 898.16 | 2483.56 | 960,758,624 |
| all, fastPrediction hint | 60.82 | 244.11 | 578.72 | 79,610,480 |

All configurations passed the FP32 comparison. Warm medians exclude iteration
zero. These sequential runs had thermal state 1; they are device/fixture-specific,
not a universal optimum. Retain default specialization and `.all`; explicit ANE
has no demonstrated speed benefit. Reject fastPrediction (slower predictions,
22.4s cold load versus 7.6s baseline) and forced GPU. No quantization or manual
fusion is retained. Core ML's standard ML Program passes fold constants and lower
the residual/dense graph; all 351 convolutions, 279 leaky ReLUs, 276 concatenations,
93 additions, 23 multiplications and two nearest-neighbor upsampling operations
prefer ANE in the compute plan. 2,765 constants have no device assignment.
No transpose/reshape fallback is identified in this graph. An ANE rewrite or
pixel-shuffle substitution therefore has no demonstrated justification.

The Core ML Instruments CLI trace is retained at
`.build/realesrgan-x4/coreml-placement.trace`; inspect exported device events
before describing the plan as actual hardware placement. Process footprint is
not total system/ANE-service memory. No memory improvement is claimed from
changing specialization or precision.

## Completed model and remaining context limitation

The CLI trace contains **100 actual RealESRGAN ANE prediction intervals**,
totaling 6.495 seconds of ANE activity, matching the 100 profiled native
predictions. This corroborates the compute plan; CPU host/transfer work remains.
See `realesrgan-x4-optimization.json` for the complete collected measurements.

The already-running context experiment finished: 128px cores with halos
0/64/128 had maximum differences 0.764407 / 0.004887 / 0.002028 against a
384px whole-image FP32 reference. All failed the predeclared .002 maximum gate;
the mean errors were .010671 / .00002218 / .00000559. No tile approximation is
accepted, and the sealed model explicitly records `tilingValidated: false`.
The converted model itself passes native FP16 parity at all supported tensor
shapes. Its current deliverable is a model package, not an integrated tiled app
backend. No additional benchmark runs or context tuning were started after the
scope change.

## Resumed integration: correctness without profiling

The 64px-core / 160px-context hypothesis passed a new rectangular FP32 check:
321×65 source, whole reference padded to 512×256, tiled model tensors 384×384.
Maximum context difference 0.000004053; mean 0.000000352. This avoids accepting
any previously rejected halo candidate or relaxing the .002/.0002 gate.
Native reconstructed 1284×260 pixels passed the quantized FP32 comparison
(maximum 0.006102, mean 0.001030 against .01/.002 bounds).

The production authenticated server executed the actual installed x2 and x4
networks over encrypted loopback, returning 66×38 and 132×76 pixels from a
33×19 source. Local/remote RGB pixels were identical. This proves server/client
execution on one machine; it is not a two-Mac LAN or performance measurement.

Focused integration checks: 6 passed, 0 failed; installer and optional existing
real-video fixture skipped until publication. Further regression checks include
photo stream cancellation and rapid Off/model changes. New profiling is excluded.

The runtime has an explicit 16,384px-per-side / 256-million-output-pixel bound;
a standard 4032×3024 source is geometrically supported at native x4. The x4
assembler stores core RGB pixels directly without full-image Float sums or
weights, and returns an opaque RGB bitmap. These are allocation design facts,
not measured process-memory improvement claims. Results larger than the 128 MiB
preview cache budget remain displayable but are not retained in that cache.
Remote messages retain the existing 64 MiB encoded payload bound; oversized
encoded output fails rather than changing the protocol or silently resizing.

## Final release verification

| Stage | Status | Evidence |
|---|---|---|
| 0 Audit/baseline | Pass | Real x2 baseline; 20 Uncover checks passed |
| 1 Conversion | Pass | Official FP16 x4 ML Program; all original numerical cases passed |
| 2 Optimization | Pass | Earlier M3 compute/ANE evidence retained; conservative tiling correctness passed; no new profiling |
| 3 Runtime/installation | Pass | Independent x2/x4 registry, configuration and native inference; x4 Uncover restores original ROI geometry |
| 4 Preview | Pass (noninteractive) | Off/x2/x4, held-image/cache behavior, stale navigation/model/Off protection; fixed existing 350px panel |
| 5 Remote | Pass | Real model execution over authenticated loopback, exact local/remote RGB parity, cancellation and invalid reply rejection |
| 6 Hardening | Pass with scope limits | 509 passed / 0 failed / 13 optional fixtures skipped; final release build and code-signature verification passed |
| 7 Publication | Pass | Pinned public revision downloaded, hash-verified, installed by real app installer, and executed |

Published repository:
https://huggingface.co/alastorid/RealESRGAN-x4plus-Optimized-CoreML

Revision: `6a598182cd47d83a307fa056db876d2fb23d1b0e`.
Manifest SHA256: `a95379588b54346070eb98fe86d9ddadac294bc3c43b50fb02c8a6eb1f83a3e1`.
The existing x2 repository and release descriptor were not changed.

Published-model checks: **8 passed, 0 failed, 1 optional real-video fixture
skipped**. The actual app installer downloaded both public pinned releases into
an isolated model root, compiled and verified them, preserved the x2 activation
when adding x4, and ran each installed network. The x4 published download was
also independently verified by the existing native release downloader. Its
release descriptor is included in the built app.

Final app: `.build/realesrgan-x4/WaifuFinder.app` (release configuration).
The normal x4 setup helper also verified and configured the published model
locally. It wrote only the separate `realesrgan-x4plus-model.txt`; the existing
x2 installation/configuration was preserved. The complete local verified release
bundle is `.native-runtime/realesrgan-x4plus-optimized`.

No interactive UI test or launcher was created or run. Existing noninteractive
preview geometry checks and the new state tests passed; interactive narrow-window,
keyboard and visual inspection was not performed. No fresh latency, repeated-memory
or hardware-placement profiling was run after the user excluded profiling. The
recorded numbers are the earlier tensor measurements, not performance claims for
the new large-image integration. Remote validation is loopback, not a two-Mac LAN.

The isolated offline conversion/publishing dependency closure and staged release
passed the repository licence-artifact audit. This does not erase the historical
finding about a separate pre-existing conversion environment recorded above.

## Implementation file inventory

- Runtime: `NativeSpatialUpscaler.swift`, `RealESRGANBackend.swift`, new
  `RealESRGANPhoto.swift`, and `Model/UncoverTypes.swift`.
- Installation: `Models/ModelInstaller.swift`, `Models/ModelLibrary.swift`,
  `tools/realesrgan-x4plus-release.json`, `tools/setup-native-realesrgan.sh`.
- Preview: `UI/FramePreviewView.swift`; existing zoom/pan renderer is reused.
- Remote: `RemoteCapabilities.swift`, `RemoteInferencePool.swift`,
  `RemoteServerJobs.swift`, `RemoteUncoverBackend.swift`.
- Offline/release: converter, sealer, native benchmark, compute-plan detail,
  new `stage-realesrgan-x4.py`, existing stage/upload/restore scripts, `build.sh`.
- Checks: new `Tests/RealESRGANTests.swift`, existing remote tests and test entry,
  `run-tests.sh`. Shared concurrent tagging work needed a small compile-only repair
  in `TaggingScalingTests.swift` (typed invalid byte fixtures and the shared
  throwing assertion helper); its behavior was preserved.
- Attribution/evidence: README, Runtime notices, Credits, Licensing, Uncover,
  Remote-inference, website credits, this report and the x2/x4 JSON reports.

All pre-existing/concurrent TruFor and tagging implementation changes were preserved.
Original network authors: Xintao Wang, Liangbin Xie, Chao Dong and Ying Shan;
reference architecture by BasicSR contributors. WaifuFinder supplies conversion,
validation and integration. BSD-3-Clause / Apache-2.0 notices are retained.
