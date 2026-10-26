# Development on a clean Mac

A fresh clone contains the application and native-worker sources, pinned dependency
versions, model release descriptors, conversion tools and synthetic data generators. The previous machine's `.build`, `Runtime/.build`, `.native-runtime`,
`.native-toolchain`, `dist` and `models/published` directories are disposable once
these portability changes have been committed and pushed.

## Build the app

Use an Apple Silicon Mac running macOS 15 or later, with Git, curl and a Swift 6
toolchain/macOS SDK from Xcode or Command Line Tools. Install Apple's developer
tools with `xcode-select --install` if they are missing. Complete any Xcode license
or first-launch prompts before building.

```sh
git clone https://github.com/alastorid/WaifuFinder.git
cd WaifuFinder
./tools/setup-development.sh
open dist/WaifuFinder.app
```

Setup builds the native workers and app, downloads the checksum-pinned MLX Metal
library and SeedVR2 positional embedding, draws the icon, and signs the bundle
locally. It recreates the SwiftPM cache and a clone-local toolchain workaround;
it does not require copying an installed toolchain from the old machine.
Internet access to GitHub and Hugging Face is required for missing dependencies.
The app has no Python runtime dependency.

Install the inference models you need through **Settings → Models**. Their release
revisions and checksums are tracked in `tools/*-release.json`. Settings, selected
videos, tagging caches and saved results under `~/Library/Application Support/Waifu`
are separate user data; a Git clone does not restore them.

The existing [release workflow](../.github/workflows/release.yml) also builds from
a checkout using only a cache of downloadable Swift package sources.

## Swift 6 build notes

The app and native workers treat compiler warnings as errors. Keep whitespace around
operators when the right side begins with a dot (`value == .case`, `value = .init`).
Use Metal compute pipelines for custom image kernels rather than GLSL `CIColorKernel`
source; copy through `withUnsafeMutableBytes` instead of deprecated `Data(bytes:count:)`.
Worker-specific MLX graph and Metal-kernel rules are recorded in
[`Runtime/README.md`](../Runtime/README.md).

## Restore model bundles for offline development

The standard-library downloader requires Python 3.9 or later and curl. It restores
the exact published model files, licenses and validation reports, checking every
file against the committed release descriptor and manifest:

```sh
python3 tools/restore-model-bundles.py wd-tagger wd-tagger-coreml pgtformer realesrgan
python3 tools/restore-model-bundles.py --all
python3 tools/restore-model-bundles.py --verify-only wd-tagger wd-tagger-coreml
python3 tools/restore-model-bundles.py --check-remote --all
```

`--all` also downloads the multi-GB SeedVR2 and FlashVSR bundles. The default
destination is `models/published`; `--destination DIRECTORY` selects another
staging root. Downloaded bundles contain portable source packages/weights. Core ML
compiled models are generated on the destination Mac. The remote metadata check
verifies pinned manifests and support files without downloading model weights;
full downloads verify the weights too.

The older `setup-native-*.sh` tools can additionally register a model with the app.
They are optional for building the application. `--convert-official` reproduces
the offline conversion using the authors' checkpoints and an external Python
environment; `tools/face-conversion-requirements.txt` records the conversion
dependencies. Normal development uses the verified published bundles.

## Generate synthetic WD Tagger inputs

```sh
./tools/setup-development.sh --benchmarks
# Or, after building the app:
./tools/setup-wd-tagger-benchmarks.sh
```

This additionally requires Python 3.9+. It draws 64 synthetic inputs using source
code, downloads the native and Core ML WD Tagger bundles, compiles the benchmark
tools, and generates model predictions locally. Patterns include solid colors,
gradients, checkerboards, edges and deterministic noise. No local images or videos
are read. Generated files remain under ignored `.build` directories.

The Core ML comparison model lives in `.build/wd-optimization/coreml`, independent
of app settings. `WAIFU_WD_COREML_DIRECTORY` can override that directory.
To generate or check only the inputs, without downloading a model:

```sh
python3 tools/generate-wd-tagger-inputs.py
python3 tools/generate-wd-tagger-inputs.py --verify-only
```

`--output-root DIRECTORY` selects an isolated build directory. Different existing
inputs are refused unless `--force` is explicit; full benchmark setup replaces the
previous generated inputs. Comparison predictions are produced by
`tools/generate-wd-tagger-baselines.sh` after the models/tools have been prepared.
They are not an independent FP32 reference or controlled performance measurements.
No generated images, tensors, predictions or compressed data archives are uploaded.

For cooled comparison scripts, install Node.js and `macmon` and then run:

```sh
./tools/build-native-wdtagger.sh --check-kernels --check-gemm-tiles --report .build/wd-native/checks.json
node tools/run-wd-tagger-paired.mjs --tile 32x64x32 --fused-qkv --compare-original --label clean-machine
node tools/run-wd-tagger-next-pass.mjs clean-machine
```

These use the generated synthetic inputs. They do not reproduce the historical
real-video measurements. Historical batch-2/batch-16/compression
experiments require exporting those candidate models with `convert-wd-tagger.py`
or `experiment-wd-tagger-optimization.py` in the external conversion environment;
setup restores the production batch-8 comparison. Image-list benchmarks can use
new images, while compute comparisons accept the restored `--tensor-input` file.
The pipeline benchmark's synthetic video can be regenerated with the Apple-native
generator in [WDTagger-native-pipeline.md](WDTagger-native-pipeline.md).

The application regression harness is `./run-tests.sh`; its generated video fixtures
use Apple system frameworks. See [Tests/README.md](../Tests/README.md).

## Preserved source

The [October 16 source archive](development-archive/2026-10-16/README.md) preserves
text environment inventories, Swift source snapshots and experiment helpers.
Raw local reports, binary test data, Instruments captures, model-conversion
outputs and compiled programs are excluded. Use the source generators and pinned
model releases to create new local test artifacts as needed.

The optional independent Torch parity fixture is generated locally with
`tools/make-wd-tagger-test-fixture.py --weights /path/to/original/model.safetensors`
in the external conversion environment. Its outputs live in ignored
`.build/test-fixtures/wd-tagger`. The Swift tests generate the gradient image in
memory and explicitly skip FP32 parity when that local reference is absent.
