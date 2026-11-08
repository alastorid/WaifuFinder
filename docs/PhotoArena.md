# PhotoArena native MiniCPM integration

Select two or more photos or frames, right-click a selected item and choose
**Compare Selected in PhotoArena**. **Ask MiniCPM** runs the actual
MiniCPM-V 4.7 1B checkpoint. Human A/B/Tie/Unknown votes and AI votes remain
separate in the persistent history and JSON export.

Install **PhotoArena MiniCPM-V 4.7** in Settings → Models on the Mac executing
inference. A matching connected remote server is preferred; clients need no local
weights for remote judging. Backend selection is captured when Ask MiniCPM is
pressed. Failures are reported rather than retried on another machine.

## Actual model and execution

- Upstream model: `openbmb/MiniCPM-V-4.7-1B`.
- Official GGUF revision: `345adbfa02c4a4671e61b7c75502dc9bf24fa93d`.
- F16 language weights: SHA256 `71bc942bf5e07b839a7e73e4c9faa2fa27f97993cddd124c63b5dcd66a432ee1`.
- F16 16x-compression vision/projector: SHA256 `fb9cc047a3cdad5317fd8433a4aeda2d22f00f02ca4bd93b34ad84165b25a900`.
- Native llama.cpp revision: `7071ba378afa96dcd69c41e1ec2b4912364dbee4`.

The native worker loads both real model components and performs vision encoding,
multimodal prefill and autoregressive generation. llama.cpp handles the native
448px vision tiles, visual merging, canvas M-RoPE and hybrid linear/full-attention
decoder. The wrapper uses the pinned upstream single-turn non-thinking chat suffix.
An explicit grammar constrains the generated comparison object. Runtime inference
uses C++/Objective-C++, Apple ImageIO/CoreGraphics, Metal and CPU. There is no Python
inference, cloud inference dependency, video decoder or external media tool.

This is **Metal/CPU**, not an ANE/Core ML backend. No Core ML conversion, ANE
placement measurement or FP16-reference tensor parity is claimed. F16 describes
the upstream GGUF weights, not a guarantee that every intermediate is FP16.

The overview interface sends bounded 1600px images; the regional interface sends
native crops. Inputs travel through anonymous pipes and both A/B orders are judged.
One child process owns the complete pair, reusing model/context and bounded image
computations between orders. All recurrent/KV state is cleared for each order.
The process exits after the action; no model remains resident between actions.
Local requests are serialized, retain a 300-second watchdog per pass and terminate
only their own worker on cancellation. No image or activation scratch is written.
See [runtime optimization and measured limits](PhotoArena-optimization.md).

## Reliability limits and measured evidence

`photoarena-native-benchmark.json` records two actual native judgments of an
identical NASA Collins image pair. The smoke test passed execution/structured
output, but the raw model incorrectly preferred A in both trials. The app detects
byte-identical overview inputs and returns an exact-input tie without inference.

`photoarena-watermark-initial.json` and `photoarena-watermark-benchmark.json` retain
synthetic watermark trials. The final prompt selected the clean source overall in
both orders. Its cleanliness category still disagreed after reversing the image
order; the app therefore returns **unknown** for that category. Other categories
can also disagree. Agreement between two orders does not establish correctness.
Evidence text may be mistaken, and the model's uncertain flag is not calibrated.
The UI labels this explicitly and does not reconstruct pixels from these votes.

Individual measured native judgments took roughly 8–10 seconds on this Mac after
loading, with around 1.56 GB physical footprint and 75–95 MB compressed memory at
the end of inference. These are worker samples, **not peak or whole-app memory**.
Results include first-load/compile effects and are not a controlled performance
comparison. No ANE speed, GPU-free operation or universal optimality is asserted.

## Installation, transport and publication

The manager downloads the immutable release descriptor in
`tools/photoarena-release.json`, verifies the manifest and every file checksum,
loads the worker to validate compatibility, then activates the configuration.
The original OpenBMB Apache-2.0 weights are redistributed with attribution, the
license, native source and validation evidence. Native third-party notices travel
with the app. Only Apple system dynamic libraries are linked by the worker.

Remote capability negotiation pins the PhotoArena runtime revision. Two bounded
image containers travel inside the existing authenticated encrypted protocol.
The server uses the same local backend, with resource reservation, stream-local
cancellation and model-in-use accounting. The response retains category judgments,
observable evidence, timing and worker footprint. Encrypted loopback tests cover
transport and cancellation; these are background tests with no interactive UI.

## Reproduction

- `./tools/build-native-photoarena.sh` builds the native worker with Apple Clang,
  CMake and Ninja. Source and build caches live under `.native-runtime`.
- `./build.sh` bundles it into the standard `dist/WaifuFinder.app`.
- `./run-tests.sh --only PhotoArena` runs model-contract and encrypted remote tests.
- `node tools/benchmark-photoarena.mjs` runs repeated real native inference using
  `models/photoarena/model.gguf` and `models/photoarena/mmproj.gguf`.
- `xcrun swift -sdk "$(xcrun --sdk macosx --show-sdk-path)" tools/fixtures-photoarena.swift Tests/Fixtures/RealESRGAN/nasa-collins-896x1134.jpg .native-runtime/photoarena-fixtures`
  generates the synthetic overlay with Apple frameworks.
- `node tools/benchmark-photoarena.mjs --watermark` runs the watermark/reversal trial.
- `python3 tools/publish-photoarena.py` is offline release tooling using only
  standard-library HTTP; it verifies both upstream weight hashes before uploading.

## Integrated PhotoArena workspace (2026-11-08)

Open **File → PhotoArena…** (Shift-Command-P) and choose 2–16 local photographs.
The existing frame-grid comparison remains available as the overview interface
above. The dedicated workspace adds:

- Arena: corresponding native regions, synchronized display pan/zoom, 100% pixel
  display, overview differences, category-specific human votes and budgeted native
  MiniCPM region comparisons.
- Ranking: authenticity preference order (ties until sufficient evidence), separate
  technical order and retained category evidence. Relative scores are not calibrated
  confidence. Human and AI records remain distinguishable.
- Region Atlas: rectangular analysis grid with preferred source, measured differences,
  OCR text candidates, alignment correlation and unresolved-region overlays.
- Reconstruction Studio: before/composite overview, region source overrides and undo,
  native TIFF export with source-index TIFF, transforms, evidence and exact source-file
  archives with SHA256. All automatic and manual selections remain unresolved for
  authenticity. Hard cell boundaries can show seams.

The largest input defines native reference dimensions. Apple Vision homography is
accepted only after correlation and overlap validation; local deformation is not
validated. Native tile metrics are independent of model input resolution. Exports
are 8-bit sRGB, preserving dimensions but converting high-bit-depth/wide-gamut inputs.
No generated detail, restoration or neural enlargement participates in reconstruction.

Sessions use source-file-content identities and preserve votes/overrides. The model
cache keys actual submitted crop pixels, geometry and runtime revision. Export writes
into a new folder and includes category rankings, technical rankings, votes and judgments.

Build with `./build.sh`; the standard output is `dist/WaifuFinder.app`. Building does
not launch or quit the running app. Run background checks with
`./run-tests.sh --only PhotoArena`. To retain the real NASA example and run the
separate 40MP export measurement:

```sh
WAIFU_ARENA_EXAMPLE="$PWD/docs/PhotoArena-example" \
WAIFU_ARENA_40MP=1 \
WAIFU_ARENA_40MP_REPORT="$PWD/docs/photoarena-40mp-export.json" \
./run-tests.sh --only PhotoArena
```

The example exporter deliberately refuses to overwrite an existing recovery folder;
use a new example directory for subsequent runs. The fixture generation is part of
`Tests/ArenaReconstructionTests.swift` and uses Apple frameworks only.

See `PhotoArena-architecture.md` for the three-agent debate and shared contract, and
`PhotoArena-model-runtime.md` for model/runtime boundaries.

## Remaining work from the full specification

This is an executable evidence/reconstruction prototype, not a validated universal
forensic detector. Semantic segmentation, learned manipulation masks, verified original
provenance, dependable watermark classification, fine-boundary seam optimization,
exposure matching, full-resolution tiled viewer, model embedding reuse and persistent
worker reuse remain unfinished. The Atlas uses rectangular cells, not object masks.
The comparison viewport uses native crops; composite inspection is an overview.

No labeled evaluation of real AI retouching, object replacement, inpainting or AI
upscaling is supplied. Synthetic conventional overlays test source recovery, not AI
recognition. Shared/correlated edits can defeat consensus. Original-framework tensor
parity and actual Core ML/ANE operation placement remain unimplemented. No ANE speed
or fully recovered/AI-free output claim is made. No interactive UI test is run.

The published revision was independently downloaded into a temporary directory, every
manifest file was SHA256-verified, and real native inference completed successfully.
See `photoarena-published-verification.json`. The temporary download was removed.

Regional and overview judging now share the same remote-first route. Install the
model through Settings → Models on the inference server; a compatible connected
server permits client judging without local weights. Native crop routing and the
real Hugging Face installation check are described in PhotoArena-model-runtime.md.
