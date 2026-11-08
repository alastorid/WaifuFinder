# PhotoArena delivery status

PhotoArena is integrated into WaifuFinder as a native, executable evidence and
reconstruction prototype. This delivery does not complete every research and
validation requirement of the mission.

## Implemented and tested without UI interaction

- Shared oriented native coordinate contract and source-to-reference transforms.
- Apple Vision global homography with correlation/overlap acceptance and rejection.
- Native tiled RGB-median residuals, texture/clipping metrics, OCR text candidates.
- Programmatic validity/difference masks, aligned region crops and overview differences.
- Sparse native-crop MiniCPM comparisons, structured outputs, reversal abstention,
  content/geometry/runtime-keyed persistent scalar result cache, total request budget.
- Category-relative pairwise ranking, ties, unknown outcomes, disconnected-graph
  abstention, and a distinct technical ranking.
- Regional source selection, human preference/explicit overrides, unresolved status.
- Native-dimension TIFF reconstruction, exact source-index TIFF, portable source-file
  archives with SHA256, coordinate transforms and structured provenance manifests.
- Synthetic complementary-patch pixel correctness and real NASA photo reconstruction.
- Separate 40MP pixel-analysis and full-native export measurements.

## Implemented but not interactively validated

The app exposes Arena, Ranking, Region Atlas and Reconstruction Studio, direct
local photo-group import, corresponding native crops, shared display pan/zoom,
Retina-aware 100% crop display, region evidence, multiple Atlas overlays, human
category votes, manual source override/undo, session persistence, and exports.
These views are compiled. No interactive UI tests were created or run; the repository
permanently prohibits them. Full-resolution composite viewing remains an overview;
100% inspection applies to selected native crop comparisons.

## Experimental and not a forensic guarantee

Alignment confidence is a correlation score, not calibrated probability. Cross-source
median agreement can prefer correlated edited versions. Texture can reward noise or
artificial sharpening. OCR text candidates include actual scene text. MiniCPM category
preferences and evidence can be wrong even when both image orders agree. Every region
remains unresolved for authenticity, including human-selected regions.

Hard cell boundaries preserve actual selected samples but can produce seams. Output is
8-bit sRGB; source files themselves are archived unchanged. Geometric resampling of a
smaller source does not recover its missing detail. No generated pixels, generative
restoration, neural super-resolution, or inpainting enters reconstruction.

## Unimplemented mission requirements

- Validated AI manipulation segmentation/classification and reliable watermark masks.
- Semantic segmentation, local-flow alignment and independently trusted provenance.
- Fine contour source selection, graph cuts, seam optimization, multiband blending,
  exposure/color matching, HDR/wide-gamut/high-bit-depth output.
- A labeled real AI retouching/object replacement/inpainting/upscaling evaluation;
  localization precision/recall, forensic accuracy or calibrated confidence on such data.
- Core ML conversion and reference tensor/decision parity, verified ANE operation
  placement, controlled CPU/GPU/ANE comparative measurements.
- Model-instance/embedding reuse in app inference and a trained specialized PairJudge.
- Exhaustive native VLM coverage: model judging is budgeted disputed-region inspection.

## Prioritized next work

1. Build an independently labeled real edit/watermark dataset; measure alignment and
   localization failures before interpreting model preferences as forensic evidence.
2. Add edge-aware finer source masks and seam validation, keeping exact provenance and
   unresolved flags. Preserve source bit depth and color profiles in output.
3. Add a full-resolution tiled composite viewer and exportable localized candidate maps.
4. Reuse model workers and embeddings while preserving cancellation isolation; measure
   complete app+worker physical footprint, compression and process swap-ins.
5. Export and parity-check the real vision encoder/projector before any ANE experiment;
   profile actual placement. Validate decoder recurrent state/M-RoPE/KV separately.
6. Evaluate PairJudge only with independently reviewed labels and calibrated abstention.

## Reproduction and evidence

Build: `./build.sh` → `dist/WaifuFinder.app`.
Open File → PhotoArena… (Shift-Command-P). Import local still photographs.
Install the existing PhotoArena MiniCPM model through Models settings for judging.
Analysis and reconstruction do not require the model.

Background checks: `./run-tests.sh --only PhotoArena`.
The test fixture generators are in `Tests/ArenaForensicsTests.swift` and
`Tests/ArenaReconstructionTests.swift`; they use only Apple image frameworks.

- `PhotoArena-architecture.md`: independent proposals, critique and chosen contract.
- `PhotoArena-model-runtime.md`: verified runtime and conversion limitations.
- `photoarena-forensics-40mp.json`: native tile analysis coverage, timing and memory.
- `photoarena-40mp-export.json`: separate export-only resolution/timing/memory.
- `PhotoArena-example/evaluation.json`: real-photo complementary overlay recovery.
- `PhotoArena-example/ranking.json`: technical best-to-worst order, not originality.
- `PhotoArena-example/recovery/manifest.json`: sources, hashes, transforms, evidence,
  selections and unresolved regions.
- `PhotoArena-example/recovery/source-map.tiff`: full-resolution source indices.

Final build and background verification passed; measurements and scope follow below.

## Measured outcomes (2026-11-08)

| Check | Actual result | Scope |
| --- | --- | --- |
| Difference patch | 1,024 changed pixels in the expected 32×32 region | Controlled native fixture, not learned manipulation classification |
| Registration | Translation/native crop tests pass; NASA rotation2°, centered/off-center crop and mild perspective accepted with correlation >0.999 | Small real-photo sample; no universal alignment guarantee |
| Multi-source pixels | Exact match against held-out synthetic original | Three complementary known damaged strips |
| Real-photo recovery | 896×1134; RGBA8 MAE 0.6241/255; three sources, six unresolved regions | Conventional translucent overlays on NASA photograph; sRGB conversion |
| 40MP analysis | 8000×5000; 216 regions; 6.675s; sampled peak physical footprint 1.210GB | Two identical Apple-generated sources; no VLM inference; compressed peak 184.8MB, swap-ins unmeasured |
| 40MP export | 8000×5000; 0.505s; sampled peak physical footprint 640.6MB | Single-source export only; compressed memory/swap-ins unmeasured |
| Native448 model trial | Clean source preferred overall both orders; watermark and quality abstain after reversal disagreement | One real-photo synthetic-overlay crop pair; not calibrated accuracy |
| Native448 worker | Judgments 6.156s and 5.944s; end footprint about 1.686GB | Reused CPU/Metal worker; not whole-app/peak memory; no ANE placement measured |

The native-crop trial does **not** solve watermark category positional bias. Its
retained report is `photoarena-native-roi-validation.json`; the reproducible generator
and harness are `tools/fixtures-photoarena-roi.swift` and
`tools/benchmark-photoarena-roi.mjs`. No accuracy percentage across real AI edits is
reported because no such labeled evaluation was performed.

## Final integrated verification

`./build.sh` succeeded in 107 seconds and produced the standard
`dist/WaifuFinder.app`; deep strict signature verification passed. The running user
app was neither launched nor quit. `./run-tests.sh --only PhotoArena` reported
**17 passed, 0 failed, 0 known bugs** in 12.15 seconds. Its optional 40MP case was
not enabled in that integrated run; the separately enabled optimized reconstruction
harness ran all four cases, including the actual 40MP export measurement above.
The integrated run compiled 205 production sources and 61 test sources.
`git diff --check` passed. No interactive UI test was created or run.

Machine-readable summary: `photoarena-integration-verification.json`. At that initial integration step, the remainder of the repository regression suite
was not run; the complete release regression below subsequently passed.

## Remote inference and Hugging Face follow-up

Regional and overview judging now share one captured remote-first backend. Native
region crops travel over the existing authenticated encrypted protocol; remote
failures are surfaced without local retry. Only the executing machine reserves its
model. Install through Settings → Models → PhotoArena MiniCPM-V 4.7 → Download on
the server for remote inference, or on this Mac for local inference. Clients with a
compatible connected server do not need local model weights.

A fresh actual Hugging Face download through ModelInstaller passed every checksum,
native worker model-load validation, isolated activation and repeated-installation
reuse in 121.17s. The user's configuration was unchanged, and temporary downloaded
weights were removed. Evidence: `photoarena-huggingface-setup-verification.json`.
The integrated suite including that opt-in setup test reports **19 passed, 0 failed**
in 133.03s. Regional remote routing, cache reuse, failure propagation and cancellation
passed encrypted loopback tests using a deterministic server catalog. No claim of a
new two-physical-Mac inference benchmark is made. The standard app build succeeded
in 181s and deep strict signature verification passed. No interactive UI tests ran.

## Release v0.1.104 verification

The full background regression suite passed: **585 passed, 0 failed, 0 known bugs,
16 skipped** in 97.59s. The opt-in Hugging Face setup test is skipped in that ordinary
run and was independently executed successfully as documented above. Version-stamped
`dist/WaifuFinder.app` built successfully; deep strict signature verification and the
dependency/license audit passed. No interactive UI tests ran. Housekeeping removed
stale test leftovers and loose compiler outputs, preserving models, caches, research
evidence and the running app. The model folder/symlink is excluded from Git.
