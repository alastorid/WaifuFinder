# PhotoArena 2.0 implementation

PhotoArena evaluates all candidates together and recommends one source when the evidence supports a choice. An original file can be the result; reconstruction is a fallback. There are no voting matches in either entry point. When the evidence cannot separate candidates or support a repair, the workspace says so and preserves a source.

The audit found three useful foundations: the validated native MiniCPM worker, Apple Vision registration with pixel-based rejection, and real-source TIFF/provenance export. The old workspace fit pairwise votes to a ranking, switched sources along grid boundaries, and displayed a source-to-reference heatmap beside its composite. The new workflow keeps the low-level components, uses independent observations for each source, masks patches at pixel level, and computes new output-to-source comparisons. The ORB inspector informed the sensitivity and zoom interactions; it is not an engine dependency.

## Review and recovery

- `ArenaForensics` reads oriented input dimensions and SHA256 identities, validates global alignment, and analyzes bounded native tiles. Distinct source contents contribute to the consensus; byte-identical copies reuse alignment and cannot inflate support. It retains exact changed-pixel runs and local connected components within each tile. A 500,000-run budget defers excess masks explicitly. Unreliable correspondence is excluded. OCR is supporting evidence and can be scene text.
- `ArenaGroupEvidence` makes one overview montage, then bounded regional packages with labeled context and focused detail montages. The model sees unannotated source pixels; the inspection area is described by native coordinates and focus crops can be magnified for readability. The red overlays remain available in the human inspector. Regional evidence includes native coordinates, coverage, texture, clipping, residuals, OCR coverage, and measured RGB and variation inside the exact change mask. The versioned MiniCPM contract requires a finding for every candidate in one group request, with comparative originality, cleanliness, detail, alteration, enhancement, degradation and consistency scores plus visible observations, defect categories, usability and confidence. Unaligned sources appear for independent assessment, never as corresponding patches.
- `ArenaStudio` publishes analysis, review, planning, verification, partial completion, cancellation and error states. The first action reviews at most 13 packages. The model state is reset between packages; one local child owns the batch and is reaped after it. A compatible remote server accepts the same group request over the authenticated route. Remote packages currently reuse the connection, with a separate worker per package. Cache keys include the ordered source IDs, actual encoded evidence, geometry and contract revisions. Only structured results are persisted.
- `ArenaRecoveryPlanner` evaluates evidence across the set. Equal bytes receive the more cautious observation across copies while raw model outputs remain inspectable. A clean eligible source is preferred before patching, even when a dirtier source has a higher raw originality score. Eligibility requires completed evidence, adequate quality, supported alignment and sufficient coverage. A difference below three comparative points does not establish a unique winner among otherwise equivalent choices.
- When no clean original is established, recovery starts from one covering source. It only replaces changed pixels when a clean aligned donor is supported by the regional evidence. `ArenaMergeVerifier` checks the exterior ring of a proposed patch for agreement before allowing an automatic replacement. A rejected or unreviewed region remains unresolved. Manual source overrides are recorded separately. No operation adds synthesized pixels, sharpens, or silently blends sources.
- `ArenaOutputInspector` generates actual result-to-source differences in reference coordinates and records changed, uncovered and expected source-selection pixels. Export verification reads the written output, checks it against the archived sources and rejects unexpected changes over the comparison threshold. The one-byte TIFF map identifies the owner per output pixel. The manifest records transforms, region masks, pixel operations and source hashes. Direct-source export preserves file bytes, metadata, profile and bit depth; its source map is in oriented native coordinates. Originals remain untouched.

## Verification and measured scope

Run the optimized noninteractive PhotoArena suite:

```sh
WAIFU_TEST_OPTIMIZED=1 ./run-tests.sh --only PhotoArena
```

To include the actual native model benchmark with the already installed weights:

```sh
./tools/build-native-photoarena.sh
WAIFU_TEST_OPTIMIZED=1 WAIFU_ARENA_GROUP_BENCHMARK=1 \
WAIFU_ARENA_GROUP_REPORT="$PWD/docs/photoarena-group-review.json" \
./run-tests.sh --only 'PhotoArena recovery'
```

`photoarena-group-review-initial.json` retains the first unoptimized three-candidate smoke run. Its model disagreed about byte-identical copies; this prompted deterministic equivalence handling. `photoarena-group-review-v2.json` retains the first optimized run: the model treated the three candidates as equivalent despite a measured 7,000-pixel overlay, and the planner abstained. The next run (`photoarena-group-review-v3.json`) brought the changed area into focus and detected the cover, but confused a context guide for a source edit. The final contract leaves guides out of model inputs and distinguishes an opaque overlay from a tonal change. `photoarena-group-review.json` records that run, including independent output comparisons, preprocessing, inference, export and verification timings, client footprint and native worker memory/Metal samples. These tests use a NASA photograph with conventional overlays, not labeled AI edits. The previous paired benchmark in [PhotoArena-optimization.md](PhotoArena-optimization.md) is a different workload; there is no claimed direct speedup against it. The legacy pair grammar, reversal path, Metal fusion, F16 weights and limits remain available. No inference uses Python.

Noninteractive fixtures cover selection and abstention, identical inputs, precise mask geometry, registration rejection, exact real-pixel recovery, conservative seam rejection, source indices, encrypted group transport, cached native requests and byte-preserving export. No interactive UI tests were created or run.

The final Apple M3 (16 GB) run reviewed three 896×1134 files in **30.11 s** end to end. Preprocessing took 0.275 s and output comparison against all three sources took 0.212 s. The native worker peaked at 1.619 GB physical footprint; the sampled headless client peaked at 0.107 GB. Their conservative sum is 1.726 GB, not a simultaneous GUI measurement. The worker reported 3.218 GB Metal allocation, zero compressed bytes at the sampled replies, and zero on its raw swap-in ledger. Metal allocation is a separate counter, not additional physical footprint.

The pixel inspection found exactly 7,000 changed pixels in each altered copy. MiniCPM recognized the opaque overlay but also attributed it to the clean candidate. The planner withheld a unique recommendation; the fallback file happened to match the known clean input byte-for-byte. This is a model-quality failure on the fixture, not a successful automatic winner claim. Separate fixture tests supplied known regional labels to the planner and verified an automatic complementary-source composite against the known original. The final PhotoArena suite reported 28 passed, zero failures and one skipped model download; the optional 40 MP export was not enabled. Native Metal fusion, sampling and canvas-position correctness checks also passed.

## Limits

MiniCPM findings are uncalibrated. A shared watermark can be missed even when every source was reviewed. The application cannot certify original provenance, infer a particular editing tool, or recover detail obscured in every source. The benchmark does not establish AI-edit, denoising or upscaling detection accuracy.

Global alignment remains the basis for comparison; no local deformation is applied. Region packages are bounded by tiles and can split one object across tiles. Exposure, compression, geometric resampling and real edits can all cause red differences. The displayed mask threshold does not rewrite source pixels or retroactively alter earlier model findings. Small changes can be harder to see in an overview; native crops retain reference sampling.

A composite still uses the reference canvas, transforms donors as required and writes 8-bit sRGB TIFF. High-bit-depth and wide-gamut source containers are archived intact. Boundary agreement is checked, but there is no feathering, color matching, HDR composite pipeline, or general seam optimization. The full output and source-map buffers are held for TIFF encoding; native inspection is tiled. The 16-source and 160MP admission limits are implementation bounds, not performance targets proven for every combination. No whole-GUI process tree or ANE placement measurement is claimed.
