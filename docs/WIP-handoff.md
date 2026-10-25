# WIP handoff — session record (2026-10-25)

**Review correction:** the FP16-default instructions below are historical.
Production now pins FP32 (`trufor-fp32-v2` execution capability) and separates its
scalar cache from the old precision. See `TruFor.md` review corrections. Metal
renderer and persistent scalar-cache bugs have also been fixed; follow current
regression results rather than assuming the earlier completion claim.

Everything described here is **uncommitted work on `main`**. The commit decision
belongs to the repository owner. This note exists so a fresh session can resume
without re-deriving any of it.

## State of the two feature streams

**Image Lineage / SAFIRE** (Photos forensics, Git-style ancestry graph): complete
and previously delivered. Sources in `Sources/Waifu/Lineage/`, native runtime in
`Runtime/Sources/SAFIREMLX` + `WaifuSAFIRE`, docs in `docs/ImageLineage.md`.
Published package: `alastorid/SAFIRE-Swift-MLX` @ revision
`6422600bd8a8b361ec4869bd597267b7553b5533`. The package is installed and
configured in the 2026-10-25 development workspace. The opt-in installer test
still depends on being explicitly requested; see `docs/Native-model-status.md`
for the current version, parity and performance summary.

**TruFor forensic scores** (Photos-only, second forensic model): complete,
verified, optimized. Feature doc: `docs/TruFor.md`; measured evidence:
`docs/TruFor-native-validation.json`; app code in `Sources/Waifu/Forensic/`,
runtime in `Runtime/Sources/TruForMLX` + `WaifuTruFor`, publisher
`tools/publish-trufor.py`, descriptor `tools/trufor-release.json`.

## TruFor facts a new session must not re-derive

- **Distribution**: `alastorid/TruFor-Swift-MLX` @ revision
  `80cdfb30d990a115b309793851435b8907ecd648`, immutable path
  `releases/trufor-native-v1-46b5190799530f2f`, public, card licence
  `grip-unina-trufor-nonprofit`. The user explicitly authorized
  redistribution under the GRIP-UNINA licence (credit + licence compatibility),
  and on 2026-10-25 a TruFor collaborator confirmed in
  [grip-unina/TruFor#36](https://github.com/grip-unina/TruFor/issues/36) that the
  pretrained weights are covered by the same licence as the code, which restricts
  use to nonprofit purposes, and that all redistributions retain the original
  licence and remain subject to its terms, including the nonprofit-use
  restriction.
  Weights stay `isNonCommercialRestricted` (commercial builds exclude them).
- **Verified**: clean anonymous download with per-file hash checks; bit-identical
  native inference from the downloaded copy; opt-in installer loopback test
  passes: `WAIFU_TEST_TRUFOR_INSTALL=1 ./run-tests.sh --only "TruFor integration"`
  (needs `dist/` built first). Suite: 465 passed / 0 failed / 11 fixture skips.
- **Runtime profile (measured, paired)**: production pins FP32 after the FP16
  experiment failed the strict score/spatial parity gates. The worker compiles
  per-shape graphs with the noise body outside the compiled graph, pre-slices
  `linear_fuse` weights and uses a 1024-pixel noise tile. Current warm 1024×768
  timings are 1.586 s (M3) / 1.108 s (M4); the earlier native FP32 path measured
  1.79 s / 1.43 s. FP16 measurements are historical only. RAM was not compared
  against upstream PyTorch. See `docs/TruFor.md`.
- **Historical FP16 measured deviations** vs official FP32 reference (516×644 fixture):
  score ≤ 0.002 (M3) / 0.0007 (M4), anomaly ≤ 0.012 (M3) / 0.032 (M4),
  noiseprint relRMS 0.087 (M3) / 0.236 (M4), masks identical at 0.5. FP32 strict
  path passes the 1e-5 score gate on both devices (score err 3.81e-06, identical
  on M3 and M4).
- **Rejected on evidence** (do not redo): custom direct 3×3 conv Metal kernel
  (92–96 ms/layer vs MLX 26 ms — weight reuse doesn't amortize at 64 channels;
  MLX conv is ~1.5× off compute bound, not memory-bound); joint batching beyond
  the 1M-pixel envelope (slower per image); attention query-chunk changes
  (neutral).
- **Env knobs** on the worker: `WAIFU_TRUFOR_PRECISION=fp32|fp16`,
  `WAIFU_TRUFOR_QUERY_ROWS=0|128|256|512|1024` (0 = no chunking),
  `WAIFU_TRUFOR_NOISE_TILE=128|256|512|1024`, `WAIFU_TRUFOR_PROFILE=1` (phase
  timing to stderr). Not installed locally: the user picks it up via
  Settings → Models (download) or Choose Existing for
  `.native-runtime/trufor-candidate-v1` (validated package with embedded
  `validation.json`).

## Environment cheat sheet

- **HF token**: `export HF_TOKEN=…` lives in `~/.zshrc` (the tool shell is
  non-interactive — source it explicitly). Visibility changes:
  `HfApi(token).update_repo_settings(repo_id=…, private=…)` — the REST route is
  `/api/models/<id>/settings`, not `/api/repos/settings`.
- **M4 host**: `thefreebao.local`, SSH BatchMode works. A workspace needs
  `Runtime/Sources`, `Runtime/Package.swift`, `tools/native-toolchain/`,
  `tools/build-native-trufor.sh`, `.native-runtime/mlx.metallib` (pinned
  checksum in the build script) and the model dir; rsync failures are silent
  with `-q`, verify afterwards. Clean the workspace when done.
- **Model pointers**: `~/Library/Application Support/Waifu/{safire,trufor}-model.txt`
  (paths only). App config dir is that same `Waifu/` folder; the app is
  unsandboxed, bundle id `com.alastorid.waifufinder`.
- **Parity tooling**: `tools/compare-trufor-activations.py` needs torch
  (available on this M3). Reference activations:
  `.native-runtime/trufor-reference-v1/`. A torch-free comparator pattern is in
  this session's history; both agree.
- **Tests**: `./run-tests.sh [--only SUITE]`; opt-in install tests need `dist/`
  built first and stage a fake app from it. Interactive UI tests remain
  prohibited (AGENTS.md).

## Build traps learned this session (warnings-as-errors app build)

- Swift's lexer rejects `x==.member` / `x=.member` and any whitespace-imbalanced
  `=` near a leading `.` — write `x == .member`, `x = .member`, or `x=.init` is
  also rejected; use `= .init`.
- `CIColorKernel(source:)` (GLSL CI kernels) is unusable under
  `-warnings-as-errors`; `TruForEvidenceRenderer` now uses a raw Metal compute
  pipeline (`TruForPreviewModel.swift`). `Data(bytes:count:)` is deprecated —
  copy via `withUnsafeMutableBytes` + `memcpy`.
- `MLXFast.metalKernel` takes the **function body only** (MLX generates the
  signature from input/output names; outputs are `out`, indexing via
  `thread_position_in_grid`); template args support Bool/Int/DType only — bake
  float constants into the source.
- `MLX.compile` array form: `compile { arrays in … } -> ([MLXArray]) -> [MLXArray]`
  for multi-input/multi-output graphs. Never `eval()` inside a compiled closure.
- Build.sh compiles the app only; native workers build via
  `tools/build-native-*.sh` into `.native-runtime/`, and `dist/` ships copies
  from there — rebuild the app after changing a worker.

## Open items

- Commit decision for the whole uncommitted tree (user's call).
- Untracked files `Sources/Waifu/UI/GoogleLensSearch.swift` and
  `Tests/GoogleLensSearchTests.swift` / `Tests/GroupRowNavigationTests.swift`
  belong to earlier work in this tree — untouched here, but they ride along in
  any commit.
- Optional installer and fixture suites can still be skipped unless explicitly
  enabled and supplied with the required fixtures. The published SAFIRE and
  TruFor install tests have passed; see `docs/ImageLineage.md` and `docs/TruFor.md`.
- If the official TruFor checkpoint is ever re-converted, bump
  `TruForConfiguration.weightsSHA` and re-run the whole verification chain
  (publisher asserts the pinned source SHA).

## ANE feasibility sweep (2026-10-25, measured on this M3)

User question: "can we remodel any of our models into ANE to run on ANE?"
Answer settled with measurements; do not re-derive. Repo was not modified
(read-only probes; downloaded weights noted below).

**Verdicts per model**

- **RealESRGAN x2plus — already on ANE.** Conv-dense, fixed tiles;
  0.175→0.050 s/image with hardware-counter confirmation (`docs/Uncover.md`).
  The one proven ANE win, and the archetype: ANE loves conv-dense fixed-shape
  CNNs.
- **WD SwinV2 tagger — feasible but fails the accuracy gate; keep native
  MLX.** Published `WDTagger.mlpackage` under `.cpuAndNeuralEngine`:
  Core ML places the real compute on ANE (all 24 SDPA, 100 linear, 53
  layer_norm, patch conv, sigmoid head = 951 ops ANE-preferred; 1,322
  plumbing ops "unknown"). Batch-8 on this M3: cpuAndGPU 12.5 fps / 693 MB,
  cpuAndNE 15.2 fps / 120 MB, cpuOnly 7.7 fps. ANE ≈ native MLX speed
  (14.5–15.5 fps on M3) at 5.8× less memory than cpuAndGPU. Blocker: against
  one torch fp32 reference, ANE mean Δp 2.1–2.6e-3 vs GPU 4.9e-4 (~5× worse);
  per-image top-15 worst 8/15 (uniform noise) and 2/15 (patterns) vs GPU's
  11 and 8. Gate is `MIN_TOP15_OVERLAP = 12` (`tools/convert-wd-tagger.py`).
  Core ML has no per-op placement/precision pinning, so "SDPA on GPU, rest on
  ANE" cannot be forced; re-conversion experiments have uncertain payoff and
  no speed win over the native path. Not worth redoing.
- **PGTFormer — already rejected** (recorded in
  `docs/pgtformer-optimization-benchmark.json` under "ANE"): fp32 fell back to
  CPU (1,059+ CPU-preferred ops), fp16 ANE compile exceeded the 6 GB budget
  (6.21 GB observed). `Runtime/Sources/WaifuPGTFormer/ComputeProfiling.swift`
  holds the reusable MLComputePlan probe pattern.
- **SAFIRE — poor fit.** ViT-B window+global attention (the op mix that
  degrades on ANE); already 26–33 ms/image median; porting the compiled MLX
  graph to Core ML is large work for no measured gap.
- **TruFor — poor fit.** Measured FP32 islands (all BN, LayerNorms, detection
  head) are load-bearing for the forensic gates (1e-5 strict score error);
  ANE compute paths are fp16/W8A8 only, no fp32.
- **SeedVR2 / FlashVSR — categorical no.** 4-bit quantized MLX weights,
  dynamic shapes/tiles, temporal control flow; ANE is fixed-shape fp16/W8A8,
  no branching.

**Gotchas a fresh session must not re-derive**

- Degenerate-input artifact: pure-black or flat/pattern frames produce
  near-tied tag probabilities, so ANY numeric difference reshuffles top-15 —
  even `cpuOnly` vs `cpuAndGPU` disagreed on the synthetic pattern frames.
  First ANE measurement (0/15) was this artifact, not ANE breakage. Adjudicate
  with the canonical uniform-noise draw (`np.random.default_rng(20261013)`,
  integers 0–255) and judge configs only against the SAME reference run in
  the same session.
- Torch-reference drift: the release recorded mean Δp 1.06e-5 / top15 15/15
  for cpuAndGPU; today's torch 2.14 + current timm reference gives cpuAndGPU
  mean 4.9e-4 / worst 11 on the same package and input. The reference moved,
  not the package — absolute cross-time comparisons are meaningless.
- ANE cold compile is ~150 s per load (ane d caches per signature); an app
  integration would need the fingerprint+OS-keyed compiled-model cache that
  PGTFormer already uses (`~/Library/Caches/Waifu/PGTFormerCompiled`).
- Original upstream weights downloaded for the torch-reference setup
  (`SmilingWolf/wd-swinv2-tagger-v3` model.safetensors, plus the /tmp probe
  scripts, the benchmark's compiled .mlmodelc copies, and the python
  e5bundlecache) were all removed in the same session's housekeeping;
  `tools/convert-wd-tagger.py` re-downloads the weights on demand.
- Probe scripts were /tmp throwaways (now deleted); the method survives in
  the pointers above (ComputeProfiling pattern + converter's
  `build_model`/`Tagger` + top-15 metric). The H15G (M3) ANE identity appears
  in Core ML compiler logs; the pasted M4 article covers H16G only.
