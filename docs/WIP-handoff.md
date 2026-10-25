# WIP handoff — session record (2026-10-25)

Everything described here is **uncommitted work on `main`**. The commit decision
belongs to the repository owner. This note exists so a fresh session can resume
without re-deriving any of it.

## State of the two feature streams

**Image Lineage / SAFIRE** (Photos forensics, Git-style ancestry graph): complete
and previously delivered. Sources in `Sources/Waifu/Lineage/`, native runtime in
`Runtime/Sources/SAFIREMLX` + `WaifuSAFIRE`, docs in `docs/ImageLineage.md`.
Published package: `alastorid/SAFIRE-Swift-MLX`. Not installed locally — the
"published SAFIRE installs…" loopback test skips until the user installs the
model through Settings → Models.

**TruFor forensic scores** (Photos-only, second forensic model): complete,
verified, optimized. Feature doc: `docs/TruFor.md`; measured evidence:
`docs/TruFor-native-validation.json`; app code in `Sources/Waifu/Forensic/`,
runtime in `Runtime/Sources/TruForMLX` + `WaifuTruFor`, publisher
`tools/publish-trufor.py`, descriptor `tools/trufor-release.json`.

## TruFor facts a new session must not re-derive

- **Distribution**: `alastorid/TruFor-Swift-MLX` @ revision
  `80cdfb30d990a115b309793851435b8907ecd648`, immutable path
  `releases/trufor-native-v1-46b5190799530f2f`, public, card licence
  `grip-unina-trufor-non-commercial`. The user explicitly authorized
  redistribution under the GRIP-UNINA licence (credit + licence compatibility).
  Weights stay `isNonCommercialRestricted` (commercial builds exclude them).
- **Verified**: clean anonymous download with per-file hash checks; bit-identical
  native inference from the downloaded copy; opt-in installer loopback test
  passes: `WAIFU_TEST_TRUFOR_INSTALL=1 ./run-tests.sh --only "TruFor integration"`
  (needs `dist/` built first). Suite: 465 passed / 0 failed / 11 fixture skips.
- **Runtime profile (measured, paired)**: per-shape MLX compile with the noise
  body outside the compiled graph; FP16 weights+activations by default with FP32
  islands (all LayerNorms, all BN incl. DnCNN rows, detection head);
  `linear_fuse` kernels pre-sliced at load; noise tile default 1024. Ratios:
  FP16/FP32 = 0.919 (M3) / 0.922 (M4); quiet-machine M3 1.79 s → 1.30 s; M4
  1.43 s → 1.02 s.
- **FP16 measured deviations** vs official FP32 reference (516×644 fixture):
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
- Fixture-gated skips (SAFIRE/TruFor installs, Image Lineage photographic
  fixtures, real video inference) resolve when the user installs models and
  supplies fixtures; the TruFor opt-in test is proven green.
- If the official TruFor checkpoint is ever re-converted, bump
  `TruForConfiguration.weightsSHA` and re-run the whole verification chain
  (publisher asserts the pinned source SHA).
