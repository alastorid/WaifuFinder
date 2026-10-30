# Repository instructions

## Build artifact housekeeping

Use `dist/WaifuFinder.app` as the standard app output. Do not leave additional app
variants, superseded installers, or ad hoc benchmark executables in the checkout.
Build and run standalone benchmarks in temporary directories with cleanup on exit;
retain their source and measurement reports. Keep build-system-managed caches and
required native runtimes. Do not quit the user's running app for housekeeping.

## Interactive UI testing

The user permanently prohibited interactive UI tests on 2026-10-26: no
interactive UI tests may be created or run now or in any future task. Do not
create or run them, request approval for them, or use the UI-test launcher as a
workaround.
Background builds, automated noninteractive tests, rendering, and inference
benchmarks may continue normally when they do not interact with the user's UI.

## Licensing and media dependencies

GPL and LGPL software are prohibited in this repository. Do not add, link, bundle,
download, invoke, or depend on GPL/LGPL components in the app, build, tests, CI,
development tools, generated fixtures, or distribution. Prefer Apple system frameworks
or alternatives with licenses compatible with this project. Check both direct and
transitive licenses before adding any dependency; if a license is unclear, stop and
resolve it before proceeding.

Media fixtures use Apple-native generation. Keep app, build, tests, benchmarks and
CI independent of FFmpeg libraries and command-line tools. Do not add such dependencies.

## TruFor Core ML, MLX and ANE experiments

Before repeating or extending TruFor Core ML, MLX or ANE work, read
`docs/TruFor-M4-experiment-notes.md`. It records M4 measurements, rejected
memory probes, the 1024×1024 Core ML safety stop and the limits of counting
ANE-service memory. Keep Python out of every inference path; Python may be used
only for explicitly offline conversion/reference tooling. The nonprofit TruFor
licence and the GPL/LGPL prohibition above still apply.

Production TruFor, including the Noiseprint/noisemap body, must remain FP32
throughout in native Swift/MLX with its validated optimized kernels. Do not
reintroduce Core ML/FP16 companions, reduced-precision runtime overrides,
activation scratch files, file-backed mappings, compression or disk spill to
lower reported RAM. App/worker inputs and evidence travel through anonymous
pipes. Persistent weights and scalar result caches are not calculation storage.
Report whole-process physical footprint alongside MLX allocations, compressed
memory and process swap-ins; do not describe RAM-only app storage as disabling
macOS paging or claim universal optimality from limited benchmarks.

## WD Tagger ANE and Core ML trials

Before repeating or extending a WD SwinV2 ANE/Core ML experiment, read
`docs/WDTagger-ANE.md`, `docs/wd-tagger-ane-results.json`, and
`docs/wd-tagger-open-license-sample.json`. They record the direct-ANE block
failure, the CPU+ANE batch-8 parity failure, and the single-image batch-1
real-image measurements. The batch-1 CPU+ANE path was faster in the measured
sample but failed numerical and tag-decision parity, so it is not an accepted
backend. The shipped Core ML model remains fixed at batch 8 and the app remains
on Metal. Do not rerun the same benchmark without a new hypothesis, a relevant
implementation change, or different hardware/checkpoint; preserve the existing
measurements and add new evidence to these reports.

## Human-facing selection controls

Do not use dropdown list boxes, popup pickers, or dropdown menus to choose an
option in app settings, inspectors, or analysis controls. Show the available
choices directly with preset-style segmented buttons, wrapping button rows,
radio buttons, or another always-visible selection control. All TruFor display
modes must remain visible together; never hide them in a dropdown. Context menus
for actions (such as Open in Photos.app) remain appropriate.

Photo analysis rows put TruFor before SAFIRE. Each row keeps its enable toggle,
mode segments, progress and score/status inline. Do not put additional model
status labels or mode controls on separate lines; detailed explanations belong
in the row's tooltip.
