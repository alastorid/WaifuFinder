# Repository instructions

## Interactive UI testing

The user requested on 2026-10-24 that routine work must not include interactive
UI tests, now or in future tasks. Do not run them unless the user explicitly
requests interactive UI testing again. Background builds, automated noninteractive
tests, rendering, and inference benchmarks may continue normally.

If interactive UI testing is explicitly requested again, the following approval
requirements still apply:

The user actively shares this computer. Save interactive UI tests as reviewable,
executable scripts and invoke them through `./run-ui-tests.sh SCRIPT [ARGUMENTS...]`.
The launcher itself shows a native macOS message box naming the exact script and
arguments, and executes that invocation only after the user clicks OK. Approval is
for that script invocation, not an agent, conversation, or general testing session.
Do not perform separate ad hoc UI actions under a script's approval. Another script
or rerun needs its own prompt. Cancel, no response, elapsed time, and dialog failure
must never authorize execution; do not add a bypass flag or cache approvals.
After an approved script exits, the launcher must show a native macOS notification
that the UI steps have finished and control is back with the user, including when
the test script exits unsuccessfully.
Background code edits, builds, and tests that do not interact with the user's UI
can continue without this prompt.

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
