# Repository instructions

## Permission before interactive UI testing

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
