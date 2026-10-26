# TruFor M4 experiment notes

Read this before repeating a similar TruFor/Core ML/MLX/ANE measurement. The
measurements below were taken on `thefreebao.local`, an Apple M4 Mac mini with
16 GB RAM, using the same TruFor weights (`3b2c7146303e984c43b51f26e38d093c8b2ebba9a13e41bb029a98aa5b29783c`).
The production app remains on the validated MLX FP32 path; Core ML and direct
ANE findings here are experiments. These are worker measurements, not claims
about image-level forensic accuracy.

## 512×512

Both workers used the same 512×512 input and emitted/materialized the spatial
maps. Native MLX FP32 had a median request peak of **815,924,184 bytes** (778.1
MiB) over three requests. The fixed 512 Core ML FP16 graph
with FP32 detection statistics and score heads had a worker physical peak of
**116,671,256 bytes** (111.3 MiB). The three ANE-related services together
reached **227,488 KiB RSS** (222.2 MiB). Adding worker physical footprint to
service RSS gives a rough **333 MiB** combined estimate; it is not an exact
whole-system physical-footprint measurement.

The earlier **7.65×** worker-only comparison omitted ANE services and should
not be reused as a whole-stack result. The 512-only Core ML package cannot be
used to predict memory at larger input sizes.

## 1024×1024

The same native MLX FP32 path completed three evidence requests. Physical peaks
were **919,372,856**, **920,880,184**, and **920,896,568 bytes** (median
**878.2 MiB**); all returned score **0.08974204**.

Two flexible Core ML packages support this shape by declaration: a fully FP16
graph and an FP16 graph with FP32 detection statistics/score heads. Both
triggered the measurement worker's **1.6 GB** physical-footprint stop. The
fully FP16 worker reached **1,617,823,760 bytes** before it was stopped, with
ANE-service RSS at up to **101,920 KiB** (99.5 MiB). The additive worker-plus-
service estimate at the stop was about **1.60 GiB**. The Core ML run did not
produce a final report or score, and its completion below the 2 GB target was
not established. Treat this as a near-budget failure, not a completed benchmark.

The first 1024 probe used `MLMultiArray` NSNumber subscripting for every output
element. That probe itself grew to several GiB and crashed; discard its memory
figures. The later direct-buffer probe removed that harness overhead and still
hit the 1.6 GB guard. The output-map check should continue to use typed buffers,
not per-element object access.

## Measurement and cleanup rules

- Use the same opaque PNG, preprocessing (`uint8 / 256`), weights hash and
  output mode for each backend. The 1024 fixture used SHA-256
  `3907df2497dbc29c50c82c6938c71b5ea30046c6d6663f68729818da2b237917`.
- Measure the worker's `task_vm_info` physical footprint and sample all three
  ANE services (`aned`, `aneuserd`, `ANECompilerService`) during Core ML runs.
  macOS denied physical-footprint inspection of the root-owned services to the
  logged-in account, so their RSS is only a proxy; do not present the sum as an
  exact system-wide number.
- Use a safety stop below 2 GB. Do not rerun the same flexible 1024 Core ML
  graph without a new memory-reduction hypothesis or implementation change.
- Keep inference in native Swift/Core ML or Swift/MLX. Python conversion and
  reference scripts stay offline and outside the inference chain. Do not add
  GPL/LGPL dependencies. Preserve the upstream TruFor nonprofit licence and
  notices.
- Stage temporary model packages and binaries under a uniquely named `/tmp`
  directory on the M4. Remove only that task's staging directory and compiled
  artifacts after recording results. Keep valuable `.native-runtime` model
  artifacts unless the user explicitly asks to remove them.
