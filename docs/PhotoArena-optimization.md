# PhotoArena memory-constrained inference optimization

This work retains the official MiniCPM-V 4.7-1B F16 weights, the exact prompt and
GBNF contract, all categories and reversal reconciliation, the 512-token output
budget, batch/ubatch 512, four CPU threads, Flash Attention, and FP32 recurrent
state. It introduces no new dependency or alternate precision/backend.

## Implemented changes

- One serialized child process owns a complete forward/reverse action. The model,
  projector and decoder are reused within that action and the process is reaped
  afterward. No persistent idle model pool or additional concurrency is introduced.
  Legacy workers remain usable through two requests on that same process.
- Image decoding/preprocessing and text tokenization run once per action. The
  pinned canvas processor produces one complete image chunk per input; reversing
  the two image chunks preserves text/role slots. Absolute canvas positions are
  regenerated with the new `n_past`, using the existing upstream decoder helper.
- Vision embeddings are reused only for the second image order, only within the
  action, with an 8MB absolute limit and a stricter limit of one quarter of the
  full-attention KV bytes removed by context sizing. Without that memory margin,
  the encoder recomputes. No cross-action image/embedding cache is added.
- Context capacity is the unchanged full prompt plus all 512 permitted output
  tokens, rounded to the allocator's 256-position boundary and bounded by the
  original 8192 positions. Larger contexts are destroyed before replacement.
  Each pass still synchronizes and clears all KV, recurrent and convolution state.
- Encoded request strings, decoded RGB bitmaps and the IPC input-line allocation
  are released before model computation. Preprocessed chunks are released after
  their final prefill. This removes overlapping input allocations instead of
  changing their precision, compressing calculation storage or using disk spill.
- Greedy sampling first validates the global argmax against the grammar. A legal
  argmax is exactly the result of full-vocabulary grammar filtering followed by
  greedy sampling. Rejected/nonfinite candidates take the unchanged full filter
  path; grammar state advances only on accept. This avoids constructing/decoding
  248,094 vocabulary candidates for each already-legal output token.
- One reusable four-position decode batch replaces per-token allocation/free.
  All four M-RoPE decode position components remain populated exactly as before.
- Swift streams one Base64 input at a time and uploads the pair once. A bounded
  POSIX reader handles coalesced frames without byte-at-a-time syscalls or a
  Foundation pipe-fill deadlock. Both pass watchdogs remain 300 seconds; queue
  serialization, per-action cancellation and worker reaping are retained.

## Actual Metal fusion

`Runtime/PhotoArena/patches/0001-gated-rms-metal.patch` applies reproducibly to
pinned llama.cpp revision `7071ba378afa96dcd69c41e1ec2b4912364dbee4` in its private
build checkout. The build checks whether the exact patch is already applied.

The Qwen3.5 graph exposes gated RMS normalization as RMS_NORM → weight MUL →
SWIGLU. A Metal fusion-table pattern recognizes this subgraph and dispatches
`kernel_rms_norm_gate_f32` or its float4 variant once, combining reduction,
normalization, learned scale, SiLU and gate multiplication. It reuses the original
RMS reduction, stride/broadcast metadata, SIMD-group reduction and threadgroup
sizing. It keeps F32 intermediates and performs no extra persistent packing.
Generic subgraph-use and allocation-alias checks remain in force. Unsupported
layouts/types use the ordinary backend path.

The pinned backend already supplies fused SSM convolution/SiLU, GDN/recurrent
cache writes, Flash Attention, norm/weight/bias fusion, gated MLP operations,
MMA matrix kernels and eligible matmul/residual fusion. Those were inspected and
retained rather than replaced with unvalidated duplicates. Repacking QKV/vision
weights or adding another resident backend would introduce additional copies.
A decoder-prefix cache is unsafe without complete hybrid recurrent/convolution
and multimodal position snapshots; none is introduced. Quantization, a precision
change and an ANE/Core ML split lack compatible numerical/placement evidence and
are not used to claim a speed or memory improvement.

## Correctness checks

- Six actual Metal gated-normalization shapes include width128 (real per-head
  dimension), odd-width scalar fallback, width1024, decode-sized and prefill-sized
  rows. Fused/unfused GPU outputs agree within relative/absolute 3e-6 and an
  independent F32 reference within 1e-5. Six actual fused GPU dispatches are counted.
- Forty-seven actual vocabulary/grammar token decisions exercise both legal-argmax
  and full-vocabulary fallback, including escaped evidence text, with exact equality
  to the original grammar/greedy selection.
- Independent reverse preprocessing matches reused role/text tokens and every
  canvas decoder position for unequal native crops (129 image positions) and a
  multi-tile overview/crop pair (521 image positions). No model forward pass or
  performance comparison is part of these checks.
- Background PhotoArena tests: 21 passed, no failures, one opt-in setup check skipped.
  New tests cover coalesced/truncated frames, paired/legacy worker protocols,
  streamed payload equality, reconciliation and cancellation. Existing encrypted
  remote transport/cache tests pass with the unchanged result contract.

Run `./tools/test-photoarena-fusion.sh` for the native correctness suite and
`./run-tests.sh --only PhotoArena` for integration checks. These do not interact
with the user's UI. The native suite uses the existing model only for vocabulary
and processor validation; it performs no neural-network inference benchmark.

## One final comparative session

The original worker was copied, hashed and archived before modification; the
original Swift judging implementation is from commit
`6a8de440b27413d8dac276d35ac30058541fbb02`. The final harness compiles both actual
Swift judging implementations with only model/worker path lookup redirected to
that temporary archive. It starts at the pre-encoded image boundary used by app
judging. PNG encoding and the SwiftUI GUI are outside the measured client path.
The original and optimized implementations use the same model files, images,
output budget, batches, CPU threads, queue concurrency and instrumentation.

`tools/photoarena-memory-observer.mm` is loaded identically into both headless
clients and their workers. It records whole-process physical footprint and its
kernel lifetime peak, sampled compressed memory, Metal device allocated bytes,
and the raw process swap-in ledger. No image, tensor or activation is written.
Metal/compressed-memory observations are sampled every 10ms; instantaneous spikes
between samples can be missed. Metal allocation is already part of the process
footprint and is not added to it again. Peak client plus maximum sequential worker
peak is a conservative bound, not a simultaneous peak measurement of the GUI app.

The single session covers native448, full overview and unequal native crop pairs,
with both input orders. Fresh production client actions are counterbalanced;
both versions also run repeated jobs in one native worker for warm/state-isolation
checks. Output comparison includes exact raw evidence/category dictionaries and
reconciled judgments. No cache purge or paging manipulation is performed: cold
means a new process/model instance, not a cold filesystem or Metal compiler cache.
A successful workload must be faster and have no greater client/worker bound or
worker physical peak, retain exact judgments, show stable repeat state, and execute
the new fused pipeline in the real model.

Preparation does not run inference:

```sh
./tools/prepare-photoarena-optimization.sh
node tools/benchmark-photoarena-optimization.mjs
```

A missing reference archive is rebuilt from immutable original source by
`tools/archive-photoarena-reference.sh`. All standalone executables and observers
live in temporary storage. The successful session deletes them; sources, final
JSON and scalar traces under ignored `.build/reports/` are retained.

The final measured report is `photoarena-optimization-final.json`. Results are
workload/hardware evidence, not a universal memory proof, a comprehensive accuracy
calibration, or a claim that every possible Apple Silicon optimization is exhausted.
No ANE placement or ANE speed is claimed. Model-quality uncertainty documented in
PhotoArena-model-runtime.md still applies.

## Final results — Apple M3

All three shipped-action gates passed in the single final session. The native
worker binary archived before editing was the actual reference, not an estimated
baseline. Raw category/evidence dictionaries and reconciled judgments matched
exactly; repeated warm jobs retained the same judgments. The new gated-RMS GPU
pipeline was observed in the real model, not only isolated kernel tests.

| Input | Original complete pair | Optimized complete pair | Speedup | Original client/worker physical bound | Optimized bound |
| --- | ---: | ---: | ---: | ---: | ---: |
| native448 | 19.47s | 8.36s | 2.33× | 1.695GB | 1.592GB |
| overview | 52.39s | 25.23s | 2.08× | 1.773GB | 1.664GB |
| unequal-native-crops | 20.11s | 11.24s | 1.79× | 1.703GB | 1.590GB |

These memory figures are conservative sums of production-path headless client
peak plus its maximum sequential worker peak, **not a whole SwiftUI GUI measurement**.
Worker physical peaks and Metal allocations were also lower on every measured
workload. The largest worker physical peak across the session fell from 1.817GB
to 1.682GB; the largest observed Metal allocation fell from 3.300GB to 3.214GB.
Metal allocations include mapped/reserved weight buffers and are not added to
physical footprint as separate resident RAM. Compression and raw swap-in ledger
measurements are retained in the JSON and scalar traces; no paging policy changed.

Cold first-judgment latency and aggregate prefill/decode throughput improved in
these production-action samples. The aggregate prefill metric includes vision
and preprocessing work, not only language-decoder token throughput. Both input
orders still receive a complete independent decoder pass.

### Warm limitation

Long-lived native-worker warm pair latency improved from 32.27s to 19.21s for
overviews and from 16.00s to 9.52s for unequal native crops. **The native448 warm
trial regressed from 15.02s to 17.72s.** Its judgments and memory still passed.
The cause of that slower warm sample was not established, and no extra performance
session was run to select a better result. This prevents a uniform warm-speed
improvement claim. The shipped application uses one worker per complete action
and does not retain a worker between actions. The acceptance flag in the JSON
explicitly applies to that primary action path, not to every warm latency metric.

No M4 hardware, GUI process-tree peak, ANE placement, broad accuracy calibration
or instantaneous sampled Metal/compression peak guarantee is established by this
session. A persistent worker pool or additional precision/backend changes require
new memory and quality evidence and are not enabled.

## Release validation: v0.1.105

The full background regression suite passed: **588 passed, 0 failed, 0 known bugs,
16 skipped** in 100.20s. Existing native fusion/sampling/position correctness results
and the one final performance session above are retained; no inference performance
session was repeated for release packaging. The dependency/license audit passed.

The preceding v0.1.104 release job exposed an older-SDK concurrency annotation gap
for CIContext. A narrowly scoped shared-context wrapper now expresses Apple's
thread-sharing contract without disabling concurrency checks across CoreImage.
The optimized runtime still uses the existing published F16 weights; no model
redownload or weight conversion is required. Local-network permission diagnostics
and the photo clip action label are also included in this release.
