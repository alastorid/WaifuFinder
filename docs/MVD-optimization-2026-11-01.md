# MVD optimization check — 2026-11-01

Hardware: Apple M3. Native release Swift, Apple Accelerate already linked by the app.
SME instructions are not an execution target for this hardware; no new dependency or hardware-specific runtime was added.

## Changes

- Reuse one vector-buffer access across seed construction and CPU distance-matrix generation.
- Cache only the distinct undirected edges used by the three native route validations. This remains O(n) storage (at most 3(n−1) entries), and does not change the Vision validation metric or acceptance rule.
- Avoid decoding feature prints for zero/one-photo input.

## Focused measurement

Command: `tools/benchmark-visual-sort.sh --distance-access`.
Executables are built in a temporary directory and removed on exit. Each result is the median of nine paired trials with 768-dimensional FP32 vectors, including symmetric matrix validation. The baseline uses per-pair buffer access; the optimized path holds the buffer once. Matrices match exactly.

| Photos | Baseline ms | Optimized ms | Ratio | Extra matrix bytes |
| --- | ---: | ---: | ---: | ---: |
| 128 | 0.412 | 0.401 | 1.026 | 0 |
| 256 | 1.645 | 1.605 | 1.025 | 0 |
| 511 | 6.343 | 6.256 | 1.014 | 0 |

This is a small matrix-kernel improvement, not an end-to-end library speed claim. The shared-edge regression verifies that an identical/reversed six-photo route needs five native edge evaluations instead of repeated evaluations, with the same deterministic result. Large-album routing retains its existing bounded-memory CPU/Metal choices.
