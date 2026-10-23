# Tag summary computation

The tag browser ranks general traits by conditional average confidence over the
photos currently shown. For each tag, it sums probabilities **strictly greater
than the active threshold** and divides by the number of those probabilities.
Values at or below the threshold do not enter either the sum or the denominator.
Common tags must occur on every shown photo, including photos without tag data.

Only general traits participate. Ratings and character heads remain separate.
Older or damaged results can use their saved top tags for availability and common
membership, but contribute no invented confidence to the average. Unscored tags
sort after numerically scored tags; equal means use alphabetical ordering.

## Background work and reuse

`TagSummaryComputer` owns UInt8 general-tag vectors on its own actor. Its cache
keys are photo identifiers, probability payloads and vocabulary identity. Display
order and thresholds do not invalidate vectors. A changed probability payload
refreshes that photo's row; changing the vocabulary clears the vector caches.

Metal keeps reusable row buffers. A new view sends small row-selection masks, so
filtering and threshold changes reuse previously uploaded vectors. Modified rows
are copied again, released source photos free reusable slots, and overflow rows
are processed in bounded temporary chunks. Completely untagged photos do not
allocate probability rows, but remain in the common-tag denominator.

CPU score rows and their Metal mirror each have a 64 MiB budget per session.
CPU rows additionally retain populated-column indices within the same budget;
reductions visit only those columns. Metal score buffers use one byte per value,
a quarter of the previous Float32 matrix size, and overflow chunks contain at
most 8 MiB of scores. Both reducers sum integer score codes; the final conditional
means are converted to Double once per tag. Two boundary counters preserve exact
0.35 equality and just-above-cutoff scores. GPU chunks are bounded to prevent
UInt32 sum overflow. CPU and GPU produce the same counts and means.

Automatic selection considers frame count and retained-score density. Sparse
selections use CPU iteration over populated columns. Dense selections become GPU
candidates at 512 frames when the estimated retained-head count exceeds one
eighth of the general-head matrix. Missing Metal support, shader setup failures,
allocation failures or failed GPU commands fall back to the same conditional
average on the CPU. Cancellation exits instead of falling back or publishing an
obsolete result. Both paths run off the main actor and require only Apple's
system frameworks, with no additional packages or media dependencies.

The session caches completed labels by shown-photo membership, result revision
and threshold. Pure ordering changes reuse those labels. Pending jobs carry a
generation token, including when undoing a filter/threshold returns to a cached
result. Publishing a summary never rebuilds or re-ranks the photo grid.

## Verification

`./run-tests.sh --only 'frame arrangement'` includes native Metal/CPU agreement,
strict threshold boundaries, legacy and missing rows, multiple chunks, filtered
row masks, GPU slot reuse, edited vectors, bounded overflow, unavailable-GPU
fallback, cancellation and obsolete session results. These checks do not open or
control the user's UI.

`./tools/benchmark-tag-summary.sh` compiles an optimized background benchmark with
synthetic UInt8 tag data at multiple frame counts and score densities. It reports
cold decoding separately from warm CPU and Metal summaries. Warm measurements
include cache lookup, view selection, reduction and result construction; they
exclude label sorting and the app UI. Both CPU and Metal caches keep their actual
64 MiB limits, so large cases include overflow decoding, packing and GPU transfers.

Override its matrix with `WAIFU_SUMMARY_BENCH_COUNTS` (comma-separated positive
frame counts), `WAIFU_SUMMARY_BENCH_TAGS` (retained general heads per frame), and
`WAIFU_SUMMARY_BENCH_REPEATS` (at least three). For example:

```sh
WAIFU_SUMMARY_BENCH_COUNTS=512,1536,8192,32768,65536 \
WAIFU_SUMMARY_BENCH_TAGS=32,256,2048 \
WAIFU_SUMMARY_BENCH_REPEATS=3 ./tools/benchmark-tag-summary.sh
```

The 2,048-frame prototype comparison with 32 retained general tags per frame,
using optimized production code on Apple M3, measured a median warm CPU reduction
of 10.266 ms with Float32 rows versus 1.067 ms with sparse UInt8 rows. Metal fell
from 5.885 ms to 4.298 ms. Changing to bytes alone does not determine a crossover:
CPU work scales with populated scores, whereas the current Metal matrix scans
all 8,106 general heads. Hardware and score distributions affect the result.

### UInt8 CPU/GPU crossover on Apple M3

The expanded benchmark used actual 64 MiB cache budgets, a threshold of 0.35,
and median timings over three repetitions. The means agreed exactly between
CPU and Metal. Timings are milliseconds; dense overflow cases include decoding
and staging that cannot stay cached.

| Frames | 32 tags: CPU / GPU | 256 tags: CPU / GPU | 2,048 tags: CPU / GPU |
| ---: | ---: | ---: | ---: |
| 512 | 0.428 / 1.403 | 0.767 / 1.070 | 3.970 / 1.588 |
| 1,536 | 0.876 / 4.119 | 2.083 / 2.301 | 11.618 / 3.917 |
| 8,192 | 5.610 / 14.631 | 16.380 / 23.755 | 208.707 / 173.948 |
| 32,768 | 34.825 / 103.652 | 133.087 / 190.411 | 1,052.286 / 924.454 |
| 65,536 | 78.304 / 221.746 | 288.897 / 398.425 | 2,183.595 / 1,922.320 |

GPU already wins at 512 frames in the densest tested distribution. No GPU win
was observed for 32 or 256 retained heads through 65,536 frames. The automatic
policy therefore uses both density and frame count, rather than forcing every
large sparse album through a dense GPU scan.
