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

`TagSummaryComputer` owns decoded general-tag vectors on its own actor. Its cache
keys are photo identifiers, probability payloads and vocabulary identity. Display
order and thresholds do not invalidate vectors. A changed probability payload
refreshes that photo's row; changing the vocabulary clears the vector caches.

Metal keeps reusable row buffers. A new view sends small row-selection masks, so
filtering and threshold changes reuse previously uploaded vectors. Modified rows
are copied again, released source photos free reusable slots, and overflow rows
are processed in bounded temporary chunks. Completely untagged photos do not
allocate probability rows, but remain in the common-tag denominator.

Decoded CPU vectors and their Metal mirror each have a 64 MiB budget per session.
Overflow chunks contain at most 8 MiB of probability values, plus allocation and
transfer buffers. General-tag vectors for an album of 1,800
photos fit in both caches. The reducer uses compensated Float32 sums on the GPU
and combines chunk totals as Double. The CPU fallback accumulates in Double;
small floating-point differences can occur in very close confidence means.

Selections with fewer than 1,536 photos carrying tag data use the CPU. Missing
Metal support, shader setup failures,
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
synthetic native Float32 tag data. It reports cold decoding separately from warm
CPU and Metal summaries. Warm measurements include cache lookup, view selection,
reduction and result construction; they exclude label sorting and the app UI.

Optimized Apple M3 measurements with 8,106 general heads and a threshold of 0.35
(median of seven measurements with sparse synthetic probabilities):

| Photos | Cold decode + CPU (ms) | Cached CPU (ms) | Cached Metal (ms) |
| ---: | ---: | ---: | ---: |
| 64 | 3.913 | 0.451 | 0.649 |
| 512 | 28.803 | 2.519 | 1.635 |
| 1,024 | 56.615 | 4.974 | 2.977 |
| 1,536 | 84.361 | 7.387 | 4.762 |
| 1,800 | 98.512 | 8.650 | 5.533 |

The maximum mean difference in these cases was 0.000000021. The automatic cutoff
is conservative because dispatch overhead varied for smaller sets in repeated
runs. Device performance and actual probability distributions can change the
crossover; the benchmark can be rerun without operating the user's UI.
