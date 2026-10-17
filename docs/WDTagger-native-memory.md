# WD Tagger memory and input pass — 2026-10-16

Start with the [performance matrix](../README.md#wd-tagger-performance-matrix).
The earlier [fusion measurements](WDTagger-native-tuning.md) established faster
inference; they also showed higher process memory than Core ML.

## Why fusion did not guarantee lower process memory

Logical traffic, live GPU allocations, allocator cache and whole-process physical
footprint measure different things. Removing a tensor's write/read can improve
speed while the runtime still queues other operations and retains their buffers.

The previous native report recorded 225.4 MB active MLX allocations after the run,
versus 870.9 MB peak active allocation. Its requested cache budget was 128 MiB.
The local MLX implementation retains input buffer references until command-buffer
completion; its soft allocation budget throttles queued work. That makes buffer
lifetimes and submission scheduling relevant even with fused arithmetic. The
model's stored weight file is only 224.5 MB. These observations identify a large
temporary-allocation component; they are not a breakdown of Core ML's internals.

Initial identical-math trials, using the original full-corpus mapping:

| Native configuration, batch 8 | Inference fps | Peak process MB |
|---|---:|---:|
| Previous 768 MiB / cache 128 MiB | 15.25 / 15.17 | 972.8 / 956.3 |
| 512 MiB / cache 64 MiB | 15.35 | 755.2 |
| 384 MiB / cache 64 MiB | 15.02 | 599.9 |
| 512 MiB, fast synchronization trial | 15.27 | 755.4 |
| 512 MiB, 16 MiB command-buffer threshold trial | 15.46 | 759.1 |

These are exploratory runs. Small speed differences require confirmation.
All six runs produced byte-identical predictions on all 64 inputs. Runtime
synchronization/submission trials were not promoted from this sweep alone.

## Direct byte input

The app previously rendered RGBA bytes, deinterleaved and expanded them on the
CPU to RGB Float32, then copied the expanded input into MLX. The new path renders
directly into the shared input mapping. The patch-input Metal kernel reads RGBA
UInt8 and performs the existing normalization and patch routing on the GPU.

For batch eight the shared input shrinks from 19,267,584 to 6,422,528 bytes, and
the app's separate 802,816-byte RGBA staging array is removed. The worker's input
allocation also shrinks. Alpha is already composited onto white by preprocessing;
the model consumes RGB only. Weight format and output-head order are unchanged.
Worker protocol WDT3 prevents pairing the new app with an older Float32 worker.

The new byte kernel exactly matches Float32 patch preparation on two deliberately
different batches, including varying alpha bytes. All 139 kernel checks passed.
All 13 WD Tagger/app-adapter contract cases passed through the byte worker,
including tail batches, concurrent instances, cancellation and image orientation.

## Comparable process-memory measurements

The earlier model probes map the entire 64-frame corpus, whereas the app supplies
one batch at a time. New `--stream-input` modes explicitly own only one batch of
file-input data in both native and Core ML probes. File-backed mappings and private
read buffers have different accounting; do not subtract the corpus file size from
physical footprint or assume streaming lowers it. Compare the fresh process-memory
figures with each other, rather than mixing them with old input strategies.
Timing still covers prediction and GPU completion, excluding file input and bulk
output decoding. This is not whole-app throughput or host-app-plus-worker memory.

Reproduce with `node tools/run-wd-tagger-native-memory.mjs rgba-LABEL`; the command
requires the retained corpus, byte-input corpus and compiled Core ML probe. Data
and measured source identities are recorded with the final comparison report.
No additional GPU trace was recorded.

## Selected result

| Fresh streaming-input process | Prediction fps | Peak process MB |
|---|---:|---:|
| Core ML controls | 14.24 / 14.48 | 702.5 / 702.5 |
| Previous native Float32, 768 / 128 MiB | 15.28 / 15.51 | 998.2 / 998.4 |
| RGBA UInt8, 512 / 64 MiB | 15.36 | 755.7 |
| **RGBA UInt8, 448 / 64 MiB — selected** | **15.52 / 15.53** | **665.3 / 665.3** |
| RGBA UInt8, 384 / 64 MiB | 15.29 | 609.9 |

The selected default preserves approximately 15.5 fps and cuts peak process memory
by **33.35%** against the previous native configuration in the same fresh input
strategy. Native peak active MLX allocation fell from 870.9 to 565.8 MB. The result
also measured **5.29% lower peak process memory and 7.18–9.05% higher inference
throughput than Core ML**. Small additional model-speed changes versus the closing
native control are within run variation; a further inference-speed gain is not
claimed. Removing CPU input expansion is not timed by this inference-only probe.

All six native configurations produced byte-identical predictions on all 64 inputs.
Both Core ML controls matched the retained production Core ML predictions exactly.
All starts and ends were normal thermal state. Background builds prevented the
original 55°C cooling gate from completing; the fresh eight-process comparison
used the same bound of 65°C and sampled CPU scaled use <=40% for every process.
Core ML and previous-native controls bracketed the candidate trials. These are
separate processes rather than within-batch interleaved timings, and background
workloads and clocks remain a limitation.

Defaults are now shared constants in WDNativeTagger: 448 MiB scheduling budget
and 64 MiB cache. Fast synchronization, alternate command submission settings,
compact bias and alternative GEMM geometry remain unselected. The prepared
weights and label CSV are unchanged and remain in the public v2 model release.
The app and worker must be updated together for the WDT3 byte-input protocol.

[Raw comparison and source identities](wd-tagger-native-memory.json).
