# WD Tagger: overlap thumbnail fetching with inference — 2026-10-16

Start with the [performance matrix](../README.md#wd-tagger-performance-matrix).

## Photos input update — 2026-10-17

Still photos now use a dedicated PhotoKit request with a 448-pixel longest edge,
preserving aspect ratio. The decoded CGImage goes directly to tagging, bypassing
the 512-pixel scoring/grid JPEG. When its longest edge is already 448, preprocessing
adds white padding without running a scaling filter. Smaller local fallbacks still
resize to model geometry. Photos tag results use a new cache identity.

Image preparation runs off the main actor, with at most four requests in flight
and one following batch prepared during inference. At most two batches of eight
decoded images are retained. Video inputs continue using bounded JPEG decoding.
The measurements below predate this Photos path; no new throughput gain has been
measured for it.

## Selected change

On-demand tagging now fetches **one following batch** while the native worker
processes the current batch. This overlaps thumbnail cache misses and video-frame
regeneration with inference. The current image preparation stage retains at most
two batches of decoded images; it does not queue a whole album or video.
The native batch size remains eight.

The queue is marked finished only after completed results have been committed.
Newly queued frames are rechecked while draining. A failure fetching the following
batch preserves the current successful result. Cancellation joins the pending
fetch before releasing the model.

Probability decoding now copies whole rows into aligned Float storage, replacing
one unaligned scalar load and conversion per output head. It preserves every bit
on the two supported little-endian macOS architectures, including unaligned Data.

## Confirmed M4 result

Mac mini M4, 16 GB, macOS 26.4.1. A public synthetic 1920×1080 H.264 source supplies
64 requested frames. The **production thumbnail regenerator** rebuilds every
thumbnail at analysis width 512. Both paths use the same native kernel, weights,
batch-eight inputs, 448 / 64 MiB budgets and bulk probability decoder.

| Tagging-stage path | Frames/sec, two fresh processes | Peak combined process MB |
|---|---:|---:|
| Serial thumbnail fetching, controls | 14.362 / 14.379 | 726.8 / 727.6 |
| **One batch ahead — selected** | **20.026 / 20.509** | **729.4 / 733.0** |

This is **39.3–42.8% more tagged frames per second** across the measured bounds.
Additional combined footprint was **2.6–6.2 MB**, under 0.9% of the control
footprint. The model's arithmetic, weights and output order are unchanged.

The timer includes thumbnail retrieval/regeneration, JPEG decoding, Core Image
preprocessing, native worker inference and IPC, vocabulary processing and encoding
the saved probability payload. It excludes warmup, interactive UI, session frame
lookup/progress publication, full-video scoring/grouping and export. This is a
tagging-stage component benchmark, not a whole-video analysis measurement.

All 64 encoded outputs matched across serial/prefetch configurations by SHA256,
and all three sweeps in each process were compared byte for byte.

### Cached thumbnails

When the compressed thumbnails already exist in memory, there is little fetching
work to hide. The initial controls measured 22.091 fps serial and 21.765 fps with
prefetch. A reverse-order confirmation measured **21.697 fps with prefetch and
21.735 fps serial**, effectively tied. No cached-throughput improvement is claimed.
The cached fixture constructs JPEGs from the retained RGBA corpus before timing;
its memory includes residual fixture-construction allocations. Compare footprints
within each fixture mode, rather than between cached and video-backed modes.

The earlier [model-only M4 comparisons](WDTagger-native-m4.md) remain separate:
the current native kernel was 11.0–12.2% faster than Core ML there. This pass does
not measure a new model-only Core ML advantage.

## Shared input experiment: not selected

An optional worker path wraps the shared file mapping in a persistent MLX/Metal
input buffer. It removes a 6,422,528-byte input allocation/copy per batch. A pointer
alias check detects MLX's fallback-to-copy behavior; copied snapshots are never
reused after the app overwrites the input. A retained mapping owner protects GPU
buffer lifetimes, including error paths. WDT3's response still acknowledges that
GPU reads are complete before the app writes another batch.

Six M4 worker processes alternated copy/shared/shared/copy/copy/shared. Every
64-frame sweep, alternating input order and tails of 1, 5 and 8 frames matched the
retained native predictions exactly:

| Input mode | Prediction + IPC fps | Worker peak MB |
|---|---:|---:|
| Copy controls | 23.396 / 23.042 / 22.883 | 654.6 / 654.7 / 654.7 |
| Shared prototype | 22.886 / 23.077 / 22.814 | 648.1 / 648.1 / 648.2 |

The M4 saved about 6.5 MB, with no established speed gain. An earlier M3 trial did
not reproduce the memory saving consistently. **The app keeps copy input by
default**. The shared implementation remains experimental through `--shared-input`
or `WAIFU_WD_SHARED_INPUT=1`; `--copy-input` forces the control. The final pipeline
result above uses the selected copy path. No further M3 tests were run after the
request to restrict testing to the M4.

## Validation and measurement limits

- The final app's signature passed deep, strict verification on the M4.
- Its packaged worker passed all 64 retained inputs, repeated overwrites,
  reversed input order, tails 1/5/8 and clean EOF on the M4.
- All **18 WD Tagger/contract tests passed on the M4**, including ordering,
  bounded overlap, cancellation, a growing queue, completed-result preservation,
  unaligned probability decoding and native adapter fidelity.
- No additional GPU trace was recorded. The Metal model kernels are unchanged.
- Each M4 comparison started in normal thermal state, GPU ≤55°C and sampled
  maximum P/E scaled CPU usage ≤40%. macOS background services and clocks still
  vary. Controls and candidates ran in separate fresh processes.
- Memory is the peak simultaneous sum of the benchmark parent and direct-child
  physical footprints, sampled every 10 ms with `proc_pid_rusage`. Shared mappings
  may count in both processes; this is not unique DRAM usage. OS codec/XPC service
  memory is excluded. MB means decimal MB.
- Cache miss rate, source codec, seek pattern, thumbnail size and other workloads
  affect the benefit. These results do not establish an absolute performance
  ceiling or an improvement for every input workload.

The first app sampler attempt was discarded after correcting its interpretation
of the child-PID count. Only corrected measurements appear in the tables and raw
report; the sampler now requires observed worker memory before accepting a run.

## Reproduction and published resources

These measurements describe the historical v4 source checkpoint. The clean repository
does not republish its checkpoint archives or test video. Build the current native
worker from source and generate the synthetic video with the command below.
Download the app from [current releases](https://github.com/alastorid/WaifuFinder/releases/latest).
Model weights and labels are pinned by [`tools/wd-tagger-release.json`](../tools/wd-tagger-release.json)
to Hugging Face. The installed app and settings on the test machine were left untouched.

Build the native benchmark tools:

```sh
./tools/build-wd-tagger-pipeline-benchmarks.sh
```

With the built runtime, weights/CSV directory and generated synthetic video:

```sh
.build/wd-native/pipeline-tools/app-benchmark \
  --model /path/to/model-directory \
  --worker .native-runtime/waifu-wdtagger \
  --video /path/to/tagging-source.mp4 --width 512 \
  --prefetch --report /path/to/prefetch.json
```

Omit `--prefetch` for the serial control. Each fresh process warms up, then measures
three complete 64-frame sweeps. Generate a new synthetic video with:

```sh
./tools/make-synthetic-video.sh tagging-source.mp4 1920 1080 30 10
```

The historical video is not distributed; encoder versions can produce different
files and hashes. The current generator uses Apple system frameworks and a new
original drawing pattern, so it does not reproduce the historical clip byte-for-byte. Cached/worker corpus checks used local
64-frame input and reference files. Those were separate from the synthetic
video experiment.

The [remote app matrix](../tools/run-wd-tagger-app-matrix.mjs) reproduces the
standalone cases in an isolated root with `bin-copy/`, `bin-shared/`, `bin/` and
`data/`. Its `selected` mode takes the earlier matrix label to verify output
identities; `cached` reverses the cached comparison order. The
[worker matrix](../tools/run-wd-tagger-worker-matrix.mjs) checks copy/shared input
using the same executable. All inference and image work is native.

[Full raw samples, hashes, telemetry and validation](wd-tagger-native-pipeline.json).

## UInt8 output transport

The production worker uses WDT4 readiness. A request byte `0x80 | count` selects
CPU quantization inside the worker. Each response contains `count * 10,861`
UInt8 values. The first four values in each row are the four rating probabilities, without a winning-class
byte or a storage cutoff. Remaining heads use the fixed trait storage cutoff.
The app packs these bytes into sparse LZ4 rows directly, and remote protocol v2
sends those same rows. The session keeps the encoded bytes instead of quantizing
again. Plain count requests retain Float32 output for numerical model validation.

`waifu-wdtagger --check-quantization` checks CPU output codes on an eight-row
synthetic batch, including four low rating scores and values immediately around
the trait cutoff. It needs no model weights and opens no windows. It also verifies
tails and invalid inputs and reports Float32 readback plus validated CPU
quantization timing.

On the development Apple M3, the fair synthetic comparison measured 0.216 ms
for Float32 readback plus CPU quantization and 0.347 ms for GPU quantization plus
UInt8 readback (31 runs, median, 86,888 scores). The slower GPU quantization pass
was removed. The production CPU path, including score validation, measured
0.225 ms in its self-check. Production reads Float32 output once and quantizes only real rows
on CPU inside the worker. IPC still transfers one byte per score; the app and
remote server reuse the UInt8/LZ4 payload without quantizing it again.

An eight-frame sparse remote-response fixture measured 347,593 Float32 payload
bytes versus 1,049 bytes of UInt8/LZ4. Reply-payload measurements exclude uploaded
images and transport headers.
