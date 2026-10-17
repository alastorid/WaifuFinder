# Analysis result memory

WaifuFinder can reuse per-frame Vision results for a video that has already
been analyzed. A matching cached frame skips the aesthetics, face-quality, and
FeaturePrint requests. The video is still decoded to recreate thumbnails, and
grouping and ranking run again so current settings take effect.

## Settings and location

Open **WaifuFinder → Settings → General → Analysis memory**.

| Setting or statistic | Behavior |
|---|---|
| **Remember analysis results** | On by default. When off, results are neither read nor written. Turning it off keeps existing entries. |
| **Delete results older than N days** | Defaults to 30 days. The value is limited to 1–36,500 days; zero and negative values are not accepted. Expiration runs when the cache is refreshed, read, or written. |
| **File location** | `~/Library/Application Support/Waifu/analysis-results.db` |
| **Current size** | Physical file size, including its header and any append history not yet compacted. |
| **Entries** | Number of current unique video/frame results after duplicate history is resolved. |

The enable and retention preferences are stored in `UserDefaults`, separately
from the cache file.

## Stored data

The cache keeps raw per-frame analysis inputs used by ranking and grouping:

- Aesthetics score (`Float32`)
- Selected face quality (`Float32`, when a qualifying face exists)
- Face definition score (`Float32`, when available)
- Detected face count and selected face area fraction
- Encoded FeaturePrint data, when available

It does not store final group assignments, combined group-relative ranks,
thumbnails, or tag text. Group-relative ranks depend on the frames in each group
and the current ranking settings, so those are recalculated from the remembered
scores. No image pixels or vocabulary labels are written here.

## Binary format

All integer fields are unsigned little-endian. The file is an append-only
sequence preceded by the four-byte signature `WAM1`. Each record has a 48-byte
fixed header, followed by the UTF-8 source path and a compressed frame payload.

| Width | Field | Meaning |
|---:|---|---|
| 4 bytes | `pathLength` | Number of bytes in the source path |
| 8 bytes | `sampleIndex` | Decoded-frame ordinal (`f000123` is index 123) |
| 8 bytes | `sourceSize` | Source video size in bytes |
| 8 bytes | `sourceModifiedMilliseconds` | Source modification time since Unix epoch, in milliseconds |
| 4 bytes | `analysisWidth` | Width used to decode the frame |
| 4 bytes | `minimumFaceAreaBits` | Exact Float32 bit pattern of the face-selection threshold |
| 8 bytes | `savedAtMilliseconds` | Time this changed result was written, in milliseconds since Unix epoch |
| 4 bytes | `payloadLength` | Number of bytes in the compressed payload |

The record key is the resolved source path, source size and modification time,
analysis width, face-selection threshold, and sample index. Replacing a video at
the same path or changing analysis width or the face threshold causes a cache
miss. Quick and Full modes can reuse overlapping sample indices because these
raw frame results do not depend on sampling mode. Grouping thresholds, group
size, and ranking weights do not invalidate the cache.

The current payload starts with `WAF2`, followed by a UInt32 uncompressed byte
count and zlib-compressed bytes. The count lets the reader allocate an exact
decompression buffer. Before compression the payload contains a Float32
aesthetics score, a flags byte, a UInt32 face count, optional Float32 face
quality/definition/area values in flag order, and an optional UInt32 FeaturePrint
length followed by its encoded bytes. All stored scores preserve their Float32
bit patterns. Legacy `WAF1` records remain readable with a buffer that grows only
as needed.

## Read, write, and expiry behavior

- The disk cache memory-maps the file and indexes record headers. Matching
  payloads are copied in their compressed form; each frame is decompressed only
  when it reaches a scoring worker.
- On a cache hit, the scoring worker rebuilds an `AnalyzedFrame` using the
  current decoded frame's timestamp and thumbnail aspect ratio, plus cached
  analysis scores and FeaturePrint. It still builds the in-memory thumbnail for browsing.
- A record can be reused only for a matching source identity and sample index.
  Missing, malformed, expired, or incompatible records fall back to Vision.
- Results are saved after analysis finishes. If analysis is cancelled or fails
  after producing frames, those completed frame results are saved once as the
  run exits. Enabling memory while an analyzed video is already open also saves
  its in-memory results. App termination flushes the current completed frames.
- Rewriting a frame appends a record only when its compressed payload changed.
  The newest record is used; expired entries and superseded history are removed
  during compaction.

This cache is a performance aid. The video remains the source of truth, and a
cache miss simply causes the frame's analysis to run again.
