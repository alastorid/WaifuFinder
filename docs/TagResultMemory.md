# Tag result memory

WaifuFinder can remember WD Tagger results between analyses. The cache avoids
running the tagger again for frames whose source video, analysis size, and tag
vocabulary still match. It stores numeric vocabulary indices and FP8 scores;
tag names are recovered from the installed vocabulary when a result is loaded.

## Settings and location

Open **WaifuFinder → Settings → General → Tagging memory**.

| Setting or statistic | Behavior |
|---|---|
| **Remember tagging results** | On by default. When off, cache results are neither read nor written. Turning it off does not erase existing entries. |
| **Delete results older than N days** | Defaults to 30 days. The value is clamped to 1–36,500 days, so zero and negative values are not accepted. Expiration is checked when the cache is refreshed, read, or written; there is no background timer. |
| **File location** | `~/Library/Application Support/Waifu/tag-results.db` |
| **Current size** | Physical file size, including the file header and any append history not yet compacted. |
| **Entries** | Number of current, unique video/frame results after duplicate history is resolved. |

The enable and retention preferences are stored in `UserDefaults`, separately
from the result file.

Preparing cache records (decoding full probabilities and compressing sparse
scores) runs on a background task. Disk indexing and writes run on the cache
actor, so saving a completed tagging run does not block the UI thread.

## Binary format

All integer fields are unsigned little-endian. The file is an append-only sequence
of records preceded by a four-byte signature:

```text
file = "WTR1" record*
```

Each record has a 52-byte fixed header, followed by the UTF-8 source path and a
compressed score payload. The header fields appear in this order:

| Width | Field | Meaning |
|---:|---|---|
| 4 bytes | `pathLength` | Number of bytes in the source path |
| 8 bytes | `sampleIndex` | Decoded-frame ordinal (`f000123` is index 123) |
| 8 bytes | `sourceSize` | Source video size in bytes |
| 8 bytes | `sourceModifiedMilliseconds` | Source modification time since the Unix epoch, in milliseconds |
| 4 bytes | `analysisWidth` | Width used to analyze the video |
| 8 bytes | `vocabularyFingerprint` | Numeric fingerprint of the ordered vocabulary and categories |
| 8 bytes | `savedAtMilliseconds` | Time this changed result was written, in milliseconds since the Unix epoch |
| 4 bytes | `payloadLength` | Number of bytes in the compressed payload |

The source path follows the header for `pathLength` bytes. It is needed to identify
the video; vocabulary/tag names are not written. The payload follows the path for
`payloadLength` bytes. Thus a record occupies `52 + pathLength + payloadLength`
bytes.

The record key is the complete source identity plus `sampleIndex`: resolved,
standardized source path, source size, source modification time, analysis width,
vocabulary fingerprint, and decoded-frame ordinal. A video replaced at the same
path, a different analysis width, or a different vocabulary cannot reuse that
entry. The vocabulary fingerprint is a 64-bit FNV-1a hash over the normalized
label bytes in model-head order, with a separator and numeric category for each
label. It does not contain the label text.

### Score payload (`WDF9`)

The current payload begins with the four-byte `WDF9` signature, followed by a
UInt16 uncompressed byte count and one zlib-compressed sparse vector. Storing the
decoded size lets the reader allocate an exact buffer. Before compression, the
vector contains:

| Width | Field | Meaning |
|---:|---|---|
| 2 bytes | `vocabularyCount` | Must equal 10,861 for the current WD Tagger vocabulary |
| 2 bytes | `pairCount` | Number of retained scores |
| 2 bytes | `vocabularyIndex` | Output-head index, strictly ascending within the payload |
| 1 byte | `scoreFP8` | Positive E4M3 FP8 probability for that index |

The final two fields repeat `pairCount` times. The score is saved only when it is
**greater than 0.35**. All other vocabulary positions are omitted and mean zero
when restored. The threshold is the fixed default used by the cache. E4M3 rounds
the retained probability to FP8 precision, so restored scores are compact
approximations rather than the original Float32 values.

Each frame payload is compressed independently. A result with no scores above
the threshold still has a valid payload with a zero pair count.
Older `WDF8` payloads remain readable; their decompression buffer grows only as
needed instead of reserving the maximum vector size for every frame.

## Read, write, and expiry behavior

- On cache checks, WaifuFinder memory-maps the file and scans record headers to
  build an in-memory index of the newest record for each key. Duplicate older
  records are ignored for lookup.
- Payloads stay compressed in the file. For the currently loaded video, matching
  payload bytes are read and decompressed when results are restored. The sparse
  pairs expand into a zero-filled vector, then the current vocabulary regenerates
  rating classes and visible trait names.
- The disk-cache actor keeps the mapped file and its key/offset index in memory;
  it does not build score arrays for other videos. During restoration, one
  matching payload at a time is expanded for result reconstruction, then released.
  The frame model retains an uncompressed Float32 runtime probability vector
  (`WDT2`, 43,448 bytes per tagged frame) for filtering and sorting. Only preparing
  a database save compresses the sparse FP8 values. Legacy compressed `WDT1`
  frame metadata remains readable.
- While tagging is active, completed batches stay in the session's runtime
  frames; they are not written to disk batch by batch. Cache-sync requests caused
  by published tag updates or newly added candidates are deferred while the run
  is active.
- When tagging finishes or fails, the accumulated results are synchronized once.
  On cancellation, WaifuFinder cancels and drains the worker first, includes any
  batch that completed as cancellation arrived, then synchronizes once. Analysis
  completion also checks for remembered results.
- Before WaifuFinder exits, it synchronously flushes the current tagged frames
  and any pending completed batch so the final update is not lost at shutdown.
- Rewriting an entry appends a new record only when its compressed score payload
  changed. Lookups use the newest record. Expired entries are removed during a
  cache check; compaction also removes superseded record history when it grows
  beyond the configured threshold.
- Expiration uses `savedAtMilliseconds`. An unchanged payload is not appended
  again just to refresh its age. Results for another video identity or vocabulary
  remain in the file until they expire or compaction removes superseded records.

The cache is a performance aid, not the source of truth for video analysis. If a
record is missing, expired, malformed, or does not match the current identity,
the corresponding frame is tagged again when tagging is started.

## Runtime filter-result reuse

For the currently loaded video, the session keeps a small in-memory LRU of the
frame indices matched by recent tag and numeric-filter combinations. A query key
includes selected normalized tags, tag threshold, vocabulary fingerprint, face
filter, face-selection threshold, and all numeric ranges. Repeating the same
query returns its cached indices directly.

When the query is narrower, the session can start from a compatible cached set:
the cached tags must be a subset of the selected tags, its face filter must be
the same or broader, and each cached numeric range must contain the current
range. It then applies the current predicates only to those cached frame indices.
This allows, for example, a `1boy` result to be narrowed to `1boy + 1girl`, or
an `Art 0...1` result to be narrowed to `Art 0.1...0.5`. The most specific
compatible set is chosen; exact matches take priority.

Tagging eligibility has a separate small cache keyed by its face filter,
face-selection threshold, and time/face/art/definition ranges. Rating and tag
score ranges do not affect which frames are sent to the tagger. Both runtime
caches store compact sample indices only and are not written to `tag-results.db`.
The display-filter cache is cleared when tag scores change. The tagging-eligibility
cache keeps its frame indices across tag updates because tag scores do not affect
tagger eligibility; its untagged counts are invalidated and refreshed. Both caches
are cleared when a new analysis replaces the current frame set. Sorting, grouping,
and the per-group cap are still applied after a cached filter result is retrieved.
