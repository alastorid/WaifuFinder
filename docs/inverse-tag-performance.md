# Inverse tag snapshot and tag-panel performance

Measured 2026-11-03 on Apple M3, macOS 27.0.1, optimized Swift 6 production sources.

The creation action now computes the complement directly from the ready published
summary. It does not decode photos, recalculate filtering, rebuild the grid, or
recalculate the snapshot later. The generated set is stored on the chip; changing
Want/Unwant changes only matching semantics. Creation is unavailable while a new
summary is pending, so the action cannot capture an obsolete summary by accident.

The AND-only summary can include more photos than the visible OR-filtered grid.
This is intentional: creation snapshots the summary shown by the tag browser.
Every currently shown photo is covered by that summary. Newly excluded tags are
absent from it, so creating the chip preserves the current grid. Subsequent filter,
threshold, source and layout changes evaluate the frozen clause normally.

## Creation latency

Run `./tools/benchmark-inverse-tags.sh`. The executable and fixture files are
created in temporary directories and removed on exit. It performs background
session/model work without windows, UI automation or photo-library access.

The benchmark uses the actual 8,106-general-tag vocabulary shape, compressed
synthetic score fixtures with 32 retained general scores per photo, and waits for
the current summary before measuring `generateUnwantedTagGroup()`. Timings include
synchronous publication and tag-browser snapshot preparation. They exclude SwiftUI
rendering; no interactive scroll or frame-rate tests were performed.

| Shown photos | Creation | Frozen inverse tags |
| ---: | ---: | ---: |
| 1,000 | 1.638 ms | 7,947 |
| 10,000 | 1.632 ms | 7,947 |
| 100,000 | 1.754 ms | 7,947 |

Each case asserts unchanged shown-photo IDs and unchanged grid revision. These
are one measurement per fixture size, not percentile or universal latency claims.
All measured creation calls are below the requested 333 ms limit.

## Scrolling and later filtering

- `TagBrowserSnapshot` prepares inverse visibility, alphabetic lists, OR lists,
  available selections, rarity ranking and rarity bands only when summary/filter
  inputs change. View evaluation reads the prepared arrays.
- `TagChipFlow` caches intrinsic label measurements and wrapping arrangements by
  width. Scroll position is not a cache input. Size and placement passes reuse the
  same measurements; content/size changes invalidate them.
- Summary publication assigns the final arrays once. It no longer computes and
  replays a potentially quadratic collection difference on the UI actor. SwiftUI
  retains tag identities through the existing stable tag IDs.
- Later filtering compiles normalized names and vocabulary indices once per batch,
  then reads only retained sparse scores per photo. The previous path repeatedly
  normalized and looked up thousands of absent inverse tags for every photo.

## Validation

`./run-tests.sh --only 'tag summary'`: 33 passed, 0 failed. Coverage includes dense
reference parity across thresholds, saved tags, unknown heads and invalid payloads;
snapshot visibility and rarity; frozen mode changes; wrapping geometry; creation
without a grid revision; later filtering and cache invalidation. These are
noninteractive model/geometry checks, not interactive UI tests.

`./build.sh` completed successfully at `dist/WaifuFinder.app`.
