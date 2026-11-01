# Photos loading investigation — 2026-11-01

Read-only native macOS `sample` profiles were captured from the running app. No UI interaction or interactive tests were performed.

## Confirmed findings

- Before the scheduling fix, every progressive result snapshot scheduled full-source PhotoNearestDistance work, even when the Photo Distance control and MVD were disabled. That calculation changed from linear center distances to all-pairs nearest distances in b5ba7ce. Progressive snapshots now skip distance and rarity jobs; nearest-distance work is requested only by its actual sort/filter control.
- Live profiles caught `runAnalysisPipeline → completeAnalysis → TagResultMemory.remember → TagResultDiskCache.remember → pruneExpired`. Expiry cleanup traversed the entire cache once for each per-photo identity. The batch API now sweeps once, writes each identity with bounded per-photo buffers, and compacts once. Regular writes and shutdown use the batch API.
- One profile also caught MVD/Metal work after scoring. The optimized MVD commit did not modify VisionFrameScorer, FrameScorePipeline, or FrameProcessingConcurrency.

## Limits

The user reports ANE power falling from about 3 W to 0.5 W while the processed count is still advancing. The captured profiles covered completion/cache work and a later idle state, not that active slow-scoring interval. These findings do not establish the exact cause of that reported power transition or prove restored throughput.

## Verification

The cache regression writes 256 independent photo identities and verifies one expiry sweep per batch, unchanged-file deduplication on repeat writes, and preservation of all results. The progressive-loading regression verifies that growing prefixes launch no distance or rarity jobs and that explicit Photo Distance activation still schedules its calculation.
