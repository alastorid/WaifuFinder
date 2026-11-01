# Profiling findings — compact summary

Source of truth for each item is the linked detailed doc. This file exists so a
reader can scan the current state of memory/performance investigations without
re-reading every raw measurement.

## Photos memory investigation — 2026-11-01
`docs/Photos-memory-profile-2026-11-01.md`

- Read-only `vmmap -summary` / `heap -s` / `leaks` inspection of a running process; no UI interaction, no restart.
- Physical footprint 1.6 GiB, peak 2.0 GiB. Default malloc zone ≈ 1.1 GiB allocated.
- Two largest heap allocations: 279,035,904 bytes each, reverse-referenced to native Espresso (Photos private framework) blobs. No established safe release path; no private-framework cache flushing or model/device/concurrency changes were made. These blocks are **outside app-controlled malloc** and dominate the footprint.
- Tag-summary retained dense general-tag byte rows (~53 MB across 6,468 UInt8 arrays) even when sparse CPU work was selected. **Fixed:** retain only populated columns; GPU rows zeroed/scattered into bounded Metal buffers; excess GPU rows reuse transient chunks.
- `leaks`: only 165,568 bytes (SwiftUI context-menu responder cycles) — not the driver.
- Same-album footprint after rebuild **not yet measured**; changes must not be described as verified restoration below 1 GiB.

## Photos loading investigation — 2026-11-01
`docs/Photos-loading-profile-2026-11-01.md`

- Read-only `sample` profiles; no UI interaction.
- Fixed: progressive snapshots no longer schedule full-source PhotoNearestDistance work when Photo Distance / MVD are disabled.
- Fixed: tag-result disk-cache expiry now sweeps once per batch instead of once per identity; writes use bounded per-photo buffers; compacts once.
- Unconfirmed: reported ANE power drop (3 W → 0.5 W) was not captured during the active slow-scoring interval; no proven cause or restored throughput.

## Memory management already in the app (self-managed allocation)

- `malloc_zone_pressure_relief(nil, 0)` via `TruForMemoryPolicy.releaseUnusedCPU()` — `Runtime/Sources/TruForMLX/MemoryPolicy.swift:18`.
- `Memory.clearCache()` — MLX allocator cache release after each inference pass (TruFor/SAFIRE/FlashVSR/SeedVR2).
- `MemoryMonitor` polls `task_vm_info.phys_footprint` every 20 ms; stops at 1.7 GB (`stopBytes`), hard budget 2 GB (`budgetBytes`) — `Runtime/Sources/NativeVideoIO/WorkerIO.swift:148-169`.
- Bounded `NSCache` cost+count limits everywhere (e.g. `PreviewPhotoSource` 128 MiB/9, `PhotosMedia.renditionCache` 64 MiB/512, `previewOriginals` 256 MiB/9).
- `autoreleasepool` around large decode/encode temporaries.
- Speculative full-size pixel decode gated to a 7-photo / 128 MiB window; larger originals stay warm as encoded.

## What could lower app-controlled memory further (trade-offs)

- Add a `DISPATCH_SOURCE_TYPE_MEMORY_PRESSURE` handler that calls `releaseUnusedCPU()` and shrinks `NSCache.totalCostLimit` values proactively (currently relief only runs at inference-step boundaries). Trade-off: more re-decode/re-fetch after pressure passes.
- Lower existing `NSCache` cost limits directly.
- Use purgeable/volatile VM (`vm_purgeable`) for rebuildable caches (thumbnails/previews) so the kernel reclaims under pressure.
- Extend Metal buffer/texture reuse pools beyond the tag-summary GPU rows.
- Tighter `autoreleasepool` scoping around large `Data`/image temporaries.

**Not recommended:** flushing Photos/Espresso private-framework caches — no safe release path was established, and those blocks are outside app-controlled malloc.
