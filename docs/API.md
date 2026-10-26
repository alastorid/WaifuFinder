# WaifuFinder — layer API

The one file that says what the engine exposes and what the UI consumes. Both sides are
written against this document, so they can land in either order and still link.

Nothing here is a design suggestion; the engine **must** provide this shape and the UI
**must** use only this shape. If a change is needed, change this file first and both sides
follow.

## `WaifuSession`

```swift
@MainActor
final class WaifuSession: ObservableObject, ThumbnailStore {
    // Configuration
    @Published var settings: WaifuSettings          // persisted
    @Published private(set) var video: VideoInfo?   // nil until a file is dropped
    @Published private(set) var progress: AnalysisProgress

    // Results
    @Published private(set) var frames: [AnalyzedFrame]   // ascending by timeSeconds
    @Published private(set) var groups: [FrameGroup]      // descending by member count
    @Published private(set) var selectedFrameIDs: Set<FrameID>
    @Published var errorMessage: String?

    // UI state
    @Published var focusedFrameID: FrameID?
    @Published var showExportSheet: Bool
    @Published var previewFrame: AnalyzedFrame?     // non-nil while the zoomed preview is up
    @Published private(set) var uncover: UncoverSession? // exclusive editing overlay
    func openUncover(at seconds: Double)
    func closeUncover()

    // The grid's arrangement (see `FrameArrangement`)
    @Published private(set) var displayedFrames: [AnalyzedFrame]
    @Published private(set) var displayedRows: [FrameRow]
    @Published private(set) var beyondCapByGroup: [FrameGroupID: Int]
    @Published private(set) var limitBounds: FrameLimitBounds   // the sliders' spans
    @Published var frameOrder: FrameOrder           // visual factors, then independent rating scores
    @Published var frameLimits: FrameLimits         // low/high per signal
    @Published var faceFilter: FrameFaceFilter      // .all, .withFaces, .withoutFaces
    @Published var groupRows: Bool                  // one group per line, or one grid
    @Published var revealedGroups: Set<FrameGroupID>

    var hiddenFrameCount: Int                       // capped frames not currently shown
    var beyondCapFrameCount: Int                    // capped frames, revealed or not

    // Grouping, live
    @Published var groupingThreshold: Float         // distance within which two frames match
    @Published private(set) var regroupPending: Bool
    func regroup()                                  // re-cluster under the current threshold

    // Commands
    func loadVideo(at url: URL)                     // drops a file in; starts analysis
    func analyze()                                  // re-runs under current settings
    func cancel()
    func selectFrame(_ id: FrameID, command: Bool = false, shift: Bool = false)
                                                    // replace, toggle, or rendered-order range
    func selectAllDisplayedFrames()                 // current filters and group caps
    func select(_ frame: FrameID)                   // context-menu membership toggle
    func selectAll(in group: FrameGroup)
    func clearSelection()
    func frame(id: FrameID) -> AnalyzedFrame?
    func members(of group: FrameGroup) -> [AnalyzedFrame]

    // Keyboard destination — the rule, so both the grid and the preview can call it
    func focusNeighbour(_ delta: Int)
    func focusGroupEdge(forward: Bool)
    func togglePreview()

    // Thumbnails — async access regenerates an LRU-evicted frame from its source.
    // Grid tasks call this only for materialized cells and stop when cancelled.
    func thumbnail(for frame: AnalyzedFrame) -> NSImage?
    nonisolated func cachedThumbnailData(forKey key: String) -> Data?
    func thumbnailData(for frame: AnalyzedFrame) async -> Data?

    // One frame at native resolution, for the zoomed preview
    func nativeImage(at timeSeconds: Double) async throws -> CGImage

    // Export
    func exportSelection(to directory: URL) async throws -> [ExportResult]
    func exportFrames(_ ids: [FrameID], to directory: URL) async throws -> [ExportResult]
}
```

### Invariants the UI may rely on

1. `frames` is **always sorted by `timeSeconds` ascending**. Re-sorting in the UI is a bug,
   not a convenience.
2. `progress.phase` moves `idle → probing → sampling → scoring → grouping → complete`, and
   reaches exactly one of `complete / failed / cancelled` at the end of a run. It never
   goes back to `idle` once a video is loaded. `sampling` and `scoring` are one streamed
   pass, so `sampling` is reported at its start and `scoring` takes over from its first
   frame. Results are accumulated off the main actor and published once after grouping;
   scalar progress updates are limited to four per second. `progress.total` is sized by `settings.analysisMode`: every frame for
   `full` (the default), one a second for `quick`.
3. Every `memberIDs` entry of every group names a frame that is in `frames`. A group is
   never partially populated.
4. `selectedFrameIDs` only ever contains identifiers that are in `frames`. Loading a new
   video clears the selection.
5. `thumbnail(for:)` is safe to call from a `View.body` and returns `nil` unless the
   thumbnail is already in memory. Async UI requests regenerate evicted thumbnails.
6. `displayedFrames` is a **subsequence of `frames`** (never a frame the analysis did not
   produce), **already ordered** by `frameOrder` and capped per group; `displayedRows` is
   the same frames split into grid lines. Both are stored, and change only when their
   inputs change — `frames`, `settings`, the arrangement fields, or a regroup.
7. `frameOrder` always has at least one signal switched on, and its signals are always in
   the app's precedence (time, then face, then art). A caller cannot build an order that is
   empty or out of order, so a grid position always means the same thing.
8. `beyondCapByGroup` counts every frame the per-group cap puts past the limit, **including
   for groups in `revealedGroups`** — a revealed group still has to be listed, or there
   would be no way to hide it again. No frame is discarded to produce it.
9. `focusedFrameID` may name a frame that is **not** in `displayedFrames`: the face filter,
   a range or the per-group cap can hide it, and the preview must keep showing it rather
   than the UI clearing the focus behind the user's back.

### Grouping contract

`regroup()` re-clusters the frames already analysed. It does not re-sample or re-score, and
it rebuilds every frame's `groupID` **from scratch** — a frame that has left a group must
lose the identifier, not keep a stale one. A group's identifier is derived from its
members, so regrouping changes identities and `revealedGroups` is cleared with them.
Expensive (thousands of distance computations), so it is driven by a released slider or a
key press, never by a drag in progress.

### Memory contract

The engine retains a bounded window of image work: one sequential reader feeds up to
two concurrent crop/resize tasks, and `FrameScorePipeline` pulls their results into up
to four scoring workers. Preparation results are delivered in source order. The
`FrameProcessingConcurrency` policy reduces concurrency for large native frames or
limited RAM, estimating source pixels, a native raster and the analysis image per slot against
256 MiB or installed RAM / 32. At least one preparation and one scoring slot remain
available. This is a concurrency estimate, not a hard process-memory cap. Pixel memory
depends on source dimensions and active stages rather than video duration.

`AnalysisMode.quick` spends less Vision work, not less decode: the reader still
walks every frame, and the mode is chosen before a video loads — on the drop zone —
because the decode is the same either way.

Two things that is easy to mistake for, both of which a maintainer has to keep true:

- **It is image memory that is bounded, not total memory.** Every analysed frame's
  `AnalyzedFrame` is retained for the session's lifetime — timecode, the two scores, the
  compressed FeaturePrint (~2.5 KB) and a thumbnail key, under 10 KB in all. That grows
  linearly and is affordable: measured at 4.3 MB for 599 frames, which extrapolates to
  ≈360 MB at the reference video's 84,067.
  Tagging additionally retains an LZ4-compressed sparse FP8 row per tagged frame.
  Values below 0.35 are discarded. The retained size follows the number of scores
  above the cutoff, rather than allocating 43,448 Float32 bytes per frame.
- **The reader decodes every frame of the file**, one at a time; a `full` pass
  analyses every one, a `quick` pass one a second. Memory is per frame in
  flight; *work* is per frame of the file. The reader advances serially;
  raster preparation, scoring and thumbnail encoding overlap on background tasks.

Feeding the pipeline an input that is already an array is legal and scores correctly but
gives up the bound; the session feeds the sampler's stream in directly. The per-group frame
cap controls what is drawn, while thumbnail JPEGs are held separately in a bounded LRU.

### Thumbnail contract

Thumbnails are JPEG encoded and retained in a 64 MiB in-memory LRU keyed by
`AnalyzedFrame.thumbnailKey`. The cache is cleared when a new analysis starts and does not
create thumbnail files or directories. An evicted thumbnail is regenerated from the source
when requested; the grid's lazy cells keep the visible area and nearby cells most recently
used. Decoded images have a separate 64 MiB cache budget.

The still preview reads its selected timestamp from the original video through
`PreviewFrameReader`, without the analysis thumbnail's resolution limit or crop.
Reads are cancellable; stale frames cannot replace a newer selection. Zoom converts
the pointer from clip coordinates to viewport coordinates before applying the new
scale, preserving its image point after the view has been panned.

## Why the session is `@MainActor`

Every published property is read from a `View.body`, and SwiftUI's rendering is main-thread
work. Making the session main-actor isolated means the UI can read it without a hop, and the
long-running decode/scoring loop and cancellable grouping are explicitly nonisolated
and run in a detached task. Generation tokens discard stale progress/completions after
cancellation or replacement. Only scalar progress snapshots and the final results cross
onto the UI actor. Both analysis and regrouping unmount the grid/player while running.

`AnalysisProgress` also carries `facesProcessed`, `framesWithFaces`, `groupedFrames`
and `groupsIdentified`. During grouping, the latter counts linked components before
final maximum-size splitting; completion replaces it with the final group count.
`ExportRequest.frameTimes` maps identifiers to actual decoded presentation timestamps,
so browser exports remain exact for variable-frame-rate videos.

## Uncover contract

`WaifuSession` integrates the mode; `UncoverSession` owns its reconstruction lifecycle.
Opening clears `previewFrame` and pauses the existing player. While editing, grid
navigation and the still-preview key monitor are inactive. Video replacement closes and
cancels Uncover. Existing analysis, ranking, arrangement and still exports retain their
contracts.

`UncoverSession` publishes selection (`roi`, `scale`, `algorithm`, `start`, `duration`), display settings
(`feather`, `strength`, `comparison`, `showOriginal`, `showB`, `editing`, `zoom`, `loop`),
and read-only `busy`, `playing`, `status`, `error`, `original`, `imageA`, `imageB`,
`displayedIndex`, `result`, `capabilities` and `allowUnsafeMemory`. Commands are
`reconstruct()`, `cancel()`, `seek(index:)`, `step(_:)`, `togglePlayback()`, `pause()`,
`export(to:)` and `dispose()`. The UI relies on these invariants:

1. ROI coordinates are normalized in the oriented native video, with a top-left origin.
   Preparation snaps to integer pixels and records the exact geometry in the manifest.
2. `hasCurrentResult` is true only when algorithm/backend identity, selection, scale and requested interval match the
   successful job. Failed/cancelled replacements retain the previous success; incompatible
   results cannot be exported as a new request.
3. Feather, strength, comparison, viewport and scrubbing never call model inference.
4. `original`, A, B and `displayedIndex` are published together. A single monotonic clock
   chooses a source frame identity for both alternatives; rendering may skip preview frames
   if it is slow, but preparation and export keep every source frame.
5. Only a validated real result becomes ready. The subprocess receives degraded ROI frames
   only; its output must match source frame count and backend output geometry. FlashVSR
   returns a 4× ROI and composites/export frames at four times the source dimensions.
6. Native images are disk-backed and decoded a frame at a time. Intervals are limited to
   five seconds/300 frames. SeedVR2's selected 1×–64× reduction is the source input size;
   its native temporal model generates the exact ROI dimensions. SeedVR2 uses overlapping
   spatial tiles and has a 16-megapixel output
   ROI limit. FlashVSR accepts the selected 1×–64× source reduction, keeps 4× output,
   applies a capability-specific padded-area limit and preserves whole-region attention.
   Both monitor physical memory.
7. Export writes a new MOV, uses source rational presentation timestamps relative to its
   first visible frame, preserves matching audio, and publishes the finished file without
   overwriting an existing path. HDR/non-square-pixel sources are explicitly unsupported.

`TemporalVideoUpscaler` exposes `cacheIdentity`, asynchronous `capabilities()` and
`reconstruct(_:allowUnsafeMemory:progress:)`. Its request is an `UncoverPrepared` directory
and manifest, and its result is an `UncoverResult`. The manifest records source identity,
oriented dimensions, snapped ROI, scale and ordered rational frame timestamps/durations.
The subprocess writes ROI PNGs and a `report.json` with version/model, actual dimensions,
frame counts, effective temporal configuration, duration and available memory measurements.
Cancellation terminates the helper's process group; incomplete jobs are removed after
termination. Native setup is `tools/setup-native-uncover.sh`. Optional configuration fields
`nativeExecutable`, `nativeEmbedding`, and `nativeMemoryBudgetGB` select
the SeedVR2 native runtime. RealESRGAN uses a separate `realesrgan-model.txt` path;
these remain backward-compatible with the original Python configuration. A configured
native failure never falls back to Python. The worker writes PNGs directly and enforces
one active job with a process lock. The default monitored physical budget is 6 GB.

See [`Uncover.md`](Uncover.md) for the full design and acceptance criteria.
See [`FlashVSR.md`](FlashVSR.md) for the second native backend. `UncoverAlgorithm` selects
the existing protocol implementation. Preparation caches are independent of algorithm;
result caches include algorithm and backend configuration. Input directories are shared
through symlinks, while reconstructed outputs remain separate. Active changes cancel the
preceding job and reject stale publication. Disposal waits for a cancelled worker to exit
before releasing its shared inputs. A shared native lock prevents simultaneous models.

`PGTFormerBackend` uses the same preparation, preview and export contract. Its
configuration points to a native worker and a validated original-model package.
Model-required 512-pixel face resizing does not alter true 1× source preparation.

## Layer ownership

| Directory      | Layer            | Tests target |
|----------------|------------------|--------------|
| `Model/`       | shared contract  | yes          |
| `Video/`       | sampling         | yes          |
| `Vision/`      | scoring          | yes (pure parts) |
| `Grouping/`    | clustering/rank  | yes          |
| `Engine/`      | orchestration    | yes (pure parts) |
| `Export/`      | writing stills   | yes (naming) |
| `Uncover/`     | temporal reconstruction, ROI images, clip export | yes |
| `UI/`          | SwiftUI          | no           |
| `App/`         | `@main` scaffold | no           |

## Environment notes for agents

- Build the app with `./build.sh`; it is compiled directly with `swiftc`, and the root
  `Package.swift` is for IDE navigation because SwiftPM cannot load it with the target
  machine's Command Line Tools. Isolated native workers under `Runtime/` use SwiftPM and
  pinned dependencies; prepare that clone-local toolchain workaround with
  `tools/native-toolchain/prepare.sh` (or `tools/setup-development.sh`).
- Test the app with `./run-tests.sh`. It compiles the production sources **except**
  `Sources/Waifu/App/WaifuFinderApp.swift` (it owns `@main`) together with every file under
  `Tests/` into one throwaway executable. The suite is a hand-rolled harness, not XCTest or
  Swift Testing; see `Tests/README.md`.
- Sample video: `sample-video.mp4` — 1920×1080, 29.97 fps, 84,067
  frames, 46.75 min, 2.86 GB. **Never read the whole file.** Decoding
  it is the app's job, not yours. This is a scale reference; recorded validation does not
  include a complete analysis of the full 46.75-minute clip.

Uncover source navigation: `sourceTimeline` stores actual asset presentation times,
with media cursor timestamps mapped through container edit lists. `seekSource(index:)`
keeps the ROI and sets the reconstruction start; reads use the cancellation/deadline
protected `PreviewFrameReader`. `sourceRequestedIndex` advances immediately, while
`sourceDisplayedIndex`/`sourceTime` update only after the matching read succeeds.
`step(_:)` targets source frames when editing and processed frames during comparison.
The source slider rounds to integer indices without drawing a tick for every frame.
