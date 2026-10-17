# WaifuFinder — working notes

This describes the code **as it is now**. It is not a changelog: where a decision looks
surprising, the reason sits next to it, because the reason is the part worth keeping.

## What it is

WaifuFinder takes one long video, decodes it, scores each chosen frame with Apple's Vision
models, clusters frames that show the same shot, ranks within each cluster, and lets you
browse, compare and export the survivors. It exists for the case where the interesting
material is scattered through an hour of footage.

It also has **Uncover Waifu**, an isolated region-reconstruction editor entered from the
native frame preview, with native SeedVR2 and FlashVSR 1.1 algorithms. The incorporated
designs are in `docs/Uncover.md` and `docs/FlashVSR.md`; the shipped layer
boundary is in `docs/API.md`. Analysis supports Full/Quick passes and still exports
carry exact decoded presentation timestamps.

Reference case: `sample-video.mp4` — 1920×1080, 29.97 fps, 84,067
frames, 46.75 min, 2.86 GB. Every frame is decoded, scored and grouped.

## Remote inference

`RemoteTaggerBackend` and `RemoteUncoverBackend` implement the existing inference
interfaces. Settings → Remote controls the independent listener and client; capability
snapshots drive the raised R marks and backend selection. A client receives the tag
vocabulary and applies its own threshold to full Float32 rows. Reconstruction uploads
only degraded frames; compositing and export use the client's originals.

The transport uses Apple Network, CryptoKit, Compression and Security frameworks.
Keys are stored separately for the server and each client address, with a private file
fallback. The real-socket regression suite injects model catalogs rather than downloading
weights. See `docs/Remote-inference.md` and `Tests/RemoteProtocolTests.swift`.

## Commands

```sh
./build.sh                                        # dist/WaifuFinder.app
open dist/WaifuFinder.app
./run-tests.sh [--only SUITE] [--strict] [--list]
```

SwiftPM does not work on this machine — `swift package dump-package` fails to link
`libPackageDescription` — so `Package.swift` exists for IDE navigation only, and neither
XCTest nor Swift Testing can run. The suite is a hand-rolled harness in `Tests/`,
compiled **into the same binary as the production sources** by `run-tests.sh`: every file
under `Sources/` except `Sources/Waifu/App/WaifuFinderApp.swift` (it owns `@main`), plus every
file under `Tests/`, into one executable in a temp directory, with `-swift-version 6`.
A failing case therefore means the real code misbehaved; there is no mock to drift.

Cases that need a real container now use Apple-native fixture generation. The
legacy optional external encoder dependency has been removed.

## Layout

```
Sources/Waifu/
  App/       @main and the window; owns nothing else
  Model/     Contracts (types + layer boundaries), WaifuSettings, FrameArrangement
  Video/     probe, sampler, thumbnail cache, the drop rule
  Vision/    scorer, FeaturePrint codec, face rule, scoring pipeline
  Grouping/  clustering, ranking
  Engine/    WaifuSession integrates browsing; UncoverSession owns editing jobs
  Uncover/   oriented temporal extraction, ROI images, SeedVR2 subprocess, clip export
  Export/    native-resolution extraction and errors
  UI/        SwiftUI: root, grid, preview panel, order panel, sliders
Tests/       harness, fixtures, one file per suite, README
docs/API.md  the engine↔UI contract
tools/       make-icon.swift
```

Layer order, and the only direction dependencies may point:

```
Video ──► Vision ──► Grouping ──► Engine ──► UI
                       │                      │
                       └────── Export ◄───────┘
```
`Model` is depended on by all and depends on none.

## The pipeline

`probe → sample+score (one streamed pass) → group → rank → arrange → UI`

### Probe — `Video/VideoProbe.swift`

Reads duration, dimensions, nominal frame rate and an *estimated* frame count with the
modern async `load(...)` APIs. Rejects non-file URLs, missing paths, directories, files
with no video track, zero-or-NaN durations and zero-size tracks with a typed
`VideoProbeError`, before AVFoundation can return something opaque. The frame count is an
estimate (`duration × rate`) because `nb_frames` is not reliably present in MP4 metadata;
it bounds work and never addresses a frame.

### Sampling — `Video/FrameSampler.swift`

The file to read first, because two things about it are load-bearing.

**Frames are produced on demand, not pushed.** `sample` returns a stream built with
`AsyncThrowingStream(unfolding:)`, so a frame is decoded when the consumer pulls one and
not before. That is a memory property, not a micro-optimisation: the obvious alternative —
a producing `Task` yielding into an `AsyncThrowingStream` — has no backpressure, so a
decoder that outruns the scorer holds every decoded frame at once (576 KB
each at 512 px: 46 GB for the reference video's 84,067 frames). Note the
**no `onCancel:`** (only `AsyncStream` does), so cancellation is observed at the next pull,
which ends the stream and releases the reader. The producer is an actor holding the reader;
`next()` returns `nil` at the end of the file.

**One sequential pass, scaled by the decoder.** There is no seeking alternative: the
decode walks every frame — a quick pass buys back the scoring, not the decode — and
in a long-GOP codec a seek decodes from the previous keyframe
anyway. One `AVAssetReader` walks the file front to back, and the output's
`outputSettings` name a pixel format *and the target dimensions*, so the hardware
decoder scales during decode rather than a software downscale afterwards. `nil`
output settings would vend the track's stored format — a compressed sample with no
image buffer at all, which is how the exporter once failed on every frame.

Ownership: the mode — `AnalysisMode`, chosen on the drop zone before a video loads —
decides which decoded frames are yielded. The full pass (the default) takes every
frame; the quick pass takes one frame every `frameStep` decoded frames, always
including the first, so a 29.97 fps quick pass keeps frames 0, 30, 60…. A kept
frame carries its *decoded* ordinal as its identifier — the same frame has the same
identifier in both modes, so the per-video cache is shared — and the pass ends when
the reader is exhausted, not when a planned count is reached, so a variable-frame-rate
file's extra frames are considered too. The first decoded frame is therefore the first
frame of the analysis by construction: there is no grid to prime and no frame to skip.

### Scoring — `Vision/`

`VisionFrameScorer` runs three requests concurrently per frame: aesthetics (required;
its failure fails the frame), face capture quality and FeaturePrint (both best-effort — a
landscape is a valid answer, not an error).

`FrameScorePipeline` runs those over a stream with `min(4, activeProcessorCount)` workers
and emits in **input order** — the order frames were submitted, tracked as a queue, so
a quick pass's gapped indices (0, 30, 60…) emit correctly — buffering out-of-order
completions. It *pulls*: it calls
`next()` only when a worker slot frees, which is the other half of the memory bound. Its
progress count is therefore also the sampler's output count, and the session reports that
one number rather than minting two.

### Grouping — `Grouping/FrameGrouper.swift`

A funnel, not a matrix: chronological window (±120 s) → FeaturePrint distance →
single-linkage union-find. Comparing by time first is what makes it affordable and is also
more correct — two frames of the same composition minutes apart are not alternates of one
shot. Single linkage chains, which is right for a slow pan and is why oversized components
are **split into contiguous chronological runs** rather than truncated; a run of one left
over from a split is dropped, because this module's contract is that a lone frame is never
a group. A frame with no FeaturePrint is ungroupable, not "matches nothing".

`FeaturePrintCodec` is the only way in or out of a `FeaturePrintObservation`: JSON (the
only public route in, verified bit-exact) plus zlib plus a one-byte marker; a payload whose
first byte is not the marker is read as plain JSON.

### Ranking — `Grouping/FrameRanker.swift`

Ranks, never raw scores. `artScore` (−0.996…+1.000) and `faceScore` (0…1) are different
instruments, so each is reduced to its **rank within the set being ranked** (1 best, 0
worst) and the ranks are blended. `faceWeight` applies only to frames with a significant
face, so a landscape group is not dragged around by a portrait's absence.

`FrameRanker.faceIsSignificant(_:settings:)` is **the** definition of "this frame has a
subject face" — the ranker, the order and the faces-only filter all call it, because two
definitions would let the filter offer a frame the ranking ignores. `faceScore` is only
ever set for a face that cleared the area threshold (`FaceSelectionRule` keeps the largest
*significant* face, not a mean); the area is re-checked for frames built elsewhere.

### Arrangement — `Model/FrameArrangement.swift`

The grid's contents, decided by value so the view decides nothing. Three steps, in this
order — the order is the behaviour:

1. **filter** — faces-only, then per-signal value limits;
2. **order** — time, face, face definition, art;
3. **per-group cap** — the best N of each group *by that order*.

Capping before ordering would keep whichever N the file listed first. Frames in no group
are never capped; they are distinct moments, not alternates. `Result` also reports
`beyondCapByGroup` — counted **including for revealed groups**, or revealing one would
remove it from the list that offers to hide it again. Nothing is discarded to produce it.

`rows(of:)` splits the ordered visible frames into grid lines: one per group, in the order
that group's best frame appears, with the frames in no group on a final line of their own.

`FrameLimits` are low/high bounds per signal, and their slider spans come from
`FrameLimitBounds.bounds(of:)` measured over the frames the **face filter** admits — not
the displayed set. Measuring the displayed set would be circular: dragging the art thumb
would narrow art's own bounds and move the thumb again.

## The UI

`RootView` is a `ZStack`: the content (grid + right column) or the drop zone, plus the
preview overlay. No left sidebar (its two lists answered "which frames are alike?" —
already legible from the order panel — and "what are the top ones?") and no timeline (a
ten-pixels-per-minute track is a worse way to land on a moment than the grid).

- **Grid** (`UI/FrameGridView.swift`) — one group per line, switchable to one continuous
  grid of everything in the same order. A cell is a thumbnail, an `art/face` reading in
  its corner, a 1 px border and, when selected, a tick. No timecode, no group badge, no
  score pills: forty copies of "0.31" is noise you scroll past.
- **Preview panel** (`UI/VideoPreviewPanel.swift` + `UI/PreviewController.swift`) — the
  video, parked on the focused frame. Clicking a frame seeks here (zero seek tolerance, so
  it lands on *the sampled frame*); `VideoPlayer` scrubs. The player lives in a small
  controller so the grid and the panel can move it without touching AVFoundation.
- **Order panel** (`UI/FrameOrderPanel.swift`) — order switches with per-signal ranges, a
  live grouping threshold, faces-only, one-group-per-line, the per-group stepper, the
  beyond-the-cap list, counts and selection actions.
- **Frame preview** (`UI/FramePreviewView.swift`) — Space opens the focused frame at
  **native resolution**, re-read from the video through the same `FrameExporter` reader
  path an export writes. Zoom is a pinch, the buttons, or a double-click; panning is the
  scroll view's. See below for the three decisions in it that are not obvious.
- **Sliders** (`UI/RangeSlider.swift`, `UI/ValueSlider.swift`) — SwiftUI has no two-thumb
  range slider on macOS and the system single-thumb one is a different weight from its
  neighbours, so both are hand-drawn: a slim ruled track, the same fill. The range
  slider's handles are pills **printed with their own value** (a timecode for time,
  `%.2f` for a score) rather than bare knobs over a separate readout line — same
  information, one line instead of two, and the number travels with the thing it
  describes. `Timecode.swift` is the one timestamp formatter.
- **Toolbar** — `Replace Video` (also clickable: a drop target that does nothing on click
  promises an action it does not have), then Analyze/Cancel, the selection count and
  Export, **one `ToolbarItem` per control**: an `HStack` inside a single toolbar item is
  laid out and compressed as one item, which is what truncated them to "Analy…" and
  "Expor…" in a toolbar with room to spare.

### Keyboard

←/→ step the neighbouring frame — **inside the current group's line** when one group per
line is on, otherwise the displayed order. ↑/↓ jump to the next/previous group. Space
toggles the preview, Return focuses and selects, Escape leaves the preview.

The *rule* lives in `WaifuSession` (`focusNeighbour`, `focusGroupEdge`, `togglePreview`)
rather than in the grid, because two views need it: the grid, and the preview overlay,
whose zoom buttons take the keyboard focus when clicked. While the overlay is up it also
installs an `NSEvent` local monitor, because a focused SwiftUI `Button` does not pass the
arrow keys up to an ancestor's `onKeyPress` — so without it, setting a zoom and then
stepping through frames silently did nothing. The monitor consumes the keys it handles
(returns `nil`) so nothing is handled twice, and is removed when the overlay goes.

#### One owner, derived from state

**The monitor is the *only* thing that handles the preview's keys.** The overlay used to
carry a second, `onKeyPress` copy of the same six rules as a "fallback". It was not a
fallback — nothing ever focused the overlay (`.focusable()` makes a view *eligible* for
focus; there is no `@FocusState` in it or in `RootView`), so those handlers ran only in
the window between `previewFrame` becoming nil and `onDisappear` finally tearing the view
down. Two owners for one key is what made this hard to reason about.

The rule that matters: **ownership is derived from state, never from view lifetime.**
Closing clears `previewFrame` first and SwiftUI removes the overlay afterwards, so for a
moment the overlay is still mounted, still in the responder chain and still running
handlers, while the preview it belongs to no longer exists. Apple is explicit that
`onDisappear` does not run "until the view disappears from the interface", and a removed
view with a `.transition` stays in the interface for the transition — the 50 ms fade here
measured closer to 0.7 s in practice. A handler that claims a press in that moment eats
the Space that *reopens* a preview, which is why leaving took two presses.

Returning `.ignored` instead was tried and is **worse**: it does not mean "undo", it means
"not mine", and dispatch continues to the *next responder* — here an inert hosting view —
so the key then reached nothing at all. That is the "no key works after entering the
preview" regression.

The decision is lifted into `PreviewKeyRouter` (`Sources/Waifu/Diagnostics/`) as a pure
function, so `Tests/PreviewKeyRoutingTests.swift` can pin it without a window. The monitor
calls it, so the tested and shipped behaviour cannot drift.

Two related fixes fell out of the same investigation:

- **`RootView` no longer applies `.disabled()` to `mainContent`.** It read like a tidy way
  to make the background inert, but it defeated the keeping-mounted goal directly:
  disabling a container drops keyboard focus from its `.focusable()` descendants and
  re-enabling does not give it back, so the grid came back from every preview with no
  keyboard focus. `opacity` + `allowsHitTesting` + `accessibilityHidden` already give the
  full visual result. Keyboard input is not left unguarded: the overlay's monitor runs
  ahead of the responder chain and consumes those keys while the preview is up.
- **Space/Escape key-up is consumed while the preview is up.** AppKit activates a focused
  control on Space *key-up* (Enter is key-down) via `performKeyEquivalent`, and the
  preview's header is a row of buttons — so the release used to land on whichever button
  held focus as the preview closed.

### Tracing

`Sources/Waifu/Diagnostics/Diag.swift` is runtime tracing for UI and keyboard work. **Off
unless asked for**: every call is one branch and writes nothing, so the shipping app pays
almost nothing. On it writes a timestamped per-process file under the temporary directory
and prints the path to stderr.

- `WAIFU_TRACE=1` or `open -a WaifuFinder --args --waifu-trace` — for launches.
- `WAIFU_TRACE=1 --waifu-trace keys,focus` (or the bare flag for everything) — narrow a
  keyboard investigation to the categories that matter. An unmatched category is reported
  rather than silently ignored, because a typo that disables logging is worse than none.
- **Debug ▸ Start Tracing…** — for attaching mid-session to a bug that only sometimes
  reproduces. Also offers Reveal in Finder and Mark This Moment.

Categories: `keys`, `focus`, `preview`, `uncover`, `grid`, `analysis`, `loading`, `export`.
Only the environment/argument forms mirror to stderr; the menu and tests write the file
alone so a trace never lands in unrelated tools' output. Rotation is by size (4 MB).

`Diag.tracedKeyHandler` wraps every `NSEvent` monitor in the app, so all of them report
the same things — the press, whether it claimed it, and what the responder chain thought
at that moment. A press that vanishes shows up as a line that stops.

### The frame preview's three decisions

1. **An overlay in the same window, not a sheet.** A sheet slides, and it takes the
   keyboard focus the grid needs to receive the Space that closes it. The overlay is
   inserted by an `if` and fades over 50 ms (`.transition(.opacity)` keyed on *whether* it
   is up, not on which frame, so stepping does not replay the fade).
2. **Stable identity, reload through `.task(id: frame.id)`.** Giving the view a per-frame
   `.id()` tears it down and rebuilds it on every step — blank, then thumbnail, then the
   full-resolution image — which is a flicker you cannot unsee. The old image also stays on
   screen until the new one is decoded rather than being cleared to a placeholder, and
   `.task(id:)` cancels an in-flight read when another step arrives.
3. **Held arrows wait for decoding.** A fresh Left/Right press can replace the pending
   request; repeat events wait, then advance after each successful frame. Key release,
   closing the overlay, and application deactivation stop repeat navigation. Each preview
   seek owns an asynchronous AVFoundation image generator. Cancellation completes its
   waiter immediately; a 15-second deadline turns a stalled read into a visible error.
   Request identity checks keep old completions from changing the current picture, note
   or loading indicator. Failed reads stop held navigation; another press can retry.
4. **Navigation preserves the view.** Only the frame you *open* is fitted; stepping keeps
   the same zoom factor and the same place in the picture, because stepping is a comparison
   and a view that resets compares nothing. Frames from one video are all the same size, so
   keeping the scroll offset keeps the same centre too.

### The preview's fit — `UI/ZoomableFrameImage.swift`

The preview opened a frame **at the wrong scale and the wrong place**: cropped, sitting in a
corner of the window, at its own pixel size. One cause, found by rendering the surface
offscreen and measuring the pixels rather than by reasoning about it:

> `NSScrollView.magnification` does not scale the document view's frame. It scales the
> **clip view's bounds** — how much document the viewport shows.

So a scroll view that fits itself with `setMagnification(fitScale)` reports, on its next
layout pass, a "viewport" of `physical ÷ magnification`; the code took that for the
viewport, refitted to it, and the magnification went to 1 — leaving the frame at 1:1,
cropped, centred on whatever point the last step had stored. The fit was being computed
from its own output. `applyRequestedZoom` in the old code even called `setMagnification` and
then re-read `contentView.bounds.size`, which is the same feedback written twice.

The rules the surface now follows, as a pure value (`FrameZoomGeometry`, pinned by
`Tests/FrameZoomGeometryTests.swift` without a window):

* **Fit touches, then centres.** `min(vw/iw, vh/ih)` fills the limiting axis exactly and
  the frame sits in the middle of the other. AppKit will not do the second half: a document
  view smaller than the clip view is *pinned to the scroll origin*, so the padding that
  makes centring possible has to live in the document view. The document view is therefore
  a plain **canvas**, never smaller than the viewport, with the image view centred inside
  it at the fitted rectangle. `.scaleNone` also had to go — it draws an image at *its own*
  size whatever the view's frame says, which is what overflowed the viewport.
* **No AppKit magnification at all.** `setMagnification(_:centeredAt:)` is overridden to
  pin 1, so the clip view's bounds size is the real viewport and nothing can put the old
  feedback back. `allowsMagnification` stays on only so a trackpad pinch still reaches
  `magnify(with:)`.
* **Anchoring is not centring.** A pinch reports a delta *and* the pointer position; the
  scroll for it is `focus·scale − pointer`, not `focus·scale − viewport/2`. Re-centring
  slid the picture out from under a hand that had not moved.
* **The state published is the state on screen.** The clip view's bounds give back the zoom
  and the image point at the middle of the window — except when nothing can scroll, where
  the stored point is left alone so magnifying back in returns to it.

`zoom` is a **multiple of the fit**, so 100 % is the fit and a window resize refits without
losing the zoom level; the old absolute-scale reading (`100 %` = one image point) is gone.

## Uncover Waifu

`WaifuSession` owns opening/closing the exclusive overlay, not inference state.
`UncoverSession` owns one job, up to four successful algorithm-specific results, shared
prepared ROI inputs, display settings and the shared
playback clock. Opening removes the still-preview keyboard monitor and pauses the existing
player; closing releases temporary results. Grid navigation/toolbar actions are inactive
while editing. Video replacement cancels and closes the mode.

**Three stages, with inference kept out of display.** Preparation reads consecutive native
frames sequentially, applies the preferred transform, extracts a fixed normalized ROI,
and downsamples once with Core Image Lanczos. Originals and ROI images go to temporary
PNG files. Lanczos, SeedVR2 and FlashVSR receive the same degraded pixels; original ROI detail
never enters the model. Reconstruction writes validated ROI results. Display composites
them with identical inward feather masks, separately from blend strength. Changing feather,
blend, wipe, mode, zoom, pan or frame position never reloads the model. Changing model
inputs invalidates the result; a failed/cancelled replacement retains the last success.

**Source times, not nominal-FPS addresses.** Rational PTS and duration travel with every
frame. `AVAssetReader.timeRange` can clip/retimestamp a partial opening sample, so extraction
decodes one second of preroll and selects by actual PTS. This fixed a real six-instead-of-
five proof failure. One monotonic clock chooses both comparison images; if preview
rendering is slow it skips display frames together, while preparation/export keep all
frames. Playback is currently visual; corresponding audio is preserved in export.

**Native Swift/MLX inference is now the preferred backend.** `NativeSeedVR2/` extends
xocialize's archived image port with the pinned Python reference's temporal video path:
5D latents and masks, shared global noise, causal VAE caches/slices, temporal upsampling
head trimming, 29/8 chunks and exact frame ownership. It loads the existing prepared
`AbstractFramework/seedvr2-3b-4bit` revision `9b2f6c3c1b1b4548c66f302ec5a4cd3cd9937eb5`
through the packed projection bundle. Quantized linears are constructed from checkpoint arrays, without quantizing
random training weights. Explicit graph boundaries free evaluated intermediate buffers.

`./tools/setup-native-uncover.sh` configures it. Normal setup downloads the self-contained
`alastorid/SeedVR2-3B-Swift-MLX-4bit` bundle (~2.4 GB), with the already-quantized upstream
projections, one-block transformer shards and positive embedding. Logical packed q4
values/scales/offsets are preserved without dequantization or re-quantization; tensor
layout and indices change for fusion. The embedding is unchanged from
`mlx-community/SeedVR2-3B-mlx` revision `46f851fe23b2a9fbefee6419b94e4fdf07b92965`.
`tools/seedvr2-release.json` pins the hosted release and checksums. The Swift download
helper verifies every file, reuses matching local shards and resumes partial downloads.
Build-only embedding setup fetches the same pinned bundle’s manifest and embedding.
No Python, further conversion or re-quantization is performed. Upstream attribution and
Apache-2.0 model licensing are preserved in `docs/SeedVR2-model-card.md` and the release.
Published verification: all six hosted model files matched checksums; the hosted installer,
verified cache reuse, offline partial recovery, checksum rejection, safe destination checks
and embedding-only bootstrap passed. Real five-frame native inference and export passed
frame-count, PTS and audio assertions using the installed bundle. `./build.sh` builds and bundles the worker,
MLX Metal kernels and the constant text embedding. No Python process is used for native
setup, building, inference or video IO. The app remains Swift 6 with warnings-as-errors;
the model library is isolated in a Swift package with pinned mlx-swift 0.31.2 / MLX core
0.31.1. A project-local compiler wrapper uses the CLT's current public PackageDescription
interface in place of its stale private interface; the installed toolchain is untouched.

**Larger ROIs are spatially tiled, still temporal.** Model tiles are at most 256×256 with
64-pixel overlap, consistent spatial layouts and shared global noise. Weighted overlapping
outputs are normalized. Fixed ROIs can span the source; there is no 256×256 selection cap.
Odd dimensions are normalized to exact selected geometry after padding removal. The prior
worker enlarged every degraded PNG to the ROI size before temporal inference, reducing source
detail but not SeedVR2's spatial workload. The new worker keeps the temporal model at the
actual degraded dimensions and writes same-size restored frames. The app then applies the
separately installed Real-ESRGAN x2plus Core ML FP16 model once per doubling to restore each
frame to the ROI dimensions. The 2× through 64× controls correspond to one through six learned
x2 passes. The spatial model is framewise and general-image, not anime-specific; it adds no
temporal consistency and high factors can invent detail or flicker. The ROI output is capped
at 16 megapixels. Setup downloads the pinned ~30 MB archive, verifies its SHA-256, compiles the
model locally and deletes both the archive and temporary `.mlpackage`; no Python is used.
The native worker reads/writes PNGs directly; no intermediate lossy MP4 or PyAV process is
involved. Spatial tiling changes context, so it is not
claimed pixel-identical to whole-region inference. Intervals remain five seconds/300
frames, sources SDR with square pixels. Scroll panning follows event deltas on both axes.
While editing, the ROI itself previews the selected 2×–64× Lanczos downsample, enlarged
without smoothing and labelled with reduced input dimensions. The preview caches one
crop by source image, snapped ROI and scale; it does not rerun on pan/zoom or invoke MLX.
Comparison removes the overlay; Original shows the untouched source.

**Memory is measured and bounded.** The default 6 GB budget is checked before loading and
monitored using physical footprint every 20 ms; macOS memory-pressure events stop the job.
Only one native worker can hold the model at once. MLX's cache/allocator limit alone is
not a hard process-memory cap, and system-wide zero swap is not guaranteed. Failed or
cancelled jobs preserve the previous successful result. The legacy Python runtime and
unsafe-memory preference remain available for reference, but a configured native job
never silently falls back to the Python backend.

The same preserved five-frame 256×256 proof measured 5.16 GB native peak physical
footprint vs 12.45 GB Python. The one-second 30-frame proof measured 5.31 GB vs 14.07 GB,
and 128 seconds vs 164 seconds. Native/Python output correlations were at least 0.984
across the 30 frames, including the chunk boundary; comparison includes the legacy
backend's intermediate MP4 compression and is not a claim of exact numeric parity.

**Export is a new MOV, never a source replacement.** The native compositor and writer
stream cached source/result frames, preserve source-relative PTS, encode original oriented
dimensions, and mux the corresponding source audio with AVFoundation. Source color
metadata is carried into encoding. A sibling temporary output is verified and published
with a same-filesystem hard link, which cannot race into overwriting another file.
Outside-ROI pixels are unchanged before encoding; the compressed stream is re-encoded.

### Uncover verification

The real SeedVR2 figures below are historical measurements from the previous implementation.
Their report `inputWidth`/`inputHeight` fields described the degraded PNGs; the earlier worker
resized those images to ROI dimensions before passing them to SeedVR2. They are not memory or
quality proof for the new actual-low-resolution inference plus Core ML spatial-upscale pipeline.
New real-video inference and output-quality checks are pending on the Mac mini.

- Real source: five consecutive frames around 10 seconds in the reference video, 256×256
  ROI; exercised 2× (128×128 degraded file), 4× (64×64 degraded file) and 8×
  (32×32 degraded file). The historical worker enlarged these PNGs to 256×256 before
  SeedVR2, so those measurements do not prove reduced-size temporal inference.
- A one-second 2× job passed real chunk-boundary processing, context trimming and native
  1920×1080 interval export. The reopened output contains 30 frames and corresponding audio.
- Native Swift MLX real runs passed all three scales. A five-frame 528×537 ROI at 8×
  used nine spatial tiles, restored exact odd dimensions and exported with audio at
  5.33 GB peak physical footprint. The final integrated five-frame 2× run used 5.18 GB
  and passed explicit native-backend, budget, exact PTS/frame count and audio assertions.
  Measurements and comparison limits are recorded in `docs/native-uncover-validation.json`.
- The packaged worker passed singleton locking, preflight refusal at a 1 GB budget,
  and real process-group cancellation in 25 ms. The app bundle passes deep strict
  ad-hoc signature verification; Metal kernels are colocated with the signed worker
  under Resources rather than treated as nested executable code.
- Actual offscreen canvas renders at 2×/4×/8× were inspected: the preview aligns with
  the source crop and shows increasingly coarse pixels inside the same selection.
- Native UI: a short moving 512×384 audio/video fixture; drew a 128×128 region, used a
  six-frame interval, and separately completed 2×/4×/8× inference. Wipe, side by side,
  instant toggle, Original, zoom, stepping after zoom, playback and feather adjustments
  worked. The UI's exported MOV reopened and decoded with six frames and 0.25 seconds of
  video/audio. Cancelling a real replacement restored usable controls; switching back to
  the prior 8× inputs reused the successful result and re-enabled export.
- Harness: snapped geometry/letterboxing/rotation, odd degradation sizes, exact outside
  and zero-strength pixels, full-strength interiors, consecutive PTS, VFR/audio export,
  process cancellation, cache reuse and failed-result retention. The deterministic backend
  is defined only in `Tests/`; real inference is separately opt-in.

Not yet verified: the new reduced-input SeedVR2 + learned Core ML x2 path at any scale,
five-second inference/quality at every scale, ROIs beyond the validated
528×537 envelope,
HDR or non-square-pixel support, and long-video editing/export. Optional PSNR/SSIM and
temporal-stability diagnostics, automatic scene-cut avoidance and degraded
spatial context are deferred. Choose intervals within a single shot. Object tracking and
whole-source export remain outside this first implementation.

### Native FlashVSR 1.1

The compact Algorithm selector defaults to SeedVR2 and offers FlashVSR 1.1. Both use the
existing ROI, downsample preview, compositor, A/B viewer and export path. FlashVSR supports
4× only; selecting it preserves a 2×/8× experiment and explains the unsupported combination.
Changing algorithms reuses decoded/downsampled inputs and restores a matching cached result
without inference. A/B labels name the engine. Active changes cancel the preceding job,
and stale results cannot publish. Shared inputs are released after workers exit.

`NativeSeedVR2/` now builds separate SeedVR2 and FlashVSR Swift/MLX Metal executables with
shared native IO/budget monitoring. FlashVSR ports the official Tiny Streaming graph:
causal LQ projection, 30-block BF16 DiT, locality-constrained sparse attention, temporal
KV caches, Tiny Conditional Decoder and ADAIN. The Metal online-softmax kernel preserves
the trained masks/top-k semantics and does not allocate dense token attention matrices.
Geometry metadata invalidates when aspect ratio changes; every clip resets temporal
memories. First context is 25 frames/21 visible, then eight frames per chunk with four
lookahead frames. Padding never changes visible frame counts.

Official checkpoints were explicitly authorized and downloaded (6.44 GB), checksum checked
and converted to about 3.51 GB BF16 shards. Offline Python conversion/oracle is allowed;
production inference has no Python/PyTorch/CUDA dependency. No unverified low-bit
quantization or spatial DiT tiling is introduced. Both native engines share one active
worker lock. FlashVSR checks padded output area against a conservative 6 GB budget before
loading and stops on physical footprint/OS pressure. Model resources disappear when the
isolated worker exits; the app does not keep both models resident.

Verification on this 16 GB Apple Silicon Mac:

- All **149 tests passed**, including real native FlashVSR inference/export at an odd
  159×253 ROI: five input/output frames, 40×64 degraded input, 256×256 model padding,
  exact native ROI restoration, source PTS and audio. Peak physical footprint 4.30 GB.
- A 30-frame moving video crossed three streaming chunks and exported exactly 30 frames.
  Official BF16 graph parity passed on every visible frame: worst MAE 0.02336 in [-1,1]
  and minimum correlation 0.99726 (preset tolerances <0.025 and >0.99).
  Five-frame preserved inputs also passed (worst MAE 0.01586, minimum correlation 0.99156).
  The oracle uses official CPU projection/DiT and a PyTorch MPS decoder with equivalent
  masked SDPA; this is not a measured CUDA comparison. Local pixel differences can exceed
  the aggregate MAE; full metrics are recorded in `docs/flashvsr-validation.json`.
- Loaded-model repeat: 30 frames in 3.13 s then 2.65 s, excluding 3.14 s weight loading;
  outputs were bit-identical after resetting clip state. Peak physical footprint 3.83 GB.
  These timings include PNG IO, not app decoding/compositing/export, and are small-ROI
  measurements rather than whole-video throughput predictions.
- FP32/BF16 sparse attention matched the small dense masked reference over initial and
  cached temporal windows and transposed aspect ratios. BF16 maximum error 0.0009766.
- SeedVR2's preserved five-frame outputs remained **byte-identical** after extracting
  shared IO. Algorithm switching tests verified preparation reuse, cache ownership,
  preserved display settings, active cancellation and unsupported-factor preservation.
- Actual offscreen Uncover UI inspected with configured FlashVSR, 4× input preview and
  enabled Reconstruct. Packaged helper passed single-worker exclusion, preflight budget
  rejection, cancellation (~0.025 s) and real 30-frame inference. App Swift 6 build passed
  with warnings as errors and deep strict signing verification.

Setup: `./tools/setup-native-flashvsr.sh`, then `./build.sh`.
The UI exposes an explicit setup action; no large model download occurs silently.
Runtime weights are in `.native-runtime/flashvsr/` (about 3.51 GB), configured by
`~/Library/Application Support/Waifu/Uncover/flashvsr.json`. Original unconverted files
in `.native-runtime/flashvsr-official/` were removed after verifying the pinned converted
manifest, all model files and support files, reclaiming 6.44 GB. Only reconversion needs
those originals; normal inference does not. Setup does not delete them automatically. Normal setup now downloads only the converted model from
`alastorid/FlashVSR-v1.1-Tiny-Swift-MLX-BF16` on Hugging Face. It requires no Python,
avoids original checkpoint downloads and verifies a pinned manifest plus every file.
`tools/flashvsr-release.json` records the release pin; `docs/FlashVSR-model-card.md`
contains the published card. Safetensors modification/license metadata was added with
all tensor payloads verified unchanged. Original reconversion remains explicitly opt-in.
Publication/setup verification: all 34 remote tensor object checksums matched; the hosted
manifest, missing prompt and supporting documents downloaded successfully while verified
shards were reused. Completed partial downloads recovered without a network request;
bad downloaded checksums preserved the previous file and removed the bad partial. Normal
setup built/configured the native workers. Five reconstructed PNGs remained byte-identical
after adding file metadata. The updated app built/signature-verified and the default suite
passed 148 cases with one explicit real-inference skip. Temporary upload copies were removed.
See `docs/FlashVSR.md` for source pins, full computation, conversion and reproducible
validation. Whole-frame FlashVSR, scene-cut quality, long clips, HDR/non-square pixels
and CUDA parity remain unverified. Its supported region size is constrained by the
monitored memory budget. There is no claim of zero system swap or A100 performance.

## Memory

A decoded analysis frame is ~576 KB at 512 px. **The engine never buffers the video**: the
sampler produces on demand, the pipeline pulls, and at most its four workers' worth of
images exist at once. Measured on one clip (599 frames, 512 px, `/usr/bin/time -l`):

| Path | Peak RSS |
|---|---|
| Buffered — every sampled frame held at once | 420 MB, of which 336 MB was frame pixels |
| Streamed | 67 MB |
| Streamed, with every analysed frame's record retained (what the app does) | 72 MB |

What "frame by frame" does and does not mean:

- **Decode memory** is per sample in flight — bounded at four frames, flat with the length
  of the video.
- **Decode work** is still per frame *of the file*: the reader hands over all 84,067 frames
  of the reference video and the sampler keeps the candidates. That is the point of one
  front-to-back pass.
- **Decode is serialised** — one reader — and is the throughput floor; the scorer's workers
  overlap with it.
- **Every sample's record is retained**: timecode, the two scores, the compressed
  FeaturePrint (~2.5 KB) and a thumbnail key — 4.3 MB for 599 frames, under 10 KB each,
  which extrapolates to ≈90 MB at 2,805 frames and ≈170 MB at 14,400.
- **Pixels are decoded on demand**: a JPEG per grid cell, a native-resolution read for the
  preview.

Because the images are never all resident, nothing is evicted — so the per-group cap is a
*display* decision (`beyondCapByGroup`) rather than cache management.

## Platform traps

Each of these cost real time; the symptom is the useful half.

| Symptom | Cause | Fix |
|---|---|---|
| Dropping a file on the window does nothing at all | `.dropDestination(for: URL.self)` does not match a Finder file drag; the closure never runs | `.onDrop(of: [.fileURL])` with the rule in `VideoDrop`, which is testable |
| Every export fails with "decode failed" | `AVAssetReaderTrackOutput(outputSettings: nil)` vends the track's *stored* format — a compressed sample with no image buffer | name a pixel format (`kCVPixelFormatType_32BGRA`); omit dimensions to keep native resolution |
| A stray ~3.5 pt blue bar draws down the window, only while the window is key, only after clicking a frame; moves when the window is resized | SwiftUI's focus effect for the grid's `.focusable()` `ScrollView`, drawn in the wrong place | `.focusEffectDisabled()`; the focused *cell*'s border is the indicator |
| `EXC_BREAKPOINT` in a binding's getter, from `NSHostingView.layout()` | a `Binding(get:set:)` captured a **position** into a list that had shrunk since the row was built | address mutations by key and make them total; whole-value bindings, never nested paths |
| Clicking a custom slider never focuses it, so its arrow keys do nothing | a zero-distance `DragGesture` begins on touch-down and swallows a simultaneous `onTapGesture` | focus inside the drag's `onChanged` |
| Arrow keys stop stepping the preview after using its zoom buttons | a focused SwiftUI `Button` does not pass keys to an ancestor's `onKeyPress` | the overlay owns the keys with an `NSEvent` local monitor while it is up |
| The preview flickers on every step | a per-frame `.id()` rebuilds the view (blank → thumbnail → native) | stable identity + `.task(id:)` + keep the old image until the new one is decoded |
| Toolbar buttons read "Analy…" / "Expor…" with plenty of room | the controls were one `HStack` inside a single toolbar item, laid out and compressed as one item | one `ToolbarItem` per control |
| `Float`/`Double` conversion errors on the grouping threshold | `GroupingSettings.similarityThreshold` is a `Float` | keep the live knob a `Float` too |
| A view that holds the only reference to a cancelled stream keeps decoding | the throwing stream has no `onCancel:` | check cancellation at the top of each pull |

## Verified, and how

| Check | Result |
|---|---|
| `./run-tests.sh --strict` | 149 registered: 149 passed with real FlashVSR opt-in, 0 failed; default run skips one real-inference case; SeedVR2 real inference verified separately at 2×/4×/8× |
| Swift 6 compile (production app and custom harness) | 0 warnings; app build uses `-warnings-as-errors` |
| Peak RSS, streamed vs buffered | 67 MB vs 420 MB (measured, `/usr/bin/time -l`) |
| Real-Vision end-to-end (historical throwaway harness: synthetic clip → sampler → scorer → pipeline) | every sampled frame emitted, in order, with matching progress ticks |
| `./build.sh` + launch | Swift 6, warnings treated as errors; builds, runs, one window; Uncover helper bundled |
| Driven through the app | a folder drop opens the panel; a frame click seeks the player to that frame; order and range changes re-sort the grid; the cap shows `7 of 19` and reveal shows 19; Space opens the preview (and closes it); Escape closes it; a Right arrow steps the preview and the fit zoom is unchanged (51 % → 51 %); the toolbar shows full labels |

### Not verified

- The 50 ms fade is wired and demonstrably not instant, but screen capture latency
  (~150 ms) is too coarse to photograph a mid-fade frame, so the *duration* is asserted by
  construction rather than measured.
- No end-to-end pass over the 47-minute reference video through the app: the per-layer
  behaviour is covered and the pipeline is exercised end to end on short clips, but a full
  analysis run has not been watched.
- `Per group` is a persisted preference. Automated keyboard testing walked it down twice;
  it is back at its documented default of 7. If preview stepping seems dead, check it: with
  a cap of 1 every group's line holds one frame and ←/→ have nowhere to go.

## Conventions

- **Pure logic lives outside the views.** `FrameArrangement`, `FrameOrder`, `FrameLimits`,
  `VideoDrop`, `RankMap` and `FaceSelectionRule` are values with tests; the views route.
- **Every non-obvious decision gets a paragraph** saying what it is and what was rejected,
  matching the app’s existing style.
- **Tests defend behaviour, not implementation.** `Tests/README.md` says which suite covers
  what and why each seam exists.
- **No dead code.** Anything never instantiated is deleted rather than kept "for later".

## Public repository preparation

The MIT license and all upstream notices remain unchanged. Local source paths and
private fixture names have been replaced with generic examples; real inference tests
require an explicit `WAIFU_TEST_VIDEO` path. Model weights, application builds, local
configuration and caches remain ignored. Publication uses a single clean root commit
with a GitHub noreply author address, excluding the previous development history.

Preview regression validation: the strict suite passes 150 cases with zero failures;
the real model inference case remains an explicit opt-in. The new cases cover held-key
waiting, release and direction changes, 100 rapidly cancelled native frame seeks,
pre-cancelled reads, deadline recovery and successful full-resolution reads afterward.

## Frame selection

Plain click replaces the selection with the clicked frame; clicking it again keeps it
selected. Command-click adds or removes one frame and updates the selection anchor.
Shift-click selects an inclusive range in rendered order; repeated Shift-clicks retain
the anchor so a range can shrink or reverse. Command-Shift-click adds a range. Row mode
uses the actual group-row order, with ungrouped frames last. A hidden anchor starts a
new selection rather than including hidden frames. Command-A selects exactly the
current visible set, including caps and filters, and uses the focused visible frame as
the range anchor. Clearing results or selection also clears its anchor.

Validation: the strict suite passes 153 cases with zero failures and one explicit
real-inference skip. The app builds and passes strict signature verification.

## Three-state face filter

The Show section uses a small native segmented Faces control: All (default), With
faces, Without faces. `FrameFaceFilter` replaces the old Boolean in the session and
arrangement. Both the visible frames and slider bounds use the chosen filter, with
Without faces exactly complementing the significant-face predicate used by ranking.
Existing value limits and group caps still apply afterward.

Validation: 153 strict regression cases pass, including absent and sub-threshold
faces in Without faces; one real-inference case remains opt-in. The signed app
builds successfully. A rendered 340-point sidebar confirms all three labels fit.

## WaifuFinder branding and public release

The app, Swift package, executable and GitHub repository are named WaifuFinder.
Build output is `dist/WaifuFinder.app`, with bundle ID `com.alastorid.waifufinder`.
The source-drawn icon combines an abstract lens, mint viewfinder corners and a
best-frame sparkle on a plum tile. A generated preview is in `docs/assets/`.
Existing Application Support/Waifu paths, worker names and model release pins stay
compatible with installed setups. Uncover Waifu remains the reconstruction feature
name. Both hosted model cards use the new app name and repository link, retaining
upstream attribution and their existing model repository names. The app remains MIT;
model and upstream Swift notices keep their existing licenses.

## Review: full/quick analysis, processing UI and source seeking

The full/quick pass work was reviewed and retained. Full scores every decoded frame;
Quick scores every rounded nominal-FPS stride, preserving decoded ordinals and actual
PTS. Removed SamplingPlan tests were retired with that type; quick-mode gapped ordinal
emission now has a dedicated pipeline regression. Older settings missing analysisMode
decode with Full while retaining every saved preference. Long frame IDs also parse
beyond the former six-digit limit. Quick-pass cancellation is checked even while
skipping decoded buffers.

The growing grid was rebuilt/sorted on every 256-frame or half-second publication.
Those publications are removed: a nonisolated detached pass collects results and runs
cancellable grouping, then publishes/arranges the results once. Analysis and regrouping
show a dedicated scalar-only screen at most four times a second, with frame counts,
face counts, frames with faces, timing/rate, compared frames and provisional linked
groups. Generation guards prevent cancelled/replaced jobs from overwriting newer state.
The player/grid are unmounted while processing. Inference/export progress also replaces
the Uncover canvas. No end-to-end throughput improvement is claimed from the synthetic
layout previews. Grouping skips edges inside an already joined component, preserving
single-linkage groups and contiguous size-limit splitting; the controlled 200-frame
all-matching case needs 199 distances instead of 19,900. Regrouping is also off-main.

Uncover adds a source-frame seek bar before reconstruction, previous/next buttons and
held-arrow navigation that waits for frame reads. Fresh presses/scrubbing supersede
reads; generation/deadline protection keeps the latest result and avoids stuck waits.
Actual source frame timestamps come from sample timing metadata, with container edit
lists applied (raw media times differ for ordinary B-frame lead-in too); compressed
packet timing is the fallback. Changing the source frame preserves the ROI and sets
the interval start. Result navigation keeps its separate processed-frame transport.
The model selector is now a small native segmented SeedVR2/FlashVSR 1.1 control.
Browser still exports carry decoded timestamps rather than deriving VFR timing from
frame ordinal divided by nominal FPS. Global thumbnail clearing no longer races a
new analysis pass; cached files are retained for reuse.

The icon is now abstract: plum tile, mint viewfinder corners and lens handle, cream
lens/sparkle. Both model cards are published with WaifuFinder links; installer pins
advance to the updated cards while weight and manifest checksums remain unchanged.
MIT and upstream model licenses/attribution remain intact.

Validation: 140 strict cases pass, zero failures, one explicit real-inference skip.
New checks cover saved-settings migration, grouping cancellation/redundant comparisons,
quick ordinal gaps, actual VFR frame timing and exact export pixels, and 100 rapid
Uncover source seeks followed by forward/backward/clamped navigation. Rendered native
UI previews confirm the progress layout, seek bar and segmented model labels fit.

## Grouping: remove repeated feature-print decoding

The production comparison loop previously decompressed and JSON-decoded both prints
for every candidate pair. It now lazily decodes each print once per pass and calls
the same Vision distance on cached observations. Finished rows are discarded, so
decoded storage spans only the active 120-second forward window, rather than the
whole video. Invalid payloads are also cached as unavailable to avoid retries.
Single-linkage, component skipping, cancellation, thresholds, splitting and IDs
remain unchanged. This is a CPU optimization; no GPU path or changed distance
formula is introduced.

Controlled optimized-build benchmark: 12 real revision-2 Vision prints from generated
128×128 patterns, repeated into 600 frames at 30 fps with threshold 0.01. The prior
per-pair decoder took 10.652 s; the window cache took 0.0535 s (about 199× faster),
with identical groups. A 4,000-pair comparison alone took 0.264 s versus 0.000523 s.
These measure a synthetic workload, not end-to-end performance on a long video.
The vectors are 768 float elements; a raw squared-distance check matched Vision on
these prints, while ordinary Euclidean distance did not. Any future GPU path needs
format/revision and threshold-boundary validation before replacing Vision.

Validation: 142 strict cases pass, zero failures, one opt-in real-inference skip.
New cases verify one decode per valid/corrupt entry, release of completed rows,
bounded active-window retention, and equality to the per-pair Vision reference
across threshold boundaries, chronological windows and size-limit splitting.

## Grouping overlaps scoring

Analysis now feeds each ordered scored frame into an incremental union-find pass
while the pipeline's scoring workers continue. A new frame compares with earlier
frames within 120 seconds, using the same Vision distance, threshold and connected
component rule as the forward batch pass. Decoded prints leave the cache when they
age out; only IDs, timestamps and union-find metadata survive for final membership.
Missing-print frames also advance cache expiry. Final IDs and contiguous size-limit
splits are applied at EOF, using the shared presentation helper. Regrouping existing
results keeps the batch path. Both paths retain component skipping and cancellation.

The scalar progress screen reports grouped frames and provisional linked groups
during scoring; there is no second full comparison pass afterward. The grid still
receives results only once. Incremental input must remain chronological; the existing
scoring pipeline preserves sampler order, including gapped Quick-mode ordinals.

New regressions compare incremental and independent batch results across chained
components, thresholds, exact window boundaries, missing/corrupt prints, large gaps,
size splitting and real Vision observations. A gated scorer proves the first group
exists before later scoring workers are allowed to finish. Cancellation and active
cache expiry are also covered.

Validation: 146 strict cases pass, zero failures, one opt-in real-inference skip.

## Inline regrouping activity

Changing the grouping factor after analysis keeps the browser and player mounted.
A small determinate progress ring beside Grouping fills as comparison rows complete.
Its own observable progress object receives throttled updates, so those pulses do
not invalidate the grid or player. Regrouping uses a separate activity flag and
does not publish full-screen processing progress or intermediate group counts.
The current groups remain visible until the replacement
is ready. Completion, cancellation and failure clear the indicator; generation
guards prevent an older cancelled request from clearing a newer request's state.

Range sliders now accept clicks anywhere along the track: mouse-down selects the
nearest limit handle and moves it to the pointer. That handle stays captured until
release, with bounds/crossing clamps unchanged. Overlapping handles can expand in
either direction depending on which side is clicked.

Preview left/right holds now have their own repeat driver: after a 250 ms hold,
navigation continues without requiring OS repeat events. Each advance waits for
the current native frame read; fresh presses can still supersede pending reads.
Release, focus loss, read failure and closing the preview cancel repeat work. The
read gate uses the session's current preview ID rather than the frame captured
when its key monitor was installed. A short tap still makes just one step.
Validation: 147 strict cases pass, zero failures, one opt-in real-inference skip.

## Face definition sorting and filtering factor

Quick and Full compute Face and Face definition together during initial scoring,
with one native DetectFaceCaptureQualityRequest per retained frame. Its observations
supply both capture quality and the face boxes for definition. A native
DetectFaceLandmarksRequest(.revision3) refines those same boxes only when a face
was found, without repeating detection or decoding. No deferred scan runs.
Old faceDefinition-mode settings decode as Full.
Aesthetics, feature prints and thumbnails use the cropped analysis image.
AnalyzedFrame stores the cropped thumbnail aspect ratio during scoring. Grid cells
use that ratio for both their loading placeholder and loaded image, so portrait
crops are not clipped into fixed 16:9 boxes. Old records fall back to JPEG dimensions.
It applies the track's preferred transform before detection/cropping. Native images
are retained only by two in-flight scoring workers; no full-video image cache or
repeated random native seeks is introduced.

Each fully bounded face box at least 8×8 native pixels is inset by 10% to reduce
background influence, cropped before resizing to 64×64 grayscale, and measured by
contrast-normalized Laplacian energy. The monotonic bounded mapping is E/(E+0.1).
Native flat-block coverage discounts mosaic texture. Landmark placement estimates
then discount sharp texture in uncertain facial features or partial-face false
detections: mean placement error e applies a bounded 1/(1+(e/0.02)^2) reliability
factor. Missing, negative or nonfinite estimates leave definition nil; landmark
request failure leaves capture quality and the other scores available.
The frame receives the maximum verified face score, independent of the largest
face, frame position, Vision capture quality and the normal 1% area threshold.
Missing detections remain nil. This is a relative detail heuristic; it does not
verify occlusion, and noise/compression/sharpening can cause false positives.

The optional faceDefinitionScore is separate from faceScore and old records decode
without it. Face definition is an always-visible fourth ordering/range control,
with canonical precedence Time → Face → Face definition → Art. All four default
to enabled; absent definition scores fall through to Art. Each factor can be
disabled or reversed independently, and ranges filter independently of sorting.
Face filters include measured small faces. Native decoding first identifies large
opposing near-black borders using a <=256px grayscale raster and crops the input
once for all scoring requests. Face area is relative to this visible picture, not
the padded video canvas; native definition pixels are retained. Analysis rasters
are bounded on their longest edge. Uniform black/dark frames remain untrimmed.
Full checks every ordinal, Quick retains its original sampling cadence. Sorting,
filtering and presets reuse stored scores without rerunning face detection.
Presets → Face definition only changes ordering without initiating a scan.
Tiles label the value D; full-resolution preview labels it Face definition.

Regression coverage includes small sharp edge crops versus large blurred centre
crops, translation invariance, flat/tiny/clipped regions, group caps and score
thresholds independent of area/capture quality, old-record decoding, full frame
coverage with scaled thumbnails and native pixels, video rotation, black-padding
cropping, portrait thumbnail metadata, no-face detection, readable features versus
sharp uncertain detections, and missing/invalid landmark evidence. Real-video
ranking validation uses local private inputs; their paths, frames and derived
images are excluded from the repository.
Validation: the strict regression suite and the app build pass. Definition ranking
was checked over an entire local video with the production detection and landmark
scoring path. Private diagnostic images are temporary and are removed afterward.

## Uncover source loading indicator

Source-frame loading is indicated by a small overlay in the preview's lower-right
corner, outside the seekbar layout. Showing or hiding it cannot resize the slider,
and the overlay does not intercept clicks or drags in the canvas.

## Uncover original versus AI comparison

Completed results compare the untouched, time-matched original frame against the
AI reconstruction in wipe, side-by-side and toggle views. The reference image is
shared with imageA, independent of feather/blend settings. Rendering no longer
reads or composites the Lanczos baseline, reducing per-frame work for preview and
export. Baseline files remain preparation/validation artifacts. Completing or
restoring a result clears the full-panel Original override to reveal comparison.
Canvas labels and the toggle action explicitly name Original and AI.

Validation: 154 strict cases pass, zero failures, one opt-in real-inference skip.
Coverage verifies exact original reference pixels, rendering without baseline files,
blend independence, and comparison becoming visible after reconstruction completes.

## SeedVR2 fusion and targeted runtime optimization — 2026-10-11

The default prepared HF release now merges 3B SwiGLU gate/value rows and both models'
VAE QKV rows, including ordinary bias, quantized scale and offset arrays. Transformer
QKV was already packed. Nonlinear layers and data-dependent normalization cannot be
folded into constant weights. Export remains entirely native Swift/MLX and writes the
manifest last; setup performs no conversion. 7B uses GELU with bias and retains its
last text-stream checkpoint parameters and updates both streams in its final layer.

The runtime matches F16 VAE activations to F16 convolution weights, avoids repeated
buffer-cache purges (cache capped at 64 MB), compiles the small SwiGLU pointwise chain
and loads each bounded parameter set in one submission. 3B transformer activations
use F16 with F32 normalization/rotary math. The 7B transformer stays F32: a tested
F16 version overflowed and was rejected. Streamed mode loads only the active VAE
encoder/decoder and one local transformer block, releasing it after evaluation.
198 byte-identical shared attention tensors use aliases, removing 325 MB of duplicate
3B weights without changing output bytes. 7B uses normalized pixel position grids,
updates both streams through the final block and bypasses the 3B output Ada stage.
Temporal context, causal tails, source frame ownership and tile geometry are preserved.

`./tools/setup-native-uncover.sh` installs 3B into `.native-runtime/model-optimized`.
`./tools/setup-native-uncover.sh --7b` installs 7B into `.native-runtime/model-7b-optimized`.
The optional bundle is the `7b/` subdirectory of the same credited HF repository;
`tools/seedvr2-7b-release.json` pins its files separately. Automatic streaming applies
to 7B, 8 GB machines and single-pass jobs; repeated 3B passes on larger machines
retain resident weights. The budget is the smaller
of 6 GB and half installed RAM. Worker reports include stage times/loading/fusion
flags and a model identifier, without local absolute paths.

`docs/seedvr2-fusion-validation.json` records paired real-video timing, peak physical
footprint, pixel comparisons, tested scales and chunk-boundary coverage. Measurements
were made on a 16 GB M3 with a 4 GB streamed-process budget, not an actual 8 GB Mac.
Packing preserves stored numeric values; F16 runtime math has small pixel differences.
Runtime variability and system swap remain dependent on the other applications in use.

Final short-clip paired medians: original 3B resident 11.576 s / peak 5.186 GB;
prepared 3B resident 9.849 s / peak 4.029 GB; prepared streamed 9.033 s / peak
2.266 GB. Original 7B streamed 13.491 s / peak 2.893 GB; prepared 7B streamed
12.602 s / peak 2.244 GB. Every prepared path won all three paired runs. These are
five-frame 128→256 fixtures, not universal timing guarantees. F16 pixel error is
about 0.13/255, maximum 4. Final alias/batch changes also preserve the optimized
30-frame chunk-boundary output byte-for-byte. Isolated committed-source suite:
154 pass, zero fail,
one separately opted-in inference test; the app builds and passes signature checks.

Final 30-frame default comparison: 197.191 → 155.452 seconds, physical peak
5.341 → 4.371 GB. Separate streaming chunk proofs for both models stayed under
a 4 GB budget, and 4×/8× 3B proofs passed. The final app-backend/export proof
passed with automatic single-pass streaming, exact PTS, matching audio and no Python.

The pinned hosted manifests, embeddings and one new LFS shard per model were
downloaded and verified from the published revision. Default native setup passed.
Obsolete original/prototype checkpoints and the publication checkout were removed;
legacy local model paths are links to the verified optimized bundles.

## Optimized SeedVR2 model-size selection — 2026-10-11

Uncover's compact segmented algorithm selector offers SeedVR2 3B, SeedVR2 7B and
FlashVSR 1.1. Both SeedVR2 choices require packed-projections-v1 / qkv-packed-v1
q4/group64 native bundles; original checkpoints and Python inference cannot be
selected or substituted. Native setup registers both sizes independently in the
local backend configuration, so installing one does not erase the other's path.
Existing optimized installs are discovered from the active path and its standard
sibling directories. A missing size shows its own setup command.

The existing algorithm-switch path cancels in-flight work, preserves experiment
settings and shares prepared inputs while retaining separate per-size result caches.
Coverage checks independent model resolution, rejecting unfused/Python fallbacks,
missing-7B guidance and result ownership/input reuse when switching 3B↔7B.

Both installed sizes were registered and exercised through the app's backend
factory, with optimized architecture/fusion flags checked in each native report.
The real 3B and 7B inference/export proofs passed, preserving source-relative PTS
and matching audio. The workspace strict suite passes 160 cases with zero failures.

Isolated staged-source validation: 156 strict cases passed, zero failures, one
opt-in real-inference skip. Both separately opted-in native model/export proofs
also passed, including selected-size and optimized-fusion assertions. The rebuilt
app passes deep signature verification.

## Native Metal VAE normalization fusion — 2026-10-11

Both optimized SeedVR2 sizes now use hierarchical F32 Welford statistics directly
on channel-last F16 activations. The epilogue combines affine scaling, the existing
F16 boundary and SiLU. This removes full activation-sized F32/group-layout copies.
Temporal axes, causal tails and per-frame attention normalization remain unchanged.
Published model weights and setup pins remain compatible; no model re-download or
conversion is needed. The worker reports `fusedVAENormalization`; the app's real
integration proof checks it. `--no-vae-norm-fusion` retains the original path for
benchmarks, and `--benchmark-vae-fusion` checks component accuracy/timing.

The component proof covers batch/rank changes, small/large tensors, affine parameters,
large offsets and exact constant-input parity. Six paired five-frame comparisons
per model include shader compilation in each fresh worker, three orders each way.
All outputs repeat byte-for-byte within each path. Median paired time reduction:
3B 6.57%, 7B 4.65%; each wins five of six pairs. Timing drift is visible during
sustained runs, so a speed improvement on every invocation is not guaranteed.
Median physical peaks: 3B 1.977→1.535 GB, 7B 1.947→1.537 GB. RGB byte errors:
3B max 2 / PSNR ≥62.19 dB; 7B max 1 / PSNR ≥65.84 dB. Measurements use a
16 GB M3 and 4 GB streamed worker budget, not an actual 8 GB Mac. The complete
sanitized record is `docs/seedvr2-vae-fusion-validation.json`.

The 30-frame / two-chunk 3B pair passes with max RGB error 2 and PSNR ≥60.45 dB
across all 30 frames. Time 108.874→102.862 s; physical peak 2.327→1.825 GB.
This is one streamed long pair, not a statistical guarantee. The reduction affects
runtime kernels; no new model release or setup revision is needed.

The strict suite passes all 161 cases, including the opted-in real 3B native
reconstruction/export proof. The separate real 7B proof also passes. Both assert
Metal normalization fusion is enabled, preserve exact source-relative PTS and
matching audio, produce finite frames and run without Python.
The rebuilt app and embedded workers pass deep, strict signature verification.

## Causal preparation, pointwise and residual VAE optimization — 2026-10-11

Both SeedVR2 sizes now prepare causal padding and compact channel-last history in
one Metal pass, use matrix multiplication for eligible 1×1×1 convolutions, and
compile bias/residual additions while preserving both F16 rounding boundaries.
Existing optimized weights, Hugging Face releases and setup pins remain compatible.
Controls: `--no-vae-causal-fusion`, `--no-vae-pointwise-conv`,
`--no-vae-residual-fusion`; component proof: `--benchmark-vae-convolution`.
The proof covers active history, stride variants, zero-padding fallback,
noncontiguous inputs and exact rounded residual parity. It passes.

All five short-clip PNGs match byte-for-byte before/after for 3B and 7B, including
repeat runs. All 30 frames in the longer 3B comparison also match byte-for-byte.
The strict suite passes 160 cases, zero failures, with real app inference skipped.
The new real export assertions are added but await desktop inference validation.

On a fanless 16 GB M3 MacBook Air, four short pairs observed median reductions of
11.9%/6.5% for 3B/7B and median worker peaks 1.535→1.444 GB. The long 3B pair
instead slowed 81.191→120.167 s and increased physical peak 1.854→1.958 GB; the
unchanged transformer stage also slowed.

Mac mini follow-up: a four-pair 3B comparison on a 16 GB M4 Mac mini used five
prepared 128→256 frames with a 4 GB worker budget. Median paired time reduction was
2.383%, with 3/4 pairs faster; median physical peak was 1.495→1.415 GB. All five
reconstructed PNGs matched byte-for-byte in every pair. The first baseline run took
9.860 s, while the other three took 6.844–7.244 s, so this short result remains
provisional. A long-clip 3B comparison on the Mac mini, long-clip 7B timing, and new
runtime app export integration remain pending. All measurements are retained in
`docs/seedvr2-causal-fusion-validation.json`.

`tools/benchmark-native-uncover.swift` runs prepared jobs serially in alternating
baseline/candidate order, checks PNG byte parity, and emits a path-free summary.
It accepts either optimized model size; README gives fixture preparation and usage.

The app rebuild succeeds and passes deep, strict signature verification. The
benchmark tool passes Swift type checking and a CPU-only two-pair harness check
covering alternating order, complete PNG counts, summary metrics and parity;
it refuses existing output directories. The Mac mini VAE worker comparison above
ran on `adae4c5`, before the reduced-ROI changes in `5aa454b`; it did not exercise
this revision’s learned restoration path.

## True low-resolution Uncover input and learned ROI restoration — 2026-10-11

The prior SeedVR2 worker read downsampled PNGs, enlarged them back to ROI dimensions, then
ran temporal inference. This change removes that pre-inference enlargement. The native MLX
worker now tiles, conditions and restores at the actual degraded dimensions. It writes the
temporal result at that size; the app then applies Real-ESRGAN x2plus through Core ML once per
doubling and crops the final frame to exact ROI dimensions. Downsample choices are 2×, 4×, 8×,
16×, 32× and 64×. For example, a 1920×1080 ROI at 32× gives SeedVR2 about 60×34 input pixels,
then five learned x2 image passes return to 1920×1080; it does not create a 61,440×34,560
video. Output-region size is limited to 16 megapixels. Large factors synthesize more detail,
and the spatial model processes frames independently, so quality and temporal consistency need
review on moving footage.

Setup pins and verifies the upstream ~30 MB Core ML x2plus archive, compiles it to a native
`.mlmodelc` directory with Swift/Core ML, removes the downloaded archive and stores its path
alongside the selected optimized SeedVR2 configuration. No Python is involved. The app bundle
includes Real-ESRGAN attribution and its BSD 3-Clause license. The upstream model is general
image restoration and runs on Apple GPU through Core ML; it is not an anime model or an ANE
model. See `docs/Uncover.md` for the complete geometry and limitations.

Verification on this Mac: `./build.sh` succeeds, including the native temporal planning and
attention checks; Swift typechecking of the Core ML compiler helper succeeds; `./run-tests.sh`
passes 161 cases with one real-inference case skipped. Added checks cover 64× extraction
geometry and the output-area guard. The Core ML package was not downloaded/compiled here,
and no end-to-end real app or learned spatial inference was run on this revision. The
earlier Mac mini VAE worker comparison above predates this change and does not validate
it; real quality, memory and speed validation of the reduced-ROI path remains pending
on the Mac mini.

## Hosted PGTFormer and optimized RealESRGAN releases — 2026-10-12

Every model the app installs now comes from a pinned release under our own Hugging Face
organisation. SeedVR2 3B/7B and FlashVSR already did. This change adds the two that did
not: PGTFormer arrived as a folder the user had to supply, and RealESRGAN was pulled from
a third-party GitHub release — and, worse, that release was not the model this project had
been validating. `NativeSpatialUpscaler` already contained a `verified` path keyed on
`waifu.optimizationVersion`, the validated flag and the exact weights hash, and nothing in
the shipping download path ever produced a model that satisfied it. The optimized conversion
was built, sealed and benchmarked, then only ever exercised by the developer setup script.

**RealESRGAN — our optimized conversion replaces the upstream package.** The upstream
`RealESRGAN_x2plus_522_fp16` model is fixed at 522 pixels with a 10 pixel context halo, so
every small tile is padded up to 522. Ours fuses the first convolution, declares 256, 384
and 522 as enumerated shapes with a 160 pixel halo, and records the measured original-model
parity in the model's own metadata, which the app requires before using the fast path. It is
therefore a different model, not a repackaging, and the "if not changed, use original"
fallback did not apply. Weights are unmodified: `manifest.json` pins the SHA-256 of the
original `RealESRGAN_x2plus.pth` and of the BasicSR architecture file, and
`nativeValidation` records a maximum absolute error of 0.0069 across all three sizes against
a 0.01 bound. The import path for a bare legacy `.mlmodelc` is kept, so an existing external
RealESRGAN install still works.

**PGTFormer — hosted as the original network, and non-commercial.** The bundle is the
original encoder and decoder, split at the original function boundary, with the encoder's
validated fingerprints copied verbatim. `tools/stage-hf-bundles.sh` refuses to stage a
package whose bytes disagree with the conversion's own `splitPackageSHA256`, and the upload
path refuses to run without `WAIFU_HF_ALLOW_NONCOMMERCIAL=1`, creating the repo private.
Upstream is BSD-style with a non-commercial restriction: redistribution for non-commercial
use is permitted, commercial use needs permission from Xidian University. That is a
decision about our distribution, so it is made explicitly by a person, not implied by a
default. It is not the same as the RealESRGAN BSD 3-Clause case and was not treated as one.

**One manifest shape, not two.** `manifest.json` pins model files only; licences,
attribution, the model card and the validation record are pinned by the release descriptor
in `supportFilesSHA256`, exactly as SeedVR2 and FlashVSR already do. An earlier draft of the
staging script hashed everything into `fileSHA256`, which made every support file verify
twice. `tools/upload-hf-bundles.py` asserts the two sets are disjoint before writing the
descriptor, and the downloader fetches each member once — 9 requests cold, 0 on a rerun.

**Compiled Core ML is not published.** Both releases ship `.mlpackage` sources. The app
compiles the RealESRGAN package at install time and the PGTFormer worker compiles and caches
its own, because a `.mlmodelc` is tied to the compiler and macOS version that built it.

**Verified here.** `tools/download-coreml-release.swift` was run against a local server
serving the staged bundle: 9 cold downloads, 0 on rerun, and a one-byte change to a support
file was detected and the file replaced. `waifu-pgtformer --configure --model` accepted the
staged PGTFormer bundle and wrote its config. The staged RealESRGAN package compiled with
`MLModel.compileModel` and `NativeSpatialUpscaler` loaded the result. `./run-tests.sh --only
Models` passes 6 cases including two new ones. Production sources typecheck under Swift 6
with `-warnings-as-errors`.

**Published.** Both repositories exist and are pinned in `tools/*-release.json`:
`alastorid/PGTFormer-Swift-CoreML` at `160a4b92`, and
`alastorid/RealESRGAN-x2plus-Optimized-CoreML` at `56fcfbbe`. `./build.sh` completes and
bundles all five descriptors.

**PGTFormer is private, and the app cannot download it.** The app fetches anonymously, with
no token and no Hugging Face login. An anonymous request for the private PGTFormer repo
returns 401; the public RealESRGAN repo returns 200. PGTFormer in Settings → Models therefore
fails at the download step until that repository is made public. That is a deliberate gate
rather than an oversight: the upstream licence restricts redistribution to non-commercial
use, and publishing a non-commercial model publicly is a distribution decision. The licence
permits it *provided* the use stays non-commercial.

**Verified against the live Hub.** `tools/download-coreml-release.swift` fetched the pinned
RealESRGAN release over HTTPS, verified every hash, and the downloaded `.mlpackage` compiled
and loaded through `NativeSpatialUpscaler` — the sequence the app performs. The PGTFormer
bundle was confirmed earlier against a local server and through `waifu-pgtformer
--configure`, but not anonymously from the Hub, for the reason above.

**Three things were caught while publishing, each of which would have failed only at the
last moment.** `git config --global --get-regexp filter.lfs` was empty, so the 250 MB
encoder weights would have been committed as plain git blobs, which Hugging Face rejects
above 100 MB. The publisher now runs `git lfs install --local` inside a throwaway checkout,
leaves no filter behind in global config, and refuses to push if nothing became LFS. The
committed pointer's SHA-256, `0de003e9…fc1b9`, matches the encoder entry in the manifest's
`splitPackageSHA256`, confirming the published bytes are the validated ones. Second, a new
Hub repository already holds a commit carrying its own LFS `.gitattributes`, so the first
push was rejected; the publisher now fetches `main` and commits on top of it rather than
orphaning or force-pushing that history, and the staged bundle no longer carries a second
`.gitattributes`. Third, Hugging Face validates model cards and rejected
`license_name: "PGTFormer BSD-style, non-commercial"` against `/^[a-z0-9-.]+$/`; the card
now carries the lowercase slug `pgtformer-bsd-style-non-commercial`, and the full terms live
in LICENSE and NOTICE.

Separately, `ModelInstaller.package` first returned the package's `Manifest.json` rather
than the package directory; `MLModel.compileModel` would have rejected that on every
RealESRGAN install, and only the end-to-end run exposed it.

## Licence inventory and distribution modes — 2026-10-12

Audited every upstream the shipped app depends on. The result is a table in
`docs/Licensing.md`; the short version is that WaifuFinder's code is MIT and almost
everything it drives is Apache-2.0, BSD-3-Clause or MIT, with exactly one exception.

**PGTFormer is the incompatible one.** Its upstream licence permits redistribution and use
for non-commercial purposes only, and commercial use needs permission from Xidian University.
MIT cannot absorb that, and attribution does not help: the licence restricts the *use*, not
the packaging. The exposure is sharper than an ordinary dependency conflict, because the app
fetches models anonymously on request. There is no bundle to omit — a commercial user taps
Download and an MIT-licensed app hands them a non-commercial model. Refusing at download
time would not fix it; the offer is the problem.

So the restriction is enforced at build time. `WAIFU_COMMERCIAL_DISTRIBUTION=1 ./build.sh`
compiles with `-D WAIFU_COMMERCIAL_DISTRIBUTION`, which makes `ManagedModel.offered` omit
PGTFormer from the Settings list, the algorithm selector and the release catalogue, and the
build omits `tools/pgtformer-release.json` from the bundle. `UncoverAlgorithm.allCases`
deliberately stays complete so a session saved while PGTFormer was available still decodes
and reports itself unavailable instead of failing to load. The default build is the
non-commercial one. `Sources/Waifu/Model/WaifuDistribution.swift` records the reasoning.

**Two facts that a build flag cannot resolve, and that need a human.** First,
`kepeng/pgtformer-base` declares no licence at all in its card metadata and ships no LICENSE
file — the repository contains only `.gitattributes`, `README.md`, `config.json` and
`model.safetensors`, and the README says nothing about terms. The only available terms are the
non-commercial LICENSE in the author's GitHub repository. Inferring that they cover the
weights he published himself is reasonable and is not a grant; without one, default copyright
applies and our redistribution rests on nothing documented. The author should be asked.
Second, whether a use is commercial is a fact about the user, not about the build.

**Three bugs caught while building this, each found only by running it.** The `offered`
filter's condition was inverted, so the *default* build was the one dropping PGTFormer and
the commercial build kept it — the opposite of the intent, and invisible to inspection of the
code alone. `build.sh` used `mkdir -p` on the release catalogue, so building the default mode
and then the commercial mode into the same `dist/` left the PGTFormer descriptor sitting in
the commercial bundle; the directory is now cleared before it is repopulated. And the new
licence test asserted commercial expectations unconditionally, so it failed in the default
build until the assertions were made mode-aware. The suite passes under both flags:
`./run-tests.sh --only Models` is 7/7, and the same sources compiled with
`-D WAIFU_COMMERCIAL_DISTRIBUTION` are also 7/7.

Verified: both modes build to a signed `.app`; the default bundle carries five release
descriptors and the commercial bundle four.

## WD-Tagger Swift integration — 2026-10-13

Implemented opt-in native WD SwinV2 tagging in batches of eight, sharing the original
sampled images with scoring and tagging before incremental grouping. Tail batches
are padded inside the backend; only real frames are emitted. General probabilities
are stored losslessly in versioned zlib Float32 payloads; old frame and settings JSON
continue to decode with tagging disabled and optional frame fields absent. Tagged
thumbnail keys carry `-tagger-v1`.

The drop zone has the fixed-width on/off control and threshold slider. The order
panel supports normalized autocomplete, keyboard selection, removable tokens and
all-trait filtering against the complete probability vector. Rating and general
confidence have independent sorting/range controls, and preview has a Traits row.
Search keyboard ownership is limited to its focused field and live overlay state.

ModelLibrary installs, validates, compiles and tracks the tagger independently of
Uncover models. Its Apache-2.0 release is included in commercial and default builds.
The offline native parity fixture uses original, unfused Torch weights with a runtime
BGR input swap, independently of the conversion implementation.

Integration exposed two defects in the prepared conversion: loading the state dict
overrode the early BGR weight permutation, and Core ML defaulted its I/O to Float16.
The converter now permutes after loading, explicitly exports Float32 I/O, and validates
against original weights with a runtime BGR swap. Native geometry was checked against
`/Users/henry/wd-tagger-photos-app/wd_daemon/server.py`: floor resize dimensions,
white alpha compositing before downscale, and floor top/left letterbox padding.

Final release: `alastorid/WD-SwinV2-Tagger-v3-CoreML` revision
`608b1a7cf3c4` (descriptor pins the full revision). Independent BGR validation
matched 15/15 top heads, maximum absolute error 0.0024964. General traits use
the daemon's strict `probability > threshold` boundary.

Final verification: `./run-tests.sh` passed 196 cases, zero failures, with four
pre-existing opt-in Uncover model cases skipped. WDTagger native parity and all
geometry/migration cases passed. `./build.sh` produced the signed app under `dist/`;
the embedded release descriptor matches the corrected published revision and deep
strict signature verification passed. Real hardware keyboard smoke for the new
search field with preview/Uncover overlays remains a manual check.

## WD-Tagger optimization audit — 2026-10-13

Full native batch benchmarking exposed padded Core ML output rows: mixing physical
strides with logical NSNumber indexing could crash. The decoder now copies Float32
rows directly, preserving actual row/column strides. Model-free padded/strided tests
and mixed-image full native batch tests cover all eight rows. Preprocessing reuses
its RGBA buffer, and autorelease pools scope frame/batch intermediates.

Measured GPU path: 11.65 frames/s, 973.6 MB peak physical footprint on the benchmark
fixture. Corrected boxed-decoder baseline: 11.79 frames/s, 972.4 MB; these runs do
not establish material overall speed or memory gains. Neural Engine eligible mode
used about 4 GB and failed parity (13/15 top heads, max error 0.1537), so the default
remains CPU and GPU. Disabling intermediate caches showed no benefit. Weight/model
compression and attention-memory fusion remain unimplemented candidates requiring
independent validation. Details: `docs/wd-tagger-runtime-benchmark.json`; reproducible
native benchmark: `tools/wd-tagger-benchmark.swift`. Final full suite: 197 passed,
zero failures, four existing Uncover cases skipped. Signed app rebuilt and verified.

## Explicit SwinV2 SDPA trial — 2026-10-13

Built isolated `.tagger-sdpa/WDTagger.mlpackage` using
`tools/experiment-wd-tagger-sdpa.py`. Preserved all 24 cosine-attention blocks,
Q/K normalization, learned temperature, relative-position bias, window masks and
batch 8, while exporting 24 native SDPA operations. PyTorch rewrite parity max
error 1.55e-6 and top15 overlap 15/15 for all 8 initial frames.

Paired GPU runs (15 warm batches per process; baseline/candidate/candidate/baseline)
measured baseline 11.66 fps/944MB peak average versus candidate 14.21 fps/727MB,
about 21.8% higher throughput and 23.0% lower peak footprint. Package grows from
about 232MB to 287MB because of precomputed position-bias/shift-mask constants.
Synthetic validation covered 24 diverse frames; every frame retained all 15 top
heads. Maximum error 0.01569 versus baseline 0.02138; both had one threshold
classification difference. This exceeds the earlier gradient-test 0.01 bound on
some flat-color inputs, so the trial report does not call broader tolerance passed.
No representative anime-image corpus was tested. Native isolated suite 9/9 passed.
An earlier concurrent run aborted with an internal MPSGraph tensor shape/stride
assertion; standalone Onone inference and the isolated suite passed, but the
cause is not established. The production release remains unchanged.

Baseline compute plan prefers 602 operations on ANE and 409 on GPU, with 99 device
changes in listed operations; GPU-only plan prefers 1012 on GPU. These are compiler
placement counts, not a trace. Baseline all-mode memory was 280.8MB after loading
and 3516.7MB after first prediction; latest peak 3820MB/12.1fps, top15 overlap13/15.
This implicates inference workspaces/intermediates plus hybrid layout handling,
not just model weights; exact allocation attribution remains unmeasured. Apple
documents 64-byte ANE last-axis alignment and reshape/transpose copy risks.
Reports: `docs/wd-tagger-sdpa-experiment.json`, `docs/wd-tagger-sdpa-validation.json`.

## Fused tagger activation and Metal assertion investigation — 2026-10-13

Promoted the rank-four SDPA variant, published and pinned revision
`95615611e0cb63e081b6c9904ee9d4047c90e7d3`, rebuilt the signed app, and installed
and selected the verified compiled model in Application Support/Waifu/Models/wd-tagger.
The compiled model reports `swinv2-sdpa-rank4-v1` and `waifu.sdpaRank4=true`.
The project `.tagger-build` package now matches this variant; the earlier package
and validation record are retained with `pre-sdpa` names for rollback/reference.

The original crash report located the assertion in Core ML/Espresso's
`MpsGraphInferenceOperation::EncodeMemoryBuffers` while preparing the prediction,
not in the Swift probability decoder. The initial rank-five variant also passed
later replays, so the precise original runtime trigger was not conclusively isolated.
The release replaces all rank-five SDPA/broadcast views with rank-four queries,
keys, values and masks, folding window and head axes together without dropping masks
or changing cosine-attention math. Conversion and staging now enforce this invariant.
There is no exception swallowing, assertion disabling, serialized global workaround,
or CPU fallback. This is a tested encoding remediation; the original internal cause
remains unconfirmed, and no claim of conclusive reproduction/fix is made.

The new encoding passed 72 concurrent mixed batches across three independent models,
15 repetitions of the exact native test sequence under concurrent GPU load, a permanent
two-model native concurrency regression, and six further batches from the installed model.
Final suite: 198 passed, zero failed, four unrelated Uncover cases skipped. Model loading,
full batches, tails, cancellation, input identity and probability parity were exercised.
Native warm GPU benchmark: 15.13 fps, 686.8 MB peak footprint. All 24 broader synthetic
frames keep all 15 reference top heads; flat-color probability deviation remains documented.
Release and local install manifests/support hashes match; hosted manifest and validation
hashes were verified. The signed app embeds the exact new descriptor and passes deep
strict signature verification. Details are in `docs/wd-tagger-sdpa-release.json`.


### Model and development-file cleanup (2026-10-13)

Archived exploratory tagger scripts under `reference/model-development/wd-tagger/` and PGTFormer diagnostic scripts under `reference/model-development/pgtformer/`. Historical native conversion outputs and fixtures moved to the ignored `reference/model-development/native-artifacts/` archive. Removed all three `.tagger-*` directories after retaining the final published rank-four SDPA release, plus obsolete PGTFormer/RealESRGAN model candidates and original upstream PGTFormer weights.

Retained all six published bundles under `models/published/`, with SeedVR2 3B/7B and FlashVSR using links to default installed bundles to avoid duplicate weights. Moved final restoration models into `~/Library/Application Support/Waifu/Models/{seedvr2-3b,seedvr2-7b,flashvsr,pgtformer,realesrgan}`; final fused WDTagger remains in `wd-tagger`. Updated existing app configurations and ownership records. Verified every installed bundle against its pinned descriptor and every model/support-file checksum. Legacy native-runtime model paths are compatibility links. No remote releases were modified.

WDTagger discovers a complete installation in the default/custom Models storage folder when its configured path is absent or stale; ModelLibrary and offline tests use that selection. Added a regression for default discovery, stale configuration, custom preference and incomplete installations. Active tagger conversion now uses `.build/model-conversion/wd-tagger`, and staging/publishing uses `models/published/`. See `models/README.md` and `reference/model-development/README.md` for the layout.

Cleanup validation: 199 regression cases passed, 0 failed; 4 opt-in restoration-video cases skipped. Native fused WDTagger inference and concurrency cases ran against the default installed model. Signed app rebuilt successfully and deep/strict code-signature verification passed. Maintained Python/shell scripts passed syntax checks; publishing CLI help checked with the existing conversion Python runtime.

## WD Tagger batch and further-optimization measurements — 2026-10-15

Built isolated rank-four SDPA models for fixed batches 1, 2, 4, 16 and 32, plus
reciprocal-square-root Q/K normalization, folded input normalization, INT8 and
INT4 weight candidates. Native Swift probes also compared fast-prediction
specialization, low-precision GPU accumulation, and Accelerate RGB packing.
Production model, descriptor and batch 8 were retained. The converter now accepts
`--batch-size`, defaulting to 8; the production renderer is accessible internally
so the benchmark uses its exact geometry and reusable buffer.

On the M3/16 GB machine, cooled batch 8 processes measured 14.25–14.75 fps and
640–654 MB peak footprint. Batch 32 gave 14.59 fps / 1,809 MB. The final independent
batch 2 / batch 8 pair measured 13.38 / 14.64 fps and 464 / 654 MB: 29.1% lower
peak memory at 8.6% lower throughput. Earlier batch 2 measured 14.43 fps / 450 MB;
the roughly 2% speed penalty in that comparison did not hold across repeats.
Batch 2 matched production probabilities exactly on 48 video frames plus 16
synthetic inputs, using original FP32 unfused weights with runtime BGR as an
independent reference and exact Swift-prepared tensors. It remains an isolated
candidate because the current backend/model input contract is fixed at eight.

No tested graph/runtime/packing change established an additional speed benefit.
INT8/INT4 packages shrank but used about 1.94/1.91 GB at runtime and increased
natural-frame maximum probability error to 0.074/0.808 versus production 0.00989.
An initial sustained sequence entered elevated thermal state and was excluded;
cooled comparisons began and ended at nominal thermal state. These are short
warm measurements, not a sustained whole-video throughput claim.

The first tagger regression run aborted inside Core ML/Espresso EncodeMemoryBuffers
with the prior MPSGraph `shape.count = 0 != strides.count = 2` assertion signature.
It overlapped CPU reference generation; a causal connection is not established.
An isolated rerun passed 13 tagger/contract cases, including concurrent independent
models. The intermittent assertion is unresolved, so the earlier rank-four
encoding remediation is not a conclusive fix. Python syntax and installed model
manifest-file hashes passed. Details and reproduction tools are documented in
`docs/WDTagger-optimization.md` and `docs/wd-tagger-batch-optimization.json`.

### Further WDTagger layout/runtime probes and GPU tools — 2026-10-15

Tested combined shift/window gathers and persistent window order through residual,
norm and MLP. Both rewrites matched Torch and all 64 native production outputs
exactly. Persistent layout reduces logical transpose/reshape/gather counts but
showed no reliable speed or footprint gain across two cooled orderings; gather
layout was slower. Production package and backend settings remain unchanged.
Results: `docs/wd-tagger-layout-optimization.json`.

Installed official Apple-silicon Xcode 26.6 (17F113) in `/Applications/Xcode.app`.
Apple signature/Gatekeeper checks passed; user completed license/first-launch
setup. Metal Toolchain 17F109 is installed, reports Metal 32023.883, and compiled
and linked a sample kernel. Global xcode-select remains CLT; profiling commands
set DEVELOPER_DIR explicitly. Metal/Core ML timeline and M3 Performance Limiters
counter captures succeeded. A separate GPU Counters instrument initially gave an
unsupported-profile warning; configuring Metal Application's counter set worked.
Shader interval export remains empty, so no per-kernel hotspot is established.
Trace exports created large temporary caches and exhausted disk once; removed
only disposable export caches and retried the interrupted toolchain download.
Saved captures and setup evidence remain under ignored `.build/` directories.

An isolated MLX 0.32.4 port uses fused LayerNorm, compiled blocks and standard or
zero-padded forced-fused SDPA. Its eight-input FP32 smoke check matched original
FP32 probabilities within 2.3e-6. Cooled 64-input prediction-only throughput:
Core ML 14.80 fps, MLX standard 9.74 fps, padded fused 10.20 fps. Padded MLX peaked
at 2689.6 MB physical footprint versus native tensor replay's 686.4 MB, with a
large untuned allocator cache. FP16 outputs differ from production; neither port
is promoted. Native tensor replay matches previous probabilities byte-for-byte.
Report: `docs/wd-tagger-mlx-optimization.json`; experiment source:
`tools/experiment-wd-tagger-mlx.py`. Native benchmark now supports retained
`--tensor-input` because original thumbnail cache paths became unavailable.

Native benchmark compilation, exact tensor replay, MLX numerical checks, sample
Metal compilation/linking, Python syntax and diff-whitespace checks passed.

### Hidden Python environment removal (2026-10-13)

Removed `.uncover-runtime/` and `.native-runtime/face-conversion/` after confirming active inference is native and the SeedVR2 configuration has no Python/helper paths. Conversion setup scripts now require explicit external Python interpreters. Archived the legacy Python SeedVR2 helper and setup script under `reference/model-development/seedvr2/`; the active legacy setup entry point delegates to native setup. Removed generated Python bytecode. Installed final models and published bundles remain available.

Verified after removal: native build/self-tests succeeded without Python environments; 199 regression tests passed, 0 failed, 4 optional video tests skipped. Explicit conversion commands fail before downloading when their external Python interpreter is unset. Existing signed app still passes strict/deep signature verification.


### Development archive removal and native package rename

Removed the temporary `reference/` archive and its local generated source/validation artifacts after preserving the final published model bundles and benchmark reports in `docs/`. Production conversion paths now generate upstream sources and fixtures under ignored `.build/model-conversion/` directories. Renamed the Swift package directory `NativeSeedVR2/` to `Runtime/` to reflect its SeedVR2, FlashVSR and PGTFormer workers; updated bundle resource lookup, license paths and build tooling. The executable names remain stable for the app and command-line users.

### Native WDTagger optimization follow-up (2026-10-15)

Added isolated `WDTaggerMLX` Swift library and `waifu-wdtagger-validation`
executable. Original weight preparation and inference run natively without
Python. Specialized Metal kernels fuse input patch packing, QKV cosine packing,
32-dimension tiled attention, norm/residual/window routing and GEMM bias/GELU.
All 28 kernel checks passed; batches 2 and 8 produce identical corpus outputs.
Against FP32, max error 0.0137874, two general threshold changes, zero ratings.

One post-freeze GPU trace succeeded. Native pipeline labels are present, but
shader intervals and GPU counter-profile export are empty, preventing per-kernel
hotspot attribution. Cooled prediction-only fps / peak decimal MB: CoreML8
14.02/683.6; native8 14.73/1246.2; native2 14.18/619.3; CoreML2 13.59/512.8.
This single ordering does not establish a repeatable native win: prior Core ML
replay reached 14.80 fps. Production remains Core ML. Native batch 2 is the more
promising native memory/speed balance. Global xcode-select now points to Xcode.
Build helper, release build, numerical validation and native linkage passed.
Report: `docs/WDTagger-native-optimization.md` and companion JSON. Local weights,
probabilities and the single capture are in ignored `.build/wd-native/`.

Checkpoint build verification: native release build and all 28 kernel checks
passed again. The exact staged app sources compiled and linked in Swift 6 with
warnings as errors and optimization. Fixed pre-existing thumbnail build errors:
parenthesized guard closures, propagated throwing store calls, and migrated
deprecated synchronous frame generation to AVFoundation's asynchronous API.
Separate active UI edits are excluded from this checkpoint and its build check.

### WDTagger speed pass after checkpoint 9ba0516

Checkpoint `9ba0516` was committed and pushed to `origin/main` before further
optimization. Added a native Metal kernel fusing QKV projection, cosine
normalization and layout packing while preserving the original 32×64×32 GEMM
tile and FP32 accumulation. Two balanced, shared-weight three-way comparisons
confirmed 2.1–2.2% over original native inference and 7.9–8.1% over production
Core ML. Fusion won 47/48 adjacent batches against original native execution.

Standalone original/fused processes measured 15.17/15.53 prediction fps at the
same 1536 MiB budget. A 768 MiB soft budget preserved 15.50 fps and reduced peak
process memory from original 1259.6 MB to 949.9 MB (24.6%). Fused QKV and the
768 MiB budget are now native defaults. Weight format remains v1; implementation
is `swinv2-native-metal32-fused-qkv-v2`. The app still uses Core ML.

All 138 native kernel checks passed. Allocation budgets and repeated fused
runs produce identical fused probabilities on all 64 inputs. Fusion changes
reduction order: max probability difference from original native is 0.00355348,
with no additional general-tag threshold or rating decisions on the corpus.
No new GPU trace was recorded. Failed tile, normalization, bounds-check,
compact-bias and narrower-fusion trials remain documented and disabled.
Evidence: `docs/WDTagger-native-tuning.md` and companion JSON. These are inference
benchmarks, not a whole-app or sustained-video performance guarantee.

## 2026-10-16 — native WD Tagger app cutover

- Selected fused QKV Metal backend for Apple Silicon tagging, batch 8,
  768 MiB soft allocation budget and 128 MiB cache. Core ML model prediction
  is removed from the app tagger path.
- Persistent native worker, shared input mapping, serial requests, cancellation
  termination, exact tail-row handling, bundled Metal sources and MLX library.
- App release build and separate exact-index source build both succeeded.
  All 13 WDTagger/contract cases passed with the native worker; 64 packaged-worker
  predictions exactly match the previously validated optimized path.
- Prepared weights, CSV, manifest/support files, MLX library and runtime archive
  are published at the GitHub `wdtagger-native-v2` release. The app's descriptor
  SHA-256 pins all model/support bytes. Existing Core ML model folders remain intact.
- Performance matrix appears before Build and run in README.md and is linked
  from the Metal source README. Detailed historical measurements remain in
  docs/WDTagger-native-tuning.md; app integration does not establish end-to-end fps.
- Unrelated working UI edits are excluded from this checkpoint.

Post-publication verification: the real app installer downloaded the public
native release, verified every pinned file, activated it, ran a one-frame tail
through the app adapter and reused the same verified release on repeated setup.
Model suite: 8 passed, 0 failed; unrelated opt-in PGTFormer download was skipped.
The local verified native model is now configured. Published checkpoint source:
`98e03f9`; a signed/ad-hoc-verified checkpoint app ZIP is included in the same
GitHub release. `docs/wd-tagger-native-integration.json` records asset hashes.

## 2026-10-16 — native WD Tagger weights moved to the Hub

The optimized tagger's weights were published as a GitHub release page
(`wdtagger-native-v2`) while every other model came from a pinned Hugging Face
release. That was the one model a user could not fetch with `hf_hub_download`, and it
made the descriptor structurally incompatible with our own tooling: `downloadBaseURL`
is understood by `ModelRelease` but not by `download-coreml-release.swift`, which also
demands a 40-hex revision and so rejected the `wdtagger-native-v2` tag outright.

**Weights now live at `alastorid/WD-SwinV2-Tagger-v3-Swift-MLX`.** All seven files were
downloaded back from the Hub and matched the descriptor by SHA-256, and the weights pass
`waifu-wdtagger --validate`. `manifest.json` is byte-identical to the previously published
one (`697fdea9…`), so installs verified against the old descriptor stay valid. Only the
model card changed, because the publisher writes the card before hashing it. The
descriptor's `downloadBaseURL` is gone; the app now takes the ordinary Hub resolve path,
and the revision being a real commit closes the tooling gap above.

**The MLX kernel library is a build artifact, not a weight, so it moved separately.**
`mlx.metallib` is not produced by `swift build` — `build-native-wdtagger.sh` symlinks the
downloaded file into the build directory — so it has to be hosted. Deleting the GitHub
release without noticing that broke the native build on a clean runner, which is invisible
locally because the file is cached in the gitignored `.native-runtime/`. It now lives at
`alastorid/WaifuRuntime-Metal` (MIT, from MLX; redistribution of the compiled library is
permitted provided the notice travels with it) and `build-native-uncover.sh` fetches it
by pinned revision and digest. Hosting it on the Hub is a deliberate exception to keeping
the weights repo weights-only: revisions there are immutable commits, and we pin by
digest, so the same guarantee we already rely on for weights applies.

**Hub rejects a `.metallib` above 10 MiB unless LFS routes it.** The default
`.gitattributes` covers `.safetensors` and other weight extensions but has no
`*.metallib` pattern, so the kernel committed as a plain blob and the push was declined.
`upload-hf-bundles.py` now appends that pattern after restoring the Hub's own attributes,
so a future runtime asset cannot fail the same way silently.

**No published app build ever referenced the GitHub release.** `v0.1.2` shipped before
this tagger work landed and still points at the Core ML repository, which is untouched and
still resolves; deleting the release broke no user. The native path ships in the next
release.
