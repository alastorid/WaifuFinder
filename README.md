# WaifuFinder

<img src="docs/assets/WaifuFinder.png" alt="WaifuFinder abstract viewfinder icon" width="128" height="128">

Find the best frames in a long video.

**[Visit the WaifuFinder showcase](https://alastorid.github.io/WaifuFinder/)** ·
**[Get the latest macOS release](https://github.com/alastorid/WaifuFinder/releases)**

Public release builds include PGTFormer and are for non-commercial use. See
[the licensing notes](docs/Licensing.md) for the model terms and the separate commercial build.

For the pinned WD Tagger, TruFor and SAFIRE versions, reference parity and
measured speed/memory comparisons, see the
[native model status summary](docs/Native-model-status.md).

The pre-squash commit hash record is backed up in a [private GitHub archive](https://github.com/alastorid/waifufinder-history-archive/blob/main/secret%20commit%20hash.md); access requires authorization to the private repository.

## SAFIRE for Photos

The Photos preview offers a **SAFIRE** overlay/mask toggle for inspecting the
model’s flagged regions. Install the model through Settings → Models. SAFIRE’s
map does not identify editing software or establish photo chronology.

## TruFor forensic scores

The Forensic section in the Photos ordering panel scores still photos with
[TruFor](https://github.com/grip-unina/TruFor) (Guillaro et al., CVPR 2023):
the official trained detection head produces the manipulation score at native
resolution, and the Photos preview offers a **TruFor** toggle with genuine
anomaly, reliability and Noiseprint++ layers. Install the model through
Settings → Models. TruFor is redistributed with credit under the GRIP-UNINA
licence: informational and nonprofit use only, and public results must credit
the original authors. See [TruFor](docs/TruFor.md) for validation and
measurements.

The native TruFor worker targets a 2 GB process budget while preserving its
original 24 MP input envelope. Large images stream FP32 activations through
temporary disk storage, preserving full resolution and global model context.
This trades speed and disk traffic for RAM. See the [RAM policy and validation](docs/TruFor.md#ram-budget-2026-10-25).

## Install on a Mac

Requires Apple silicon and macOS 15 or later. Download the DMG from the
[official releases page](https://github.com/alastorid/WaifuFinder/releases/latest),
then drag **WaifuFinder** onto **Applications** in the disk image window.

Open **Start Here.html** in that window for instructions to copy the command and paste it into Terminal.
The developer does not have an Apple Developer ID certificate. This release needs
a one-time local signature on your Mac; read the guide’s explanation before running
this command in Terminal after copying the app:

```sh
xattr -dr com.apple.quarantine "/Applications/WaifuFinder.app" && codesign --force --deep --sign - "/Applications/WaifuFinder.app"
```

This removes only the app’s download quarantine flag and applies an ad-hoc local
signature. It does not download scripts, upload files, or launch the app. Local
signing removes a protection; it does not verify the app’s safety. Release builds
are not Apple-notarized. Continue only if you trust the official download.
After setup, open WaifuFinder from Applications as usual and eject the disk image.

WaifuFinder takes one or more video files, decodes it frame by frame, asks Apple's Vision models
what is interesting about each frame, clusters the frames that show the same shot, ranks
within each cluster, and lets you browse, compare and export the survivors. It is built
for the case where the interesting material is scattered through an hour of footage and
watching the whole thing to find it is not an option.

The reference case is a 46.75-minute 1080p clip — 84,067 frames at 29.97 fps — which
Full mode processes frame by frame. It defines the intended workload; a full analysis
of this entire clip is not part of the recorded end-to-end validation. While processing,
the grid is replaced by a progress screen; results appear after grouping finishes.

## Remote inference

Use **Settings → Remote** to share one Mac’s installed models with another Mac.
On the server, enable **Serve inference from this Mac**, then copy its address and
access key. On the other Mac, enable **Use an inference server**, enter that address,
paste the key, and connect. Nearby servers also appear automatically when the network
allows discovery. Allow local-network access when macOS asks.

Use **Add inference server** in the same Remote panel to configure more servers,
with a separate address and access key for each. Once connected, each server appears
as a compact card above the add-server form, with its capabilities and an on/off switch.
Swipe a card left to reveal **Disconnect**, which removes it. An off switch shows
“Swipe left to disconnect” as its tooltip. Switching off retains the card and its
last known capabilities. Tagging sends
each batch to the connected server with the smallest outstanding batch queue.
Servers with different tag labels are excluded from that run. The primary server
continues to provide remote reconstruction.

Hold **Option** to change the tagging action to **Start Taggerᴿ⁺ᴸ** and include
this Mac’s local WD Tagger alongside the remote servers. The local model must be
installed and use the same tag labels. Each frame is tagged once, by one worker.

A raised **R** beside tagging or reconstruction means that operation will run on the
connected server. The server must have the corresponding model installed; the client
can tag without downloading the model. Tag thresholds, Apple Vision analysis,
grouping, compositing and export remain on the client. Only model inputs are sent,
with authenticated encryption after connection setup. Each server accepts one client.
Use **Kick** to disconnect it, or **Rotate…** to replace its access key.

See [the protocol and implementation design](docs/Remote-inference.md) for limits,
key storage and the client/server boundary.

## WD Tagger performance matrix

**Optimizing WD Tagger? Start with the [latest pipeline gain](docs/WDTagger-native-pipeline.md)**,
[memory/input pass](docs/WDTagger-native-memory.md) and
[M4 confirmation and further tuning](docs/WDTagger-native-m4.md).
See also [fusion measurements, rejected trials and reproduction guide](docs/WDTagger-native-tuning.md).

### Current WD Tagger model and backends

Both rows refer to the same WD SwinV2 Tagger and label vocabulary, packaged for
their respective runtimes. The app's active route is the native worker; Core ML is
retained as the production-reference comparison.

| Model / backend | Current use | Optimizations in the path |
|---|---|---|
| **WD SwinV2 Tagger — native Swift + MLX + Metal** | **Active app backend** on Apple Silicon; batch 8, 448×448 | FP16 prepared weights; direct RGBA UInt8 input with GPU normalization and patch preparation; fused QKV projection, cosine normalization and Q/K/V packing; custom Flash32 attention; fused residual, normalization and routing; fused MLP GEMM, bias and GELU; 448 MiB soft allocation budget and 64 MiB cache. The app also fetches one following thumbnail batch while tagging the current batch. |
| WD SwinV2 Tagger — Apple Core ML (CPU + GPU) | Production-reference backend in the benchmark; legacy Core ML package retained | Compiled by Core ML for the test Mac and executed on Apple's CPU/GPU runtime. No WaifuFinder custom Metal kernels are applied to this path. |

The thumbnail prefetch overlaps app-side input preparation with inference; it does
not change the model or its predictions. Batch size 8 and the native kernel defaults
are the best confirmed configuration in the recorded M4 comparisons, not a proven
hardware-wide optimum. See the tables below for measured rates and memory.

M3 MacBook Air, 16 GB, batch **8**, 448×448 input. Fresh process comparisons use
one batch of file input at a time. Rates measure model inference and GPU completion;
they exclude image preparation, file I/O and app/worker communication.

| Implementation / configuration | Inference fps | Peak process memory | Status |
|---|---:|---:|---|
| Core ML, CPU + GPU | 14.24 / 14.48 | 702.5 MB | Opening / closing controls |
| Previous native, Float32 input, 768 MiB budget | 15.28 / 15.51 | 998.2 / 998.4 MB | Opening / closing controls |
| **Native Metal, direct RGBA bytes, 448 MiB budget** | **15.52 / 15.53** | **665.3 MB** | **Selected app backend**, repeated |
| Native RGBA, 384 MiB budget | 15.29 | 609.9 MB | Lower-memory trial; slower than selected |

The Core ML controls above allowed CPU and GPU only. A later CPU+ANE run did invoke
ANE execution, but measured 14.62 fps versus 15.66 fps for native and failed the
output-agreement check; the `.all` cold-load compilation did not complete. The
[WD Tagger ANE investigation](docs/WDTagger-ANE.md) records the measurements and
the numerical failure.

The selected path preserves approximately 15.5 fps while using **33.4% less peak
process memory** than the previous native path. It measured **7.2–9.0% faster and
5.3% lower peak memory than Core ML** in these comparisons. All 64 native predictions
remain byte-identical. Small additional inference-speed differences versus the
previous native path are not established beyond run variation.

These figures cover individual benchmark processes, not combined host-app-plus-worker
memory or whole-app throughput. Allocation budgets are soft scheduling controls,
not process-memory caps. Normal thermal state was required; the fresh comparison
allowed a starting GPU temperature up to 65°C amid active desktop work. See the
[full method and raw samples](docs/wd-tagger-native-memory.json). Historical
full-corpus-mapping measurements remain in the earlier fusion report.

Kernel sources: [Runtime/Sources/WDTaggerMLX](Runtime/Sources/WDTaggerMLX).

### Tagging pipeline gain

On the M4, fetching one following thumbnail batch while inference runs improved
the tagging stage when thumbnails need regeneration from video:

| Tagging-stage path | Frames/sec, repeated | Peak combined process memory |
|---|---:|---:|
| Serial thumbnail fetching | 14.36 / 14.38 | 726.8 / 727.6 MB |
| **One batch ahead — selected** | **20.03 / 20.51** | **729.4 / 733.0 MB** |

**39–43% more throughput**, with at most 6.2 MB additional combined footprint in
these measurements. This uses a public synthetic 1080p H.264 source, 64 requested
frames and production thumbnail regeneration at width 512. Cached-thumbnail
confirmation was effectively tied. The timer includes fetching, preprocessing,
inference, IPC and result encoding; UI and full-video analysis are excluded.
Combined footprint sums parent and worker measurements and can count shared
mappings twice. [Method, limits and raw samples](docs/WDTagger-native-pipeline.md).

[Download the latest app release](https://github.com/alastorid/WaifuFinder/releases/latest).
Model weights are downloaded separately from pinned Hugging Face repositories.

### M4 confirmation

Mac mini M4, 16 GB, batch 8; same corpus and prediction-only timing:

| Implementation | Inference fps, opening / closing | Peak process memory |
|---|---:|---:|
| Core ML, CPU + GPU | 19.53 / 20.95 | 675.3 / 685.5 MB |
| **Current native Metal** | **23.34 / 23.35** | **671.8 / 671.4 MB** |

Three subsequent interleaved comparisons measured the current native kernel
**11.0–12.2% faster than Core ML**. Compact bias, larger tiles, submission changes
and reduced synchronization did not establish an additional speed gain; defaults
remain unchanged. [Full M4 results and limitations](docs/WDTagger-native-m4.md).

## Build and run

For a fresh machine, run `./tools/setup-development.sh`. Optional
`--benchmarks` generates synthetic WD Tagger inputs and local comparison outputs,
downloads published models, and builds benchmark tools. See [clean-machine development setup](docs/Development-setup.md)
for requirements, model restoration and the preserved experiment archive.

Clone [WaifuFinder](https://github.com/alastorid/WaifuFinder), then run the commands below.

The app is built with `swiftc`; native GPU workers use the pinned MLX Swift package
in `Runtime/`. An Apple Silicon build includes the optimized WD Tagger worker.

```sh
./build.sh                 # builds dist/WaifuFinder.app
open dist/WaifuFinder.app   # launches it
```

`build.sh` compiles every file under `Sources/` into one binary, writes the bundle's
`Info.plist`, draws the icon (`tools/make-icon.swift`), and ad-hoc signs the result.
Repeated builds reuse the app executable and icon when their inputs are unchanged.
Release builds (the default) retain full optimization. For faster edit/build cycles:

```sh
WAIFU_BUILD_CONFIGURATION=debug ./build.sh
```

Debug builds use incremental compilation and disable app optimization; native GPU
workers remain optimized. Both modes write the same `dist/WaifuFinder.app`, with
separate compiler caches under `.build/app/`. Delete `.build/app/` to force an app
rebuild. Native worker self-tests rerun when their binaries, pinned resources,
build script or macOS version change. `WAIFU_SWIFT_JOBS` controls parallel compilation
(default 3, suitable for a 16 GB Mac).

The app keeps its existing `~/Library/Application Support/Waifu` settings, caches and
model configuration locations, so renaming does not reset an installed setup.

`WAIFU_ARCH` pins the architecture and `WAIFU_VERSION` stamps the version, so a release
built for one architecture carries that architecture's name rather than the runner's.

To package a release disk image after building:

```sh
WAIFU_VERSION=0.1.0 ./tools/package-dmg.sh
```

The release workflow runs this automatically. Packaging regenerates the background
and writes the Finder layout directly, without opening Finder or showing dialogs.
It requires macOS, Xcode Command Line Tools, and Python 3; its MIT-licensed Python
libraries are vendored, so no package installation or network access is needed.
`python3 tools/test-dmg-packaging.py` checks unattended packaging and the background
alias after remounting; the release workflow runs this check before packaging.

## Tests

```sh
./run-tests.sh                  # compile + run everything
./run-tests.sh --only grouping  # one suite, matched as a substring of its name
./run-tests.sh --list           # how many cases are registered
./run-tests.sh --strict         # also exit non-zero when a known bug still fails
```

The app is built directly with `swiftc`; the root `Package.swift` is for IDE navigation
because SwiftPM cannot load it with the target machine's Command Line Tools. Native model
workers under `Runtime/` use SwiftPM with the clone-local workaround prepared by
`tools/native-toolchain/prepare.sh`. App regression cases use a hand-rolled harness rather
than XCTest or Swift Testing. `run-tests.sh` compiles the production sources — everything
except the `@main` scaffold — together with everything under `Tests/` into one throwaway
executable. Because the production code is compiled in, **a failing case means the real
code misbehaved**; there is no mock to drift out of sync.

See [`Tests/README.md`](Tests/README.md) for the harness and which suite covers what.

## How it works

One pass, in stages, each in its own directory:

| Layer | Directory | What it does |
|---|---|---|
| **Model** | `Sources/Waifu/Model/` | The types every layer exchanges, plus the grid's arrangement. No dependencies. |
| **Video** | `Sources/Waifu/Video/` | Probes metadata, decodes every frame on demand with one sequential `AVAssetReader` pass, keeps thumbnails in memory. |
| **Vision** | `Sources/Waifu/Vision/` | Aesthetics, face capture quality and FeaturePrint over each frame, four at a time. |
| **Grouping** | `Sources/Waifu/Grouping/` | Single-linkage clustering on FeaturePrint distance; group-relative ranking. |
| **Engine** | `Sources/Waifu/Engine/` | Orchestration: probe → sample → score → group, with progress and cancellation. |
| **Export** | `Sources/Waifu/Export/` | Re-extracts the selected frames at native resolution; PNG or JPEG. |
| **UI** | `Sources/Waifu/UI/` | SwiftUI: drop zone, frame grid, video preview, order panel, export panel. |

### Every frame, not a sample

The analysis is frame by frame: every frame the file holds is decoded and scored, and
none is skipped. The grid stays navigable at that size because frames are grouped into
shots and each group contributes its best few by default — the per-group cap is a
display decision, not a filter, and a group can be revealed in place.

The drop zone offers **Quick | Full** before a video loads. Full — the default — is
the frame-by-frame pass above. Quick keeps the same decode (a sequential reader
cannot skip frames cheaply) but scores one frame a second, so the 47-minute
reference is ~2,800 scored frames rather than 84,067. The choice precedes the load
because the decode itself is the same either way.

**WD Tagger** is optional. Download it in **Settings → Models**. Video analysis
never waits for or runs the tagger: after analysis, choose **Start tagging** in
the Rating or Tags panel. It tags frames admitted by the active face filter and
time, face-quality, face-definition, and art ranges, in fixed batches of eight.
While tagging is active, loosening those filters adds newly matched frames. After
a successful run, later filter loosening starts a follow-up run for new matches.
The general-trait threshold defaults to 0.35. Every selected trait must
match; Preview shows the rating and strongest general tags. The Rating panel
shows each category's share of tagged frames. Character tags are excluded.
Tagger results can be reused across analyses; the retention settings and binary
cache format are documented in [`docs/TagResultMemory.md`](docs/TagResultMemory.md).
Batch-size, fusion and compression measurements are documented in
[`docs/WDTagger-optimization.md`](docs/WDTagger-optimization.md).
The experimental native Swift/Metal backend and its measured tradeoffs are in
[`docs/WDTagger-native-optimization.md`](docs/WDTagger-native-optimization.md).
The subsequent fused QKV speed gain and selected memory budget are documented in
[`docs/WDTagger-native-tuning.md`](docs/WDTagger-native-tuning.md).
Frame analysis scores and image features are calculated directly from decoded frames
on every analysis run. Only tagging results are saved for reuse.

**Face definition** is an independent sorting and filtering factor for finding
detailed faces even when they are small or near the edge. Its row is always present
under **Order**, between Face and Art, with its own checkbox, direction and range.
Both **Quick** and **Full** compute it during their initial scoring pass, using the
same face detection as the Face score, then refining landmarks only when a face
is detected. Full checks every frame; Quick samples once a second.
Large opposing black borders are cropped from the
scoring input first, so padding does not hide a small subject behind the 1% area
cutoff. Face detection uses original-resolution picture pixels, measuring detail
in each face's interior at a common scale and keeping the highest score. Landmark
placement uncertainty discounts sharp mosaic boundaries and false detections around
only part of a face: readable facial features must accompany sharp pixels. It does
not weight frame position or face area, and bypasses the normal 1% subject-area
filter for faces with a definition score. Faces need at least 8 native pixels in
both dimensions to be measured. There is no deferred detection or second scan
when sorting or filtering. To prioritize definition, turn on Face definition and switch off Minimal Visual Distance, Time and Face,
or use **Presets → Face definition only**; its range filters independently of
whether its sorting checkbox is enabled. Tiles show `D`, and preview names the
score. Original export pixels remain unchanged.

Loading uses up to two parallel frame-preparation tasks and four scoring workers,
with lower concurrency for large sources or limited memory. Native frames stay resident only while in flight;
other models and thumbnails use a smaller version of the cropped picture, bounded
on its longest edge by the analysis size. Grid tiles retain the cropped thumbnail's
aspect ratio, including portrait pictures. Preview and export retain the original
canvas. The score combines contrast-normalized sharpness with landmark reliability.
It remains a heuristic: detection can miss faces, and noise or oversharpening can
raise detail energy. There is no extra model download or Python runtime.

During analysis, decoding/scoring and grouping run in the background. The UI shows
only progress and numeric frame/face counts, updating at most four times a second.
During grouping it also reports compared frames and linked groups (before final
size-limit splitting). The frame grid, thumbnails and player are not mounted; results
are arranged once at completion. Cancel remains available.

After analysis, the title-bar **Thumbnail shape** control switches between square
crops and full aspect ratios. Changing the grouping factor keeps the browser visible.
A small progress ring beside Grouping fills as it works; finished groups replace the
old ones when ready.

Grouping now runs as scored frames arrive, alongside the scoring workers. Each
new frame compares with earlier neighbours in the 120-second window; decoded
prints are released when they leave that window. The live counters include
grouped frames and provisional linked groups. Only group IDs and size-limit
splitting remain after scoring finishes, and results still appear once at completion.
This avoids decompressing and parsing both prints for every pair, while retaining
Vision's exact distance and existing groups. A local
600-frame synthetic benchmark using real Vision prints fell from 10.65 s to
0.053 s for batch grouping; this is a controlled comparison, not a full-video speed guarantee.

Frame ownership is deterministic: the same video always yields the same frame identifiers
(`f000123` is the 124th decoded frame), which is what makes the per-video cache worth
writing and a frame stable enough to cite in a URL, an export filename or a cache row.

### One sequential decode, parallel frame preparation

One reader walks the file from front to back. Native pixels supply face scoring;
background preparation tasks crop the visible picture and resize it to the analysis
size for aesthetics, grouping and thumbnails. The bounded preparation window overlaps
with scoring and JPEG encoding, delivering results in source order. This avoids
repeated seeks and retains native detail for face measurement and export.

### Two scores, never summed

`artScore` is Apple's `CalculateImageAestheticsScoresRequest` `overallScore`, stored
verbatim (it measures roughly −0.996…+1.000 and is not a probability). `faceScore` is
Vision's `DetectFaceCaptureQualityRequest` capture quality, 0…1. Different instruments on
different scales, so they are never blended at capture. Ranking happens later, on
**group-relative ranks**: each signal is reduced to its rank within the set being ranked
(1 best, 0 worst) and the ranks are blended — rank is unit-free, so the blend means what it
says.

A face counts only if it covers at least 1% of the frame, and the **largest** significant
face is the one that speaks for the frame — not a mean, which an incidental background
face would drag around.

### Grouping is single-linkage with a time window

Frames are compared only within ±120 s (a composition minutes apart is not an alternate
capture of the same shot), then chained: A merges with B and B with C, so the ends of a
slow pan end up together even though they are further apart than the threshold. Oversized
groups are split into contiguous chronological runs, and a run of one is dropped — a lone
frame is not a group.

## The window

The grid holds the candidates; the right column decides what is in it and shows the
focused frame in motion.

- **Preview** (top right) — the video, parked on the frame the grid points at. Clicking a
  frame seeks here; the transport is `VideoPlayer`'s. Pressing **Space** opens the
  original frame at native resolution full-window, fitted so that it touches the border on its limiting axis
  and is centred on the other, with pinch or the −/+ buttons to magnify (100 % is the
  fit), the scroll view to pan and a double-click to toggle fit/2×. Pinch and double-click
  zoom keep the image point under the mouse in place, including after panning; Space or Escape
  leaves it. Stepping to another frame keeps the zoom and position you set up, because
  stepping is a comparison.
- **Sort and Filter** (right) — **Minimal Visual Distance** is the only sorting signal
  on by default. It arranges all
  shown photos along an approximate shortest visual path using the same FeaturePrint
  distance as grouping. A native Swift/Metal port of cuOpt's 2-opt routing search
  explores several initial paths; small sets use the CPU and larger sets use Metal.
  Its forward/reverse control follows the same path in either direction, with no
  filtering range. An inline ring spins while the route is computed in the background.
  Photos appear in one continuous grid across group boundaries. Photos with missing feature
  data remain visible at the end of the forward sequence. Other sorting signals—time,
  face, face definition, art, and four independent WD Tagger rating probabilities—are
  off by default. When enabled alongside visual distance, they rank candidates, choose
  group representatives and provide the route's starting order; visual distance then
  arranges the shown photos. Turn visual distance
  off to sort directly by the other enabled signals and restore group rows. Each can be
  selected, reversed and filtered by its own range. Frames without a score sort last.
  See [the cuOpt source audit and benchmark](docs/CuOptMetal.md) for the port's scope,
  metric checks, memory limits and CPU fallback.
- **Ranges** (right) — under each filterable signal, a two-ended slider that keeps only the frames
  between its ends. Score ranges show the observed span; the four rating probabilities
  each use their own 0–1 range.
- **Tags** (right) — the WD Tagger traits on the frames currently shown, as chips. A
  bold **Common** chip leads and discloses the traits every shown frame carries; those sit
  in their own inset group, and the group's label collapses it again. Click a chip to
  require its trait, click it again to sort the grid by that trait's confidence, and
  right-click to exclude frames carrying it. The traits offered track the frames shown,
  so the list narrows as the grid does. Available traits rank by their average
  confidence among values strictly above the tag threshold. Summary work runs in
  the background and reuses cached vectors on Metal for larger selections, with
  the same average on the CPU for small selections or when Metal is unavailable.
  See [tag summary computation](docs/Tag-summary.md) for caching and fallback rules. Right-click
  the **Tags** header to choose **Basic** or **Advanced** (default). Advanced adds inline
  **Top** (1–99, default **all**) and **Threshold** (0.35–1, default 0.35)
  sliders above the tag search. Top combines the top N frames for every tag when none are selected; selected tags
  keep their AND rule with a separate Top N ranking for each, before group caps. Basic leaves Top off.
  Threshold filters stored UInt8 scores without discarding scores below the chosen value,
  so lowering it restores matches. Mode and slider values persist and are saved in presets. While Top is active,
  thumbnails show the tags whose Top N rankings admitted them in place of the frame number,
  in white text on stable dark colors. Click the tag label to show the frame number;
  click the number to restore the tags. This toggle is available only in Advanced mode
  with a Top limit active. A +N count and tooltip disclose additional tags.
- **Grouping** (right, off by default) — the FeaturePrint distance within which two frames are the same
  shot, as a live slider: tighter makes more, smaller groups, looser makes fewer, larger
  ones. It re-groups when released (drag) or on each step (←/→), without re-scoring.
- **Top** (Grouping panel) — an inline slider for how many frames of each group the grid keeps,
  best first by the current order (7 by default). Choose 1–33, or move fully right to **all**
  to reveal every group. Similarity and Top show their values on their handles.
  Frames in no group are never capped; the tail of each capped group's line has a
  "+N more" tile to reveal that group. **Group view** switches one group per line on or off.
- **Show** (right) — inline Faces and Favorite controls. Faces offers All (default), With
  faces, or Without faces; Favorite offers All (default), Favorite, or Not Favorite in
  Photos mode. Also choose one-group-per-line vs one continuous grid. The face filter uses the same
  significance threshold as ranking, so tiny detected faces count as without faces.

**Keyboard**: ←/→ step the neighbouring frame (inside the group's line when the grid is one
group per line, otherwise the displayed order), ↑/↓ jump groups, Space toggles the preview,
Return focuses and selects.

Click a frame to select only it. ⌘-click toggles individual frames; Shift-click selects
the inclusive range from the selection anchor in the displayed order. ⌘A selects all
frames currently shown by the grid, respecting filters and group caps.

## Uncover Waifu

In the still preview, holding Left or Right waits for each full-resolution frame before
advancing. Releasing and pressing again can skip a pending read. A stalled read shows
an error after 15 seconds, and navigation remains available.

From the native-resolution preview, click **Uncover Waifu**. The Source seek bar and
previous/next buttons navigate every source frame before reconstruction, using actual
presentation timestamps. Seeking sets the interval start and keeps the region. Held
arrows wait for each read; fresh presses or scrubbing can supersede it. The algorithm
uses an algorithm selector including **SeedVR2 3B**, **SeedVR2 7B**, **FlashVSR 1.1** and **RealESRGAN**. Inference temporarily
replaces the canvas with progress. Draw, move or resize a
region, choose **1× through 64×** downsampling and a short interval, then **Reconstruct**.
While selecting, the ROI shows the reduced input pixels and dimensions; changing scale
or resizing the region updates that preview immediately, without inference.
Choose **SeedVR2 3B** (default), **SeedVR2 7B** or **FlashVSR 1.1** for temporal video restoration. RealESRGAN restores frames independently. FlashVSR accepts the selected source downsampling and always generates **4× output**; its model is tuned for 4× super-resolution, so other input reductions change its effective input-to-output ratio. Its export is four times the source frame dimensions. Compare the original against the AI result using a wipe, side-by-side view or instant toggle. The preview has original-pixel, fit-width and fit-height modes, and remains pannable and zoomable.
Playback and frame stepping share one clock. Scroll/Option-drag
pans and pinch/buttons zoom; feather and blend adjustments reuse the inference result.
**Export** writes a new MOV with the processed region and corresponding source audio.

Open **WaifuFinder → Settings… → Models** to download, choose, or remove models.
No Terminal, developer tools, Python, or local conversion is needed in the packaged
Apple Silicon app. The library shows model status, disk size, ownership, and location.
Choose a download folder (including an external drive); existing models keep their
locations. **Choose Existing…** verifies a supported model folder before activating it.
App-managed models move to Trash when removed; external models are disconnected without
deleting their files. Models in use cannot be changed until reconstruction finishes.
Apple Vision's analysis models are included with macOS and listed separately.
Every model is downloaded from our own pinned Hugging Face releases:
[SeedVR2](https://huggingface.co/alastorid/SeedVR2-3B-Swift-MLX-4bit),
[FlashVSR](https://huggingface.co/alastorid/FlashVSR-v1.1-Tiny-Swift-MLX-BF16),
[PGTFormer](https://huggingface.co/alastorid/PGTFormer-Swift-CoreML) and
[RealESRGAN](https://huggingface.co/alastorid/RealESRGAN-x2plus-Optimized-CoreML).

**Licensing.** This project's code is MIT, but **PGTFormer is licensed for non-commercial
use only**, and the app downloads models on your behalf. A build that offers PGTFormer
therefore cannot be used commercially. For a commercially usable build run
`WAIFU_COMMERCIAL_DISTRIBUTION=1 ./build.sh`, which omits PGTFormer entirely. See
[`docs/Licensing.md`](docs/Licensing.md) for the full inventory, the attribution we owe,
and two questions that need a lawyer rather than a build flag.

For developer setup from source, `./tools/setup-native-uncover.sh` remains available.

Inference runs in an isolated **Swift + MLX** executable, with no Python process. Setup downloads a pinned, checksum-verified
[ready-to-use 4-bit bundle](https://huggingface.co/alastorid/SeedVR2-3B-Swift-MLX-4bit)
(~2.4 GB), including the text embedding, and reuses matching files already installed.
The prepared bundle fuses packed 3B SwiGLU inputs and VAE attention projections without
re-quantization. Matching F16 precision, bounded loading and a small reusable buffer cache
reduce inference cost. Original authors and conversion credits are preserved.
Choose **SeedVR2 3B** or **SeedVR2 7B** in the Uncover algorithm selector; both
always use these optimized native bundles. For the optional **7B** bundle (~5.14 GB),
download **SeedVR2 7B** in Settings → Models. Both sizes can stay installed, and setup
registers each separately. Its GELU architecture uses VAE attention fusion and
streamed transformer blocks. Switching sizes preserves the region and degradation,
reuses prepared frames and keeps separate result caches.
See [optimization measurements](docs/seedvr2-fusion-validation.json). Setup needs
no Python, source checkpoint download, conversion or re-quantization. `./build.sh` bundles the native worker and Metal kernels.
Both sizes also use native Metal VAE normalization/activation fusion to reduce
temporary memory and processing time; see the
[runtime comparison](docs/seedvr2-vae-fusion-validation.json). Existing model downloads
remain compatible with this optimization.
Regions may span the video; larger regions use 256-pixel model tiles with 64-pixel overlap,
shared noise and temporal context. SeedVR2 receives the actual **1× / 2× / 4× / 8× / 16× /
32× / 64×** degraded ROI pixels, enlarges them as conditioning, and generates video at the
selected ROI's native dimensions. Output is capped at 16 megapixels. High reduction factors
invent more detail and still need quality and memory validation. **RealESRGAN** is a separate
framewise Core ML x2plus option, downloaded from its pinned release in Settings → Models.

**PGTFormer** is a validated native face-restoration option using its original
pretrained network through Core ML. Download it in Settings → Models, or convert it
yourself with `./tools/setup-native-pgtformer.sh --convert-official`; Python is
restricted to that offline conversion and reference validation. **1× Original** keeps
the source crop unchanged before the required face resizing. PGTFormer is redistributed
under a **non-commercial** licence. See [Uncover](docs/Uncover.md) for details.
Intervals remain bounded to
**5 seconds / 300 frames**, with SDR, square-pixel sources.

The native worker uses a physical-memory budget of **6 GB or half installed RAM**,
whichever is smaller. It streams weights for 7B, 8 GB Macs and single-pass jobs,
keeps 3B resident for repeated passes on larger Macs, checks the estimated working
set before loading, stops on macOS memory-pressure events, and permits one inference
process at a time. The MLX allocator limit alone is not a hard process cap. This reduces
memory pressure but cannot guarantee zero system swap when other apps are busy.
Reconstruction may generate details that differ from the original. The previous Python
adapter is retained as a reference/legacy fallback only when native inference is not
configured; a failed native job never silently falls back to it.
See [`docs/Uncover.md`](docs/Uncover.md) for the design and
[`docs/native-uncover-validation.json`](docs/native-uncover-validation.json) for the
recorded configurations, verification results and limits.

### SeedVR2 7B. Tested under 4 GB.

Native Swift + MLX, prepared 4-bit weights and Metal fusion bring SeedVR2 to a
smaller memory footprint on Apple Silicon. Both **3B and 7B** were tested with a
**4 GB worker budget on a 16 GB M3**. This describes inference-process memory for
our bounded video regions, not total system RAM or unrestricted full-frame video;
an actual 8 GB Mac has not yet been tested.

<details>
<summary>Optimization details, measurements and further opportunities</summary>

These optimizations are already in use:

- **Native execution:** Swift + MLX runs inference without a Python process;
  setup downloads checksum-verified prepared weights directly from Hugging Face.
- **Packed projections:** 3B combines SwiGLU gate/value inputs; both sizes combine
  VAE Q/K/V projections and biases, preserving stored quantized values.
- **Shared 3B weights:** 198 identical tensors use aliases, removing about
  325 MB of duplicates; shared video/text projections execute together.
- **Selective precision:** VAE and 3B transformer activations use F16, with F32
  normalization/rotary math. The 7B transformer keeps F32 because tested F16
  execution overflowed.
- **Metal normalization fusion:** F32 Welford statistics read F16 activations
  directly; affine scaling, the F16 boundary and SiLU share one epilogue, avoiding
  full-size F32 activation copies and grouped-layout rearrangements.
- **Causal-buffer fusion:** one Metal pass creates padded convolution inputs and
  compact frame history in channel-last layout, replacing separate copies and
  history evaluation passes.
- **VAE channel projections:** unit-stride, unpadded 1×1×1 convolutions use matrix
  multiplication; general 3D convolutions retain their original implementation.
- **Residual fusion:** compiled convolution-bias and skip additions preserve both
  F16 rounding boundaries while avoiding a separate full activation pass.
- **Bounded weight loading:** streaming holds only the active VAE half and one
  transformer block. Repeated 3B passes on larger Macs retain weights when faster.
- **Less allocation overhead:** a 64 MB reusable MLX buffer cache, batched parameter
  evaluation and compiled SwiGLU arithmetic reduce repeated work.
- **Bounded video processing:** overlapping temporal chunks and 256-pixel spatial
  tiles retain causal context, shared noise and source-frame ownership.

The Metal normalization change was compared with the same prepared models
using the previous normalization path. Six paired five-frame tests per size used
128×128 inputs and 256×256 outputs; fresh-worker shader compilation is included.

| Model | Median paired time reduction | Median peak worker memory, before → after |
| --- | ---: | ---: |
| SeedVR2 3B | 6.6% | 1.977 → 1.535 GB |
| SeedVR2 7B | 4.6% | 1.947 → 1.537 GB |

Each size was faster in five of six pairs. A separate 30-frame, two-chunk 3B pair
reduced time from 108.9 to 102.9 seconds and peak memory from 2.327 to 1.825 GB.
The latest kernel's output differences stayed within 2/255 for 3B and 1/255 for
7B in the short tests; the long 3B test also stayed within 2/255. Timing varies
with region size, heat and other applications. See the complete
[bundle optimization record](docs/seedvr2-fusion-validation.json) and
[Metal fusion record](docs/seedvr2-vae-fusion-validation.json).

The subsequent causal-buffer, pointwise-convolution and residual changes reduce
intermediate work and are enabled for both sizes. On a fanless M3 MacBook Air, four
short pairs observed median paired reductions of 11.9% (3B) and 6.5% (7B), with
median worker peaks of 1.535 → 1.444 GB. The longer 30-frame 3B pair instead slowed
from 81.2 to 120.2 seconds and increased peak memory from 1.854 to 1.958 GB. The
unchanged transformer stage also slowed substantially.

A four-pair 3B VAE-worker comparison on the 16 GB M4 Mac mini, run on `adae4c5`
before the true reduced-ROI update in `5aa454b`, used five prepared 128→256 frames.
It observed a 2.4% median paired time reduction, with 3 of 4 pairs faster and median
worker peaks of 1.495 → 1.415 GB. Every pair produced byte-identical PNGs. The first
baseline run took 9.86 seconds; the next three took 6.84–7.24 seconds, so this short
result remains provisional. It does not measure the updated app's learned spatial
restoration stage. Long-clip 3B/7B timing and end-to-end reduced-ROI quality, memory,
and speed validation remain pending.

All short-clip PNGs were byte-identical before/after for both sizes on the Air and for
the historical Mac mini 3B VAE-worker comparison; all 30 long-clip 3B frames also
matched byte-for-byte.
See the [complete causal/convolution record](docs/seedvr2-causal-fusion-validation.json).
Existing model downloads work with these runtime changes; no conversion is needed.

The command below reproduces the historical full-resolution SeedVR2 worker
comparison on `adae4c5`. Its 2×/4×/8× output geometry is restored here;
the current fused runtime and additional factors still need direct comparison.

```sh
WAIFU_TEST_SEEDVR2=1 WAIFU_TEST_VIDEO=/path/to/video.mp4 \
  WAIFU_PROOF_OUTPUT=/path/to/new-proof ./run-tests.sh --strict --only 'Uncover real'
swift tools/benchmark-native-uncover.swift --job /path/to/new-proof \
  --model 3b --pairs 4 --output /path/to/new-results
```

Use `--model 7b` with a separate output directory after installing that size.
Workers run serially in alternating order, with the same streamed 4 GB budget.
The summary records timing, memory and PNG byte parity without source paths.
Allow the machine to cool before starting; retain slower runs as well as faster ones.

**Further opportunities:** VAE encoding/decoding accounted for about 85% of the
normalization-fusion long-clip runtime. The remaining candidates are general 3D
convolution kernel scheduling, per-frame attention normalization, and additional
compiled transformer pointwise arithmetic. These are profiling targets, not
implemented speed claims; changes must pass paired speed, memory and output checks.
MLX documents [graph compilation](https://ml-explore.github.io/mlx/build/html/usage/compile.html)
and [custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)
as tools for reducing dispatch and intermediate-memory costs.

</details>

To add FlashVSR, open **Settings → Models** and click **Download** beside
**FlashVSR 1.1 Tiny** (about 3.51 GB). The app verifies the converted release and activates
it automatically. Use **Choose Existing…** to register an existing converted folder.
Developer setup scripts remain available for source builds.

Setup downloads our pinned [Swift/MLX conversion](https://huggingface.co/alastorid/FlashVSR-v1.1-Tiny-Swift-MLX-BF16)
and checks every file. **Setup and inference require no Python**. FlashVSR runs natively
through Swift/MLX Metal with BF16 weights, streaming temporal state and trained sparse attention.
It shares the 6 GB monitored budget and single-worker lock. Its padded ROI size is
limited by a conservative memory estimate. See [`docs/FlashVSR.md`](docs/FlashVSR.md)
for installation, architecture and measured reference validation.

## Memory

A decoded frame is ~576 KB at the 512 px analysis width, so an analysis that held its
frames' pixels would cost 46 GB for the reference video's 84,067 frames — past most
machines' RAM. WaifuFinder retains only its active preparation and scoring window.

The sampler reads sequentially and prepares a bounded window of up to two frames in
parallel, while up to four scoring workers run alongside it. A source-size estimate
reduces those counts for large native frames. Each slot allows for native pixels, one raster intermediate
and the analysis image, sharing an estimated working budget of 256 MiB or one
thirty-second of installed RAM, whichever is smaller. This chooses concurrency; it is
not a hard limit on decoder, Vision or process memory. Temporary image objects are
released per frame, and Core Image does not cache intermediates from unrelated frames.

Compressed thumbnails use a 64 MiB in-memory LRU; evicted frames are regenerated from
the source when requested. Decoded thumbnail images have a separate 64 MiB cache budget.
Grid cells reveal their image and controls together after the pixels are decoded,
without a grey loading tile or spinner. Initial and nearby rows are prefetched.
Visible cells retain their decoded pixels independently of cache eviction; a brief
250 ms release delay keeps view re-entry and cell moves stable, then releases images
that remain offscreen. Generation and request checks discard obsolete completions.

What stays in memory per frame is a timecode, the two scores and the FeaturePrint. The
compressed thumbnail JPEGs are held only in the bounded LRU and the UI decodes them on
demand for grid cells; thumbnails are not persisted to disk. The zoomable preview reads
the original source frame at native resolution and keeps the previous image visible
until the next read completes.

## Layout

```
waifu/
├── build.sh                 # swiftc-based app build + bundle + icon + codesign
├── run-tests.sh             # compiles Sources + Tests into one test binary
├── run-ui-tests.sh          # prompts for approval, then notifies when UI control is returned
├── docs/API.md              # the engine↔UI contract
├── tools/make-icon.swift    # draws AppIcon.icns from code
├── Sources/Waifu/           # App, Model, Video, Vision, Grouping, Engine, Export, UI
└── Tests/                   # hand-rolled harness + one file per suite
```

The engine/UI contract is in [`docs/API.md`](docs/API.md). Current subsystem details and
validation live in the model and feature documents under `docs/`.

## Requirements

macOS 15 or later. The three Vision requests WaifuFinder uses were introduced in macOS 15 and
`build.sh` sets the deployment target to match — lowering it without replacing the model is
not supported. Test media is synthesized with Apple system frameworks; no external
media executable is required.

### Dependency licensing rule

GPL- and LGPL-licensed software must not be added, linked, bundled, invoked, or relied on
by the app, its build, tests, CI, development tools, fixtures, or distribution. Check
direct and transitive licenses before adding dependencies; if a license is unclear, stop
and resolve it first. Use Apple system frameworks or license-compatible alternatives.
Test and benchmark fixtures use Apple-native generation. See [AGENTS.md](AGENTS.md)
for the repository instructions and [credits](docs/Credits.md) for project acknowledgements.

Final model files and published bundles are documented in [models/README.md](models/README.md). Final installations use `~/Library/Application Support/Waifu/Models/`; published bundles are retained under `models/published/`.

### Open media from a URL and Firefox

The empty loading panel offers **Browse files…** and **Or load from URL**.
Paste a direct HTTP(S) media URL, or drop a web link onto the window. Import inspects the
content rather than requiring a filename extension: extensionless endpoints and
misnamed downloads (including `.dms`) work when their contents are supported video.
Files chosen locally are also inspected regardless of suffix.

Drop multiple videos together, or select multiple files in **Browse files…**, to
analyze them in one workspace. The app reads each original directly, preserving
its frame rate, size, orientation and audio. Sorting, grouping, filtering and tagging
work across all results. Preview, export and restoration use the selected frame’s
original video and local timestamp. Analysis and tag caches stay keyed by the
original video and frame, so opening a video alone or in a different collection
reuses the same results. No joined video or intermediate media file is created.

Direct media URLs are downloaded to a temporary local file before analysis so
frame sampling, preview, restoration and export share a seekable source. Finite,
unencrypted MPEG-TS HLS playlists are followed through redirects, then their
segments are joined in a temporary file for native AVFoundation decoding. Direct
media downloads are inspected from their contents, independent of filename suffix
or server MIME type. Loading, frame decoding and export use Apple's AVFoundation
and CoreMedia frameworks; the app does not bundle FFmpeg or any GPL/LGPL media
libraries. Unsupported containers or codecs show an error instead of falling back
to a third-party decoder. Encrypted, live, byte-range and fragmented-MP4 HLS
playlists and DRM-protected media are not supported.

Optional Firefox forwarding is in **Settings → Firefox**. Set suffixes such as
`mp4, dms, ts, m3u8` and click **Set Up / Update Firefox**. WaifuFinder registers
its local bridge, opens Firefox's add-on loader, and highlights the packaged add-on
file. The Settings indicator turns green when Firefox is connected. Registration
and **Unregister** are managed in the app; the extension has no separate setup
switch. Matching direct tab navigations open in WaifuFinder, embedded players stay
in Firefox, and failed forwarding leaves Firefox's navigation alone. Local add-on
installation is temporary until a Mozilla-signed package is available. See
[Firefox extension setup](FirefoxExtension/README.md).

### Photos workspace

Choose **Or open Photos** on the loading panel. A dedicated workspace keeps the
album list on the left and the existing ranking, filter, tagging and preview tools
on the right. Photos integration includes still photos only; video files and URLs
remain available through the main loading flow.

My Albums follows PhotoKit's ordered top-level album/folder list and the child order
inside each folder, without alphabetizing or flattening folders. Built-in collections
such as Favorites and Animated remain in the separate Smart Albums section. Albums
with no accessible still photos remain visible but cannot be selected.
The sidebar refreshes when WaifuFinder becomes active again, so rearrangements made
in Photos appear on returning to the app without restarting the selected analysis.

Album thumbnails load lazily as you scroll. Analysis and tagging use PhotoKit
renditions bounded by 512×512 pixels, rather than decoding full-resolution originals.
Up to four small photo requests overlap with scoring. Ranked results appear and
reorder in batches during analysis. Photos always uses this ranked view.
PhotoKit renditions have a bounded memory cache. Persistent analysis results use
ordered Photos asset identifiers and revision metadata, so unchanged albums reuse
results while edits or membership changes invalidate them. Tagging results are
keyed per photo by its asset identifier, revision, and pixel geometry rather than
by its position in an album, because PhotoKit does not guarantee the order of an
album fetch. Reopening an album therefore reuses every cached tag, reordering or
adding to the album costs nothing, editing one photo re-scores only that photo,
and a photo already tagged in another album is not tagged again.

The thumbnail-shape icon switches between square crops and full aspect ratios.
Ranking and tagging share PhotoKit's
renditions requested at 512 pixels on the longest edge, preserving aspect ratio.
Smaller cached previews appear immediately while the requested rendition loads;
they are not treated as a completed 512-pixel rendition. If a larger rendition is
unavailable locally, the smaller cache remains usable with iCloud downloads off.
Thumbnail tooltips show their actual pixel dimensions. Tagging does not request originals.
The tagger prepares its 448×448 input from these cached thumbnails.

Hover over a ranked thumbnail to add or remove its Photos Favorite using the heart;
favorites keep a filled heart visible. The Unicode order button cycles through all,
ordered, and unordered photos without a dropdown; unordered uses a slashed glyph.
A small matching glyph at the top right marks photos whose order was synced.
The Photos label and both panel collapse controls live in the title bar.

Right-click a ranked thumbnail and choose **Sync order** to write the displayed
order to the album in Photos. With multiple photos selected, only those photos are
reordered, using their displayed sequence; other album members keep their slots.
The app marks photos as ordered only after a successful write. Photos must display
the album's custom order for that sequence to be visible there; date or title
sorting can override it. Smart albums do not support custom ordering. **Reset order
state** clears the app's private markers without changing the album in Photos.

**Download from iCloud** is off by default. Off uses locally available renditions
and skips missing ones rather than blocking the rest of the album. Enable it to
fetch missing media as needed. Changing the switch preserves completed analysis,
tags, selection, and cached thumbnails. Enabling it retries unfinished photos;
disabling it cancels pending analysis downloads and continues with local media.
Grouping is updated over the combined results without decoding completed photos
again. Originals are requested only for explicit full-resolution
preview, restoration or export and follow the same iCloud policy. No underlying
originals are exported into an import folder, and the app never opens the private
Photos library filesystem.

Static photos support RealESRGAN spatial restoration and export reconstructed
results as PNG. Temporal restoration models are unavailable for still photos.


### RealESRGAN photo preview

The Photos preview's right panel includes **RealESRGAN Off / x2 / x4**, directly
below Photo Info and above TruFor and SAFIRE. Both choices use the official
23-block FP16 Core ML networks. Install each independently in Settings → Models.
Off immediately returns to source pixels; changing models shows source pixels
until the new reconstruction completes. Results remain non-destructive and use the
existing zoom, pan and navigation controls.

Compatible enabled inference servers show the raised **R** beside each supported
variant. x2 support never implies x4 support. TruFor and SAFIRE continue analyzing
original pixels; their evidence views take display priority over reconstructed
pixels. AI reconstruction must not be interpreted as original forensic evidence.

Original models by **Xintao Wang, Liangbin Xie, Chao Dong and Ying Shan**, authors
of *Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic
Data*, with reference architecture by BasicSR contributors. WaifuFinder supplies
conversion, optimization investigation and native integration. See
[the integration report](docs/RealESRGAN-x4-integration.md),
[CPU + ANE optimization results](docs/RealESRGAN-ANE.md), and [credits](docs/Credits.md).
