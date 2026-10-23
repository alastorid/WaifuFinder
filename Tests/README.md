# WaifuFinder regression tests

There is no SwiftPM on this machine, and neither XCTest nor Swift Testing can run, so
these are not a target in any framework. They are compiled **into the same binary as the
production sources** by [`../run-tests.sh`](../run-tests.sh):

```sh
../run-tests.sh                 # compile + run everything
../run-tests.sh --only grouping # one suite, matched as a substring of its name
../run-tests.sh --list          # how many cases are registered
../run-tests.sh --strict        # also exit non-zero when a known bug still fails
../run-tests.sh --help          # the runner's own usage
```

`run-tests.sh` compiles every file under `Sources` **except
`Sources/Waifu/App/WaifuFinderApp.swift`** (it owns `@main`; `Tests/main.swift` owns the entry
point), together with every file here, into a throwaway executable in a temporary directory,
and removes that afterwards. The default run never touches the real cache at
`~/Library/Application Support/Waifu`, the sample video or `dist/`; every fixture is a
temporary directory.

Because the production code is compiled in, **a failing case means the real code
misbehaved** — there is no mock to drift out of sync. Where a production member is `private`
because it is an implementation detail (`FrameExporter.uniqueDestinationURL`, `RankMap`) it
is internal and documented as a test seam; nothing here re-implements the logic it is
testing.

## Remote inference

Run `./run-tests.sh --only remote` from the repository root. The suite uses real
loopback TCP connections with generated vocabulary, image fixtures and injected model
backends. It needs no downloaded weights and does not read or write the login Keychain.
Network discovery is tested as encoded datagrams and an expiring cache; the loopback
listener does not need to advertise on the LAN. Hardware model execution and interactive
Settings layout are separate checks.

## Interactive UI tests

Save interactive checks as executable scripts and launch each invocation through
[`../run-ui-tests.sh`](../run-ui-tests.sh):

```sh
../run-ui-tests.sh /absolute/path/to/ui-test-script argument
```

After the approved script exits, the launcher sends a native macOS notification
that the UI steps are finished and control is back with the user.

The launcher shows a native message box with the exact script and arguments before
starting it. **OK** approves only that invocation, ending when the script exits.
Cancel, closing the dialog, or a dialog error prevents execution. There is no
approval timeout, stored approval, or unattended bypass. This does not authorize
separate UI actions by an agent. The background regression runner is unchanged.

## `knownBug:` markers

A case may carry a `knownBug:` string. Such a case asserts the behaviour the production code
*should* have and currently fails; the string documents the defect it pins. They are
reported in their own section and, by default, do **not** fail the run — so a newly broken
invariant is never lost in the noise of a defect that is already known. `--strict` makes them
fail the run too. When one starts passing, the runner says so and the marker should be
deleted. There are none at present.

## Native media fixtures

Video and audio fixtures are synthesized with Apple's AVFoundation, CoreGraphics,
CoreVideo and CoreMedia. H.264 encoding uses the system codec; rotated and audio/video
fixtures use native composition. Original MPEG-TS packet framing wraps Apple's encoded
samples for transport-stream and HLS tests. No external media executable is needed.

## Layout

| File | Covers |
|---|---|
| `TestHarness.swift` | the assertion harness, the case registry, environment gates |
| `TestSupport.swift` | fixtures: temporary directories, native clips, images, frames, a tick log |
| `main.swift` | the runner's entry point and every suite registration |
| `ProcessingReviewTests.swift` | legacy preferences, grouping cancellation/redundant-edge reduction, VFR timeline and exact still export |
| `FrameArrangementTests.swift` | the order, rating distributions and filters, value limits and spans, the face filter, the per-group cap, the grid rows |
| `VideoDropTests.swift` | which dropped files `VideoDrop` accepts — the drag-and-drop rule |
| `VideoProbeTests.swift` | `VideoProbe.probe`: metadata, and every rejection path |
| `FrameSamplerTests.swift` | the pull-based full/quick sampler: first-frame inclusion, decoded ordinals, ascending timing and mode estimates |
| `ThumbnailCacheTests.swift` | the thumbnail actor: round-trip, key sanitisation, atomic and concurrent writes, clearing |
| `FrameZoomGeometryTests.swift` | the preview's zoom surface: the fit touches and centres, the canvas that makes centring possible, scroll origins and focus points across frames and resizes, pinch anchoring, zoom clamping and degenerate sizes |
| `FeaturePrintCodecTests.swift` | the codec: bit-exact round trip, compression, corrupt and empty input |
| `FaceSelectionTests.swift` | `FaceSelectionRule`: the largest significant face, not a mean, deterministically |
| `ArtScoreNormaliserTests.swift` | the display-only raw↔0…1 mapping and its degenerate cases |
| `FrameScorePipelineTests.swift` | the pipeline's ordering, completeness and progress, with a hand-written scorer |
| `FrameGrouperTests.swift` | single-linkage clustering, the time window, the threshold's effect, oversized-group splitting, ordering |
| `FrameRankerTests.swift` | `RankMap` ties, and how the face weight moves the ranking |
| `FrameExporterTests.swift` | export naming, the overwrite suffix, and a real PNG/JPEG round trip |
| `RemoteProtocolTests.swift` | bounded binary framing, encryption and replay rejection, keys, discovery expiry, vocabulary integrity, real TCP handshake/UInt8/LZ4 tagging/reconstruction with injected models, cancellation/model-reference cleanup, busy rejection, kick and reconnect |
| `UncoverTests.swift` | snapped ROI/letterboxing/rotation, large selections at every scale, degradation, known-value compositing, consecutive PTS, variable-timing audio/video export, cancellation, algorithm cache/preparation reuse, unsupported-factor preservation, failure retention, and opt-in real native inference |

## WD Tagger

`./run-tests.sh --only WDTagger` checks label ordering, autocomplete, threshold
boundaries, all-head compression, batch tails and original-image ordering, legacy
JSON migration, and RGB/white-letterbox preprocessing. Native model parity runs
offline when the installed default WDTagger model or `models/published/WD-SwinV2-Tagger-v3-CoreML` and its CSV are present; missing
model files are an explicit skip. Native model parity additionally needs the
`waifu-wdtagger` worker, which is a resource of the app bundle: a bare
`./run-tests.sh` cannot see it, so those two cases skip until either
`WAIFU_TEST_TAGGER_INSTALL=1` packages the runtime (see `run-tests.sh`) or
`WAIFU_TEST_TAGGER_WORKER` points at a built binary. Tests generate their asymmetric
color image in memory. Generate the optional independent FP32 probabilities with
`tools/make-wd-tagger-test-fixture.py --weights /path/to/original/model.safetensors`.
This uses original unfused Torch weights and a runtime BGR swap, and writes only
to ignored `.build/test-fixtures/wd-tagger`. The parity case explicitly skips when
that local reference is absent. No binary test data is tracked or uploaded.
The test never downloads a model.

## Real temporal inference

The default suite skips model inference explicitly. Install with
`../tools/setup-native-uncover.sh`, then run:

```sh
WAIFU_TEST_SEEDVR2=1 ../run-tests.sh --only 'Uncover real'
WAIFU_TEST_SEEDVR2=1 WAIFU_TEST_SCALE=4 ../run-tests.sh --only 'Uncover real'
WAIFU_TEST_SEEDVR2=1 WAIFU_TEST_SCALE=8 ../run-tests.sh --only 'Uncover real'
WAIFU_TEST_SEEDVR2=1 WAIFU_TEST_ALGORITHM=flashvsr WAIFU_TEST_SCALE=4 ../run-tests.sh --only 'Uncover real'
```

This reads the configured local model and the reference clip. `WAIFU_TEST_VIDEO` selects
another 1920×1080 source containing the 10-second test window.
`WAIFU_TEST_ROI_WIDTH` / `WAIFU_TEST_ROI_HEIGHT` exercise larger and odd selections; `WAIFU_TEST_SECONDS=1`
exercises chunk boundaries instead of the default five-frame window.
`WAIFU_PROOF_OUTPUT=/absolute/new/path` preserves prepared frames, backend metadata/report
and the reopened MOV independently of temporary test cleanup. No model downloads occur
inside the test. The deterministic backend lives only in `Tests/` and exercises the real
pipeline/compositor/exporter; it is never shipped or presented as SeedVR2.

## The seams, and why they exist

**`FrameGrouper.init(distance:)`.** Vision's FeaturePrint distance cannot be placed on
either side of a threshold at will — it is almost blind to colour and highly sensitive to
texture, so "these two images are clearly different" is not a distance you can name in
advance. The injection lets the *clustering* — union-find, the ±120 s window, the split, the
ordering — be pinned exactly, on an absolute difference over a controlled position.
Vision's own distance is covered separately, by the codec's round-trip cases.

**`FrameExporter.uniqueDestinationURL` and `RankMap`** are internal rather than private so
their decisions can be pinned by value without writing 10,001 files, or re-deriving a rank
map in the test.

**`FrameRanker.faceIsSignificant(_:settings:)`** is exercised through both the ranker and
the arrangement, because the point of it is that there is exactly one answer to "does this
frame have a subject face".

## The sampler is asserted as a stream, not as a batch

`FrameSamplerTests` covers the property the memory budget rests on: `sample` returns a
stream whose elements are produced when the consumer asks (`unfolding:`), so a consumer
that takes two frames gets two and the rest of the file is not decoded ahead of them.
The same cases pin the frame-by-frame count — every frame of the clip is processed,
checked against the probe's own estimate — the first-frame behaviour, and the ascending,
contiguous order the rest of the app assumes. They need a real container and a
decoder, supplied by Apple system frameworks. The measurements behind the budget (67 MB streamed
against 420 MB buffered, and what retaining the per-frame records adds) are in the README's
**Memory** section and in `docs/API.md`'s memory contract.

## The drop rule is a value, not a view

`DropZone` routes; `VideoDrop` decides. A drop handler that rejects everything and one that
never runs look identical from the outside — both produce no analysis and no error — so the
acceptance rule is a pure function that `VideoDropTests` pins by value, and the SwiftUI view
has nothing left to get wrong.

## Preview keyboard regression

`PreviewKeyRoutingTests.swift` covers preview navigation ownership and the persistent
window handler's Space/Escape gestures: one toggle per physical press, consumed release
across overlay teardown, immediate reopening, modified keys, text entry, and modal UI.
The window handler lives outside the conditional preview, scopes events to its own key
window, and leaves Space/Escape to Uncover while it is active. Preview removal uses an
identity transition. The session publisher ordering test checks that scroll/focus effects
run after the preview setter commits. Direct @Published delivery occurs before storage;
synchronous scrolling from that callback can render old state and leave the overlay up.

A hardware smoke check remains useful for macOS focus delivery: load a video, focus a
frame, and repeatedly press Space to open, close, and reopen. Each press should toggle
once. Hold Space to confirm it does not oscillate. Click a preview zoom button, then
close with Space or Escape and immediately reopen with Space. Also check text entry,
export sheets, another window, and Uncover. Automated router tests do not verify real
hardware event delivery. If a focus problem appears, enable `--waifu-trace=keys,focus,preview`
and inspect the `preview.toggle`, `preview.monitor`, and `root preview rendered` records.
Closing must render a nil root preview and trigger `overlay onDisappear`; consuming a
key and clearing session state alone does not prove the visible preview closed.

Hardware validation on October 13, 2026: the user confirmed Space and Escape dismiss
the preview on the first press after deferring the scroll/focus subscribers until the
session mutation has committed.

## The preview's fit

`FrameZoomGeometryTests.swift` covers the geometry the preview draws: the fit touches the
viewport on its limiting axis and centres on the other, the canvas that makes centring
possible, scroll origins and focus points across frames, window resizes and zoom levels,
pinch anchoring, the zoom clamp and degenerate sizes. The last case is the one that pins
the mechanism of the regression — an `NSScrollView` whose magnification is pinned at 1, so
the clip view's bounds stay the viewport the fit is measured against.

Two things still need a person. A trackpad pinch cannot be synthesised: the macOS SDK
exposes no factory for a magnify `NSEvent`, so the anchoring rule is pinned as a value and
the `magnify(with:)` call that uses it is four lines of wiring. And an offscreen render can
measure *where* the frame is but not whether it is smooth. Open a frame and check that it
reaches the border on the axis it is limited by and sits in the middle on the other, pinch
to magnify and confirm the point under the pointer does not move, resize the window and
confirm the frame refits without losing its zoom level or its place, then step to the next
frame with Right.

## URL media and Firefox forwarding

`MediaImportTests` serves generated media through a loopback HTTP fixture. It checks
extensionless and misnamed downloads, HTTP errors, content-based HLS detection,
redirect-relative playlist segments, native HLS frame sampling, unsupported-format
errors, cancellation and imported-file lifetime. Only these cases use a local network
connection; they never contact external services. Native media generation and system Python 3
supply test fixtures only; production media handling uses Apple's native frameworks.
Run `./run-tests.sh --only 'media import'` for this group.

`node tools/test-firefox-extension.cjs` checks the extension's opt-in behavior,
main-frame filter, suffix/query matching and original-navigation fallback on bridge
failure without changing any browser profile. Actual add-on installation requires
the manual steps in `FirefoxExtension/README.md`.
