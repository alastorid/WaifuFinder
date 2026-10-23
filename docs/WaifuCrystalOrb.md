# WaifuCrystalOrb

Orb is a same-window SwiftUI inspection workspace. Its grid stays mounted, including
Photos album navigation. The title-bar orb sits immediately after tagging progress
in each workspace. Paste (⌘V), an image drop, or Choose Image supplies an independent
`CrystalOrbQuery`; it never fabricates a frame or selects a query in the collection.
Text editors retain their normal paste behavior. Video URLs continue to the existing
video importer.

## Implementation

`WaifuCrystalOrbModel` owns import, source, search and selected-pair lifetimes. UUID
tickets reject stale provider callbacks, decoding, ranking, metadata and comparison
results. Closing cancels tasks and releases query and comparison buffers; the search
actor retains only compact features. A source's thumbnail generation invalidates its
index and matches. Completing analysis reruns an open query against the finished set.

`CrystalOrbSearchService` searches all analyzed source frames, independent of grid
filters, sort or representative caps. It uses Apple's revision-2 Vision feature-print
distance, reusing existing `FeaturePrintCodec` payloads. Missing features are computed
from analysis renditions sequentially, away from the main actor. It computes one
query feature per query identity and retains decoded source features across queries.
Results sort by distance then stable frame ID; distance is not a confidence percentage.
The strip displays the live top 19. Search publishes the first examined candidate
immediately, then coalesces changed rankings to at most twice per second (500 ms apart),
including the final ranking. Unchanged lists do not publish again. Rank one fills the candidate pane automatically as better matches arrive;
a deliberate user selection takes control until the next search. The summary reports
unavailable candidates and Photos assets not yet analyzed; Orb never scans unrelated
albums or the entire library.

Originals come from `PhotosMedia.image`, which handles video collections, PhotoKit,
orientation, native resolution and the existing iCloud preference. Only deliberate candidate clicks or Load original request originals; automatic rank-one presentation and gallery thumbnails stay preview-only. Up to three recently inspected pairs
share a 160 MB cost-limited cache, so revisiting a match is immediate. New pairs commit
the best already cached candidate thumbnail immediately, then replace its pixels
with a larger local PhotoKit preview up to 2048 pixels when available. Explicitly loaded
candidates then upgrade to the original in the background. Difference processing runs independently and
never substitutes a loading splash for the photograph. Small preview comparisons
remain visible until native comparisons are ready and are labelled Preview.
External queries first decode an oriented ImageIO preview up to 2048 pixels, start
searching, and upgrade to the original under the same query identity. Query replacements
preserve content until preview decoding succeeds. Headers and metadata have stable geometry during loading. Visible thumbnails share `GridThumbnailLoader`,
`GridThumbnailPresentation`, and foreground work priority with the grid; two adjacent
candidate thumbnails are prefetched. Optional Photos metadata is independent of the
first image display and uses `PhotoMetadataService`; expensive original-resource
metadata is fetched only when the inspector opens. External/clipboard metadata
uses ImageIO and the existing parser and inspector. Clipboard payloads have no
invented filename, file size, timestamp, frame ID, or library identity.

`CrystalOrbAlignment` uses a common canvas as large as the maximum source width and
height. Each source is independently aspect-fit and centered without cropping; this
upscales the smaller source without reducing the larger one. This is geometric
aspect-fit placement, not automatic content registration. Core Image Lanczos sampling
and an explicit sRGB working/output space feed absolute per-channel RGB differences.
Black means equal, full channel range is 0–255, and magenta marks unmatched regions.
The reusable red mask uses the maximum changed RGB channel with a square-root
display curve as alpha, making subtle changes visible while exact matches stay clear.
Unmatched regions have opaque red. Transparent source pixels compare over black. Slider adjustment only changes compositing opacity. The same
mask works in both side-by-side panels, swipe and toggle; Difference suppresses it
without clearing preferences. There is no hard source-size rejection. Two previews are bounded to a 1600-pixel
long edge; lazy native-resolution Core Image graphs render only the visible viewport
at the window's backing resolution for zoomed inspection. A 50 MP test retained
8192 × 6144 native coordinates, generated 1600 × 1200 previews and a 256 × 192
native-detail tile in 143 ms. Original image loading is not included in that timing.

Alignment uses declared native image dimensions even while a preview is displayed.
Upgrading query/candidate pixels preserves zoom, pan, actual-pixel mode, swipe position,
comparison mode, toggle position, and red-overlay controls. Only an explicit Fit action
resets the zoom/pan settings. The orb glows and sparkles during search and refinement;
the inspection canvas has no blocking loading state.

All modes share zoom and pan. Fit uses the viewport dimensions; 1:1 accounts for the
screen backing scale. Source toggling retains the same geometry. Swipe clips in
viewport coordinates so the divider remains at the visible boundary during panning.

Candidate clicks synchronize the session's real focused and selected frame IDs.
Closing sets a deferred grid-scroll request through the existing staged-row pipeline.
A hidden candidate is transiently pinned in displayed frames without mutating filter
or grouping preferences, then removed when normal pointer browsing moves elsewhere.
The grid restores keyboard focus after the close publication commits. Photos reveal
uses `PhotosAppIntegration.reveal` and does not change selection.

Keyboard handling is part of `PreviewToggleKeyboard`, not another overlapping local
monitor: Escape closes, Left/Right browse, S/W/T/D choose modes, R toggles the mask,
and Space temporarily toggles the opposite image. Text focus and sheets gate routing.

## Validation

Run `./run-tests.sh --only 'Crystal Orb'` for native image fixtures, real Vision
ranking/cache tests, alignment/difference checks, and exact grid reveal tests.
`./run-tests.sh` runs the existing regression suites as well. `./build.sh` builds the
app (`WAIFU_BUILD_CONFIGURATION=debug` selects the supported debug configuration).

The reviewable interactive smoke script is `tools/ui-test-crystal-orb.sh`. Invoke
only through `./run-ui-tests.sh tools/ui-test-crystal-orb.sh`. The launcher requests
approval for that exact invocation and sends the required completion notification.
The script restarts the app, imports a synthetic Apple-generated video, pastes a
native frame, checks results and all modes, closes to the grid, then checks Photos
workspace toolbar and paste activation. It restores clipboard contents and saves
accessibility trees/screenshots. It does not alter Photos assets or iCloud settings.

Synchronous ImageIO query decoding and pair difference rendering each have a serial
actor. This bounds native processing across rapid query/candidate replacements,
including calls that cannot interrupt an already-running native image operation.
Cancellation is checked before queued work begins and before publishing results.

The initial 2,048-candidate synthetic benchmark using existing stored features
measured 78 ms for the cold pass and 3 ms for the warm pass. This measures feature
loading/ranking, not original decoding or initial feature computation for unindexed
media. Interactive verification was stopped at the Accessibility-permission check;
no further UI tests were run, at the user's request.

## Changed-file inventory

New production sources:

- `Sources/Waifu/CrystalOrb/WaifuCrystalOrbModel.swift`
- `Sources/Waifu/CrystalOrb/WaifuCrystalOrbView.swift`
- `Sources/Waifu/CrystalOrb/CrystalOrbSearchService.swift`
- `Sources/Waifu/CrystalOrb/CrystalOrbImageSource.swift`
- `Sources/Waifu/CrystalOrb/CrystalOrbImageWorkers.swift`
- `Sources/Waifu/CrystalOrb/CrystalOrbImport.swift`
- `Sources/Waifu/CrystalOrb/CrystalOrbComparisonView.swift`
- `Sources/Waifu/CrystalOrb/CrystalOrbDifferenceRenderer.swift`
- `Sources/Waifu/CrystalOrb/CrystalOrbRasterView.swift` — bounded native viewport rasterization.

Integration changes:

- `Sources/Waifu/UI/RootView.swift` — workspace lifetime, input routing, video toolbar.
- `Sources/Waifu/UI/PhotosWorkspaceView.swift` — Photos toolbar insertion.
- `Sources/Waifu/UI/DropZone.swift` — image-provider delegation.
- `Sources/Waifu/UI/PreviewToggleKeyboard.swift` — paste and Orb key routing.
- `Sources/Waifu/UI/FrameGridView.swift` — navigation gating and focus restoration.
- `Sources/Waifu/Engine/WaifuSession.swift` — exact selection, transient reveal and scroll.

Verification and notes:

- `Tests/CrystalOrbTests.swift` — sixteen native search, renderer, lifecycle and integration cases.
- `Tests/main.swift` — register new suites.
- `Tests/GridThumbnailStageTests.swift` — wait for actual asynchronous sort completion,
  independently of its deliberately delayed spinner.
- `tools/ui-test-crystal-orb.sh` and `tools/ui-crystal-orb-helper.swift` — saved smoke check.
- `docs/WaifuCrystalOrb.md` — architecture, behavior, constraints and validation.

Other pre-existing or concurrently edited working-tree changes are not part of Orb.
