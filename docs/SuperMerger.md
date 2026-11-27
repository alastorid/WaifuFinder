# Super Merger

Open **File → Super Merger** or select related frames and choose **Super Merger**
from their context menu. Choose or drop 2–16 versions of one photograph. Analysis
starts automatically. **Replace Sources** starts a new set when the current work
has stopped. The persistent toolbar keeps Undo, Export and Done available without
scrolling; progress, cancellation and resume live in the bottom status bar.

Selections from a Photos album use the unmodified originals. When the evidence
for a whole-image source is comparable, Super Merger prefers JPEG over WebP.
Aligned WebP sources can still supply individual patches.

The workspace puts the photograph on the left and the region review on the right.
The source strip shows aligned previews, the base source and excluded inputs.
Selecting a thumbnail inspects that source without changing the result. **Result**,
**Source** and **Compare** are always-visible choices; Compare places the result
and selected aligned source side by side with shared zoom and pan. Browsing remains
available during review. Excluded sources can be inspected but cannot be compared
in reference coordinates or chosen as replacement pixels.

Select a red region on the image or a numbered region in the inspector. The selected
region is outlined in yellow. Previous/next buttons navigate regions; **Locate**
centers and enlarges the selected area. Aligned source crops appear in an adaptive
grid with model evidence, confidence and the source actually used in the published
result. **Use Source** changes only that region; **Reset** restores its automatic
choice. Undo reverses an earlier choice. These actions are disabled while a result
is being rendered. Uncertain and unreviewed regions remain explicitly labelled.

Use **Fit**, **100%**, or double-click the image to change the preview scale. The
region-outline toggle leaves the underlying image unobscured. Preview 100% refers
to preview pixels; exports use full resolution. **Output verification** in the
inspector checks the rendered plan against the selected source without rebuilding
the displayed preview. Switching sources or changing the result clears stale audit
results. Export creates a folder containing the image, original sources, review
records, source map and output differences. The inspector and export use the
published, verified reconstruction plan.

## Review pipeline

1. Registration estimates orientation, overlap and a global transform. Local
   agreement is checked in up to 64 sampled areas. A source with less than 70%
   support is excluded. If the initial reference aligns with no other source, the
   remaining set gets one independent attempt.
2. Median RGB offsets compensate uniform color shifts for difference detection.
   Residual thresholds use median absolute deviation and stay within 12–32 RGB
   levels. These corrections only affect detection. Reconstruction uses source
   pixels. Small edits below the measured threshold may be missed.
3. Nearby components consolidate across tile boundaries. The planner preserves the
   calibrated mask and stops before model review if more than seven regions remain,
   if mask storage is incomplete, or if differences cover more than 35% of the
   reference. A rejected set can be inspected or replaced.
4. The model provides an explicit source choice or abstention and a comparative
   explanation for every evidence package. Its per-source observations are retained.
   Contradictory or unsupported choices become abstentions. The local runtime and
   remote server must support the same review contract.
5. An eligible original is exported byte for byte. Otherwise only supported masks
   can receive replacement pixels. Automatic donors must meet the clean-evidence
   and coverage gates, then pass the boundary check. Manual overrides are labelled
   separately. Export independently verifies the written image against its recorded
   sources.

When complete difference masks are too fragmented for automatic review, select
an area on the image and click **Use Source** on an aligned version. The export
verifies coverage for each manual patch and records the source of its pixels.

## Validation scope

Registration and local agreement can reject incompatible inputs, but cannot prove
that two closely matching captures share the same photographic root. The model's
observations also cannot establish provenance or detect every shared edit. The
review and export retain uncertainty. Composite encoding uses the existing 8-bit
sRGB reconstruction path, while preserved originals retain their source format and
metadata. Region overrides reuse analysis and model decisions and re-render the
preview. Native tiled viewport rendering and incremental compositing are outside
the current implementation.

The noninteractive suite includes synthetic admission checks and a photographic
fixture with complementary overlays, duplicate content, JPEG encoding, and eight
separate edits. Run the installed-model fixture with
`WAIFU_TEST_OPTIMIZED=1 WAIFU_SUPER_MERGER_LIVE=1 ./run-tests.sh --only 'Super Merger'`.
It records measured latency and the actual source decisions in
`.build/super-merger-live-review.json`.

On the Apple M3 with 16 GB of unified memory, the installed model chose the known
clean source for the overview and both changed regions in the three-source NASA
fixture. The planner chose original-file preservation. The model still
misattributed some overlays in its per-source descriptions. This is one selection
fixture and does not establish general watermark or AI-edit detection accuracy.

The three evidence packages and model review took 42.10 s and generated 1,329
tokens. Peak worker physical footprint reported at the replies was 1,645,202,936
bytes. The largest sampled Metal allocation was 3,220,930,560 bytes, and the
largest sampled compressed memory was 117,489,664 bytes. Reported process swap-ins
were zero. Metal allocation is a separate counter. This run did not measure the
whole app's peak footprint or time from opening the images to export.
