# Super Merger

Open **File → Super Merger** or select related frames and choose **Super Merger**
from their context menu. Import 2–16 versions of one photograph. The result opens
inside the main window using the shared Preview image surface. Select a red region
on the image or in the inspector to inspect aligned crops, read the model's
explanation and see the source actually used in the rendered result. The selected
region is outlined in yellow. Region overrides can be undone or restored to the
automatic decision.

Select a source to inspect it in the same viewport. **Audit differences** measures
the rendered result against that source. Double-click the image to switch between
**Fit** and **Preview 100%**. Press Space to return to the previous view. Exports
use full resolution. The export folder contains
the image, original sources, review records, source map, and output differences.
The inspector and export both use the published, verified reconstruction plan.

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
