# PhotoArena v1 architecture decision record

The current product and architecture are described in [PhotoArena 2.0](PhotoArena-2.0.md). The following is the historical prototype decision record.

## Independent proposals and debate

Agent A proposed bounded-preview geometric registration, native-pixel tile residuals,
OCR text candidates, and conservative rejection of invalid correspondence. Agent B
proposed reusing the existing real MiniCPM native Metal/CPU runtime, judging only
corresponding native crops with reversed candidate order and a bounded sparse schedule.
Agent C proposed native-resolution hard source selection and TIFF export with a
source-index map rather than blending away evidence.

The material disagreements were resolved as follows:

- A originally proposed scale/translation first. The lead requested global rotation
  and perspective support. Apple Vision homography is now evaluated against a
  centered scale baseline; only improved correlation with adequate overlap is accepted.
  Local optical flow is excluded because it can erase evidence of geometric edits.
- B objected to treating sharpness or majority agreement as authenticity. All automatic
  regions remain unresolved; technical quality and category-specific relative preferences
  are separate. Human authenticity preferences can guide source selection but do not prove
  provenance. AI votes cannot silently override source pixels.
- C defended hard source boundaries for exact sample preservation. The lead accepted
  them as a working first reconstruction path with explicit seam warnings, not a claim of
  finished seam optimization. Blending, semantic masks and graph cuts remain unimplemented.
- The lead rejected a fresh speculative ANE conversion as a prerequisite. The existing
  tested native model runs on Metal/CPU. No actual Core ML/ANE placement is claimed.
- The lead identified per-pixel allocation overhead in the initial median implementation;
  A replaced temporary pixel arrays with tile-reused buffers before high-resolution work.

Targeted tests settle geometry, localized changes, unknown/tie rankings, content/geometry
cache invalidation, candidate-order reversal, exact complementary-source reconstruction,
source-index decoding, and 40MP export. These measure implementation correctness, not
AI-edit detection accuracy. See the test output and retained evaluation JSON for outcomes.

## Shared contract

`ArenaAnalysisTypes.swift` defines oriented native pixel edges with a bottom-left origin,
source-to-reference projective transforms, per-source acceptance/confidence, rectangular
regions, measured residuals, texture/clipping/text candidates, and uncertainty. The largest
source defines reference resolution. Unreliable sources are excluded rather than force-matched.

`ArenaForensics` owns import metadata, registration, tiled native measurements, crop
rendering and overview differences. `ArenaRegionalJudge` owns at-most-linear regional
scheduling with a strict total request cap, native 448-pixel inspection crops, scalar JSON
cache keyed by submitted pixels/geometry/runtime, and model records. `ArenaRanking`
fits regularized pairwise category preferences and a separate initial technical order.
`ArenaReconstruction` owns reversible source selection, previews, TIFF/source-map export,
streamed source SHA256 identities, and a complete analysis manifest.

`ArenaStudioView` offers file-group import and four native workspace modes. Background
analysis/export and bounded display previews keep the UI thread available. Regional
human votes and explicit source overrides are separate evidence. File-content-keyed
session storage preserves votes and overrides, while model comparisons use their own
pixel-content cache. Exports include category rankings, technical rankings, votes and
structured model judgments.

## Performance and failure policy

Analyze 2–16 local still images with bounded 448-pixel tile buffers and bounded registration
previews. Decode through Apple ImageIO/Core Image, without new third-party dependencies.
No Python inference, media utilities, neural restoration or synthetic source detail is used.
Only selected crop pairs reach the VLM. Export holds output RGBA and a one-byte source map,
with sequential tiled source rendering; it does not intentionally retain full decoded RGBA
copies of every source. Apple framework decoding behavior must still be measured.

Reject unreadable/oversized inputs and weak alignment. Registration confidence is an
uncalibrated correlation metric, not probability. Repeated or shared edits can defeat
cross-source consensus. OCR may identify genuine scene text. Default source selection
can retain watermarks and manipulated content. No output is labeled fully recovered.

Noninteractive automated tests and background builds are permitted. No interactive UI
verification is created or run, in accordance with AGENTS.md. The UI is compiled but
its interactions require user evaluation. Detailed limitations and next priorities are
recorded in PhotoArena.md and PhotoArena-model-runtime.md.
