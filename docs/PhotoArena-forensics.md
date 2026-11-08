# PhotoArena deterministic forensics

`ArenaForensics` uses Apple ImageIO, Core Image and Vision only. There are no added
third-party dependencies, model downloads or Python inference paths. The existing
TruFor and WD Tagger backends are unchanged.

## Implemented contract

Coordinates are EXIF-oriented native pixel-edge coordinates with a bottom-left
origin. The largest input defines the reference raster. `ArenaTransform` stores a
source-to-reference homography, including its two projective denominator terms;
`point()` and `alignedImage()` apply that complete mapping. The affine property
alone is insufficient for a projective transform.

Registration compares fit-to-reference and native-crop hypotheses, estimates a
Vision homography on previews capped at 768 pixels, and validates the result using
textured luminance correlation, residual and overlap. Unrelated, textureless or
poorly aligned sources are rejected. The threshold is conservative, not a
calibrated match probability. Local deformation and parallax remain unverified.

Accepted sources are scanned at native reference sampling in 448-pixel tiles.
Source metrics include absolute RGB deviation from the cross-source pixel median,
a normalized gradient, clipping, alpha validity and OCR text-candidate coverage.
Two sources have equal median deviations when they disagree: the residual cannot
identify which one is original. Noise and artificial sharpening can increase the
gradient. OCR text may be part of the photographed scene. None of these metrics
proves an AI edit or a watermark.

`differenceMask()` exports thresholded changed pixels, `validityMask()` exports
covered pixels, and `differenceOverlay()` renders a bounded visual difference.
These masks are geometric/pixel evidence, not learned semantic classifiers.
Every automatically analyzed region remains unresolved for authenticity.

The scan retains lazy CI image descriptors and bounded tile buffers rather than
explicitly collecting every original as an RGBA image. Apple decoders and rendering
can still allocate image-wide storage; physical-footprint measurements below
include those allocations. macOS paging remains enabled.

## Verified evidence

Run `./run-tests.sh --only "PhotoArena Forensics"` for five noninteractive tests:

- A 32×32 replacement patch is localized to its exact native tile. The binary
  difference mask contains exactly 1,024 changed pixels, matching the known patch;
  its bottom-left coordinates are checked against the top-down fixture raster.
- An unrelated deterministic raster is rejected.
- A seven-pixel translation is recovered in native coordinates.
- A native crop retains scale 1 and the original pixel offsets.
- Projective point mapping agrees with its analytical homogeneous coordinates.

`photoarena-registration-benchmark.json` records Apple-generated transformations
of the real NASA Collins fixture. All four cases were accepted: two-degree
rotation, centered crop, off-center crop and mild perspective. Correlations were
0.99946–0.99999. The off-center crop origin was estimated as (34.705, 126.906)
against a known (35, 127). This single photograph does not establish general
registration accuracy; no local reprojection-error distribution is claimed.
The reproducible fixture/measurement source is
`tools/benchmark-photoarena-registration.swift`.

`photoarena-forensics-40mp.json` records two identical 8000×5000 Apple-generated
checkerboards: 216 native tiles in 6.675 seconds with a 20ms-sampled whole-process
physical-footprint maximum of 1,210,386,136 bytes. This measures coverage and
throughput, not robustness on repetitive scenes or real 40MP photographic detail.
Compressed-memory samples are retained; process swap-ins were unavailable.
The generator and sampler are in `tools/benchmark-photoarena-forensics.swift`.
Compile these standalone Swift tools with the three production files named in
their headers, place executables in a temporary directory and remove that
directory after use. Reports remain in `docs/`; fixture images are temporary.

## Remaining limitations

There is no semantic segmentation, dedicated watermark classifier, learned
manipulation mask, robust feature-level match verification, local optical flow,
SSIM pyramid or native-detail authenticity classifier. Overview registration plus
native tile residuals do not substitute for those capabilities. Global matches
can fail on repeated structures or large edits; safe rejection may exclude useful
sources. No ANE execution or operation placement is measured by this module.
