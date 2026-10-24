# SAFIRE for Photos

Right-click a Photos thumbnail and choose **SAFIRE**. In **Grouping → One group per
line**, the action analyzes every photo in that thumbnail's group, including
members hidden by filters or the per-group display cap. Other layouts and
ungrouped thumbnails analyze only the clicked photo. An unrelated multiselection
never expands the request. Video frames are excluded. The SAFIRE action remains
available during album analysis and regrouping. If an inference lane is busy, the
workspace or preview displays a waiting message and resumes automatically; Stop,
closing the workspace, or preview navigation cancels that wait.

The workspace shows every requested photo, sorted from **lowest to highest
forged-mask area ratio**. Each card shows the photograph with a strong red overlay
and bright boundary, alongside its separate black-and-white mask. White pixels
are flagged. The ratio is the number of probabilities at or above 0.5 divided by
the full 1024 × 1024 mask area. Ties keep stable photo order.

The least masked versions are labeled **Original candidate** after all requested
results arrive, or **Least masked so far** while analysis is incomplete. This is
the user's selected ranking rule. It is not a learned chronology prediction or
proof of originality. A smaller mask can reflect a missed edit; natural scene
boundaries can also be flagged. All equally low-mask versions remain candidates.

Select **Read region evidence** for a photograph to inspect its genuine connected
regions and cropped evidence. Explanations describe the observed mask, its
location and threshold, and the model's limits. There are no semantic object
labels, fabricated editing-application claims, LLM stories or pixel-comparison
controls. The existing metadata inspector and Show in Photos action are reused.

In the ordinary **Photos preview**, turn **SAFIRE** on and choose **Overlay** or
**Mask**. It analyzes only the current still photo and reuses the same
content/model cache as the workspace. Fast navigation cancels stale requests and
debounces uncached inference. Results from an earlier photo cannot overlay the
new one. The existing preview zoom and navigation controls remain available.

The right-panel expand/collapse control is available in Photos, ranked media,
preview and SAFIRE. Photos retains its preview and Sort and Filter column.

## Photos deletion and investigation state

Delete Photo from Library remains available through the visible trash controls,
thumbnail context menus and Command-Delete. The app confirms the exact asset before
calling the shared PhotoKit deletion service. Photos may require additional
native authorization. Failure preserves the investigation. Successful deletion
removes evidence and references, updates surviving selection, invalidates pending
analysis and rejects late results. Photos manages recovery in Recently Deleted;
WaifuFinder does not promise an independent undo.

Investigations retain keeper/review state separately for each explicit photo or
group scope. Navigating to another album closes the old scope. Reopening one
scope does not analyze other Photos albums or groups.

## Genuine model evidence

SAFIRE supplies a binary manipulation probability map and multi-source partitions
from the official pretrained checkpoint. It does not directly identify an original,
editing program or capture order. Source-partition IDs group learned spatial
features, rather than identifying people or cameras. Exact component contours and
bounds use a 0.5 threshold; components smaller than four mask pixels are omitted
from the expanded region descriptions. Mask-area ranking includes **all** pixels
above the threshold, including smaller components.

The retained lineage data model supports multiple parents, confirmed and tentative
relationships, cycle protection, stale-content invalidation and deletion cleanup.
Classical correspondence and date hypotheses are distinct from SAFIRE evidence
and never become a neural originality verdict.

`ImageLineage-fixture-validation.json` records actual diagnostic model behavior on
one photographic scene with known edits. SAFIRE flagged 1.192% of baseline pixels,
81.3% of the watermark bounding window, and 11.9% of the local patch-fill window.
The recompressed third-generation watermark had only two thresholded pixels.
These are diagnostic observations, not precise edit-mask recall or dataset-level
accuracy. The baseline is an Apple-generated rendition, not a pristine camera raw.
The native photographic correspondence check also includes a separate adjacent
Apollo photograph, rather than mislabeling a derivative as a different capture.

## Native architecture and CPU efficiency

`Runtime/Sources/SAFIREMLX` implements the official ViT-B adapter encoder, FFT
high-pass branch, positive-point prompt encoder, two-way transformer decoder and
automatic 256-prompt prediction. Postprocessing preserves upstream raw-feature
clustering means. Deterministic farthest-first initialization makes comparisons
reproducible. Fused Metal preprocessing matches the official OpenCV uint8 linear
square-resize convention. Oriented top-left masks remain registered to the photo.

FP32 is retained for numerical and localization parity. The worker prepares
convolution weights once, caches compiled decoder batches, chunks global
relative-position attention, validates GPU resources and remains warm for 60
seconds. Per-device profiles have measured provenance and conservative fallbacks.
The allocator budget is 2 GiB with a 128 MiB cache; it is not a hard process cap.

GPU batch mask areas and feature pooling replace per-prompt scalar/readback
fences. Internal worker output, encrypted spatial responses and mask caches use
contiguous Float32/UInt16 binary data. They avoid formatting and parsing a million
JSON numbers per result. Legacy JSON caches/responses remain readable. The default
workspace does not run image registration or load a second image.

On the independently tested M4 host `thefreebao.local`, three warm diagnostic
samples gave:

| Worker path | Median wall time | Median worker CPU time per image |
| --- | --- | --- |
| Before: per-prompt readback + JSON | 3.103 s | 0.79 CPU seconds |
| After: GPU batch pooling + binary output | 2.585 s | 0.34 CPU seconds |

That is about 57% less worker CPU time and 17% less wall time on this diagnostic.
The optimized worker averaged about 13% of one CPU core during requests. CPU time
includes every worker thread and is sampled using cumulative process CPU time.
These results exclude cold loading, Photos fetch/client PNG encoding and network
transfer. They include image decode, all 256 prompts, postprocessing, serialization
and output-file I/O. See `ImageLineage-native-validation.json` for M3 and M4 results,
profile selection evidence, and validation. The M3 128/4 paired diagnostic measured 4.083 seconds and 1.62 CPU seconds before,
and 3.511 seconds and 1.12 CPU seconds after (about 31% less worker CPU time).
The original separate M3 JSON baseline was 4.305 seconds. These diagnostics do
not establish a globally fastest configuration or guarantee a busy machine’s timing.

No Python, Torch, ONNX or OpenCV runtime is shipped. Production uses the already
pinned MLX Swift/core, Swift Numerics and Apple frameworks. The offline PyTorch
oracle and its audited permissive dependencies never enter the app or package.

## Model management, publication and remote execution

Install SAFIRE in the existing **Settings → Models** manager. Download,
verification, preparation, ready/in-use, failure, update, removal and storage
states use the normal lifecycle. Every artifact is checksum-pinned; the worker
hashes the actual weights before loading them.

Published repository: [alastorid/SAFIRE-Swift-MLX](https://huggingface.co/alastorid/SAFIRE-Swift-MLX).
`tools/safire-release.json` pins revision
`6422600bd8a8b361ec4869bd597267b7553b5533`, under immutable path
`releases/safire-native-v1-7225d52e9f42b6e6`. It includes converted weights,
architecture/source hashes, compatibility metadata, validation, measured profiles,
licenses and attribution. Actual clean installation through ModelInstaller and
native inference passed; reinstall reused the verified version without changing
the user's real model configuration.

Remote requests reuse authenticated encrypted connections, request streams,
capabilities, cancellation, resource arbitration and the same native worker.
Lossless PNG preserves input pixels. Real local and encrypted-loopback mask
probabilities and labels were identical after installing the published package.
No independent remote neural implementation exists.

Asset content identity, weight fingerprint, preprocessing version and runtime
revision key the bounded 512 MiB evidence cache. Visible card previews are bounded
and released when scrolled out. Selection, mask toggles and preview reuse cache
results; they do not trigger new inference when a compatible result exists.

## Reproduction and validation

Official checkpoint SHA-256:
`65bdf6880380162d57f77d71b3fd367c3f16aebca316f942a821da16e9225b12`.
Converted FP32 weights SHA-256:
`88477a26fe3f6c5e4569ecb4b7b73b6db6a92adac093dd66641ee2e464dde62f`.
Inspected upstream revision: `ec2af6af8c789fb6359493a3e352f89c6fd8107f`.

- `tools/convert-safire.py`: restricted standard-library checkpoint conversion.
- `tools/build-native-safire.sh`: native build and shared metallib verification.
- `tools/validate-safire-reference.py` and `tools/compare-safire-activations.py`:
  all encoder blocks, decoder, automatic masks and partitions. Both M3 and M4
  passed after GPU pooling changes. Required relative RMS ≤ 2e-4; maximum error ≤
  2e-4 + 1e-3 × dynamic range; binary IoU ≥ .999; partition agreement ≥ .9999;
  minimum source-partition IoU ≥ .999. Five geometries check fused preprocessing
  to less than 1e-6. A separate gradient case checks spatial boundaries.
- `tools/benchmark-native-safire.py`: complete warm wall time and worker CPU time;
  `--transport sfm1` exercises the production binary path.
- `tools/make-lineage-fixtures.swift`: Apple-native resize, JPEG, watermark,
  removal, patch fill, sibling/missing-baseline fixtures and actual production
  Core ML RealESRGAN restoration.
- `tools/evaluate-lineage-fixtures.py`: real native outputs and diagnostic mask
  coverage; no Python ML/image dependencies.
- `./run-tests.sh --only 'Image Lineage'`: scope, ranking, masks, orientation,
  binary integrity, graph and cache regression checks.
- `WAIFU_TEST_SAFIRE_INSTALL=1 ./run-tests.sh --only 'SAFIRE'`: opt-in clean
  publication download and real encrypted native equivalence. Requires a built
  `dist/WaifuFinder.app` and network access.
- `WAIFU_TEST_LINEAGE_FIXTURES=PATH ./run-tests.sh --only 'photographic fixtures'`:
  actual known-history correspondence diagnostics.
- `python3 tools/audit-licenses.py`, `./build.sh` and the background regression
  suite check licensing and app integration. Final full verification: 460 passed,
  zero failed, seven unrelated optional model fixtures skipped. Actual clean
  SAFIRE installation and encrypted binary-mask equivalence passed.

Fixture source credit: [NASA Apollo 17 Blue Marble](https://science.nasa.gov/resource/the-blue-marble/),
AS17-148-22727. Separate capture: [AS17-148-22726](https://eol.jsc.nasa.gov/SearchPhotos/photo.pl?frame=22726&mission=AS17&roll=148).
Image courtesy of the Earth Science and Remote Sensing Unit, NASA Johnson Space
Center; see [NASA's conditions of use](https://eol.jsc.nasa.gov/FAQ/default.htm).
Derived restoration is WaifuFinder-generated; no NASA endorsement is implied.

Independently authored WaifuFinder source remains MIT. SAFIRE/SAM adaptations
retain Apache-2.0 notices, weights CC BY 4.0, resampling BSD and clustering MIT.
See `Runtime/SAFIRE-*` for terms and the offline-oracle inventory. No GPL/LGPL or
external media encoder dependency was added.

Interactive UI testing is disabled by the user's standing request. Compilation,
background state tests and rendered-mask checks do not claim interactive UI QA.
