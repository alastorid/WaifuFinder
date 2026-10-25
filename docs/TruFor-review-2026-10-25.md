# TruFor implementation review — 2026-10-25

The review found and fixed functional defects that the earlier passing suite did
not cover. Verification uses background builds, native inference and automated
rendering; no interactive UI tests were performed.

| Finding | Fix and verification |
| --- | --- |
| Overlay shader could not compile because of invalid `unsigned char(...)` casts. GPU buffers also collided at slot zero. | Valid Metal casts and distinct buffer slots. The actual renderer now runs in regression tests. |
| Outline neighborhoods clamped both axes to width, causing wrong portrait edges and out-of-range landscape reads. | Clamp by both dimensions. Compare every rendered pixel with a CPU morphology reference for portrait and landscape maps. |
| Persisted scalar records used analysis width zero, rejected by the shared cache reader on reopen. | Use a valid scalar identity. A separate cache reader now restores an exact zero score; edited assets and other models remain misses. |
| Default FP16 inference did not meet the strict score/spatial validation attached to the package. Older workers and scores shared its identity. | Production defaults to FP32, and app workers explicitly pin it. The `trufor-fp32-v2` execution capability and cache namespace separate old results and remote workers. Weight-package identity remains unchanged. |
| Compiled graphs accumulated for every photo geometry. | Retain at most four graph geometries, evicting the least recently used. |
| Old jobs could continue after leaving or changing an album; repeated Start actions could overlap during cache refresh. | Cancel on source transitions and album-snapshot changes; guard concurrent starts. Content/revision checks still protect commits. |
| Crystal Orb's shared context had a private holder type exposed across files, preventing compilation. | Correct the holder's access level. Both the regression executable and release app compile. |

Additional hardening rejects excessive evidence dimensions before multiplication,
reports allocation failure instead of force-unwrapping Metal buffers, and names
TruFor correctly in model-setup errors. Remote tests verify that cancelling
evidence leaves the authenticated connection available for another score job.

## Verification

- Full regression suite: **471 passed, 0 failed, 11 optional fixture/install skips**.
- Additional TruFor run: **9 passed**, including a fresh real-model install,
  isolated activation, and identical local/remote evidence over encryption.
- Actual compiled default worker compared with the official reference: detection
  score absolute error **0.00000380**, below the **0.00001** gate; anomaly maximum
  error **0.00004146**; reliability **0.00002450**; Noiseprint **0.00025855**.
- Release app built successfully; strict code-signature verification passed.
- All 38 native worker sections match the signed bundled copy (whole-file hashes
  differ because the bundle worker is signed).

FP16 remains an explicit standalone benchmark option, rather than a production
accuracy promise. The prior paired benchmarks put FP32 about 9% slower on M3 and
M4; those historical figures are preserved in `TruFor-native-validation.json`.
Remote Macs need the updated app to advertise the new execution capability.
