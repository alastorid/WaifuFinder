# Super Merger workspace redesign — 2026-11-15

## Scope

The SwiftUI workspace now separates the canvas, thumbnail source browser, selected
region review and persistent actions. Result/source/side-by-side comparison shares
viewport position and magnification. Browsing is read-only and remains available
during review. Explicit region choices retain the existing verified reconstruction
and export pipeline. Output audits run separately from preview reconstruction and
are cancelled/invalidated when the source or reconstruction changes. Undo and Reset
reflect actual edit history and overrides.

No model, precision, alignment, reconstruction or inference backend changed. No
new dependencies were added. Interactive UI tests were neither created nor run.

## Noninteractive checks

Production and test sources compiled with Swift 6.3.3 for arm64 macOS 15. The
compiler required an explicit writable `-module-cache-path .build/ModuleCache` in
this restricted environment; the normal scripts were left unchanged.

| Suite | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| Super Merger | 1 | 1 | 0 |
| PhotoArena | 17 | 10 | 2 |
| Frame zoom | 10 | 0 | 0 |

The image fixtures could not validate rendering/registration here. An independent
Apple Core Image probe rendering a constant opaque red-orange image returned
`[0, 0, 0, 0]` instead of `[255, 64, 0, 255]`, using both the default context and
`useSoftwareRenderer: true`. The three remote-worker tests could not authenticate
loopback; an independent `127.0.0.1` listener bind failed with `EPERM`.
These failures mean image and remote end-to-end behavior is **not verified in this
environment**. No successful model-inference or visual-interaction run is claimed.
The zoom suite uses geometry and offscreen native view checks without opening or
interacting with the user's windows.

Logs are in `.build/super-merger-redesign-tests.log` and
`.build/super-merger-redesign-coreimage-probe.log`.

## App build

The release app build succeeded in 113 seconds with warnings treated as errors.
The output is `dist/WaifuFinder.app` and reuses the existing native
runtimes. The standard packaging steps run from a temporary copy of `build.sh`
with an explicit writable compiler cache; native runtime rebuild calls are omitted
because this change only modifies the host app. The user's running app is not
quit or relaunched. Build log: `.build/super-merger-redesign-build.log`.
