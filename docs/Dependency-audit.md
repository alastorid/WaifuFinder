# Dependency cleanup

The repository prohibits GPL/LGPL software across application, build, tests,
CI, development tools and distribution. [Credits](Credits.md) covers the approved
project acknowledgements; [Licensing](Licensing.md) records component terms and
the separate model-weight licensing questions.

## Findings and removals

- A project-local legacy upstream checkout contained a bundled FFmpeg executable
  and manuals. That directory was deleted without invoking the executable.
- An old project-local Python conversion environment contained the LGPLv3
  `autocommand` 2.2.2 package vendored inside setuptools. The environment was
  deleted. A NumPy metadata notice mentioning GNU runtime terms was also found;
  that notice alone does not make the BSD-licensed NumPy project GPL. Future
  binary distributions still need transitive-component review.
- Four generated source snapshots retained the retired external-encoder test
  helpers. Those rebuildable source copies were removed.
- Active test fixtures now use Apple's AVFoundation, CoreGraphics, CoreVideo and
  CoreMedia. H.264 encoding, PCM audio, rotation, variable timestamps and composition
  are native. Original MPEG-TS framing supplies transport/HLS fixtures around
  Apple-encoded samples. No external media executable is called.
- CI no longer installs an external media encoder. Benchmark regeneration uses
  `tools/make-synthetic-video.sh`. Historical benchmark hashes remain historical;
  the new original drawing pattern is not claimed to reproduce those bytes.

The audit found prohibited development artifacts and dependency instructions,
not evidence that WaifuFinder's MIT application source had become GPL-licensed.
The cleanup follows this repository's stricter dependency policy.

## Checks

Run `python3 tools/audit-licenses.py` to inspect project-local executable/library
names, component license headers, Python package license declarations, SPDX source
headers and active media-tool calls. It uses the standard library only and does
not execute inspected software. CI runs this guard before the regression suite.

Cleanup validation: 268 regression checks passed, 0 failed, with 8 opt-in checks
skipped. The dependency guard and synthetic website sequence checks also passed.

The guard supplements the manual inventory. It does not approve arbitrary new
packages, hidden binary contents, every optional model download, or externally
installed software. Direct and transitive licenses must still be checked before
adding or installing dependencies. Historical references to GPL in a permissive
license's compatibility discussion are not automatically GPL components.

The owner requested a single clean root commit, removal of refs retaining old
history, expiration of all reflogs, and pruning of unreachable local Git objects.
Local garbage collection cannot erase GitHub caches, other clones or forks; the
owner plans to recreate the hosted repository separately.
