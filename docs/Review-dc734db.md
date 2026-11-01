# Review of dc734db

Reviewed on 2026-11-02 against parent 7161178. This review covers all production
changes in the squashed commit, its tests, and the README/site claims.

## Corrected findings

- **Broken OR-group creation:** TagFilterView replaced the keyboard monitor with
  `onDragUpward`, but no event path invoked that callback. The documented F1
  command had no implementation. Restore key-down handling using F1's virtual
  key code, reject repeats and modified shortcuts, and leave text editing and
  modal states alone. Reuse the existing grouping operation, which validates
  available tags and clears the consumed selection. Remove tag/group drag and
  drop UI. Command-click and rectangle selection remain; F1 can also regroup
  selected members of existing groups. Documentation now describes this path.
- **Navigation buffer applied in the wrong direction:** Expanding the visible
  interval by a row suppressed scrolling even for offscreen selections. The
  intended behavior is to scroll a row earlier in either direction. Inset the
  interval and include the same buffer in the resulting offset. Limit the buffer
  to half the space left around the selected item so small viewports and oversized
  rows do not oscillate. Include inter-row/inter-cell spacing. Noninteractive
  geometry regression checks cover forward/backward early scrolling, stable
  boundaries and a viewport-sized row.

- **Duplicate pointer scrolling and stale navigation intent:** focusFromPointer
  requested a scroll before selectFrame requested it again. It also stopped
  clearing pending keyboard navigation. Restore clearing intent in pointer focus;
  selectFrame already reveals the committed selection, including range endpoints.
  The existing pointer and preview-dismissal regressions caught both failures.

## Other reviewed paths

- Sparse tag summaries: inspected CPU indexing, sparse-to-dense Metal upload,
  zeroing reused rows, bounded overflow, source pruning, vocabulary invalidation,
  threshold sentinels and codec uniqueness checks. Existing tests exercise CPU /
  Metal parity, thresholds, refresh, masks and bounded overflow. No further
  correctness defect established in this commit's changes.
- Photo previews: inspected shared software CIContext use, encoded-original
  preload fallback, overflow-safe pixel admission and cache clearing on close.
  The existing cost limit is an NSCache eviction policy, not a hard process RAM
  guarantee. No new memory benchmark or claim is made by this review.
- Remote reconnect: inspected ping-timeout handling, serverStopped handling,
  retry guards and explicit refusal states. No additional defect established in
  the changed branches; live network behavior was not interactively exercised.
- Pointer selection: selectFrame retains the clicked endpoint as the scroll
  target after selection commits. The corrected buffer applies to this path.
- Rarity button styling changes preserve the existing two-state behavior.

## Verification and history

No interactive UI tests were created or run. The regression suite and standard
background app build are the validation paths. See the completion entry below
for their final results. The initial test compilation was invalidated when the
scrolling requirement was clarified and the source changed; a clean run replaced it.

Tags v0.1.51 through v0.1.61 point into the superseded release chain outside
main's ancestry. Earlier tags remain. The corrected commit replaces dc734db;
publication uses an explicit lease on the original remote main revision.

## Completion

The final noninteractive regression run passed 552 cases with zero failures and
zero known bugs; 15 optional model/install cases were skipped. Both previously
failing pointer-navigation cases now pass. The standard background build
completed successfully at dist/WaifuFinder.app. No interactive UI validation was
performed, so actual F1 key delivery and visual scrolling remain unverified by
hands-on UI testing, in accordance with repository policy.
