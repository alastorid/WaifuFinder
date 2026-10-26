# Photos preview loading

Preview retains the largest decoded rendition of the same photo and source. A
thumbnail refresh or delayed local PhotoKit fallback cannot replace it with fewer
pixels. Navigation still changes to the new photo's own pixels. Original-image
provenance stays attached to the image used by SAFIRE and TruFor.

The iCloud switch is observed while preview is open. Enabling it restarts the
current photo read immediately; disabling it cancels that read and the speculative
preload window, then requests local data only. Cloud-capable requests made by the
app share `PhotosCloudQueue`: the viewed photo first, analysis/tagging/export
requests second, and speculative neighbours last. A higher-priority request
cancels the lower-priority PhotoKit operation and requeues its owner, so a paused
album analysis does not fail merely because the user opened preview.

Once the current preview has finished loading, preload follows the preview's
captured arrow-key order: next, previous, then forward through the next seven
items, without wrapping or crossing a navigation row boundary. Only one app-owned
cloud request runs at once. Neighbour downloads run only when no viewed-photo or
background request is waiting, and navigation/closing cancels the old window.
This schedules this app's requests; it does not inspect or control unrelated
system iCloud synchronization.

Preloaded originals are cached as encoded image data with a nine-item, 256 MiB
NSCache budget. The cache includes the PhotoKit identifier, revision and geometry;
edits do not reuse stale bytes. Full-resolution images are decoded when viewed,
not eagerly for all seven neighbours. These are memory caches only; Photos owns
the downloaded originals.

Local rendition reads bypass the cloud queue, including when cloud downloads are
enabled. Only missing or undersized local renditions request a queued upgrade;
local model-sized reads retain the four-request tagging window.

Photo context menus begin with **Open in Photos.app**. TruFor appears before SAFIRE;
each enable toggle, always-visible preset-style mode control and score/status
shares one line. Detailed explanations and preview-quality notes use tooltips. Repository UI requirements live in
`AGENTS.md`. `Tests/PhotosCloudQueueTests.swift` covers preemption, cancellation,
resumption and neighbour order without PhotoKit downloads or interactive UI.
