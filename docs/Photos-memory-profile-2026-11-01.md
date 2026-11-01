# Photos memory investigation — 2026-11-01

Read-only inspection of the user's existing WaifuFinder process (PID 97514), without UI interaction or restarting it:

- `vmmap -summary`: physical footprint 1.6 GiB, peak 2.0 GiB. Default malloc zone accounted for about 1.1 GiB allocated.
- `heap -s`: 6,468 UInt8 array allocations occupied 52,920,912 bytes. The tag-summary implementation retained a dense general-tag byte row for each cached photo even though the automatic reducer selected sparse CPU work. Its populated-index arrays occupied additional memory.
- The two largest individual heap allocations were 279,035,904 bytes each. Reverse-reference inspection included native Espresso blobs; conservative paths also included unrelated objects. This does not establish an app leak or a safe way to release these blocks. No private framework cache flushing or model/device/concurrency changes were made.
- `leaks --noContent --groupByType`: 165,568 bytes reported leaked, predominantly SwiftUI context-menu responder cycles. This is far too small to account for the reported increase. This snapshot does not prove absence of every retention problem.

## Changes

- Retain tag-summary values only for populated columns. CPU iteration stays sparse; GPU rows are zeroed/scattered directly into bounded Metal buffers. CPU and GPU admission now account for their different row costs; excess GPU rows use existing transient chunks. Automatic sparse CPU work releases an unused dense Metal cache.
- Full-quality originals for all six preview neighbours still preload. Speculative full-size pixel decoding is admitted only when a photo can fit its share of the seven-photo 128 MiB pixel window. Larger originals remain warm as encoded data, avoiding repeated decode/evict cycles. The currently viewed original retains full resolution.
- Leaving full preview releases its decoded inspection cache. Encoded original caching remains available for returning to preview.
- Reuse the thread-safe, software-rendered Core Image context for original reads, rather than creating a context per image. Rendering options and pixel format are unchanged.

## Validation

Noninteractive regressions cover sparse-cache size, repeat summary reuse, CPU/Metal parity, bounded GPU overflow and slot reuse, oversized preview dimensions, and decoded-cache release. Existing native-original preview tests verify resolution/orientation. No interactive UI tests are used.

The existing app remains running with its original executable. A same-album footprint after launching the rebuilt app has not been measured; these changes must not be described as a verified restoration below 1 GiB.

Results: standard app build passed (220 s); tag-summary suite 29 passed / 0 failed; Frame preview suite 14 passed / 0 failed. The preview suite was rerun in a temporary, automatically cleaned source snapshot after an unrelated concurrent TagFilterView edit interrupted its first compile. `git diff --check` passed.
