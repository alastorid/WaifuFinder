# M3 tagging input overlap — 2026-10-26

Local tagging now keeps two batches in flight. Each request renders into its own
6.4 MB RGBA staging buffer while the previous request runs in the native worker.
A separate inference actor copies staging into the worker's shared mapping only
after the prior response completes. GPU inference remains serialized; later
preparation cannot overwrite live input. Cancellation still terminates the worker
and the structured batch pipeline joins pending work before releasing its model.

Photos tagging probes local renditions concurrently, even when iCloud downloads
are enabled. Only missing or undersized local pixels need the cloud queue. This
avoids serializing four local reads behind cloud admission. Actual downloads
retain viewed-photo priority and speculative-preload ordering.

The native Metal kernels, model weights, fixed batch size eight, sparse UInt8
output and padding behavior are unchanged. Two local slots retain at most two
staged requests; remote request windows keep their existing bound of eight.

## Measured adapter result

On this Apple M3, the old adapter measured 15.99 and 15.97 fps; the overlapped
adapter measured 16.23 and 16.17 fps in two reversed-order comparisons. Those are
observed gains of 1.47% and 1.25% on this synthetic in-memory workload. Every
compressed output byte was identical across all rounds and implementations.
Thermal state was normal before and after each measurement.

Whole-device `ioreg` GPU utilization averaged 98.75–98.81% for the control and
98.85–98.98% for the candidate. Individual samples reached 100%. This does not
establish a constant 100% utilization guarantee or isolate this process from other
GPU activity. Photo downloads, disk reads and other apps can still limit observed
utilization. The local-read cloud-queue fix is covered by concurrent-read tests;
these throughput figures do not measure real album or network performance.

[Raw samples and output hashes](wd-tagger-m3-input-overlap.json).

## Reproduction

Build with `./tools/build-wd-tagger-overlap-benchmark.sh`, then run the resulting
tool with `--model MODEL_DIRECTORY --worker NATIVE_WORKER --depth 2 --report FILE`.
The tool generates its own original portrait/landscape pixels with Apple image
frameworks. It needs installed native weights and labels; it never downloads
models or launches UI. Each process warms up, then runs three 64-image rounds.
For the historical control, build with `WD_BENCH_BACKEND_SOURCE` pointing at
`WDTaggerBackend.swift` from commit `9aa449e` and run with `--depth 1`.
