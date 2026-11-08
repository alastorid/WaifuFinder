# PhotoArena regional visual judging

## Implemented runtime

The existing pinned **MiniCPM-V 4.7-1B** worker is reused, rather than introducing a second model. The upstream [model card](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B) declares Apache-2.0. OpenBMB's [deployment guide](https://openbmb.cn/docs/en/guides/minicpm-v/minicpm-v-4.7/deploy) explicitly supports native llama.cpp CPU/Metal. Both were checked on 2026-11-08. Runtime transitive licenses were already recorded in `Runtime/PhotoArena/THIRD-PARTY-NOTICES.txt`; this change adds no dependency.

The pinned inspected config is retained in `photoarena-model-inspection.json`: a 27-layer, 1152-wide visual encoder with patch size 14, 16x merging, and a 24-layer Qwen3.5 text decoder with three recurrent/linear-attention layers per full-attention layer. Canvas multimodal rotary positions, recurrent state, convolution history, and full-attention KV all matter. A generic old MiniCPM conversion is not a compatible substitute. The guide specifically warns that older GGUF conversion can silently get M-RoPE wrong. Existing upstream GGUF hashes and native runtime revision remain unchanged.

## Regional harness

`ArenaRegionalJudge` selects disputed regions first, accepts only aligned sources with at least 99.5% valid ROI coverage, and schedules a linear star tournament within each region. The default cap is eight crop comparisons per action, with an absolute cap of 64. Each comparison runs both image orders through the existing grammar-constrained worker and reconciles category disagreements to `unknown`. No model probability is inferred from verbal certainty or reversal agreement.

Each actual input is a native-reference-geometry crop of at most 448×448 pixels. A region larger than this is inspected at its center; the record exports the actual inspection rectangle. It must not be represented as exhaustive inspection of the larger region. Lower-resolution sources are geometrically resampled by the alignment helper; this adds no recovered original detail. The 40 MP pixel-analysis grid remains independent of model crop coverage.

Cache identities include the ordered encoded crop bytes, reference rectangle, region ID, model revision, and regional contract revision. Only structured scalar judgments and evidence are persisted. Images and activations are not written to scratch storage. Cancellation is checked between bounded jobs and delegates in-flight worker cancellation to the existing backend. Results keep provenance and actual crop geometry and can be converted into ordinary historical Arena votes.

`ArenaRanking` provides separate deterministic technical ranking and regularized pairwise logistic preference fitting. Unknown outcomes create no edges; ties contribute a 0.5 target. Later decisions by the same author for the same pair, region, and category replace previous decisions for fitting while the original vote history remains intact. Scores are relative preferences, never calibrated authenticity probabilities. Disconnected comparison graphs cannot establish a global preference between components. The technical order measures texture, clipping and discrepancy, which are not proof of original detail; authenticity remains unresolved without separate evidence.

## Architecture debate and selected fallback

A proposed all-pairs/full-frame judging approach was rejected: it costs quadratic inference, discards native detail and encourages false forensic certainty. Deterministic localized measurements plus bounded crop comparisons were selected. A proposed sharpness-first authenticity score was rejected because generated detail may be sharp. Source consensus was also rejected as proof because edited duplicates can share a common ancestor. The existing reversal test demonstrates actual positional-bias abstention; new background tests cover cache invalidation, sparse scheduling, invalid coverage, unknown/tie ranking, and reversed candidate identity.

## ANE status and conversion boundary

**No Core ML conversion or ANE operation-placement measurement is implemented or claimed.** The current runtime is Metal/CPU. Prior native measurements and their limitations remain in `PhotoArena.md` and the existing JSON reports; no identical baseline experiment was rerun. End-of-worker footprint samples are not peak whole-app memory, and there is no new CPU/GPU/ANE comparative speed claim.

A defensible future conversion must separately export the image encoder, merge/projector and decoder. Use fixed 448 visual tiles, compare encoder/projector tensors to the pinned reference, then validate canvas positions and recurrent/KV state through separate prefill/decode tests. FP16 weights alone do not ensure FP16 activations. Keep recurrent state precision compatible with the original architecture. Stateful Core ML buffers require correct ownership and reset between independent comparisons. Use Apple's operation-placement profiling before claiming ANE execution, and compare category decisions after the same crop and reverse-order protocol. PairJudge could later reuse encoder embeddings with a compact symmetric comparison head, trained on independently verified labels rather than treating MiniCPM's provisional explanations as ground truth.

## Remaining limitations

The crop center policy is budgeted inspection, not complete AI coverage. The scheduler is a sparse heuristic rather than a calibrated information-gain policy. Model/context reuse and bounded embedding reuse now apply within one forward/reverse action; workers are released between actions. See PhotoArena-optimization.md for the single before/after session and its warm-latency limitation. Reversal consistency is not accuracy. No validated dataset of real AI retouching, object replacement or AI upscaling is introduced by these tests. Unknown provenance is retained as unknown. This runtime cannot certify that a reconstruction is AI-edit-free or that text detected by OCR is a watermark.

## Remote regional inference and Hugging Face setup

Both Arena interfaces capture the same `ArenaInferenceBackend` at the beginning of
an action. A connected server advertising the exact PhotoArena runtime revision is
preferred; otherwise inference runs on this Mac. The UI identifies the captured
backend. Regional judging submits aligned native crops through the existing
authenticated/encrypted `arenaRequest` transport, with the same cancellation,
server resource reservation and structured evidence validation as overview judging.
The client needs no model weights when a compatible remote server is selected.
Remote failures surface without silently moving the job to local inference. Only
local jobs reserve the client's local model; remote jobs reserve it on the server.
Validated regional cache entries are reusable across backends with the same pinned
runtime because their identity includes the actual submitted pixels and geometry.

On the machine performing inference, open Settings → Models → **PhotoArena
MiniCPM-V 4.7** → **Download**. The built-in manager downloads the pinned release
from [Hugging Face](https://huggingface.co/alastorid/PhotoArena-MiniCPM-V-4.7-1B-F16),
verifies the immutable manifest and every file SHA256, loads both model components
with the bundled native worker, then activates the installation. A valid existing
installation is reverified and reused. **Choose Existing…** accepts the same verified
release folder. No separate Python environment or conversion step is required.

Run the real download/setup test without touching the user's model configuration:

```sh
WAIFU_TEST_ARENA_INSTALL=1 \
WAIFU_TEST_ARENA_INSTALL_REPORT="$PWD/docs/photoarena-huggingface-setup-verification.json" \
./run-tests.sh --only PhotoArena
```

This opt-in check downloads approximately 3.2GB into temporary storage, verifies all
files, runs native load validation, checks isolated activation and repeat setup,
and removes the temporary installation. Ordinary tests cover encrypted regional
routing, persistent cache reuse and surfacing inference failures; their server
catalog is deterministic, not an additional model-quality evaluation.
