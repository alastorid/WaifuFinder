# Archived development source

This archive contains source and text environment metadata only. Test tensors,
images, predictions, local experiment reports and compressed data archives are
excluded. Generate new synthetic inputs with `tools/generate-wd-tagger-inputs.py`.

- `environments.json` records package versions from the old offline conversion
  and MLX prototype environments. It is historical metadata, not an app dependency.
- `adapter-snapshot.patch` preserves five Swift source snapshots as a patch
  against `df6260d`. This is historical source, not a patch to apply to production.
- `historical-helpers` contains an old staged-app builder, WDT2/WDT3 worker probes,
  a rejected narrow QKV kernel and a minimal Metal compiler diagnostic.

Use current build and benchmark tools for new work. None of these source files
embed media or test tensors. Published model weights are restored separately
using `tools/restore-model-bundles.py` and the pinned release descriptors.
