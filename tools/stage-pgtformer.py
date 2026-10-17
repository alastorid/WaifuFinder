#!/usr/bin/env python3
"""Stage the validated attention-fused PGTFormer release without uploading."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = 'PGTFormer-Swift-CoreML'


def digest(path):
    with path.open('rb') as stream:
        value = hashlib.sha256()
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
        return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / '.native-runtime/pgtformer')
    parser.add_argument('--stage', type=Path, default=ROOT / 'models/published')
    parser.add_argument('--validation', type=Path, default=ROOT / 'docs/pgtformer-fusion-benchmark.json')
    args = parser.parse_args()
    source = json.loads((args.source / 'manifest.json').read_text())
    validation = json.loads(args.validation.read_text())
    hashes = source['splitPackageSHA256']
    if (source.get('validated') is not True
            or source.get('optimizations', {}).get('fusedCodeAttention') is not True
            or validation.get('validatedPackageSHA256') != hashes
            or validation.get('productValidation', {}).get('passed') is not True):
        raise SystemExit('Stage the fused package covered by the original-network and product validation record.')
    native = validation['fusedAttentionValidation']
    if native.get('passed') is not True or len(native['measurements']) != 8:
        raise SystemExit('All eight original-network reference windows must pass.')
    target = args.stage / BUNDLE
    if target.resolve() == args.source.resolve():
        raise SystemExit('Source and staging directories must differ.')
    for name, expected in hashes.items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or digest(args.source / path) != expected:
            raise SystemExit(f'Validated package fingerprint mismatch: {name}')
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for name in ['PGTEncoder.mlpackage', 'PGTDecoder.mlpackage']:
        shutil.copytree(args.source / name, target / name)
    shutil.copyfile(args.source / 'LICENSE', target / 'LICENSE')
    shutil.copyfile(args.validation, target / 'validation.json')
    manifest = {key: source[key] for key in ['upstreamRevision', 'weightsSHA256', 'weightsRevision',
        'inputShape', 'outputShape', 'inputRange', 'centerFrame', 'w', 'optimizations',
        'splitPackageSHA256', 'runtime'] if key in source}
    manifest.update(format=1, algorithm='pgtformer', validated=True, fileSHA256=hashes,
                    optimizationVersion='pgtformer-sdpa-gpu-v2')
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (target / 'NOTICE').write_text('''PGTFormer, prepared for WaifuFinder
Copyright (c) 2021 Xidian University and the PGTFormer contributors.

The original pretrained network is converted offline to Core ML. Both stages
allow GPU execution: an FP32 encoder with nine native scaled-dot-product
attention blocks, and an FP16 temporal decoder with native normalization and
center-frame pruning after the final temporal operator. Positional Q/K, V,
attention scaling and trained parameters retain their original meaning.
Original Conv/BatchNorm arithmetic ordering is preserved.

Original network: https://github.com/kepengxu/PGTFormer (revision e78a26c)
Original weights: https://huggingface.co/kepeng/pgtformer-base (revision bc0b7e3)

LICENCE — NON-COMMERCIAL. Redistribution and use are permitted for
non-commercial purposes only. Commercial use requires prior permission from
Xidian University or the contributors. See LICENSE.

Conversion uses PyTorch and coremltools offline. Inference uses native Swift
and Core ML; installation requires no Python or checkpoint conversion.
''')
    (target / 'README.md').write_text('''# PGTFormer — fused GPU Core ML

Self-contained face-restoration packages for
[WaifuFinder](https://github.com/alastorid/WaifuFinder).
Install from Settings → Models → PGTFormer. No Python or conversion is required.

## Optimizations

- Both encoder and decoder allow Apple GPU execution.
- The FP32 encoder uses nine native scaled-dot-product attention blocks.
  Positional embeddings apply to Q/K only; V and attention scaling are preserved.
- The FP16 decoder uses native normalization and prunes neighboring frames only
  after the final temporal operator.
- Original Conv/BatchNorm ordering is retained. Encoder GroupNorm fusion remains off.

The worker compiles the packages locally and caches them by content and macOS
version. The legacy `cpu-codebook-gpu-decoder-v1` package-format identifier is
retained for compatibility; it does not prescribe current device placement.

## Validation and measurements

`validation.json` records eight original-network reference windows, product
preview/export with audio, and alternating trials on an Apple M3 with 16 GB RAM.
The attention-fused encoder reduced encoder time by 13.2%, total inference time
by 6.3%, and peak memory by 5.7% relative to the prior GPU encoder. All eight
output PNGs matched that version byte for byte. Original-network maximum error
was 0.01538, below the unchanged 0.02 limit; mean error was below 0.000150,
against a 0.002 limit. Timings depend on hardware and system load.

## Files

`PGTEncoder.mlpackage` and `PGTDecoder.mlpackage` contain the models.
`manifest.json` pins every model file by SHA-256 and records the conversion.
`validation.json` contains qualification results. `LICENSE` and `NOTICE` contain
upstream terms and attribution. WaifuFinder pins the release and verifies all files.

## Provenance and licence

Original network: [kepengxu/PGTFormer](https://github.com/kepengxu/PGTFormer),
revision `e78a26cd463182f4a5aa96b8681110da8615d44d`.
Original weights: [kepeng/pgtformer-base](https://huggingface.co/kepeng/pgtformer-base),
revision `bc0b7e3ff10cfa0f2f335f5256919de20fe2b29d`.

**Non-commercial.** Upstream is BSD-style with a non-commercial restriction.
Commercial use requires permission from Xidian University or the contributors.
See `LICENSE` and `NOTICE`.
''')
    print(f'Staged {BUNDLE}: {len(hashes)} verified model files')


if __name__ == '__main__':
    main()
