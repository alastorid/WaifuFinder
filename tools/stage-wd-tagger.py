#!/usr/bin/env python3
"""Stage the validated WD SwinV2 Core ML release without conversion or upload."""
import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = 'WD-SwinV2-Tagger-v3-CoreML'


def digest(path):
    with path.open('rb') as stream:
        hash_value = hashlib.sha256()
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            hash_value.update(chunk)
        return hash_value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / '.build/model-conversion/wd-tagger')
    parser.add_argument('--stage', type=Path, default=ROOT / 'models/published')
    args = parser.parse_args()
    record = json.loads((args.source / 'validation.json').read_text())
    if record['validation']['top15Overlap'] != 15 or record['inputShape'] != [8, 3, 448, 448]:
        raise SystemExit('The source bundle did not pass the required reference validation.')
    if record['optimizationVersion'] != 'swinv2-sdpa-rank4-v1':
        raise SystemExit('The app release requires rank-four fused attention.')
    import coremltools as ct
    from wd_tagger_attention import check_sdpa_spec
    check_sdpa_spec(ct.utils.load_spec(str(args.source / 'WDTagger.mlpackage/Data/com.apple.CoreML/model.mlmodel')))
    target = args.stage / BUNDLE
    target.mkdir(parents=True, exist_ok=True)
    package_target = target / 'WDTagger.mlpackage'
    if package_target.exists():
        shutil.rmtree(package_target)
    shutil.copytree(args.source / 'WDTagger.mlpackage', target / 'WDTagger.mlpackage', dirs_exist_ok=True)
    for name in ['wd-tagger-labels.csv', 'validation.json']:
        shutil.copyfile(args.source / name, target / name)
    # Use the complete canonical Apache-2.0 text, not an abbreviated notice.
    with urllib.request.urlopen('https://www.apache.org/licenses/LICENSE-2.0.txt') as response:
        (target / 'LICENSE').write_bytes(response.read())
    (target / 'NOTICE').write_text('WD SwinV2 Tagger v3, converted for WaifuFinder.\n'
        'Original model: SmilingWolf/wd-swinv2-tagger-v3 (Apache-2.0).\n'
        'https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3\n'
        'Conversion stores fp16 weights and folds the BGR\n'
        'channel permutation into PatchEmbed weights. Runtime input remains RGB.\n'
        'All 24 attention blocks use native SDPA with rank-four queries and masks.\n'
        'The label CSV retains the original output-head order.\n')
    (target / 'README.md').write_text('# WD SwinV2 Tagger v3 — Core ML\n\n'
        'Native fixed-batch tagger for WaifuFinder, converted from '
        '[SmilingWolf/wd-swinv2-tagger-v3](https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3).\n\n'
        '`WDTagger.mlpackage` accepts RGB float32 [8,3,448,448] values in 0–255 '
        'and returns sigmoid probabilities [8,10861]. Use CPU and GPU compute units. '
        'All 24 SwinV2 attention blocks use native SDPA; cosine normalization, '
        'learned temperature, relative bias and shifted-window masks are preserved. '
        'Window and head axes share one head axis so all SDPA queries/masks are rank four. '
        'BGR permutation and normalization are fused; do not apply either in the caller. '
        'Resize longest side to at most 1024, center on a white square and resize '
        'bicubically to 448 before inference. Duplicate the last frame to pad tails.\n\n'
        '`wd-tagger-labels.csv` is in exact output order: 4 rating, 8,106 general, '
        '2,751 character heads. WaifuFinder emits only general tags at threshold 0.35 '
        'and the winning rating. `validation.json` records original PyTorch parity. '
        '`manifest.json` pins every model member and labels by SHA-256. '
        'The app compiles the package locally at installation.\n\n'
        'Apache-2.0. See LICENSE and NOTICE.\n')
    with (target / 'README.md').open('a') as handle:
        handle.write('\n## Fused attention qualification\n\n'
            'The release checks all 24 SDPA input/mask ranks. The initial rank-five '
            'trial encountered an internal MPSGraph memory-encoding assertion. This '
            'release removes that view/broadcast layout. Concurrent mixed-batch '
            'inference and repeated native test sequences pass; no assertion is '
            'suppressed and no CPU fallback is used. The exact original runtime '
            'trigger was not reliably reproducible.\n')
    files = {str(path.relative_to(target)): digest(path)
             for path in sorted((target / 'WDTagger.mlpackage').rglob('*'))
             if path.is_file() and not path.name.startswith('.')}
    files['wd-tagger-labels.csv'] = digest(target / 'wd-tagger-labels.csv')
    manifest = {'format': 1, 'algorithm': 'wd-swinv2-tagger-v3',
                'validated': True, 'fileSHA256': files,
                'inputShape': record['inputShape'], 'outputShape': record['outputShape'],
                'weightsSHA256': record['weightsSHA256'],
                'optimizationVersion': record['optimizationVersion']}
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Staged {target}: {len(files)} pinned files')


if __name__ == '__main__':
    main()
