#!/usr/bin/env python3
"""Offline conversion reconnaissance only; downloads JSON metadata, never model code."""
import hashlib
import json
import platform
import urllib.request
from pathlib import Path

MODEL = 'openbmb/MiniCPM-V-4.7-1B'
def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()

metadata = json.loads(fetch('https://huggingface.co/api/models/' + MODEL))
revision = metadata['sha']
config_bytes = fetch(f'https://huggingface.co/{MODEL}/raw/{revision}/config.json')
report = {
    'model': MODEL, 'revision': revision,
    'configSHA256': hashlib.sha256(config_bytes).hexdigest(),
    'declaredLicense': metadata.get('cardData', {}).get('license'),
    'config': json.loads(config_bytes),
    'inspectionHost': platform.platform(),
    'validation': {'referenceInference': 'not run', 'coreMLConversion': 'not run',
                   'anePlacement': 'not measured', 'fp16Fidelity': 'not measured',
                   'remoteInference': 'not implemented', 'publishedArtifact': None},
}
output = Path(__file__).resolve().parents[1] / 'docs/photoarena-model-inspection.json'
output.write_text(json.dumps(report, indent=2) + '\n')
print(output)
