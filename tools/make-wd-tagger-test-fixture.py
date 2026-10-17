#!/usr/bin/env python3
"""Regenerate the offline native parity fixture using original unfused BGR weights."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import timm
from PIL import Image
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--weights', type=Path, default=ROOT / '.build/model-conversion/wd-tagger/model.safetensors')
parser.add_argument('--output', type=Path, default=ROOT / '.build/test-fixtures/wd-tagger')
args = parser.parse_args()
torch.set_num_threads(4)
# Original architecture configured locally, no Hugging Face download.
model = timm.create_model('swinv2_base_window8_256', pretrained=False, num_classes=10861,
                         img_size=448, window_size=14, act_layer='gelu_tanh').eval()
model.load_state_dict(load_file(str(args.weights)), strict=True)
y, x = np.mgrid[:448, :448]
rgb = np.stack([(x * 255 // 447), (y * 255 // 447), np.full_like(x, 37)], axis=2).astype(np.uint8)
directory = args.output
directory.mkdir(parents=True, exist_ok=True)
Image.fromarray(rgb).save(directory / 'rgb-gradient.png')
# Independent daemon contract: normalize and reverse RGB to BGR at runtime.
bgr = rgb[:, :, ::-1].copy().transpose(2, 0, 1).astype(np.float32)
with torch.inference_mode():
    probabilities = torch.sigmoid(model(torch.from_numpy(bgr[None]) * (2 / 255) - 1))[0].numpy()
np.asarray(probabilities, dtype='<f4').tofile(directory / 'original-bgr-probabilities.bin')
(directory / 'reference.json').write_text(json.dumps({
    'source': 'SmilingWolf/wd-swinv2-tagger-v3 original unfused weights; BGR input',
    'weightsSHA256': hashlib.sha256(args.weights.read_bytes()).hexdigest(),
    'top15Indices': np.argsort(probabilities)[::-1][:15].tolist(),
}, indent=2) + '\n')
print('Offline reference fixture written')
