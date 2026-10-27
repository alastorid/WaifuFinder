#!/usr/bin/env python3
"""Stage the independently validated x4 release using the existing Hub pipeline."""
import argparse,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,default=ROOT/'.native-runtime/realesrgan-x4plus-optimized')
p.add_argument('--stage',type=Path,default=ROOT/'models/published')
p.add_argument('--integration',type=Path,default=ROOT/'.build/realesrgan-x4/native-integration.json')
p.add_argument('--remote',type=Path,default=ROOT/'.build/realesrgan-x4/remote-integration.json')
a=p.parse_args()
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
m=json.loads((a.source/'manifest.json').read_text())
if m.get('algorithm')!='realesrgan-x4plus' or not m.get('validated') or m.get('weightsSHA256')!='4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1':raise SystemExit('Unverified or incorrect x4 model')
for name,expected in m['compiledSourceSHA256'].items():
 if digest(a.source/'RealESRGANValidated.mlpackage'/name)!=expected:raise SystemExit('Package differs from native-validated bytes: '+name)
integration=json.loads(a.integration.read_text());remote=json.loads(a.remote.read_text())
if not integration.get('passed') or not remote.get('passed') or {r['variant'] for r in remote['results']}!={'x2plus','x4plus'}:raise SystemExit('Actual local and remote integration gates must pass before staging')
target=a.stage/'RealESRGAN-x4plus-Optimized-CoreML'
if target.exists():raise SystemExit('Stage already exists; choose a new --stage directory to preserve the prior bundle')
target.mkdir(parents=True)
shutil.copytree(a.source/'RealESRGANValidated.mlpackage',target/'RealESRGANValidated.mlpackage')
for src,name in [(a.source/'LICENSE','LICENSE'),(a.source/'BasicSR-LICENSE','BasicSR-LICENSE'),(ROOT/'LICENSE','WaifuFinder-LICENSE'),(ROOT/'Runtime/REALESRGAN-NOTICE.md','NOTICE')]:shutil.copyfile(src,target/name)
report=json.loads((ROOT/'docs/realesrgan-x4-optimization.json').read_text())
report['integration']={'native':integration,'remote':remote,'context':json.loads((ROOT/'.build/realesrgan-x4/integration-reference/validation.json').read_text())}
report['limitations']=['Measurements are from the recorded M3 fixtures; no further profiling was requested.','Remote execution is genuine authenticated loopback, not a second-Mac LAN measurement.','Photo output is bounded at 256 million pixels (at most 16,384px per side); AI output is not forensic evidence.','Process footprint does not include all Core ML/ANE service memory.']
(target/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
for name in ['tools/convert-realesrgan-optimized.py','tools/realesrgan_graph.py','tools/seal-realesrgan-optimized.py','tools/realesrgan-benchmark.swift','tools/compile-coreml-model.swift','Sources/Waifu/Uncover/NativeSpatialUpscaler.swift','Runtime/Sources/WaifuPGTFormer/ComputeProfiling.swift']:
 dest=target/'reproduce'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
m.update(format=1,tilingValidated=True,tileSize=64,contextHalo=160,overlap=0,integrationValidation=report['integration'])
m['fileSHA256']={str(f.relative_to(target)):digest(f) for folder in ['RealESRGANValidated.mlpackage','reproduce'] for f in sorted((target/folder).rglob('*')) if f.is_file()}
(target/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
(target/'README.md').write_text('''# Real-ESRGAN x4plus — FP16 Core ML

**Original authors: Xintao Wang, Liangbin Xie, Chao Dong and Ying Shan.**
Paper: *Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data*.
[Original project](https://github.com/xinntao/Real-ESRGAN). RRDBNet reference
architecture by the [BasicSR contributors](https://github.com/XPixelGroup/BasicSR).
WaifuFinder contributes conversion, optimization investigation, validation and
native integration; the original authors created the network and trained weights.

This is the official **23-block x4plus**, 64 features and growth 32, with native
4× reconstruction. It is not the anime 6B model, RealESRNet or an x2 cascade.
Weights, activation tensors and input/output tensors use FP16. Scalar operator
parameters remain in their Core ML-required types. No INT8/INT4, palettization,
compression or other weight quantization is used.

## Files and native usage

`RealESRGANValidated.mlpackage` is a portable Core ML ML Program for macOS 15+.
Compile it on the destination Mac using `MLModel.compileModel(at:)`, then load
with default specialization and `MLComputeUnits.all`. Compiled `.mlmodelc`
artifacts are machine/OS dependent and are deliberately not distributed.

Input feature `input` and output `restored` use NCHW RGB tensors normalized to
0...1. Input shapes are 1×3×128×128, 1×3×256×256, and 1×3×384×384 (default);
output width and height are each multiplied by four. Clamp output to 0...1
when encoding pixels. The native implementation is included under `reproduce/`.
Color management, orientation, padding, alpha handling and tiling belong to the
consumer; do not substitute x2 pixel-unshuffle preprocessing.

WaifuFinder uses 384px tensors with conservative 160px source context around
64px cores, reflects against the whole source image and crops core pixels into
the final RGB bitmap. Small images up to 224px per side use one 384px tensor.
It bounds output at 256 million pixels (at most 16,384px per side), checks cancellation between predictions,
and reuses its loaded model and input buffer. This is a correctness-validated
configuration, not a claim of optimal large-photo performance.

## Validation and ANE findings

`validation.json` contains the original numerical checks, native device comparison,
ANE hardware trace summary, and actual local/remote integration checks. Maximum
FP32-reference error across the original native tensor checks was 0.006221;
worst mean error was 0.000348 on normalized RGB. Integration evidence includes
native 4× reconstruction with partial tiles and pixel-identical authenticated
remote execution using the actual x4 model. The remote check uses loopback.

The M3 compute plan prefers ANE for all 1,024 executable operations. A noninteractive
Core ML Instruments trace recorded 100 RealESRGAN ANE prediction intervals.
Default `.all` was faster than forced CPU+GPU and comparable to CPU+ANE on the
recorded fixtures. Explicit fastPrediction specialization was rejected as slower.
Recorded warm tensor medians were 46.11 / 187.75 / 430.14 ms for 128 / 256 / 384
inputs. These are earlier tensor measurements, not large-image runtime timings.
Sampled process footprint excludes some Core ML/ANE-service allocations.
No additional profiling was performed during application integration.

## Provenance and reproduction

Official checkpoint: [RealESRGAN_x4plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth).
SHA256: `4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1`.
Real-ESRGAN tag revision: `4c05b20cb30aded73ed3731848a9fbbf70c7c0ac`.
BasicSR revision: `8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a`;
architecture SHA256: `6aa8eb3f77e918b90eeff4ebfaf8745abe0c40eb8fcc03abbc1928693d70ab58`.

Use an audited offline Python environment (PyTorch 2.7.0, NumPy 2.5.4,
Core ML Tools 9.0) with the included scripts or the WaifuFinder repository:

```sh
REALESRGAN_CONVERTER_PYTHON=/absolute/path/to/audited/python \\
  tools/setup-native-realesrgan.sh --variant x4plus --convert-official
```

`manifest.json` pins each model and reproduction file. It also records conversion
options and reference results. Normal native inference and app installation use
Apple frameworks and require no Python, checkpoint download or media executable.

## Licences and responsible interpretation

Real-ESRGAN weights: BSD-3-Clause (`LICENSE`). BasicSR reference architecture:
Apache-2.0 (`BasicSR-LICENSE`). Preserve `NOTICE` and all copyright notices.
WaifuFinder source excerpts are MIT; see `WaifuFinder-LICENSE`.
AI reconstruction synthesizes detail and must not be treated as original
photographic or forensic evidence. TruFor and SAFIRE in WaifuFinder continue to
analyze and display original source pixels.
''')
print(target)
