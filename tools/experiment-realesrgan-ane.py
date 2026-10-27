#!/usr/bin/env python3
"""Offline CPU reference/export experiment; never installs unvalidated packages.

Uses the same audited checkpoint loader as the production converter. Native
Swift benchmarks are required before a candidate can be sealed or installed.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path
import numpy as np
import torch
from realesrgan_graph import optimize_rrdb, fuse_x2_input, load_official, FusedUpsamplingNet


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--variant', choices=['x2plus', 'x4plus'], required=True)
    p.add_argument('--fusion', choices=['none', 'scale', 'identity'], default='identity')
    p.add_argument('--sizes', type=int, nargs='+', help='Diagnostic shapes; defaults match installed models')
    p.add_argument('--tile-size',type=int,choices=[384,512],default=384,help='x4 diagnostic tile; 512 retains the 160px halo')
    p.add_argument('--upsampling',choices=['nearest','transpose'],default='nearest')
    p.add_argument('--skip-fixtures', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(4)
    scale = 2 if a.variant == 'x2plus' else 4
    original, expected = load_official(a.source, a.variant)
    candidate = optimize_rrdb(original, a.fusion)
    if scale == 2: fuse_x2_input(candidate)
    if a.upsampling == 'transpose': candidate = FusedUpsamplingNet(candidate).eval().requires_grad_(False)
    sizes = a.sizes or ([256,384,522] if scale == 2 else [128,256,384])
    assert all(n > 0 and (scale == 4 or n % 2 == 0) for n in sizes)
    a.output.mkdir(parents=True, exist_ok=True)
    fixtures = a.output/'fixtures'
    fixtures.mkdir(exist_ok=True)
    cases, algebra = [], []
    with torch.inference_mode():
        # Small odd/rectangular and flat images also exercise zero-padding edges.
        for name,w,h in [('texture',33,19),('gradient',31,47),('flat',1,1)]:
            yy,xx = np.indices((h,w))
            rgb = np.stack([(xx*7+yy*3)%256,(yy*11+xx*3)%256,((xx//4+yy//3)%2)*255],-1).astype(np.uint8)
            check = torch.from_numpy(np.pad(rgb.astype(np.float32)/255,((0,h%2),(0,w%2),(0,0)),mode='reflect')).permute(2,0,1).unsqueeze(0).contiguous()
            error = (original(check)-candidate(check)).abs()
            row = dict(name=name,maximumAbsoluteError=error.max().item(),meanAbsoluteError=error.mean().item())
            assert row['maximumAbsoluteError'] < 0.0002 and row['meanAbsoluteError'] < 0.00002, row
            algebra.append(row)
            if a.skip_fixtures: continue
            for size in sizes:
                def reflect(count):
                    if count == 1: return np.zeros(size,np.int64)
                    z = np.arange(size)%(2*(count-1))
                    return np.where(z<count,z,2*(count-1)-z)
                x = torch.from_numpy(rgb[reflect(h)[:,None],reflect(w)[None,:]].astype(np.float32)/255).permute(2,0,1).unsqueeze(0).contiguous()
                reference = original(x)[:,:,:h*scale,:w*scale].clamp(0,1).contiguous()
                label = f'{name}-{size}'
                x.numpy().tofile(fixtures/(label+'-input.bin'))
                reference.numpy().tofile(fixtures/(label+'-output.bin'))
                cases.append(dict(name=label,size=size,width=w,height=h,scale=scale))
                print(f'Reference {label}',flush=True)
        trace = torch.jit.trace(candidate,torch.zeros(1,3,sizes[0],sizes[0]),check_trace=False)
    (fixtures/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    import coremltools as ct
    default_size = 522 if scale == 2 and 522 in sizes else 384 if scale == 4 and 384 in sizes else sizes[-1]
    model = ct.convert(trace,convert_to='mlprogram',inputs=[ct.TensorType(name='input',shape=ct.EnumeratedShapes(shapes=[[1,3,n,n] for n in sizes],default=[1,3,default_size,default_size]),dtype=np.float16)],outputs=[ct.TensorType(name='restored',dtype=np.float16)],compute_precision=ct.precision.FLOAT16,compute_units=ct.ComputeUnit.CPU_ONLY,minimum_deployment_target=ct.target.macOS15,skip_model_load=True)
    model.user_defined_metadata.update({'waifu.variant':a.variant,'waifu.scale':str(scale),'waifu.optimizationVersion':'rrdb-ane-'+a.fusion+'-v2','waifu.weightsSHA256':expected,'waifu.computeUnits':'ane','waifu.upsampling':a.upsampling,'waifu.x4TileSize':str(a.tile_size)})
    model.author = 'Xintao Wang and collaborators; BasicSR contributors'
    model.license = 'BSD-3-Clause (Real-ESRGAN); Apache-2.0 (BasicSR architecture)'
    model.save(str(a.output/'RealESRGAN.mlpackage'))
    report = dict(variant=a.variant,scale=scale,fusion=a.fusion,upsampling=a.upsampling,tileSize=a.tile_size,weightsSHA256=expected,inputSizes=sizes,algebraValidation=algebra,validated=False,computeUnits='ane',toolchain=dict(torch=torch.__version__,coremltools=ct.__version__,numpy=np.__version__))
    (a.output/'experiment.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__ == '__main__': main()
