#!/usr/bin/env python3
"""Offline reference validation of an actual Uncover PGTFormer video job."""
import argparse,importlib.util,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file,save_file
p=argparse.ArgumentParser(description=__doc__)
for name in ['upstream','config','weights','job','report']:p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--frame',type=int,help='Diagnose one source frame after a failed full-video comparison')
p.add_argument('--reference-dir',type=Path,help='Save original-network tensors for repeatable native fusion comparisons')
a=p.parse_args();torch.set_num_threads(4)
spec=importlib.util.spec_from_file_location('converter',Path(__file__).with_name('convert-pgtformer.py')); c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
model=c.load_reference(a.upstream,a.config,a.weights); manifest=json.loads((a.job/'manifest.json').read_text());results=[]
def resize(image):
    src=np.asarray(image.convert('RGB'),dtype=np.float64);h,w=src.shape[:2]
    xx=(np.arange(512)+.5)*w/512-.5;yy=(np.arange(512)+.5)*h/512-.5
    x0=np.floor(xx).astype(int);y0=np.floor(yy).astype(int);fx=(xx-x0)[None,:,None];fy=(yy-y0)[:,None,None]
    top=src[np.clip(y0,0,h-1)[:,None],np.clip(x0,0,w-1)[None,:]]*(1-fx)+src[np.clip(y0,0,h-1)[:,None],np.clip(x0+1,0,w-1)[None,:]]*fx
    bottom=src[np.clip(y0+1,0,h-1)[:,None],np.clip(x0,0,w-1)[None,:]]*(1-fx)+src[np.clip(y0+1,0,h-1)[:,None],np.clip(x0+1,0,w-1)[None,:]]*fx
    return torch.from_numpy(np.floor(top*(1-fy)+bottom*fy+.5).astype(np.float32)/255).permute(2,0,1)
with torch.no_grad():
    n=len(manifest['frames'])
    for i in ([a.frame] if a.frame is not None else range(n)):
        tensors=load_file(str(a.job/'validation-tensors'/f'{i:06d}.safetensors'))
        indices=[max(0,i-1),i,min(n-1,i+1)]
        expected_input=torch.stack([resize(Image.open(a.job/'degraded'/f'{j:06d}.png')) for j in indices])
        input_error=(expected_input-tensors['frames']).abs().max().item()
        if input_error>1e-7:raise ValueError(f'Native ROI preprocessing mismatch {input_error}')
        reference=model(expected_input,w=1.0)[0][1:2].clamp(0,1)
        if a.reference_dir:
            a.reference_dir.mkdir(parents=True,exist_ok=True)
            save_file({'frames':expected_input.contiguous(),'restored':reference.contiguous()},str(a.reference_dir/f'{i:06d}.safetensors'))
        error=(reference-tensors['restored']).abs()
        result={'frame':i,'window':indices,'inputMaximumError':input_error,'maximumAbsoluteError':error.max().item(),'meanAbsoluteError':error.mean().item()}
        results.append(result);print(result,flush=True)
        passed=result['maximumAbsoluteError']<=.02 and result['meanAbsoluteError']<=.002
        a.report.write_text(json.dumps({'model':'pgtformer','passed':passed,'frames':results},indent=2)+'\n')
        if not passed:
            save_file({'frames':expected_input.contiguous(),'restored':reference.contiguous()},str(a.job/'failed-reference.safetensors'))
            raise ValueError('Original PGTFormer video parity failed')
a.report.write_text(json.dumps({'model':'pgtformer','scope':'Actual product video inputs, preprocessing and native graph versus original pretrained network','passed':True,'frames':results},indent=2)+'\n')
