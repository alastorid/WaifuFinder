#!/usr/bin/env python3
"""TruFor native/original-reference comparison; never generates model outputs."""
import argparse,json,struct,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--packages',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--native',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.packages))
import torch
torch.set_num_threads(4)
def read(p):
    raw=p.read_bytes();size=struct.unpack_from('<Q',raw)[0];header=json.loads(raw[8:8+size]);base=8+size
    return {key:torch.frombuffer(bytearray(raw[base+meta['data_offsets'][0]:base+meta['data_offsets'][1]]),dtype=torch.float32).reshape(meta['shape']) for key,meta in header.items() if key!='__metadata__'}
rows=[]
for path in sorted(a.reference.glob('*.safetensors')):
    if path.stem=='input':continue
    original=read(path);native=read(a.native/path.name)
    for name,x in original.items():
        y=native[name];assert x.shape==y.shape,(name,x.shape,y.shape)
        error=(x-y).abs();maximum=float(error.max());relative=float(error.square().mean().sqrt())/max(1e-8,float(x.square().mean().sqrt()))
        passed=bool(torch.isfinite(y).all()) and maximum<=.0005+float(x.abs().max())*.001 and relative<=.0005
        if name=='score':passed=maximum<=1e-5
        row=dict(tensor=name,shape=list(x.shape),maximumAbsoluteError=maximum,relativeRMS=relative,passed=passed)
        if name=='anomaly':
            xm=x>=.5;ym=y>=.5;union=(xm|ym).sum();iou=float((xm&ym).sum()/union) if union else 1.0
            row['maskIoU']=iou;row['passed']=passed and iou>=.999
        rows.append(row);print(name,'PASS' if row['passed'] else 'FAIL',maximum,relative)
report=dict(status='passed' if all(r['passed'] for r in rows) else 'failed',runtimeRevision='trufor-native-v1',preprocessingVersion='trufor-native-rgb-div256-v1',sourceSHA256='ac1d90e329a72e0d66e8665e123a19e94bfae3209c3ef8a4f9ca3b91578c7844',scoreParityPassed=all(r['passed'] for r in rows if r['tensor']=='score'),spatialParityPassed=all(r['passed'] for r in rows if r['tensor'] in ['anomaly','reliability','noiseprint']),tolerances=dict(relativeRMS=.0005,maximumFloor=.0005,maximumRangeFraction=.001,scoreAbsolute=1e-5,maskIoU=.999),scope='native-resolution odd geometry; official forward math, all four fusion stages, score and genuine maps',results=rows)
a.output.write_text(json.dumps(report,indent=2)+'\n')
sys.exit(0 if report['status']=='passed' else 1)
