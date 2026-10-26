#!/usr/bin/env python3
"""Compare native tensors against the offline oracle without NumPy."""
import argparse,json,struct,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--packages',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
p.add_argument('--native',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();sys.path.insert(0,str(a.packages))
import torch
torch.set_num_threads(4)
def load(path):
    data=path.read_bytes();size=struct.unpack('<Q',data[:8])[0];header=json.loads(data[8:8+size]);base=8+size
    result={}
    for key,meta in header.items():
        if key=='__metadata__':continue
        assert meta['dtype']=='F32'
        start,end=meta['data_offsets']
        result[key]=torch.frombuffer(bytearray(data[base+start:base+end]),dtype=torch.float32).reshape(meta['shape'])
    return result
results=[];passed=True
for reference in sorted(a.reference.glob('*.safetensors')):
    if reference.name in {'input.safetensors','automatic_raw.safetensors'}:continue
    left=load(reference);right=load(a.native/reference.name)
    for key,x in left.items():
        y=right[key];assert x.shape==y.shape,(key,x.shape,y.shape)
        error=(x-y).abs();rms=float((error.square().mean()).sqrt());maximum=float(error.max())
        relative=rms/max(1e-8,float(x.square().mean().sqrt()))
        # Predeclared FP32 tolerance: relative RMS <= 2e-4; max error scaled
        # to the reference dynamic range, plus a 2e-4 absolute floor.
        ok=bool(torch.isfinite(y).all()) and relative <= 2e-4 and maximum <= 2e-4+float(x.abs().max())*1e-3
        record=dict(tensor=key,shape=list(x.shape),maximumAbsoluteError=maximum,relativeRMS=relative,passed=ok)
        if key in {'masks','probabilities'}:
            threshold=0.5 if key=='probabilities' else 0
            xm=x>threshold;ym=y>threshold;union=(xm|ym).sum();intersection=(xm&ym).sum()
            iou=float(intersection/union) if union else 1.0
            record['thresholdZeroIoU']=iou;ok=ok and iou>=0.999;record['passed']=ok
        if key=='partitions':
            agreement=float((x==y).float().mean())
            ious=[]
            for label in set(x.flatten().tolist()) | set(y.flatten().tolist()):
                xm=x==label;ym=y==label;union=(xm|ym).sum()
                ious.append(float((xm&ym).sum()/union) if union else 1.0)
            ok=agreement>=0.9999 and min(ious)>=0.999
            record.update(passed=ok,pixelAgreement=agreement,minimumSourceIoU=min(ious))
        results.append(record);passed=passed and ok
        print(key,'PASS' if ok else 'FAIL','max',round(maximum,7),'relative RMS',round(relative,7))
report=dict(status='passed' if passed else 'failed',runtimeRevision='safire-native-v2',
    sourceSHA256='65bdf6880380162d57f77d71b3fd367c3f16aebca316f942a821da16e9225b12',
    fixture=json.loads((a.reference/'reference.json').read_text())['fixture'],tolerances=dict(relativeRMS=2e-4,maxAbsoluteFloor=2e-4,maxDynamicRangeFraction=1e-3,maskIoU=0.999,partitionAgreement=0.9999,sourcePartitionIoU=0.999),
    spatialParityPassed=passed,scope='FP32 encoder, two positive-point decoder masks, and automatic 256-prompt binary/source-partition output; deterministic farthest-first clustering; not dataset quality validation',results=results)
a.output.write_text(json.dumps(report,indent=2)+'\n')
raise SystemExit(0 if passed else 1)
