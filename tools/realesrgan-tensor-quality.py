#!/usr/bin/env python3
"""Offline quality metrics on native Swift FP16/W8A8 planar output dumps."""
import argparse
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--reference',type=Path,required=True);p.add_argument('--actual',type=Path,required=True)
p.add_argument('--cases',type=Path,required=True);p.add_argument('--scale',type=int,choices=[2,4],required=True)
p.add_argument('--size',type=int,required=True);p.add_argument('--report',type=Path,required=True)
a=p.parse_args();rows=[]
for case in json.loads(a.cases.read_text()):
 if case['size']!=a.size:continue
 name=case['name'];shape=(3,case['height']*a.scale,case['width']*a.scale)
 x=np.fromfile(a.reference/(name+'.bin'),np.float32).reshape(shape).astype(np.float64)
 y=np.fromfile(a.actual/(name+'.bin'),np.float32).reshape(shape).astype(np.float64)
 assert np.isfinite(x).all() and np.isfinite(y).all()
 d=y-x;mse=np.mean(d*d);scores=[]
 # Same non-overlapping 8x8 luma SSIM convention as the native image checker.
 lx=np.einsum('c,chw->hw',[.2126,.7152,.0722],x);ly=np.einsum('c,chw->hw',[.2126,.7152,.0722],y)
 for yy in range(0,shape[1],8):
  for xx in range(0,shape[2],8):
   u=lx[yy:yy+8,xx:xx+8];v=ly[yy:yy+8,xx:xx+8];mu=u.mean();mv=v.mean()
   scores.append(((2*mu*mv+.0001)*(2*np.mean((u-mu)*(v-mv))+.0009))/((mu*mu+mv*mv+.0001)*(u.var()+v.var()+.0009)))
 edges=[np.abs(np.diff(x,axis=axis)-np.diff(y,axis=axis)).mean() for axis in [1,2] if shape[axis]>1]
 rows.append(dict(name=name,psnrDB=float(-10*np.log10(mse)) if mse else 999.0,ssim8x8Luma=float(np.mean(scores)),maximumAbsoluteError=float(np.abs(d).max()),meanAbsoluteError=float(np.abs(d).mean()),meanRGBBias=d.mean(axis=(1,2)).tolist(),edgeGradientMeanAbsoluteError=float(np.mean(edges)) if edges else 0.0))
assert rows
passed=all(r['maximumAbsoluteError']<=.02 and r['meanAbsoluteError']<=.002 and r['psnrDB']>=45 and r['ssim8x8Luma']>=.995 for r in rows)
a.report.write_text(json.dumps(dict(passed=passed,reference='native FP16 output',measurements=rows),indent=2)+'\n')
print(json.dumps(dict(passed=passed,minimumPSNR=min(x['psnrDB'] for x in rows),minimumSSIM=min(x['ssim8x8Luma'] for x in rows))))
