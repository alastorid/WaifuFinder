#!/usr/bin/env python3
"""Run genuine native SAFIRE on the Apple-generated known-history fixtures.

No Python ML/image packages. Reports observations, never invents accuracy or a
chronology label. Local-window rates measure known edit bounding windows, not
pixel-perfect edit-mask recall. The baseline is a rendition, not pristine raw.
"""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--worker',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
history=json.loads((a.fixtures/'history.json').read_text())
process=subprocess.Popen([str(a.worker.resolve()),'--model',str(a.model.resolve()),'--serve'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
rows=[]
try:
    if process.stdout.readline().strip()!='SAF1':raise RuntimeError('Native worker failed to start')
    for entry in history['images']:
        image=a.fixtures/entry['name'];result=a.output/(image.stem+'.json')
        digest=hashlib.sha256(image.read_bytes()).hexdigest()
        if digest!=entry['sha256']:raise RuntimeError('Fixture hash changed: '+entry['name'])
        start=time.monotonic()
        process.stdin.write(json.dumps(dict(input=str(image.resolve()),output=str(result.resolve())))+'\n');process.stdin.flush()
        if process.stdout.readline().strip()!='OK':raise RuntimeError('Native inference failed for '+entry['name'])
        elapsed=time.monotonic()-start;data=json.loads(result.read_text());prob=data['probabilities']
        w,h=data['width'],data['height'];flagged=sum(x>=.5 for x in prob)
        row=dict(name=entry['name'],sha256=digest,operation=entry['operation'],wallSeconds=elapsed,
            flaggedFraction=flagged/len(prob),meanProbability=sum(prob)/len(prob),sourcePartitionCount=len(set(data['partitions'])))
        window=None
        if entry['name']=='locally-inpainted.png':window=(.43,.36,.53,.46)
        elif entry['name']=='watermarked.png':window=(350/512,425/512,421/512,467/512)
        if window:
            x0,y0,x1,y1=window;inside=hits=0
            for y in range(int(y0*h),int(y1*h)):
                for x in range(int(x0*w),int(x1*w)):
                    inside+=1;hits+=prob[y*w+x]>=.5
            row.update(knownEditWindowTopLeft=window,windowFlaggedFraction=hits/inside,outsideWindowFlaggedFraction=(flagged-hits)/(len(prob)-inside))
        rows.append(row);print(entry['name']+': flagged '+format(row['flaggedFraction'],'.3%'),flush=True)
finally:
    process.stdin.close();process.terminate();process.wait(timeout=15)
report=dict(runtimeRevision='safire-native-v1',source=history['source'],sourceURL=history['sourceURL'],
    scope='One photographic scene and known edits; diagnostic behavior, not dataset-level accuracy. Baseline flagged area is an overflagging diagnostic, not a pristine-camera false-positive rate. Window rates are not exact-mask recall.',results=rows)
(a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
