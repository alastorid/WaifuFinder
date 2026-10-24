#!/usr/bin/env python3
"""Reproducible complete native inference benchmark; no interactive UI or Python ML.

Reports actual wall time including lossless input decode, fused preprocessing,
encoder, all 256 prompts, binary/source clustering, JSON serialization and file IO.
The worker is held warm. Never labels a different device's data as a measurement.
"""
import argparse,array,json,os,platform,statistics,struct,subprocess,tempfile,time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--worker',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
p.add_argument('--image',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--samples',type=int,default=3)
p.add_argument('--transport',choices=['json','sfm1'],default='sfm1')
p.add_argument('--profiles',default='128:4,64:4,256:4,128:1,128:8,256:8')
a=p.parse_args()
reports=[]
def cpu_seconds(pid):
    value=subprocess.check_output(['ps','-p',str(pid),'-o','time='],text=True).strip()
    total=0.0
    for part in value.split(':'):total=total*60+float(part)
    return total

with tempfile.TemporaryDirectory(prefix='safire-benchmark-') as work:
    for rows,batch in [tuple(map(int,item.split(':'))) for item in a.profiles.split(',')]:
        env=dict(os.environ,WAIFU_SAFIRE_QUERY_ROWS=str(rows),WAIFU_SAFIRE_PROMPT_BATCH=str(batch))
        output=Path(work)/'result.json'
        process=subprocess.Popen([str(a.worker.resolve()),'--model',str(a.model.resolve()),'--serve'],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env)
        try:
            if process.stdout.readline().strip()!='SAF1':raise RuntimeError(process.stderr.read())
            samples=[];cpuSamples=[];warmup=None
            for sample in range(a.samples+1):
                beforeCPU=cpu_seconds(process.pid)
                start=time.monotonic()
                process.stdin.write(json.dumps(dict(input=str(a.image.resolve()),output=str(output),format=a.transport))+'\n');process.stdin.flush()
                if process.stdout.readline().strip()!='OK':raise RuntimeError('worker failed')
                elapsed=time.monotonic()-start
                cpu=cpu_seconds(process.pid)-beforeCPU
                if sample==0:warmup=elapsed
                else:samples.append(elapsed);cpuSamples.append(cpu)
            if a.transport=='json':data=json.loads(output.read_text())
            else:
                raw=output.read_bytes();magic,w,h,flag=struct.unpack_from('<4sIII',raw)
                assert magic==b'SFM1' and len(raw)==16+w*h*(4+2*flag)
                values=array.array('f');values.frombytes(raw[16:16+w*h*4])
                data=dict(probabilities=values,runtimeRevision='safire-native-v1')
            record=dict(queryRows=rows,promptBatch=batch,warmupSeconds=warmup,samples=samples,medianSeconds=statistics.median(samples),
                resultRange=[min(data['probabilities']),max(data['probabilities'])],runtime=data.get('runtimeRevision'),transport=a.transport,workerCPUSamples=cpuSamples,medianWorkerCPUSeconds=statistics.median(cpuSamples),medianWorkerCPUUtilizationPercent=100*statistics.median([c/w for c,w in zip(cpuSamples,samples)]))
            reports.append(record)
            print(f'{rows} query rows / {batch} prompts: {record["medianSeconds"]:.3f}s',flush=True)
        finally:
            process.stdin.close();process.terminate();process.wait(timeout=10)
report=dict(platform=platform.platform(),machine=platform.machine(),fixture=str(a.image),
    runtimeRevision='safire-native-v1',device=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),scope='single gradient fixture, complete warm worker wall time; CPU seconds include all worker threads, measured with ps cumulative CPU around each request; 100% is one CPU core',profiles=reports)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
