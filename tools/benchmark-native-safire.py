#!/usr/bin/env python3
"""Reproducible complete native inference benchmark; no interactive UI or Python ML.

Reports actual wall time including lossless input decode, fused preprocessing,
encoder, all 256 prompts, binary/source clustering, JSON serialization and file IO.
The worker is held warm. Never labels a different device's data as a measurement.

Optional collections:
  --stage-profile  feeds STAGE stderr records through and reports per-stage medians
  --cache-ops      measures the versioned encode/decode interactive ops against a
                   warm GPU-resident embedding (encode once, then repeated decodes)
"""
import argparse,array,json,os,platform,statistics,struct,subprocess,tempfile,time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--worker',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
p.add_argument('--image',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--samples',type=int,default=3)
p.add_argument('--transport',choices=['json','sfm1'],default='sfm1')
p.add_argument('--profiles',default='256:16')
p.add_argument('--stage-profile',action='store_true')
p.add_argument('--cache-ops',type=int,default=0,help='cached decode samples (points per decode)')
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
        if a.stage_profile: env['WAIFU_SAFIRE_STAGE_PROFILE']='1'
        output=Path(work)/'result.json'
        startLoad=time.monotonic()
        process=subprocess.Popen([str(a.worker.resolve()),'--model',str(a.model.resolve()),'--serve'],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env)
        loadSeconds=None;stageRecords={}
        try:
            line=process.stdout.readline()
            loadSeconds=time.monotonic()-startLoad
            if line.strip()!='SAF1':raise RuntimeError(process.stderr.read())
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
            record=dict(queryRows=rows,promptBatch=batch,loadSeconds=loadSeconds,warmupSeconds=warmup,samples=samples,
                medianSeconds=statistics.median(samples),
                resultRange=[min(data['probabilities']),max(data['probabilities'])],runtime=data.get('runtimeRevision'),transport=a.transport,workerCPUSamples=cpuSamples,medianWorkerCPUSeconds=statistics.median(cpuSamples),medianWorkerCPUUtilizationPercent=100*statistics.median([c/w for c,w in zip(cpuSamples,samples)]))
            if a.cache_ops:
                def request(payload):
                    process.stdin.write(json.dumps(payload)+'\n');process.stdin.flush()
                    reply=process.stdout.readline().strip()
                    if not reply.startswith('OK'):raise RuntimeError(reply)
                    return reply
                request(dict(op='encode',input=str(a.image.resolve()),key='bench'))
                request(dict(op='decode',key='bench',points=[[256,256]],output=str(Path(work)/'decode.safetensors')))
                decodeSamples=[]
                for sample in range(a.cache_ops):
                    start=time.monotonic()
                    request(dict(op='decode',key='bench',
                        points=[[256+sample,256],[768,768],[100,900],[512,512]],output=str(Path(work)/'decode.safetensors')))
                    decodeSamples.append(time.monotonic()-start)
                record['cachedDecode']=dict(points=4,samples=decodeSamples,
                    medianSeconds=statistics.median(decodeSamples))
            if a.stage_profile:
                # The worker exits once its stdin closes; only then does the
                # stderr pipe reach EOF, so stage records are collected here.
                process.stdin.close()
                try: process.wait(timeout=30)
                except subprocess.TimeoutExpired: process.terminate();process.wait(timeout=10)
                stderr=process.stderr.read()
                for lineText in stderr.splitlines():
                    if lineText.startswith('STAGE {'):
                        recordJSON=json.loads(lineText[len('STAGE '):])
                        stageRecords.setdefault(recordJSON['name'],[]).append(recordJSON['seconds'])
                record['stageMedians']={name:statistics.median(values) for name,values in sorted(stageRecords.items())}
            reports.append(record)
            print(f'{rows} query rows / {batch} prompts: {record["medianSeconds"]:.3f}s (load {loadSeconds:.2f}s)'
                  +(f' cached decode {record["cachedDecode"]["medianSeconds"]*1000:.1f}ms' if a.cache_ops else ''),flush=True)
        finally:
            try: process.stdin.close()
            except Exception: pass
            process.terminate();process.wait(timeout=10)
report=dict(platform=platform.platform(),machine=platform.machine(),fixture=str(a.image),
    runtimeRevision='safire-native-v1',device=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),scope='single gradient fixture, complete warm worker wall time; CPU seconds include all worker threads, measured with ps cumulative CPU around each request; 100% is one CPU core',profiles=reports)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
