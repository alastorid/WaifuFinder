#!/usr/bin/env python3
"""Noninteractive native TruFor RAM/parity regression. Apple PNG fixtures; stdlib only."""
import argparse,array,hashlib,json,math,os,platform,re,struct,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',type=Path,required=True)
p.add_argument('--worker',type=Path,default=ROOT/'.native-runtime/waifu-trufor')
p.add_argument('--baseline',type=Path)
p.add_argument('--completed-full-score',type=Path,help='Reuse the already completed standalone score run and its sibling .log')
p.add_argument('--reference',type=Path,default=ROOT/'.native-runtime/trufor-reference-v1')
p.add_argument('--output',type=Path,default=ROOT/'docs/TruFor-memory-validation.json')
a=p.parse_args()
work=ROOT/'.native-runtime/trufor-memory-validation';work.mkdir(parents=True,exist_ok=True)
env=dict(os.environ)
for key in list(env):
    if key.startswith('WAIFU_TRUFOR_'):del env[key]
env.update(WAIFU_TRUFOR_PROFILE='1',WAIFU_TRUFOR_PRECISION='fp32')
subprocess.run(['xcrun','swift',str(ROOT/'tools/trufor-memory-fixtures.swift'),str(work/'fixtures')],check=True)
def tensors(path):
    raw=path.read_bytes();n=struct.unpack_from('<Q',raw)[0];head=json.loads(raw[8:8+n]);base=8+n
    result={}
    for key,meta in head.items():
        if key=='__metadata__':continue
        assert meta['dtype']=='F32'
        values=array.array('f');lo,hi=meta['data_offsets'];values.frombytes(raw[base+lo:base+hi])
        result[key]=(meta['shape'],values)
    return result
def compare(name,x,y,score=False,mask=False):
    assert len(x)==len(y)
    maximum=max(abs(i-j) for i,j in zip(x,y))
    relative=math.sqrt(math.fsum((i-j)**2 for i,j in zip(x,y))/max(1e-30,math.fsum(i*i for i in x)))
    passed=all(math.isfinite(v) for v in y) and maximum<=(1e-5 if score else .0005+max(abs(v) for v in x)*.001)
    passed=passed and (score or relative<=.0005)
    row=dict(tensor=name,maximumAbsoluteError=maximum,relativeRMS=relative,passed=passed)
    if mask:
        union=sum(i>=.5 or j>=.5 for i,j in zip(x,y));intersection=sum(i>=.5 and j>=.5 for i,j in zip(x,y))
        row['maskIoU']=intersection/union if union else 1.;row['passed']=passed and row['maskIoU']>=.999
    return row
parity=[]
# Exercise the same production code while exporting all original-reference intermediates.
with (work/'activations.log').open('w') as log:
    subprocess.run([str(a.worker),'--model',str(a.model),'--activations',str(a.reference/'input.safetensors'),
                    '--output',str(work/'activations')],env=env,stdout=log,stderr=log,check=True,timeout=90)
for path in sorted(a.reference.glob('*.safetensors')):
    if path.stem=='input':continue
    expected=tensors(path);actual=tensors(work/'activations'/path.name)
    for name,(shape,values) in expected.items():
        assert shape==actual[name][0]
        parity.append(compare(name,values,actual[name][1],score=name=='score',mask=name=='anomaly'))
print('Official reference:', 'PASS' if all(r['passed'] for r in parity) else 'FAIL',flush=True)
streamed_parity=[]
with (work/'streamed-activations.log').open('w') as log:
    subprocess.run([str(a.worker),'--model',str(a.model),'--activations',str(a.reference/'input.safetensors'),
                    '--output',str(work/'streamed-activations')],env=dict(env,WAIFU_TRUFOR_STREAM_ROWS='1'),stdout=log,stderr=log,check=True,timeout=90)
for path in sorted(a.reference.glob('*.safetensors')):
    if path.stem=='input':continue
    expected=tensors(path);actual=tensors(work/'streamed-activations'/path.name)
    for name,(shape,values) in expected.items():
        assert shape==actual[name][0]
        streamed_parity.append(compare(name,values,actual[name][1],score=name=='score',mask=name=='anomaly'))
print('Streamed official reference:', 'PASS' if all(r['passed'] for r in streamed_parity) else 'FAIL',flush=True)
log_path=work/'serve.log'
cases=[('full-native-score',['full-native'],False),('full-native-evidence',['full-native'],True),('odd-score',['odd'],False),('odd-evidence',['odd'],True),
       ('landscape',['landscape'],True),('portrait',['portrait'],True),
       ('maximum-score',['maximum'],False),('maximum-evidence',['maximum'],True),
       ('wide',['wide'],True),('tall',['tall'],True),
       ('four-maximum',['maximum']*4,False),('oversize',['oversize'],False),('after-rejection',['odd'],False)]
results=[]
completed_score=None
if a.completed_full_score:
    completed_log=a.completed_full_score.with_suffix('.log').read_text()
    records=[]
    for line in completed_log.splitlines():
        try:records.append(json.loads(line))
        except ValueError:pass
    memory=next(r for r in records if r.get('phase')=='request-memory')
    seconds=next(r['ms']/1000 for r in records if r.get('phase')=='predict+1')
    completed_score=dict(case='full-native-score',seconds=seconds,reply='OK',scores=json.loads(a.completed_full_score.read_text())['scores'],
                         passed=memory['peakPhysicalBytes']<2_000_000_000,memory=memory,reusedCompletedStandaloneRun=True)
    cases=[case for case in cases if case[0]!='full-native-score']
with log_path.open('w') as log:
    proc=subprocess.Popen(['/usr/bin/time','-l',str(a.worker),'--model',str(a.model),'--serve'],
                          env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True)
    try:
        assert proc.stdout.readline().strip()=='TRF1'
        for name,inputs,evidence in cases:
            output=work/(name+'.bin' if evidence else name+'.json')
            output.unlink(missing_ok=True)
            request=dict(inputs=[str(work/'fixtures'/(f+'.png')) for f in inputs],output=str(output),mode='evidence' if evidence else 'scores')
            start=time.monotonic();proc.stdin.write(json.dumps(request)+'\n');proc.stdin.flush()
            reply=proc.stdout.readline().strip();seconds=time.monotonic()-start
            rejected=name=='oversize'
            passed=(reply.startswith('ERROR ') and '24 megapixel' in reply and not output.exists()) if rejected else reply=='OK'
            values=None
            if passed and not rejected:
                if evidence:
                    with output.open('rb') as h:header=h.read(20)
                    assert header[:4]==b'TFM1'
                    w,h,maps,score=struct.unpack_from('<IIIf',header,4)
                    passed=maps==3 and output.stat().st_size==20+w*h*12 and math.isfinite(score)
                    values=[score]
                else:
                    values=json.loads(output.read_text())['scores'];passed=len(values)==len(inputs) and all(math.isfinite(v) for v in values)
            results.append(dict(case=name,seconds=seconds,reply=reply,scores=values,passed=passed))
            print(name,reply,round(seconds,3),'seconds',flush=True)
            if not passed:raise RuntimeError('Request failed: '+reply)
        proc.stdin.close();assert proc.wait(timeout=60)==0
    finally:
        if proc.poll() is None:proc.kill();proc.wait()
logs=log_path.read_text();snapshots=[]
for line in logs.splitlines():
    try:row=json.loads(line)
    except ValueError:continue
    if row.get('phase')=='request-memory':snapshots.append(row)
assert len(snapshots)==len(results)
for result,snapshot in zip(results,snapshots):
    result['memory']=snapshot;result['passed'] &= snapshot['peakPhysicalBytes']<2_000_000_000
if completed_score:results.insert(0,completed_score)
external={label:int(match.group(1)) for label,pattern in [
    ('maximumResidentBytes',r'(\d+)\s+maximum resident set size'),('peakPhysicalBytes',r'(\d+)\s+peak memory footprint')]
    if (match:=re.search(pattern,logs))}
if external.get('peakPhysicalBytes',0)>=2_000_000_000:raise RuntimeError('External process footprint exceeded 2 GB')
# Strip seams/odd geometry compared against the previous validated FP32 worker.
strip_parity=[]
if a.baseline:
    output=work/'baseline-odd.bin'
    with (work/'baseline.log').open('w') as log:
        subprocess.run([str(a.baseline),'--model',str(a.model),'--image',str(work/'fixtures/odd.png'),
                        '--output',str(output),'--evidence'],env=env,stdout=log,stderr=log,check=True,timeout=90)
    def evidence(path):
        raw=path.read_bytes();w,h,_,score=struct.unpack_from('<IIIf',raw,4)
        values=array.array('f');values.frombytes(raw[20:]);n=w*h
        return dict(score=[score],anomaly=values[:n],reliability=values[n:2*n],noiseprint=values[2*n:])
    expected=evidence(output);actual=evidence(work/'odd-evidence.bin')
    strip_parity=[compare(name,x,actual[name],score=name=='score',mask=name=='anomaly') for name,x in expected.items()]
    streamed_output=work/'streamed-odd.bin'
    with (work/'streamed-odd.log').open('w') as log:
        subprocess.run([str(a.worker),'--model',str(a.model),'--image',str(work/'fixtures/odd.png'),
                        '--output',str(streamed_output),'--evidence'],env=dict(env,WAIFU_TRUFOR_STREAM_ROWS='1'),stdout=log,stderr=log,check=True,timeout=90)
    streamed=evidence(streamed_output)
    strip_parity += [compare('streamed.'+name,x,streamed[name],score=name=='score',mask=name=='anomaly') for name,x in expected.items()]
score=next(r['scores'][0] for r in results if r['case']=='maximum-score')
evidence_score=next(r['scores'][0] for r in results if r['case']=='maximum-evidence')
consistent=abs(score-evidence_score)<=1e-5 and all(abs(v-score)<=1e-5 for r in results if r['case']=='four-maximum' for v in r['scores'])
full_score=next(r['scores'][0] for r in results if r['case']=='full-native-score')
full_evidence=next(r['scores'][0] for r in results if r['case']=='full-native-evidence')
consistent &= abs(full_score-full_evidence)<=1e-5
report=dict(status='passed' if all(r['passed'] for r in parity+streamed_parity+strip_parity+results) and consistent else 'failed',
    measuredDate=time.strftime('%Y-%m-%d'),platform=platform.platform(),chip=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),budgetBytes=2_000_000_000,stopBytes=1_700_000_000,
    maximumPixels=24_000_000,precision='fp32',inputPolicy='Original 24 MP envelope; no new RAM pixel cap; native geometry without resize; FP32 disk scratch for large activations',
    workerSHA256=hashlib.sha256(a.worker.read_bytes()).hexdigest(),physicalPeakSamplingMilliseconds=5,externalProcessMemory=external,scoreEvidenceBatchConsistent=consistent,
    officialReferenceParity=parity,streamedOfficialReferenceParity=streamed_parity,previousWorkerStripParity=strip_parity,requests=results)
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(report['status'],str(a.output),external,flush=True)
raise SystemExit(0 if report['status']=='passed' else 1)
