#!/usr/bin/env python3
"""Offline release tooling: publish the real F16 MiniCPM GGUF pair, pinned by SHA256.
Only standard-library HTTP is used. Never used by app inference.
"""
import argparse, base64, hashlib, json, os, urllib.request, urllib.error
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--repository', default='alastorid/PhotoArena-MiniCPM-V-4.7-1B-F16')
a = p.parse_args()
package = ROOT/'models/photoarena'
validation = json.loads((ROOT/'docs/photoarena-native-benchmark.json').read_text())
assert validation['status'] == 'passed', 'Run the native smoke benchmark first'
validation['watermarkTrial'] = json.loads((ROOT/'docs/photoarena-watermark-benchmark.json').read_text())
validation['qualityValidated'] = False
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()
pinned={'model.gguf':'71bc942bf5e07b839a7e73e4c9faa2fa27f97993cddd124c63b5dcd66a432ee1',
        'mmproj.gguf':'fb9cc047a3cdad5317fd8433a4aeda2d22f00f02ca4bd93b34ad84165b25a900'}
for name, sha in pinned.items(): assert digest(package/name)==sha, 'Upstream checksum mismatch: '+name
path=Path(os.environ.get('HF_HOME',str(Path.home()/'.cache/huggingface')))/'token'
token=os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN') or (path.read_text().strip() if path.is_file() else None)
if not token: raise SystemExit('No Hugging Face token; model remains available locally')
def request(url,data=None,method=None,headers=None,auth=True,raw=False):
    headers=dict(headers or {})
    if auth: headers['Authorization']='Bearer '+token
    with urllib.request.urlopen(urllib.request.Request(url,data=data,method=method,headers=headers),timeout=120) as r:
        body=r.read()
        return dict(r.headers) if raw else (json.loads(body) if body else None)
def api(endpoint,value=None):
    return request('https://huggingface.co'+endpoint,None if value is None else json.dumps(value).encode(),headers={'Content-Type':'application/json'})
account=api('/api/whoami-v2')['name']
owner,name=a.repository.split('/',1)
assert owner==account, 'Repository owner differs from authenticated account'
try: api('/api/models/'+a.repository)
except urllib.error.HTTPError as e:
    if e.code!=404: raise
    api('/api/repos/create',dict(type='model',name=name,private=False))
readme='''---
license: apache-2.0
base_model: openbmb/MiniCPM-V-4.7-1B
tags: [gguf, photoarena, image-text-to-text]
---
# PhotoArena MiniCPM-V 4.7 1B F16

Original OpenBMB MiniCPM-V 4.7 1B weights, redistributed without modification from
openbmb/MiniCPM-V-4.7-1B-GGUF revision 345adbfa02c4a4671e61b7c75502dc9bf24fa93d.
Language and vision projector use F16 weights and 16x visual compression.
Credit OpenBMB, Qwen and SigLIP authors. Model license: Apache-2.0.

This is a native llama.cpp Metal/CPU integration, not an ANE/Core ML conversion.
Pinned native source: 7071ba378afa96dcd69c41e1ec2b4912364dbee4.
PhotoArena sends two image overviews and requests constrained structured JSON.
The included benchmark proves native execution and repeated structured output only.
It does not establish FP16 tensor parity, calibrated authenticity judgments, image
registration, pixel provenance or reconstruction quality. The raw model preferred A
on identical images: see validation.json. The app detects identical inputs and
reconciles both A/B orders, abstaining on disagreements. Do not use the model's
self-reported certainty as a probability or its judgment as proof of authenticity.
No Python, video decoding dependencies or cloud inference are needed at runtime.

Files at the immutable release prefix are installed by WaifuFinder's model manager.
The native worker source and CMake configuration are included for reproducibility;
llama.cpp and all linked third-party notices accompany the app runtime.
'''
license_text=urllib.request.urlopen('https://www.apache.org/licenses/LICENSE-2.0.txt',timeout=30).read()
small={'README.md':readme.encode(),'LICENSE':license_text,'validation.json':json.dumps(validation,indent=2).encode(),
       'main.mm':(ROOT/'Runtime/PhotoArena/main.mm').read_bytes(),
       'CMakeLists.txt':(ROOT/'Runtime/PhotoArena/CMakeLists.txt').read_bytes(),
       'build-native-photoarena.sh':(ROOT/'tools/build-native-photoarena.sh').read_bytes(),
       'THIRD-PARTY-NOTICES.txt':(ROOT/'Runtime/PhotoArena/THIRD-PARTY-NOTICES.txt').read_bytes()}
manifest={'schemaVersion':1,'runtimeRevision':'minicpm-v4.7-f16-16x-llama-7071ba3-v1',
          'sourceRevision':'345adbfa02c4a4671e61b7c75502dc9bf24fa93d',
          'fileSHA256':{**pinned,**{n:hashlib.sha256(b).hexdigest() for n,b in small.items()}}}
small['manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
manifest_sha=hashlib.sha256(small['manifest.json']).hexdigest()
prefix='releases/photoarena-f16-'+manifest_sha[:16]
lines=[dict(key='header',value=dict(summary='Publish native PhotoArena MiniCPM F16 package',description='Real native inference smoke-tested; semantic reliability unvalidated.'))]
for filename,oid in pinned.items():
    path=package/filename;size=path.stat().st_size
    batch=api('/'+a.repository+'.git/info/lfs/objects/batch',dict(operation='upload',transfers=['basic','multipart'],objects=[dict(oid=oid,size=size)],hash_algo='sha256'))
    obj=batch['objects'][0]
    if obj.get('error'):raise SystemExit('LFS upload refused')
    actions=obj.get('actions',{})
    if 'upload' in actions:
        action=actions['upload'];headers=action.get('header',{})
        if 'chunk_size' in headers:
            chunk_size=int(headers['chunk_size']);urls=sorted((int(k),v) for k,v in headers.items() if k.isdigit());parts=[]
            assert len(urls)==(size+chunk_size-1)//chunk_size
            with path.open('rb') as f:
                for number,url in urls:
                    chunk=f.read(chunk_size)
                    response=request(url,data=chunk,method='PUT',headers={'Content-Length':str(len(chunk))},auth=False,raw=True)
                    etag=next(v for k,v in response.items() if k.lower()=='etag')
                    parts.append(dict(partNumber=number,etag=etag));print(filename,'part',number,'of',len(urls),flush=True)
            request(action['href'],json.dumps(dict(oid=oid,parts=parts)).encode(),method='POST',headers={'Content-Type':'application/vnd.git-lfs+json'},auth=False)
        else:
            with path.open('rb') as f:request(action['href'],data=f,method='PUT',headers={**headers,'Content-Length':str(size)},auth=False)
        if 'verify' in actions:
            v=actions['verify'];request(v['href'],json.dumps(dict(oid=oid,size=size)).encode(),method='POST',headers={**v.get('header',{}),'Content-Type':'application/json'})
    lines.append(dict(key='lfsFile',value=dict(path=prefix+'/'+filename,algo='sha256',oid=oid,size=size)))
for name,data in small.items():
    (package/name).write_bytes(data)
    lines.append(dict(key='file',value=dict(path=prefix+'/'+name,content=base64.b64encode(data).decode(),encoding='base64')))
lines.append(dict(key='file',value=dict(path='README.md',content=base64.b64encode(readme.encode()).decode(),encoding='base64')))
body=('\n'.join(json.dumps(x) for x in lines)+'\n').encode()
result=request('https://huggingface.co/api/models/'+a.repository+'/commit/main',body,method='POST',headers={'Content-Type':'application/x-ndjson'})
release=dict(repository=a.repository,revision=result['commitOid'],manifestSHA256=manifest_sha,supportFilesSHA256={},pathPrefix=prefix)
(ROOT/'tools/photoarena-release.json').write_text(json.dumps(release,indent=2)+'\n')
print('Published',a.repository,'revision',result['commitOid'])
