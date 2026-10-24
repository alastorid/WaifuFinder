#!/usr/bin/env python3
"""Publish a validated, immutable native SAFIRE package through Hugging Face's API.

Reuses the existing publisher's bundle catalog, model card, and pinned descriptor.
Uses Python's standard library and HTTP, with no version-control executables or
additional Python dependencies. Does not print credentials or signed upload URLs.
A clean checksum-verified download + native inference MUST follow publication.
"""
import argparse,base64,hashlib,importlib.util,json,os,time
from pathlib import Path
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('waifu_existing_publisher',ROOT/'tools/upload-hf-bundles.py')
publisher=importlib.util.module_from_spec(spec);spec.loader.exec_module(publisher)
bundle=next(x for x in publisher.BUNDLES if x['directory']=='SAFIRE-Swift-MLX')
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--package',type=Path,required=True);p.add_argument('--repository',required=True)
a=p.parse_args()
path=Path(os.environ.get('HF_HOME',str(Path.home()/'.cache/huggingface')))/'token'
token=os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN') or (path.read_text().strip() if path.is_file() else None)
if not token:raise SystemExit('No Hugging Face API token; validated package retained locally.')

def request(url,data=None,method=None,headers=None,authenticated=True,response_headers=False):
    headers=dict(headers or {})
    if authenticated:headers['Authorization']='Bearer '+token
    headers['User-Agent']='WaifuFinder-SAFIRE-Release/1'
    position=data.tell() if hasattr(data,'tell') else None
    for attempt in range(4):
        if position is not None:data.seek(position)
        req=urllib.request.Request(url,data=data,method=method,headers=headers)
        try:
            with urllib.request.urlopen(req,timeout=45) as response:
                raw=response.read()
                if response_headers:return dict(response.headers)
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as error:
            if attempt==3 or error.code not in [429,500,502,503,504]:raise
        except urllib.error.URLError:
            if attempt==3:raise
        time.sleep(1+attempt)

def api(endpoint,value=None):
    return request('https://huggingface.co'+endpoint,None if value is None else json.dumps(value).encode(),headers={'Content-Type':'application/json'})

validation=json.loads((a.package/'validation.json').read_text())
assert validation['status']=='passed' and validation['spatialParityPassed']
assert any(x['tensor']=='probabilities' for x in validation['results'])
assert any(x['tensor']=='partitions' for x in validation['results'])
manifest=json.loads((a.package/'manifest.json').read_text())
for name,digest in manifest['fileSHA256'].items():
    assert publisher.digest(a.package/name)==digest, 'Package integrity failure: '+name
account=api('/api/whoami-v2')['name']
owner,name=a.repository.split('/',1)
if owner != account:raise SystemExit('Repository owner differs from the authenticated project account.')
try:api('/api/models/'+a.repository)
except urllib.error.HTTPError as error:
    if error.code!=404:raise
    api('/api/repos/create',dict(type='model',name=name,private=False))
readme=publisher.card(bundle,a.package,a.repository)
# Include all package content in the immutable destination identity.
files={f.name:f.read_bytes() for f in a.package.iterdir() if f.is_file() and f.name!='model.safetensors'}
files['README.md']=readme.encode()
identity=hashlib.sha256()
for name,data in sorted(files.items()):identity.update(name.encode()+b'\0'+hashlib.sha256(data).digest())
identity.update(bytes.fromhex(publisher.digest(a.package/'model.safetensors')))
prefix='releases/safire-native-v1-'+identity.hexdigest()[:16]
weight=a.package/'model.safetensors';oid=publisher.digest(weight);size=weight.stat().st_size
batch=api('/'+a.repository+'.git/info/lfs/objects/batch',dict(operation='upload',transfers=['basic','multipart'],objects=[dict(oid=oid,size=size)],hash_algo='sha256'))
obj=batch['objects'][0]
if obj.get('error'):raise SystemExit('Hugging Face refused the model object upload.')
actions=obj.get('actions',{})
if 'upload' in actions:
    action=actions['upload'];headers=action.get('header',{})
    if 'chunk_size' in headers:
        chunk_size=int(headers['chunk_size'])
        urls=sorted((int(k),v) for k,v in headers.items() if k.isdigit())
        assert chunk_size>0 and len(urls)==(size+chunk_size-1)//chunk_size, 'Invalid multipart instructions'
        parts=[]
        with weight.open('rb') as handle:
            for number,url in urls:
                chunk=handle.read(chunk_size)
                response=request(url,data=chunk,method='PUT',headers={'Content-Length':str(len(chunk))},authenticated=False,response_headers=True)
                etag=next((v for k,v in response.items() if k.lower()=='etag'),None)
                assert etag, 'Missing part integrity tag'
                parts.append(dict(partNumber=number,etag=etag))
                print('Uploaded checkpoint part',number,'of',len(urls),flush=True)
        request(action['href'],json.dumps(dict(oid=oid,parts=parts)).encode(),method='POST',headers={'Content-Type':'application/vnd.git-lfs+json'},authenticated=False)
    else:
        with weight.open('rb') as handle:request(action['href'],data=handle,method='PUT',headers={**headers,'Content-Length':str(size)},authenticated=False)
    if 'verify' in actions:
        verify=actions['verify'];request(verify['href'],json.dumps(dict(oid=oid,size=size)).encode(),method='POST',headers={**verify.get('header',{}),'Content-Type':'application/json'},authenticated=True)
lines=[dict(key='header',value=dict(summary='Publish validated native SAFIRE package '+prefix.rsplit('-',1)[-1],description='Immutable versioned checkpoint; weights CC BY 4.0, native code MIT, upstream code Apache 2.0.'))]
for name,data in sorted(files.items()):
    lines.append(dict(key='file',value=dict(path=prefix+'/'+name,content=base64.b64encode(data).decode(),encoding='base64')))
lines.append(dict(key='file',value=dict(path='README.md',content=base64.b64encode(readme.encode()).decode(),encoding='base64')))
lines.append(dict(key='lfsFile',value=dict(path=prefix+'/model.safetensors',algo='sha256',oid=oid,size=size)))
body=('\n'.join(json.dumps(line) for line in lines)+'\n').encode()
result=request('https://huggingface.co/api/models/'+a.repository+'/commit/main',body,method='POST',headers={'Content-Type':'application/x-ndjson'})
revision=result['commitOid']
publisher.write_release(bundle,a.package,a.repository,revision,readme)
descriptor=ROOT/'tools/safire-release.json'
release=json.loads(descriptor.read_text());release['pathPrefix']=prefix
descriptor.write_text(json.dumps(release,indent=2)+'\n')
print('Published',a.repository,'at',revision,'with immutable package',prefix)
print('Clean download and native inference verification remain required.')
