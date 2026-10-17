#!/usr/bin/env python3
"""Seal offline-converted RealESRGAN only after native original-model validation."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import coremltools as ct
p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--validation',type=Path,required=True);p.add_argument('--units',choices=['all','gpu','ane'],default='all');a=p.parse_args()
r=json.loads(a.validation.read_text());manifest=json.loads((a.directory/'manifest.json').read_text())
assert r['passed'] and r['units']==a.units
assert {int(x['size']) for x in r['measurements']}=={256,384,522},'Validate every supported size'
assert all(x['maximumAbsoluteError']<=.01 and x['meanAbsoluteError']<=.001 for x in r['measurements']),'Original-model parity failed'
if a.units=='ane':assert r['plan']['preferredOperationCounts'].get('ane',0)>0,'ANE placement was not verified'
package=a.directory/'RealESRGAN.mlpackage';model=ct.models.MLModel(str(package),skip_model_load=True)
model.user_defined_metadata['waifu.validated']='true';model.user_defined_metadata['waifu.computeUnits']=a.units
model.user_defined_metadata['waifu.profileHardware']=subprocess.check_output(['sysctl','-n','hw.model'],text=True).strip()
model.save(str(a.directory/'RealESRGANValidated.mlpackage'))
manifest['validated']=True;manifest['nativeValidation']=r;manifest['computeUnits']=a.units
manifest['compiledSourceSHA256']={str(f.relative_to(a.directory/'RealESRGANValidated.mlpackage')):hashlib.file_digest(f.open('rb'),'sha256').hexdigest() for f in (a.directory/'RealESRGANValidated.mlpackage').rglob('*') if f.is_file()}
(a.directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
