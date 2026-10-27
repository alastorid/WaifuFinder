#!/usr/bin/env python3
"""Seal offline-converted RealESRGAN only after native original-model validation."""
import argparse,hashlib,json,subprocess,statistics
from pathlib import Path
import coremltools as ct
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--image-validation',type=Path);p.add_argument('--baseline-benchmark',type=Path);p.add_argument('--candidate-benchmark',type=Path);p.add_argument('--experimental',action='store_true',help='Seal shared-fusion diagnostic export after every shape passed');p.add_argument('directory',type=Path);p.add_argument('--validation',type=Path,required=True);p.add_argument('--context-validation',type=Path);p.add_argument('--units',choices=['ane'],default='ane');a=p.parse_args()
r=json.loads(a.validation.read_text());manifest=json.loads((a.directory/('experiment.json' if a.experimental else 'manifest.json')).read_text())
assert r['passed'] and r['units']==a.units
assert {int(x['size']) for x in r['measurements']}==set(manifest['inputSizes']),'Validate every supported size'
assert all(x['maximumAbsoluteError']<=.01 and x['meanAbsoluteError']<=.001 for x in r['measurements']),'Original-model parity failed'
assert r['plan']['preferredOperationCounts'].get('ane',0)>0,'ANE placement was not verified'
assert r['plan']['preferredOperationCounts'].get('gpu',0)==0,'GPU placement is prohibited'
package=a.directory/'RealESRGAN.mlpackage';model=ct.models.MLModel(str(package),compute_units=ct.ComputeUnit.CPU_ONLY,skip_model_load=True)
if not a.experimental and manifest['algorithm']=='realesrgan-x4plus':
 assert manifest['scale']==4 and manifest['firstConvFusion'] is False
 manifest['tilingValidated']=False
 if a.context_validation:
  context=json.loads(a.context_validation.read_text())
  assert context['tileSize']==128 and context['overlap']==0 and context['scale']==4
  assert any(x['halo']==128 and x['passed'] for x in context['comparisons']), 'x4 context gate failed'
  manifest['contextValidation']=context
  manifest['adaptiveContextMinimum']=128
  manifest['tilingValidated']=True
  model.user_defined_metadata['waifu.contextHalo']='128'
 assert all(x.get('scale')==4 for x in r['measurements'])
 assert {x['name'].rsplit('-',1)[0] for x in r['measurements']} >= {'square','landscape','portrait','singleton','odd-edge'}
if a.image_validation:
 quality=json.loads(a.image_validation.read_text())
 assert quality['passed'],'Full-image quality gate failed'
 manifest['imageValidation']=quality
 model.user_defined_metadata['waifu.imageValidated']='true'
assert bool(a.baseline_benchmark)==bool(a.candidate_benchmark),'Supply both benchmark reports'
if a.baseline_benchmark:
 assert a.image_validation,'Performance promotion requires full-image quality validation'
 baseline=json.loads(a.baseline_benchmark.read_text());candidate=json.loads(a.candidate_benchmark.read_text())
 assert baseline['passed'] and candidate['passed'] and baseline['units']==candidate['units']=='ane'
 assert baseline['scale']==candidate['scale'] and set(baseline['geometry'])==set(candidate['geometry'])
 assert statistics.median(candidate['secondsPerImage']) < statistics.median(baseline['secondsPerImage']),'No end-to-end improvement'
 assert candidate['peakPhysicalFootprintBytes'] <= baseline['peakPhysicalFootprintBytes']*1.05,'Process memory regressed by more than 5%'
 manifest['performanceValidation']={'baseline':baseline,'candidate':candidate,'maximumMemoryRegressionFraction':0.05}
 model.user_defined_metadata['waifu.tileTuningValidated']='true'
model.user_defined_metadata['waifu.validated']='true';model.user_defined_metadata['waifu.computeUnits']=a.units
model.user_defined_metadata['waifu.profileHardware']=subprocess.check_output(['sysctl','-n','hw.model'],text=True).strip()
model.save(str(a.directory/'RealESRGANValidated.mlpackage'))
manifest['validated']=True;manifest['nativeValidation']=r;manifest['computeUnits']=a.units
manifest['compiledSourceSHA256']={str(f.relative_to(a.directory/'RealESRGANValidated.mlpackage')):hashlib.file_digest(f.open('rb'),'sha256').hexdigest() for f in (a.directory/'RealESRGANValidated.mlpackage').rglob('*') if f.is_file()}
(a.directory/('experiment.json' if a.experimental else 'manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
