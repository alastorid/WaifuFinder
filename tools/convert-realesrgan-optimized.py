#!/usr/bin/env python3
"""Offline original RealESRGAN x2plus conversion and independent reference validation."""
import argparse,copy,hashlib,importlib.util,json,sys,types
from pathlib import Path
import numpy as np,torch
from PIL import Image
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--validate-product',type=Path,help='Validate actual native 2x video outputs against the original checkpoint');p.add_argument('--reference-only',action='store_true');p.add_argument('--no-first-conv-fusion',action='store_true')
a=p.parse_args();torch.set_num_threads(4)
def digest(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
# BasicSR initialization helpers do not affect a strictly loaded pretrained state.
class Registry:
 def register(self):return lambda cls:cls
for name in ['basicsr','basicsr.archs','basicsr.utils','basicsr.utils.registry','basicsr.archs.arch_util']:
 sys.modules[name]=types.ModuleType(name)
sys.modules['basicsr.utils.registry'].ARCH_REGISTRY=Registry()
utils=sys.modules['basicsr.archs.arch_util'];utils.default_init_weights=lambda *args,**kwargs:None
utils.make_layer=lambda block,n,**kw:torch.nn.Sequential(*(block(**kw) for _ in range(n)))
utils.pixel_unshuffle=lambda x,scale:torch.nn.functional.pixel_unshuffle(x,scale)
spec=importlib.util.spec_from_file_location('basicsr.archs.rrdbnet_arch',a.source/'rrdbnet_arch.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
assert digest(a.source/'RealESRGAN_x2plus.pth')=='49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb','Unexpected checkpoint'
assert digest(a.source/'rrdbnet_arch.py')=='6aa8eb3f77e918b90eeff4ebfaf8745abe0c40eb8fcc03abbc1928693d70ab58','Unexpected architecture'
checkpoint=torch.load(a.source/'RealESRGAN_x2plus.pth',map_location='cpu',weights_only=True)
m=mod.RRDBNet(num_in_ch=3,num_out_ch=3,scale=2,num_feat=64,num_block=23,num_grow_ch=32).eval();m.load_state_dict(checkpoint['params_ema'],strict=True);m.requires_grad_(False)
if a.validate_product:
 job=a.validate_product;metadata=json.loads((job/'manifest.json').read_text());assert metadata['scale']==2,'Use native 2x outputs before final ROI resizing'
 results=[]
 with torch.no_grad():
  for item in metadata['frames']:
   index=item['index'];rgb=np.array(Image.open(job/'degraded'/f'{index:06d}.png').convert('RGB'));h,w=rgb.shape[:2]
   def reflected(count):
    if count==1:return np.zeros(522,np.int64)
    z=np.arange(522)%(2*(count-1));return np.where(z<count,z,2*(count-1)-z)
   padded=rgb[reflected(h)[:,None],reflected(w)[None,:]].astype(np.float32)/255
   reference=m(torch.from_numpy(padded).permute(2,0,1).unsqueeze(0))[0,:,:h*2,:w*2].clamp(0,1).permute(1,2,0).numpy()
   actual=np.array(Image.open(job/'reconstructed'/f'{index:06d}.png').convert('RGB')).astype(np.float32)/255
   assert actual.shape==reference.shape
   error=np.abs(actual-reference);result={'frame':index,'maximumAbsoluteError':float(error.max()),'meanAbsoluteError':float(error.mean())};results.append(result);print(result,flush=True)
   assert error.max()<=.01 and error.mean()<=.002,result
 a.output.mkdir(parents=True,exist_ok=True);(a.output/'product-validation.json').write_text(json.dumps({'passed':True,'scope':'Actual native 2x video PNGs versus original FP32 checkpoint with canonical 522 reflection padding','weightsSHA256':digest(a.source/'RealESRGAN_x2plus.pth'),'frames':results},indent=2)+'\n')
 raise SystemExit(0)
fused=copy.deepcopy(m)
if not a.no_first_conv_fusion:
 old=fused.conv_first;new=torch.nn.Conv2d(3,64,6,stride=2,padding=2)
 with torch.no_grad():
  for c in range(3):
   for y in range(2):
    for x in range(2):new.weight[:,c,y::2,x::2].copy_(old.weight[:,4*c+2*y+x])
  new.bias.copy_(old.bias)
 fused.conv_first=new;fused.scale=4 # Original forward skips unshuffle; the fused stride retains x2 output.
fused.requires_grad_(False)
a.output.mkdir(parents=True,exist_ok=True);fixtures=a.output/'fixtures';fixtures.mkdir(exist_ok=True)
images=[('face48',Path('.build/model-conversion/realesrgan-product-proof/scale-2/degraded/000000.png')),('face96',Path('.build/model-conversion/realesrgan-product-proof/scale-1/degraded/000000.png'))]
cases=[];context=[]
def reflect(size,count):
 if count==1:return np.zeros(size,np.int64)
 x=np.arange(size)%(2*(count-1));return np.where(x<count,x,2*(count-1)-x)
with torch.no_grad():
 for name,path in images+[('edge33',None),('texture224',None)]:
  if path is not None and path.exists():rgb=np.array(Image.open(path).convert('RGB'))
  elif path is not None:
   side=48 if name=='face48' else 96;yy,xx=np.indices((side,side));rgb=np.stack([(xx*3+yy)%256,(yy*2+xx)%256,(xx+yy*3)%256],-1).astype(np.uint8)
  else:
   yy,xx=np.indices((19,33) if name=='edge33' else (167,224));rgb=np.stack([(xx*17+yy*3)%256,(yy*31+xx*5)%256,((xx//3+yy//2)%2)*255],axis=-1).astype(np.uint8)
  h,w=rgb.shape[:2];canonical=None
  for size in [522,256 if max(h,w)<=96 else 384]:
   padded=rgb[reflect(size,h)[:,None],reflect(size,w)[None,:]].astype(np.float32)/255
   x=torch.from_numpy(padded).permute(2,0,1).unsqueeze(0).contiguous()
   reference=m(x)[:,:,:h*2,:w*2].clamp(0,1).contiguous()
   if size==522:canonical=reference
   else:
    error=(canonical-reference).abs();context.append({'name':name,'size':size,'maximumAbsoluteError':error.max().item(),'meanAbsoluteError':error.mean().item()})
    # Smaller reflected context must preserve the existing 522-pixel result.
    assert error.max().item() <= .002 and error.mean().item() <= .0002,context[-1]
   operation=fused(x)[:,:,:h*2,:w*2].clamp(0,1)
   error=(reference-operation).abs()
   assert error.max().item() <= 2e-5 and error.mean().item() <= 2e-6,'First-convolution fusion changed reference math'
   label=f'{name}-{size}';x.numpy().tofile(fixtures/(label+'-input.bin'));reference.numpy().tofile(fixtures/(label+'-output.bin'))
   cases.append({'name':label,'size':size,'width':w,'height':h,'fusionMaximumError':error.max().item(),'fusionMeanError':error.mean().item()})
   print(cases[-1],flush=True)
   if size !=522:trace_input=x
(fixtures/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
manifest={'algorithm':'realesrgan-x2plus','weightsSHA256':digest(a.source/'RealESRGAN_x2plus.pth'),'architectureSHA256':digest(a.source/'rrdbnet_arch.py'),'basicsrRevision':json.loads((a.source/'basicsr-revision.json').read_text())['sha'],'blocks':23,'features':64,'growth':32,'scale':2,'firstConvFusion':not a.no_first_conv_fusion,'inputSizes':[256,384,522],'adaptiveContextMinimum':160,'contextValidation':context,'validated':False}
(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if a.reference_only:raise SystemExit(0)
import coremltools as ct
with torch.no_grad():traced=torch.jit.trace(fused,trace_input,check_trace=False);assert torch.equal(traced(trace_input),fused(trace_input))
model=ct.convert(traced,inputs=[ct.TensorType(name='input',shape=ct.EnumeratedShapes(shapes=[[1,3,n,n] for n in manifest['inputSizes']],default=[1,3,522,522]),dtype=np.float16)],outputs=[ct.TensorType(name='restored',dtype=np.float16)],compute_precision=ct.precision.FLOAT16,minimum_deployment_target=ct.target.macOS15,skip_model_load=True)
model.user_defined_metadata['waifu.optimizationVersion']='rrdb-fused-v1'
model.user_defined_metadata['waifu.contextHalo']='160'
model.user_defined_metadata['waifu.weightsSHA256']=manifest['weightsSHA256']
model.save(str(a.output/'RealESRGAN.mlpackage'))
print('Saved faithful fused flexible-shape model; native validation required before installation',flush=True)
