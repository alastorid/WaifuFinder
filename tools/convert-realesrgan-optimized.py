#!/usr/bin/env python3
"""Offline official RealESRGAN x2plus/x4plus conversion and reference validation."""
import argparse,copy,hashlib,importlib.util,json,sys,types
from pathlib import Path
import numpy as np,torch
from realesrgan_graph import optimize_rrdb, fuse_x2_input, load_official, FusedUpsamplingNet
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--validate-product',type=Path,help='Validate actual native 2x video outputs against the original checkpoint');p.add_argument('--reference-only',action='store_true');p.add_argument('--no-first-conv-fusion',action='store_true')
p.add_argument('--integration-reference',action='store_true',help='One rectangular x4 reference and conservative tile correctness gate, without profiling')
p.add_argument('--tiling-reference',action='store_true',help='Offline x4 context comparison; does not convert or seal a model')
p.add_argument('--variant',choices=['x2plus','x4plus'],default='x2plus')
p.add_argument('--x4-tile-size',type=int,choices=[384,512],default=384,help='512 is enabled only on the hardware recorded during validation')
p.add_argument('--upsampling',choices=['nearest','transpose'],default='nearest',help='Experimental nearest-conv reparameterization')
p.add_argument('--residual-fusion',choices=['none','scale','identity'],default=None,help='Default: validated identity fusion for x2; original residuals for x4')
a=p.parse_args();a.residual_fusion=a.residual_fusion or ('identity' if a.variant=='x2plus' else 'none');torch.set_num_threads(4)
def digest(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
scale=4 if a.variant=='x4plus' else 2
m,expected=load_official(a.source,a.variant)
if a.integration_reference:
 assert scale==4
 a.output.mkdir(parents=True,exist_ok=True)
 w,h=321,65;yy,xx=np.indices((h,w));rgb=np.stack([(xx*7+yy*3)%256,(yy*11+xx*3)%256,((xx//4+yy//3)%2)*255],-1).astype(np.uint8)
 def reflect(indices,count):
  z=indices%(2*(count-1)) if count>1 else np.zeros_like(indices)
  return np.where(z<count,z,2*(count-1)-z)
 def predict(ox,oy,width,height):
  padded=rgb[reflect(np.arange(height)+oy,h)[:,None],reflect(np.arange(width)+ox,w)[None,:]].astype(np.float32)/255
  return m(torch.from_numpy(padded).permute(2,0,1).unsqueeze(0).contiguous())[0].clamp(0,1).numpy()
 with torch.no_grad():
  reference=predict(0,0,512,256)[:,:h*4,:w*4].copy()
  reference.astype(np.float32).tofile(a.output/'reference.bin');rgb.tofile(a.output/'input-rgb.bin')
  restored=np.zeros_like(reference);cache={}
  for y in range(0,h,64):
   for x in range(0,w,64):
    ox=max(0,x-160);oy=max(0,y-160);key=(ox,oy)
    if key not in cache:cache[key]=predict(ox,oy,384,384)
    tw=min(64,w-x);th=min(64,h-y);restored[:,y*4:(y+th)*4,x*4:(x+tw)*4]=cache[key][:,(y-oy)*4:(y-oy+th)*4,(x-ox)*4:(x-ox+tw)*4]
  error=np.abs(restored-reference)
  report={'width':w,'height':h,'scale':4,'tileSize':64,'halo':160,'modelSize':384,'referenceWidth':512,'referenceHeight':256,'maximumAbsoluteError':float(error.max()),'meanAbsoluteError':float(error.mean()),'passed':bool(error.max()<=.002 and error.mean()<=.0002)}
  (a.output/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)
 raise SystemExit(0)
if a.tiling_reference:
 assert scale==4
 a.output.mkdir(parents=True,exist_ok=True)
 w,h=321,65;yy,xx=np.indices((h,w));rgb=np.stack([(xx*7+yy*3)%256,(yy*11+xx*3)%256,((xx//4+yy//3)%2)*255],-1).astype(np.uint8)
 def reflect_values(indices,count):
  if count==1:return np.zeros_like(indices)
  z=indices%(2*(count-1));return np.where(z<count,z,2*(count-1)-z)
 def predict(origin,size):
  padded=rgb[reflect_values(np.arange(size),h)[:,None],reflect_values(np.arange(size)+origin,w)[None,:]].astype(np.float32)/255
  return m(torch.from_numpy(padded).permute(2,0,1).unsqueeze(0).contiguous())[0].clamp(0,1).numpy()
 reports=[]
 with torch.no_grad():
  reference=predict(0,384)[:,:h*4,:w*4].copy()
  reference.astype(np.float32).tofile(a.output/'reference.bin')
  rgb.tofile(a.output/'input-rgb.bin')
  for halo,size in [(0,128),(64,256),(128,384)]:
   total=np.zeros_like(reference);weights=np.zeros((h*4,w*4),np.float32)
   starts=[0,128,256]
   for column,x in enumerate(starts):
    width=min(128,w-x);origin=max(0,x-halo);offset=(x-origin)*4
    tile=predict(origin,size)[:,:h*4,offset:offset+width*4]
    ramp=np.ones(width*4,np.float32)
    if column>0:
     count=(starts[column-1]+128-x)*4
     if count>0:ramp[:count]*=np.arange(1,count+1)/(count+1)
    if column+1<len(starts):
     count=(x+width-starts[column+1])*4
     if count>0:ramp[-count:]*=np.arange(count,0,-1)/(count+1)
    total[:,:,x*4:(x+width)*4]+=tile*ramp[None,None,:]
    weights[:,x*4:(x+width)*4]+=ramp[None,:]
   error=np.abs(total/weights[None,:,:]-reference)
   result={'halo':halo,'size':size,'maximumAbsoluteError':float(error.max()),'meanAbsoluteError':float(error.mean()),'passed':bool(error.max()<=.002 and error.mean()<=.0002)}
   reports.append(result);print(result,flush=True)
 (a.output/'context-validation.json').write_text(json.dumps({'width':w,'height':h,'scale':4,'tileSize':128,'overlap':0,'referenceSize':384,'comparisons':reports},indent=2)+'\n')
 raise SystemExit(0)
if scale==4:
 # Official x4 has a 3-channel first convolution and no pixel-unshuffle.
 # No x2 fusion or architecture rewrite is applied.
 assert m.conv_first.weight.shape==torch.Size([64,3,3,3])
 assert len(m.body)==23 and m.conv_up1.weight.shape==m.conv_up2.weight.shape
 assert not a.validate_product, 'x2 video validation is scale-specific'
 a.output.mkdir(parents=True,exist_ok=True)
 fixtures=a.output/'fixtures';fixtures.mkdir(exist_ok=True)
 sizes=[128,256,384]+([512] if a.x4_tile_size==512 else []);cases=[]
 def reflection(size,count):
  if count==1:return np.zeros(size,np.int64)
  x=np.arange(size)%(2*(count-1));return np.where(x<count,x,2*(count-1)-x)
 with torch.no_grad():
  for name,w,h in [('square',48,48),('landscape',61,29),('portrait',23,57),('singleton',1,1),('odd-edge',33,19)]:
   yy,xx=np.indices((h,w));rgb=np.stack([(xx*7+yy*3)%256,(yy*11+xx*3)%256,((xx//4+yy//3)%2)*255],-1).astype(np.uint8)
   for size in sizes:
    padded=rgb[reflection(size,h)[:,None],reflection(size,w)[None,:]].astype(np.float32)/255
    x=torch.from_numpy(padded).permute(2,0,1).unsqueeze(0).contiguous()
    reference=m(x)
    assert list(reference.shape)==[1,3,size*4,size*4]
    reference=reference[:,:,:h*4,:w*4].clamp(0,1).contiguous()
    label=f'{name}-{size}';x.numpy().tofile(fixtures/(label+'-input.bin'));reference.numpy().tofile(fixtures/(label+'-output.bin'))
    cases.append({'name':label,'size':size,'width':w,'height':h,'scale':4})
    print(cases[-1],flush=True)
 (fixtures/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
 manifest={'algorithm':'realesrgan-x4plus','sourceURL':'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth','realesrganRevision':'4c05b20cb30aded73ed3731848a9fbbf70c7c0ac','weightsSHA256':expected,'architectureSHA256':digest(a.source/'rrdbnet_arch.py'),'basicsrRevision':'8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a','blocks':23,'features':64,'growth':32,'scale':4,'firstConvFusion':False,'inputSizes':sizes,'precision':'float16','referencePrecision':'float32','validated':False,'toolchain':{'torch':torch.__version__,'numpy':np.__version__}}
 (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 if a.reference_only:raise SystemExit(0)
 import coremltools as ct
 candidate=optimize_rrdb(m,a.residual_fusion)
 if a.upsampling=='transpose':candidate=FusedUpsamplingNet(candidate).eval()
 with torch.no_grad():traced=torch.jit.trace(candidate,torch.zeros(1,3,128,128),check_trace=False)
 model=ct.convert(traced,convert_to='mlprogram',inputs=[ct.TensorType(name='input',shape=ct.EnumeratedShapes(shapes=[[1,3,n,n] for n in sizes],default=[1,3,384,384]),dtype=np.float16)],outputs=[ct.TensorType(name='restored',dtype=np.float16)],compute_precision=ct.precision.FLOAT16,compute_units=ct.ComputeUnit.CPU_ONLY,minimum_deployment_target=ct.target.macOS15,skip_model_load=True)
 model.author='Xintao Wang and collaborators; BasicSR contributors'
 model.short_description='Official RealESRGAN x4plus, 23 RRDB blocks. WaifuFinder FP16 Core ML conversion.'
 model.license='BSD-3-Clause (Real-ESRGAN); Apache-2.0 (BasicSR architecture)'
 model.user_defined_metadata.update({'waifu.variant':'x4plus','waifu.scale':'4','waifu.optimizationVersion':'rrdb-x4-v1' if a.residual_fusion=='none' else 'rrdb-ane-'+a.residual_fusion+'-v2','waifu.weightsSHA256':expected})
 model.user_defined_metadata['waifu.residualFusion']=a.residual_fusion
 model.user_defined_metadata['waifu.x4TileSize']=str(a.x4_tile_size)
 model.save(str(a.output/'RealESRGAN.mlpackage'))
 raise SystemExit(0)
from PIL import Image
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
fused=optimize_rrdb(m,a.residual_fusion)
if not a.no_first_conv_fusion:fuse_x2_input(fused)
if a.upsampling=='transpose':fused=FusedUpsamplingNet(fused).eval()
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
   assert error.max().item() <= (2e-5 if a.residual_fusion=='none' else 2e-4) and error.mean().item() <= (2e-6 if a.residual_fusion=='none' else 2e-5),'Graph fusion changed reference math'
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
model=ct.convert(traced,inputs=[ct.TensorType(name='input',shape=ct.EnumeratedShapes(shapes=[[1,3,n,n] for n in manifest['inputSizes']],default=[1,3,522,522]),dtype=np.float16)],outputs=[ct.TensorType(name='restored',dtype=np.float16)],compute_precision=ct.precision.FLOAT16,compute_units=ct.ComputeUnit.CPU_ONLY,minimum_deployment_target=ct.target.macOS15,skip_model_load=True)
model.user_defined_metadata['waifu.optimizationVersion']='rrdb-fused-v1' if a.residual_fusion=='none' else 'rrdb-ane-'+a.residual_fusion+'-v2'
model.user_defined_metadata['waifu.contextHalo']='160'
model.user_defined_metadata['waifu.weightsSHA256']=manifest['weightsSHA256']
model.user_defined_metadata['waifu.residualFusion']=a.residual_fusion
model.save(str(a.output/'RealESRGAN.mlpackage'))
print('Saved faithful fused flexible-shape model; native validation required before installation',flush=True)
