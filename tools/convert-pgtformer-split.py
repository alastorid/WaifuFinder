#!/usr/bin/env python3
"""Offline PGTFormer split: FP32 code prediction, GPU temporal decoder."""
import argparse,importlib.util,inspect,textwrap,types,json,subprocess,hashlib
from pathlib import Path
import torch,numpy as np,coremltools as ct
from safetensors.torch import save_file
p=argparse.ArgumentParser(description=__doc__)
for key in ['upstream','config','weights','output']:p.add_argument('--'+key,type=Path,required=True)
p.add_argument('--validate-existing',action='store_true',help='Recover saved packages without repeating conversion')
p.add_argument('--native-worker',type=Path,default=Path('.native-runtime/waifu-pgtformer'))
p.add_argument('--encoder-only',action='store_true')
p.add_argument('--decoder-precision',choices=['float32','float16'],default='float16',help='Decoder precision experiment; encoder remains exact FP32')
p.add_argument('--decoder-only',action='store_true',help='Reuse the validated encoder; rebuild only the decoder')
p.add_argument('--center-tail',action=argparse.BooleanOptionalAction,default=True,help='Prune non-center frames after the final temporal operator')
p.add_argument('--fuse-code-attention',action=argparse.BooleanOptionalAction,default=True,help='Use native SDPA for code-prediction attention without materializing unused attention weights')
p.add_argument('--fuse-encoder-norm',action=argparse.BooleanOptionalAction,default=False,help='Experimental encoder GroupNorm fusion; requires independent original-network validation')
p.add_argument('--fuse-decoder-norm',action=argparse.BooleanOptionalAction,default=True,help='Lower decoder GroupNorm to a native normalization operation')
p.add_argument('--disable-batchnorm-fusion',action='store_true',default=True,help='Preserve original convolution and BatchNorm arithmetic ordering (required)')
a=p.parse_args()
if a.encoder_only and a.decoder_only:p.error('Select only one stage to rebuild')
torch.set_num_threads(4)
def validate_and_seal():
 subprocess.run([str(a.native_worker.resolve()),'--model',str(a.output.resolve()),'--verify-split',str((a.output/'reference.safetensors').resolve())],check=True)
 metrics=json.loads((a.output/'split-validation.json').read_text())
 if metrics.get('passed') is not True:raise ValueError('Native split validation failed')
 manifest=json.loads((a.output/'manifest.json').read_text());manifest['runtime']='cpu-codebook-gpu-decoder-v1';manifest['validated']=True
 manifest.pop('packageSHA256',None)
 manifest['optimizations']=json.loads((a.output/'conversion-options.json').read_text()) if (a.output/'conversion-options.json').exists() else {}
 manifest['splitPackageSHA256']={str(f.relative_to(a.output)):hashlib.file_digest(f.open('rb'),'sha256').hexdigest() for package in ['PGTEncoder.mlpackage','PGTDecoder.mlpackage'] for f in (a.output/package).rglob('*') if f.is_file()}
 if 'fileSHA256' in manifest:manifest['fileSHA256']=manifest['splitPackageSHA256']
 (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if a.validate_existing:
 validate_and_seal();raise SystemExit(0)
manifest=json.loads((a.output/'manifest.json').read_text());existing_options=manifest.get('optimizations',{}).copy();manifest['validated']=False
manifest.pop('runtime',None);manifest.pop('splitPackageSHA256',None)
(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
spec=importlib.util.spec_from_file_location('c',Path(__file__).with_name('convert-pgtformer.py')); c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
m=c.load_reference(a.upstream,a.config,a.weights)
source=textwrap.dedent(inspect.getsource(type(m).forward))
marker='    self.decoder.last_z_shape = z.shape'
assert marker in source
encoder=source.split(marker)[0]+"    return z, enc_feat_dict['32'], enc_feat_dict['64'], enc_feat_dict['128'], enc_feat_dict['256']\n"
# Keep every original neural operator. Only the function boundary changes.
decoder="def forward(self,z,e32,e64,e128,e256):\n    w=1.0\n    enc_feat_dict={'32':e32,'64':e64,'128':e128,'256':e256}\n"+marker+source.split(marker)[1]
decoder=decoder.replace('return out, logits, lq_feat.permute(0,2,3,1)','return out[1:2].clamp(0,1)')
if a.center_tail:
 # norm_out is GroupNorm over each frame independently. Every temporal
 # operator has already run, so discard neighbors before the 512px tail.
 decoder=decoder.replace('h = h.view(BF*DF,CF,HF,WF)','h = h.view(BF*DF,CF,HF,WF)[1:2]')
 decoder=decoder.replace('return out[1:2].clamp(0,1)','return out.clamp(0,1)')
for expression in ['str(feat.shape[-1])','str(h.shape[-1])']:
    dimension=expression[len('str('):-1]
    encoder=encoder.replace(expression,f'str(int({dimension}))');decoder=decoder.replace(expression,f'str(int({dimension}))')
class Wrapper(torch.nn.Module):
 def __init__(self,model,method):
  super().__init__();self.network=model;scope=dict(type(model).forward.__globals__)
  exec(compile(method,'<original-pgt-split>','exec'),scope)
  self.operation=types.MethodType(scope['forward'],model)
 def forward(self,*args):return self.operation(*args)
x=torch.from_numpy(np.load(a.output/'reference-input.npy'))
with torch.no_grad():
 reference=m(x,w=1.0)[0][1:2].clamp(0,1)
 if a.fuse_code_attention and not a.decoder_only:
  # Position is added to Q/K only in the original layer; V remains the
  # pre-normalized feature. Keep those projections, masks and head scaling.
  # The original caller discards attention weights, so request the native
  # scaled-dot-product attention route instead of constructing that matrix.
  def attention_without_weights(self,query,key,value,key_padding_mask=None,need_weights=True,attn_mask=None,average_attn_weights=True,is_causal=False):
   return torch.nn.MultiheadAttention.forward(self,query,key,value,key_padding_mask=key_padding_mask,need_weights=False,attn_mask=attn_mask,average_attn_weights=average_attn_weights,is_causal=is_causal)
  for layer in m.ft_layers:
   layer.self_attn.forward=types.MethodType(attention_without_weights,layer.self_attn)
 enc=Wrapper(m,encoder).eval();features=enc(x)
 dec=Wrapper(m,decoder).eval();y=dec(*features)
 tail_error=(y-reference).abs()
 assert torch.equal(y,reference),'Split changed original network math'
 print('Original Torch center-tail maximum error',tail_error.max().item(),flush=True)
 traced_enc=None if a.decoder_only else torch.jit.trace(enc,(x,),check_trace=False)
 traced_dec=None if a.encoder_only else torch.jit.trace(dec,features,check_trace=False)
 if traced_enc is not None:assert all(torch.equal(actual,expected) for actual,expected in zip(traced_enc(x),features))
 if traced_dec is not None:assert torch.equal(traced_dec(*features),y)
print('Original and split graphs identical',flush=True)
# Register exact chunk conversion used by the original quantizer.
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
@register_torch_op
def chunk(context,node):
 x,count,axis=_get_inputs(context,node,expected=3);count,axis=int(count.val),int(axis.val)
 length=int(x.shape[axis]);size=(length+count-1)//count
 context.add(mb.split(x=x,split_sizes=[min(size,length-start) for start in range(0,length,size)],axis=axis,name=node.name),torch_name=node.name)
names=['z','e32','e64','e128','e256']
save_file({'frames':x.contiguous(),'restored':reference.contiguous()},str(a.output/'reference.safetensors'))
pipeline=ct.PassPipeline.DEFAULT
if a.disable_batchnorm_fusion:
 pipeline.remove_passes({'common::fuse_conv_batchnorm','common::fuse_conv_scale'})
from coremltools.converters.mil.frontend.torch.ops import group_norm as original_group_norm
fuse_current_norm=a.fuse_encoder_norm
if a.fuse_encoder_norm or a.fuse_decoder_norm:
 @register_torch_op(override=True)
 def group_norm(context,node):
  if not fuse_current_norm:return original_group_norm(context,node)
  x,groups,weight,bias,epsilon,_=_get_inputs(context,node,expected=6)
  grouped=mb.reshape(x=x,shape=[int(x.shape[0]),int(groups.val),-1])
  normalized=mb.layer_norm(x=grouped,axes=[-1],epsilon=float(epsilon.val))
  restored=mb.reshape(x=normalized,shape=x.shape)
  # Preserve learned per-channel affine after each independent group reduction.
  affine=mb.batch_norm(x=restored,mean=np.zeros(int(x.shape[1]),np.float32),variance=np.ones(int(x.shape[1]),np.float32),gamma=weight,beta=bias,epsilon=0.0,name=node.name)
  context.add(affine,node.name)
encoder_model=None
if not a.decoder_only:
 encoder_model=ct.convert(traced_enc,inputs=[ct.TensorType(name='frames',shape=x.shape)],outputs=[ct.TensorType(name=n) for n in names],minimum_deployment_target=ct.target.macOS15,compute_precision=ct.precision.FLOAT32,skip_model_load=True,pass_pipeline=pipeline)
 if a.fuse_code_attention:
  def count_attention(block):
   return sum(int(op.type=='scaled_dot_product_attention')+sum(count_attention(child) for child in op.blocks) for op in block.operations)
  count=sum(count_attention(block) for function in encoder_model.get_spec().mlProgram.functions.values() for block in function.block_specializations.values())
  assert count==len(m.ft_layers),'Export did not retain every native fused attention block'
 encoder_model.save(str(a.output/'PGTEncoder.mlpackage'))
options=existing_options
if not a.decoder_only:options.update(preservesBatchNormOrdering=True,fusedEncoderNorm=a.fuse_encoder_norm,fusedCodeAttention=a.fuse_code_attention)
if not a.encoder_only:options.update(centerTail=a.center_tail,fusedDecoderNorm=a.fuse_decoder_norm,decoderPrecision=a.decoder_precision)
(a.output/'conversion-options.json').write_text(json.dumps(options)+'\n')
if a.encoder_only:
 # Release Torch tensors before starting the native verification process too.
 del encoder_model,traced_enc,traced_dec,enc,dec,m,features,x,reference,y
 import gc;gc.collect()
 if (a.output/'PGTDecoder.mlpackage').exists():validate_and_seal()
 raise SystemExit(0)
fuse_current_norm=a.fuse_decoder_norm
decoder_model=ct.convert(traced_dec,inputs=[ct.TensorType(name=n,shape=f.shape,dtype=np.float32) for n,f in zip(names,features)],outputs=[ct.TensorType(name='restored',dtype=np.float32)],minimum_deployment_target=ct.target.macOS15,compute_precision=ct.precision.FLOAT32 if a.decoder_precision == 'float32' else ct.precision.FLOAT16,skip_model_load=True)
decoder_model.save(str(a.output/'PGTDecoder.mlpackage'))
# Avoid the Python Core ML bridge: macOS releases NumPy/Torch-backed input
# storage on a Core ML dispatch thread, which can segfault in Torch teardown.
del encoder_model,decoder_model,traced_enc,traced_dec,enc,dec,m,features,x,reference,y
import gc;gc.collect()
validate_and_seal()
