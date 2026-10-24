#!/usr/bin/env python3
"""Offline SAFIRE oracle. Never used by the app or distributed with the worker.

Requires separately audited PyTorch CPU wheel + permissive Python dependencies.
Does not use NumPy, OpenCV, torchvision, training dependencies or media tools.
--upstream must contain the exact official source revision recorded in the package.
"""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import sys
import types
import time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--packages',type=Path,required=True)
parser.add_argument('--upstream',type=Path,required=True)
parser.add_argument('--checkpoint',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--automatic',action='store_true')
parser.add_argument('--input',type=Path)
args=parser.parse_args()
sys.path.insert(0,str(args.packages))
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any,Dict,List,Tuple,Union

torch.set_num_threads(4)
root=args.output;root.mkdir(parents=True,exist_ok=True)

def save(name,values):
    # PyTorch's private storage writer copies bytes directly without a NumPy bridge.
    import struct
    header={}; offset=0; ordered=[]
    for key,tensor in sorted(values.items()):
        tensor=tensor.detach().cpu().to(torch.float32).contiguous().clone()
        length=tensor.numel()*4
        header[key]=dict(dtype='F32',shape=list(tensor.shape),data_offsets=[offset,offset+length])
        offset+=length;ordered.append(tensor)
    encoded=json.dumps(header,separators=(',',':')).encode();encoded+=b' '*((8-len(encoded)%8)%8)
    with (root/(name+'.safetensors')).open('wb') as handle:
        handle.write(struct.pack('<Q',len(encoded)));handle.write(encoded)
        handle.flush()
        for tensor in ordered:
            tensor.untyped_storage()._write_file(handle,True,False,1)

def package(name,path):
    value=types.ModuleType(name);value.__path__=[str(path)];sys.modules[name]=value
package('segment_anything',args.upstream/'segment_anything')
package('segment_anything.modeling',args.upstream/'segment_anything/modeling')
from segment_anything.modeling.image_encoder_with_adaptor import ImageEncoderViTAdaptor
from segment_anything.modeling.mask_decoder import MaskDecoder
# This module uses NumPy only for np.pi. Substitute the identical Python
# constant so the offline oracle needs no numerical-library binary dependency.
import math
prompt_path=args.upstream/'segment_anything/modeling/prompt_encoder.py'
prompt_tree=ast.parse(prompt_path.read_text())
prompt_tree.body=[node for node in prompt_tree.body if not (isinstance(node,ast.Import) and any(alias.name=='numpy' for alias in node.names))]
prompt_module=types.ModuleType('segment_anything.modeling.prompt_encoder')
prompt_module.__package__='segment_anything.modeling'
prompt_module.__dict__['np']=types.SimpleNamespace(pi=math.pi)
sys.modules[prompt_module.__name__]=prompt_module
exec(compile(prompt_tree,str(prompt_path),'exec'),prompt_module.__dict__)
PromptEncoder=prompt_module.PromptEncoder
from segment_anything.modeling.transformer import TwoWayTransformer
source=args.upstream/'networks/safire_model.py'
tree=ast.parse(source.read_text())
# Load only the official class; its training module imports unrelated libraries.
classes=ast.Module(body=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='AdaptorSAM'],type_ignores=[])
exec(compile(classes,str(source),'exec'),globals())
model=AdaptorSAM(
    ImageEncoderViTAdaptor(img_size=1024,patch_size=16,embed_dim=768,depth=12,num_heads=12,
        norm_layer=lambda d:nn.LayerNorm(d,eps=1e-6),use_rel_pos=True,window_size=14,
        global_attn_indexes=[2,5,8,11],out_chans=256),
    MaskDecoder(transformer_dim=256,transformer=TwoWayTransformer(depth=2,embedding_dim=256,mlp_dim=2048,num_heads=8),
                num_multimask_outputs=3,iou_head_depth=3,iou_head_hidden_dim=256),
    PromptEncoder(embed_dim=256,image_embedding_size=(64,64),input_image_size=(1024,1024),mask_in_chans=16)).eval()
checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
model.load_state_dict({k.removeprefix('module.'):v for k,v in checkpoint['model'].items()},strict=True)
# The official FFT helper hardcodes .cuda(); this sole device-placement change
# retains the exact operations, rate, dimensions, normalization and absolute value.
def cpu_highpass(x,rate=0.25,prompt_type='highpass'):
    mask=torch.zeros_like(x);w,h=x.shape[-2:];line=int((w*h*rate)**0.5//2)
    mask[:,:,w//2-line:w//2+line,h//2-line:h//2+line]=1
    spectrum=torch.fft.fftshift(torch.fft.fft2(x,norm='forward'))
    spectrum=spectrum*(1-mask)
    return torch.abs(torch.fft.ifft2(torch.fft.ifftshift(spectrum),norm='forward').real)
model.image_encoder.fft_and_ifft=cpu_highpass
for i,block in enumerate(model.image_encoder.blocks):
    block.register_forward_hook(lambda module,inputs,value,i=i:save('block_'+str(i),{'block_'+str(i):value}))
# A reproducible textured input with sharp local changes exercises spatial layout,
# Fourier filtering, adapter injection, attention and prompt-mask localization.
generator=torch.Generator().manual_seed(20261024)
x=torch.rand((1,3,1024,1024),generator=generator)*255
x[:,:,200:380,600:780]=torch.tensor([220.,40.,160.]).view(1,3,1,1)
x=model.preprocess(x)
if args.input:
    import struct
    data=args.input.read_bytes();length=struct.unpack('<Q',data[:8])[0];header=json.loads(data[8:8+length]);meta=header['input']
    start,end=meta['data_offsets'];x=torch.frombuffer(bytearray(data[8+length+start:8+length+end]),dtype=torch.float32).reshape(meta['shape']).permute(0,3,1,2)
save('input',{'input':x.permute(0,2,3,1)})
with torch.no_grad():
    start=time.monotonic()
    high=cpu_highpass(x);save('highpass',{'highpass':high.permute(0,2,3,1)})
    patch=model.image_encoder.patch_embed(x)+model.image_encoder.pos_embed
    save('patch_embedding',{'patch_embedding':patch})
    tuned=model.image_encoder.adaptor_patch_embed_tune(patch)+model.image_encoder.hfp_patch_embed_tune(model.image_encoder.hfp_overlap_patch_embed(high))
    save('adapter_input',{'adapter_input':tuned})
    embedding=model.image_encoder(x);save('embedding',{'embedding':embedding.permute(0,2,3,1)})
    points=torch.tensor([[[32.,32.]],[[512.,512.]]]);labels=torch.ones((2,1),dtype=torch.int64)
    sparse,dense=model.prompt_encoder(points=(points,labels),boxes=None,masks=None)
    masks,quality=model.mask_decoder(image_embeddings=embedding,image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,dense_prompt_embeddings=dense,multimask_output=False)
    save('decoder',{'masks':masks[:,0], 'quality':quality[:,0]})
    if args.automatic:
        low_masks=[];qualities=[];areas=[];features=[]
        for start_index in range(0,256,4):
            points=torch.tensor([[(float(i%16)*64+32,float(i//16)*64+32)] for i in range(start_index,start_index+4)])
            sparse,dense=model.prompt_encoder(points=(points,torch.ones((4,1),dtype=torch.int64)),boxes=None,masks=None)
            low,quality=model.mask_decoder(image_embeddings=embedding,image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,dense_prompt_embeddings=dense,multimask_output=False)
            full=F.interpolate(low,size=(1024,1024),mode='bilinear',align_corners=False)
            for j in range(4):
                binary=full[j,0]>0;sample=binary[::16,::16]
                if not sample.any():continue
                features.append(embedding[0,:,sample].mean(dim=1))
                low_masks.append(low[j,0]);qualities.append(float(quality[j,0]));areas.append(int(binary.sum()))
        matrix=torch.stack(features)
        # Use the official upstream clustering function, with a documented,
        # deterministic farthest-first initializer shared with the Swift port.
        cluster_path=args.upstream/'safire_kmeans.py'
        cluster_tree=ast.parse(cluster_path.read_text())
        cluster_tree.body=[node for node in cluster_tree.body if not (isinstance(node,ast.Import) and any(alias.name=='numpy' for alias in node.names))]
        cluster_ns={'torch':torch}
        exec(compile(cluster_tree,str(cluster_path),'exec'),cluster_ns)
        def initialize(values,count):
            normalized=F.normalize(values,dim=-1);chosen=[0]
            for _ in range(1,count):
                distance=1-normalized@normalized[chosen].T
                chosen.append(int(distance.min(dim=1).values.argmax()))
            return values[chosen].clone()
        cluster_ns['initialize']=initialize
        def representatives(count):
            labels,_=cluster_ns['kmeans'](matrix,num_clusters=count,distance='cosine')
            chosen=[]
            for label in sorted(set(labels.tolist())):
                members=[i for i in range(len(features)) if int(labels[i])==label]
                chosen.append(max(members,key=lambda i:qualities[i]))
            return chosen
        binary=representatives(2);multi=sorted(representatives(3))
        def probability(i):return F.interpolate(low_masks[i][None,None],size=(1024,1024),mode='bilinear',align_corners=False)[0,0].sigmoid()
        if len(binary)==2:
            foreground=min(binary,key=lambda i:areas[i]);background=next(i for i in binary if i!=foreground)
            probabilities=(probability(foreground)+1-probability(background))/2
        else:probabilities=probability(binary[0])
        partitions=torch.stack([probability(i) for i in multi]).argmax(dim=0)
        save('automatic',{'probabilities':probabilities,'partitions':partitions.float()})
        save('automatic_raw',{'masks':torch.stack(low_masks),'quality':torch.tensor(qualities),'features':matrix})
        (root/'automatic.json').write_text(json.dumps(dict(binaryRepresentatives=binary,multiRepresentatives=multi,
            initialization='deterministic farthest-first',maskCount=len(features)),indent=2)+'\n')
    (root/'reference.json').write_text(json.dumps(dict(torch=torch.__version__,seconds=time.monotonic()-start,
        fixture=str(args.input) if args.input else 'seed-20261024-texture-local-patch',device='cpu',upstream='ec2af6af8c789fb6359493a3e352f89c6fd8107f',
        changes=['FFT helper device placement: zeros_like instead of zeros.cuda','np.pi replaced by the identical math.pi constant'],numpy=False),indent=2)+'\n')
print('Official SAFIRE reference activations saved:',root)
