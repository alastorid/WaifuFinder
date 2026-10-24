#!/usr/bin/env python3
"""Offline CPU oracle from pinned official TruFor code, without training packages.

Uses the separately audited permissive PyTorch CPU environment. No application
runtime dependency, NumPy, OpenCV, timm installation or external media program.
Import-only dependencies are stripped; official forward math remains unchanged.
"""
import argparse,ast,json,math,struct,sys,types
from functools import partial
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--packages',type=Path,required=True);p.add_argument('--upstream',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--height',type=int,default=129);p.add_argument('--width',type=int,default=161);p.add_argument('--input',type=Path)
a=p.parse_args();sys.path.insert(0,str(a.packages))
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_num_threads(4)
a.output.mkdir(parents=True,exist_ok=True)
def save(name,values):
    header={};offset=0;ordered=[]
    for key,tensor in sorted(values.items()):
        tensor=tensor.detach().cpu().to(torch.float32).contiguous().clone();length=tensor.numel()*4
        header[key]=dict(dtype='F32',shape=list(tensor.shape),data_offsets=[offset,offset+length]);offset+=length;ordered.append(tensor)
    encoded=json.dumps(header,separators=(',',':')).encode();encoded+=b' '*((-len(encoded))%8)
    with (a.output/(name+'.safetensors')).open('wb') as handle:
        handle.write(struct.pack('<Q',len(encoded)));handle.write(encoded);handle.flush()
        for tensor in ordered:tensor.untyped_storage()._write_file(handle,True,False,1)
def load(path):
    raw=path.read_bytes();length=struct.unpack_from('<Q',raw)[0];header=json.loads(raw[8:8+length]);base=8+length
    return {key:torch.frombuffer(bytearray(raw[base+meta['data_offsets'][0]:base+meta['data_offsets'][1]]),dtype=torch.float32).reshape(meta['shape']) for key,meta in header.items() if key!='__metadata__'}
class RemoveImports(ast.NodeTransformer):
    def visit_Import(self,node):return None
    def visit_ImportFrom(self,node):return None
base=dict(torch=torch,nn=nn,F=F,math=math,partial=partial,trunc_normal_=nn.init.trunc_normal_,DropPath=lambda *x,**k:nn.Identity(),to_2tuple=lambda x:(x,x))
def module(path,extra={}):
    tree=RemoveImports().visit(ast.parse((a.upstream/path).read_text()));ast.fix_missing_locations(tree)
    values=dict(base,**extra);exec(compile(tree,str(path),'exec'),values);return values
net=module('lib/models/cmx/net_utils.py')
enc=module('lib/models/cmx/encoders/dual_segformer.py',dict(FFM=net['FeatureFusionModule'],FRM=net['FeatureRectifyModule']))
dec=module('lib/models/cmx/decoders/MLPDecoder.py');dn=module('lib/models/DnCNN.py');layers=module('lib/models/cmx/layer_utils.py')
class Config(dict):
    def __getattr__(self,key):return self[key]
extra=Config(BACKBONE='mit_b2',DECODER='MLPDecoder',DECODER_EMBED_DIM=512,PREPRC='imagenet',BN_EPS=.001,BN_MOMENTUM=.1,NP_WEIGHTS=None,MODULES=['NP++','backbone','loc_head','conf_head','det_head'],FIX_MODULES=['NP++','backbone','loc_head'],DETECTION='confpool')
cfg=Config(MODEL=Config(EXTRA=extra,MODS=('RGB','NP++'),PRETRAINED=None),DATASET=Config(NUM_CLASSES=2))
builder=module('lib/models/cmx/builder_np_conf.py',dict(logging=types.SimpleNamespace(info=lambda *x:None),os=__import__('os'),DecoderHead=dec['DecoderHead'],make_net=dn['make_net'],weighted_statistics_pooling=layers['weighted_statistics_pooling']))
builder['create_backbone']=lambda typ,norm:(enc['mit_b2'](norm_fuse=norm),[64,128,320,512])
def initialized(self,pretrained=None):
    # Skip discarded random initialization/pretrained downloads. Preserve the
    # original head BatchNorm epsilon; every trained parameter is loaded below.
    for head in [self.decode_head,self.decode_head_conf]:
        for layer in head.modules():
            if isinstance(layer,nn.BatchNorm2d):layer.eps=.001
modelClass=builder['EncoderDecoder'];modelClass.init_weights=initialized
model=modelClass(cfg).eval();state=load(a.model/'model.safetensors')
missing,unexpected=model.load_state_dict(state,strict=False)
assert not unexpected and all(key.endswith('num_batches_tracked') for key in missing),(missing,unexpected)
for i,layer in enumerate(model.backbone.FFMs):layer.register_forward_hook(lambda mod,inputs,out,i=i:save('stage_'+str(i+1),{'stage_'+str(i+1):out.permute(0,2,3,1)}))
model.dncnn.register_forward_hook(lambda mod,inputs,out:save('noiseprint',{'noiseprint':out.permute(0,2,3,1)}))
torch.manual_seed(20261024);rgb=load(a.input)["input"].permute(0,3,1,2) if a.input else torch.randint(0,256,(1,3,a.height,a.width),dtype=torch.int32).float()/256
save('input',{'input':rgb.permute(0,2,3,1)})
with torch.no_grad():
    out,conf,det,np=model(rgb,save_np=True)
    # Exact official output conversions, preserving the scalar detection head.
    save('localization_logits',{'localization_logits':out.permute(0,2,3,1)})
    save('confidence_logits',{'confidence_logits':conf.permute(0,2,3,1)})
    f1=layers['weighted_statistics_pooling'](conf)
    f2=layers['weighted_statistics_pooling'](out[:,1:2]-out[:,0:1],F.logsigmoid(conf))
    save('detection_statistics',{'detection_statistics':torch.cat([f1,f2],-1)})
    save('score',{'score':torch.sigmoid(det).reshape(-1)})
    save('outputs',{'anomaly':F.softmax(out,dim=1)[:,1],'reliability':torch.sigmoid(conf)[:,0],'noiseprint':np[:,0],'score':torch.sigmoid(det).reshape(-1)})
print('Official reference outputs produced; score',float(torch.sigmoid(det).item()))
