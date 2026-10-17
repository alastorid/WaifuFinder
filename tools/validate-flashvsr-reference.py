#!/usr/bin/env python3
"""Offline CPU oracle: executes pinned official projection/DiT/decoder definitions.

CUDA block-sparse dispatch is replaced only with mathematically equivalent masked
PyTorch SDPA. No unmasked attention, different decoder, quantization or model substitute.
The production app does not invoke this validation script.
"""
import argparse
import ast
import json
import math
import random
import subprocess
import time
import typing
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange,repeat
from safetensors.torch import load_file,save_file
from tqdm.auto import tqdm

REVISION="cf910c61a60733e610e9c6e8b607f80c3a6c202b"

def definitions(path,environment,names=None):
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and (names is None or n.name in names)]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),"exec"),environment)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--job",type=Path,required=True)
    p.add_argument("--fixture",type=Path,required=True)
    p.add_argument("--decoder-device",choices=["cpu","mps"],default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--compare-saved",action="store_true",help="Compare native tensors to an already generated official oracle fixture")
    p.add_argument("--mask-only",action="store_true",help="Check captured Q/K block selection against the official mask functions without loading the model")
    args=p.parse_args()
    if args.mask_only:
        commit=subprocess.check_output(["git","-C",str(args.reference),"rev-parse","HEAD"],text=True).strip()
        dirty=subprocess.check_output(["git","-C",str(args.reference),"status","--porcelain","--untracked-files=no"],text=True).strip()
        if commit!=REVISION or dirty: raise ValueError("A clean pinned official reference is required")
        environment={"torch":torch,"math":math,"rearrange":rearrange}
        definitions(args.reference/"diffsynth/models/wan_video_dit.py",environment,
                    {"build_local_block_mask_shifted_vec_normal_slide","generate_draft_block_mask"})
        def captured(name): return load_file(str(args.fixture/(name+".safetensors")))[name]
        q,k,mask=captured("q"),captured("k"),captured("mask")
        job=json.loads((args.job/"report.json").read_text())
        h,w=job["paddedHeight"]//128,job["paddedWidth"]//128
        local=environment["build_local_block_mask_shifted_vec_normal_slide"](h,w,11,11)
        topk=int((h*w)**2*2*768*1280/(job["paddedHeight"]*job["paddedWidth"]))-1
        expected=environment["generate_draft_block_mask"](1,12,q.shape[0]//(h*w),q,k,topk,local)[0]
        mismatches=int((mask!=expected).sum())
        report={"referenceRevision":REVISION,"entries":mask.numel(),"mismatches":mismatches,"passed":mismatches==0}
        (args.fixture/"mask-validation.json").write_text(json.dumps(report,indent=2)+"\n")
        print(report)
        if mismatches: raise AssertionError("Official block mask selection differs")
        return
    if args.compare_saved:
        results=[]
        for file in sorted(args.fixture.glob("official-*.safetensors")):
            name=file.name.removeprefix("official-")
            native_file=args.fixture/name
            if not native_file.exists(): continue
            x=next(iter(load_file(str(native_file)).values())).float().numpy().ravel()
            y=next(iter(load_file(str(file)).values())).float().numpy().ravel()
            if x.shape!=y.shape: raise AssertionError(f"Fixture shape mismatch: {name}")
            error=np.abs(x-y)
            result={"stage":file.stem.removeprefix("official-"),"maximumAbsoluteError":float(error.max()),
                    "meanAbsoluteError":float(error.mean()),"correlation":float(np.corrcoef(x,y)[0,1])}
            results.append(result); print(result,flush=True)
        previous=json.loads((args.fixture/"reference-validation.json").read_text())
        outputs=[x for x in results if x["stage"].startswith("output-")]
        passed=len(outputs)==previous["inputFrames"] and all(x["meanAbsoluteError"]<0.025 and x["correlation"]>0.99 for x in outputs)
        previous.update(passed=passed,comparisons=results)
        (args.fixture/"reference-validation.json").write_text(json.dumps(previous,indent=2)+"\n")
        if not passed: raise AssertionError("Saved official fixture comparison failed")
        return
    commit=subprocess.check_output(["git","-C",str(args.reference),"rev-parse","HEAD"],text=True).strip()
    if commit!=REVISION: raise ValueError("Reference source revision mismatch")
    dirty=subprocess.check_output(["git","-C",str(args.reference),"status","--porcelain","--untracked-files=no"],text=True).strip()
    if dirty: raise ValueError("Official reference has tracked modifications; restore the pinned source before validating")
    torch.set_num_threads(4)
    environment={"torch":torch,"nn":nn,"F":F,"math":math,"random":random,"rearrange":rearrange,"repeat":repeat,
                 "tqdm":tqdm,"CACHE_T":2,**vars(typing)}
    definitions(args.reference/"diffsynth/models/wan_video_dit.py",environment)
    def attention(q,k,v,num_heads,compatibility_mode=False,attention_mask=None,return_KV=False):
        qq=q.reshape(1,-1,num_heads,128).transpose(1,2)
        kk=k.reshape(1,-1,num_heads,128).transpose(1,2)
        vv=v.reshape(1,-1,num_heads,128).transpose(1,2)
        mask=None if attention_mask is None else attention_mask.repeat_interleave(128,-2).repeat_interleave(128,-1)
        return F.scaled_dot_product_attention(qq,kk,vv,attn_mask=mask).transpose(1,2).reshape(q.shape)
    environment["flash_attention"]=attention
    projection_environment=dict(environment)
    definitions(args.reference/"examples/WanVSR/utils/utils.py",projection_environment)
    from collections import namedtuple
    decoder_environment=dict(environment)
    decoder_environment["TWorkItem"]=namedtuple("TWorkItem",("input_tensor","block_index"))
    decoder_environment["init"]=torch.nn.init
    definitions(args.reference/"examples/WanVSR/utils/TCDecoder.py",decoder_environment)
    environment["TeaCache"]=object
    definitions(args.reference/"diffsynth/pipelines/flashvsr_tiny_long.py",environment,
                {"model_fn_wan_video","_calc_mean_std","_adain"})
    WanModel=environment["WanModel"]
    with torch.device("meta"):
        dit=WanModel(dim=1536,in_dim=16,ffn_dim=8960,out_dim=16,text_dim=4096,freq_dim=256,
                     eps=1e-6,patch_size=(1,2,2),num_heads=12,num_layers=30)
        projection=projection_environment["Causal_LQ4x_Proj"](3,1536,1)
        decoder=decoder_environment["TAEHV"](checkpoint_path=None,channels=[512,256,128,128],latent_channels=784)
    dit.freqs=environment["precompute_freqs_cis_3d"](128)
    manifest=json.loads((args.model/"manifest.json").read_text())
    def state(component):
        tensors={}
        files={v["file"] for k,v in manifest["weightMap"].items() if k.startswith(component+":")}
        for file in files:
            for key,value in load_file(str(args.model/file)).items():
                if value.ndim==5: value=value.permute(0,4,1,2,3).contiguous()
                elif value.ndim==4 and key.endswith("weight"): value=value.permute(0,3,1,2).contiguous()
                tensors[key]=value
        return tensors
    dit.load_state_dict(state("dit"),strict=True,assign=True)
    projection.load_state_dict(state("projection"),strict=True,assign=True)
    decoder.load_state_dict(state("decoder"),strict=True,assign=True)
    decoder.to(args.decoder_device)
    dit.eval(); projection.eval(); decoder.eval()
    projection.clear_cache(); decoder.clean_mem()
    prompt=load_file(str(args.model/"prompt.safetensors"))["context"]
    job=json.loads((args.job/"manifest.json").read_text())
    n=len(job["frames"]); chunks=1+max(0,(n-21+7)//8); total=25+(chunks-1)*8
    first=Image.open(args.job/"degraded/000000.png").convert("RGB")
    width=(first.width*4+127)//128*128;height=(first.height*4+127)//128*128
    frames=[]
    for i in range(total):
        image=Image.open(args.job/f"degraded/{min(i,n-1):06d}.png").convert("RGB")
        up=np.asarray(image.resize((image.width*4,image.height*4),Image.Resampling.BICUBIC)).copy()
        up=np.pad(up,((0,height-up.shape[0]),(0,width-up.shape[1]),(0,0)),mode="edge")
        frames.append(torch.from_numpy(up).permute(2,0,1).float().div(127.5).sub(1).to(torch.bfloat16))
    video=torch.stack(frames,1).unsqueeze(0)
    noise=load_file(str(args.fixture/"noise.safetensors"))["noise"].permute(0,4,1,2,3).contiguous()
    comparisons=[]
    def compare(name,reference):
        save_file({"reference":reference.detach().cpu().contiguous()},str(args.fixture/("official-"+name+".safetensors")))
        file=args.fixture/(name+".safetensors")
        if not file.exists(): return
        native=next(iter(load_file(str(file)).values())).float().numpy().ravel()
        ref=reference.detach().cpu().float().numpy().ravel()
        if native.shape!=ref.shape: raise AssertionError(f"{name}: shape mismatch")
        error=np.abs(native-ref)
        correlation=float(np.corrcoef(native,ref)[0,1]) if native.std()>0 and ref.std()>0 else None
        report={"stage":name,"maximumAbsoluteError":float(error.max()),"meanAbsoluteError":float(error.mean()),"correlation":correlation}
        comparisons.append(report); print(report,flush=True)
    with torch.inference_mode():
        compare("conditioning",video[0].permute(1,2,3,0))
        hook_counts={}
        for module,name in [(projection.conv1,"projection_conv1"),(projection.act1,"projection_norm1"),
                            (projection.conv2,"projection_conv2"),(projection.act2,"projection_norm2")]:
            def hook(layer,inputs,result,name=name):
                number=hook_counts.get(name,0); hook_counts[name]=number+1
                if name.endswith("1") and number<2: compare(name+f"_{number}",result.permute(0,2,3,4,1))
                elif name.endswith("2") and number==0: compare(name,result.permute(0,2,3,4,1))
            module.register_forward_hook(hook)
        dit.reinit_cross_kv(prompt)
        time_embedding=dit.time_embedding(environment["sinusoidal_embedding_1d"](256,torch.tensor([1000.],dtype=torch.bfloat16)))
        modulation=dit.time_projection(time_embedding).unflatten(1,(6,1536))
        compare("time",time_embedding);compare("time_modulation",modulation)
        cache_k=[None]*30;cache_v=[None]*30;output=[];start_time=time.time()
        for chunk in range(chunks):
            pieces=[]
            for i in range(7 if chunk==0 else 2):
                start=max(0,i*4-3) if chunk==0 else chunk*8+17+i*4
                count=1 if chunk==0 and i==0 else 4
                value=projection.stream_forward(video[:,:,start:start+count])
                if value is not None: pieces.append(value[0])
            projected=torch.cat(pieces,dim=1);compare(f"projection_{chunk}",projected)
            begin=0 if chunk==0 else 4+chunk*2;count=6 if chunk==0 else 2
            current=noise[:,:,begin:begin+count]
            handles=[]
            if chunk==chunks-1:
                for i in (0,29):
                    handles.append(dit.blocks[i].register_forward_hook(lambda module,inputs,result,i=i:compare(f"block_{i}",result[0])))
            predicted,cache_k,cache_v=environment["model_fn_wan_video"](dit,x=current,timestep=torch.tensor([1000.]),context=None,
                LQ_latents=[projected],is_stream=True,pre_cache_k=cache_k,pre_cache_v=cache_v,
                topk_ratio=2*768*1280/(height*width),kv_ratio=3,cur_process_idx=chunk,t_mod=modulation,t=time_embedding,local_range=11)
            for handle in handles:handle.remove()
            if chunk==chunks-1: compare("predicted",predicted.permute(0,2,3,4,1))
            latent=current-predicted;compare(f"latent_{chunk}",latent[0].permute(1,2,3,0))
            offset=0 if chunk==0 else 21+(chunk-1)*8;count=21 if chunk==0 else 8
            visible=min(count,n-offset)
            needed=(visible+(3 if chunk==0 else 0)+3)//4
            conditioning_count=needed*4-(3 if chunk==0 else 0)
            # Omitted trailing replicated frames cannot influence this causal decoder.
            decoded=decoder.decode_video(latent[:,:,:needed].transpose(1,2).to(args.decoder_device),parallel=False,show_progress_bar=False,
                                         cond=video[:,:,offset:offset+conditioning_count].to(args.decoder_device)).transpose(1,2)*2-1
            # Actual official ADAIN computes each frame/channel's spatial statistics.
            for i in range(visible):
                corrected=environment["_adain"](decoded[:,:,i],video[:,:,offset+i].to(args.decoder_device)).clamp(-1,1)
                if offset+i<n:
                    compare(f"output-{offset+i:06d}",corrected[0].permute(1,2,0))
                    output.append(corrected[0])
            print(f"Official CPU chunk {chunk+1}/{chunks} complete",flush=True)
    outputs=[x for x in comparisons if x["stage"].startswith("output-")]
    passed=len(outputs)==n and all(x["meanAbsoluteError"]<0.025 and x["correlation"]>0.99 for x in outputs)
    report={"referenceRevision":REVISION,"reference":f"Official BF16 graph, CPU projection/DiT, {args.decoder_device} decoder, masked SDPA equivalent to sparse dispatch",
            "inputFrames":n,"outputFrames":len(output),"secondsExcludingLoad":time.time()-start_time,
            "tolerances":{"outputMeanAbsoluteError":0.025,"minimumOutputCorrelation":0.99},"passed":passed,"comparisons":comparisons}
    (args.fixture/"reference-validation.json").write_text(json.dumps(report,indent=2)+"\n")
    if not passed: raise AssertionError("Native output parity failed; inspect intermediate comparisons")

if __name__=="__main__":main()
