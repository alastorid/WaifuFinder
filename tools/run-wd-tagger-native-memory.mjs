#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import cp from 'node:child_process';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const label=process.argv[2]??'initial';
if(!/^[a-zA-Z0-9_-]+$/.test(label))throw Error('Invalid output label');
const out=path.join(root,'.build/wd-native/memory-pass',label);
if(fs.existsSync(out))throw Error('Choose a new label to preserve prior measurements');
fs.mkdirSync(out,{recursive:true});
const native=path.join(root,'Runtime/.build/release/waifu-wdtagger-validation');
const cases=label.startsWith("rgba")?[
 {name:"coreml-start",coreml:true},
 {name:"current-float",memory:768,cache:128,stream:true},
 {name:"rgba512",memory:512,cache:64,rgba:true,stream:true},
 {name:"rgba448",memory:448,cache:64,rgba:true,stream:true},
 {name:"rgba384",memory:384,cache:64,rgba:true,stream:true},
 {name:"rgba448-confirm",memory:448,cache:64,rgba:true,stream:true},
 {name:"current-float-confirm",memory:768,cache:128,stream:true},
 {name:"coreml-end",coreml:true},
]:[
 {name:'baseline',memory:768,cache:128},
 {name:'budget512',memory:512,cache:64},
 {name:'budget384',memory:384,cache:64},
 {name:'fastsync512',memory:512,cache:64,env:{MLX_METAL_FAST_SYNCH:'1'}},
 {name:'submit16',memory:512,cache:64,env:{MLX_MAX_MB_PER_BUFFER:'16'}},
 {name:'baseline-confirm',memory:768,cache:128},
];
// Offline conversion is outside all measured processes and timers.
if(label.startsWith('rgba')){
 const source=fs.readFileSync(path.join(root,'.build/wd-optimization/inputs.bin'));
 const pixels=448*448, frames=source.length/(3*pixels*4), rgba=Buffer.alloc(frames*pixels*4);
 for(let frame=0;frame<frames;frame++)for(let pixel=0;pixel<pixels;pixel++){
  for(let channel=0;channel<3;channel++){
   const value=source.readFloatLE(((frame*3+channel)*pixels+pixel)*4);
   if(!Number.isInteger(value)||value<0||value>255)throw Error('RGBA comparison requires exact byte-valued RGB source');
   rgba[(frame*pixels+pixel)*4+channel]=value;
  }
  rgba[(frame*pixels+pixel)*4+3]=255;
 }
 fs.writeFileSync(path.join(root,'.build/wd-native/inputs-rgba.bin'),rgba);
}
const maxGPU=Number(process.env.WAIFU_BENCH_MAX_GPU_C??55);
if(!Number.isFinite(maxGPU)||maxGPU<40||maxGPU>65)throw Error('Invalid GPU temperature gate');
const records=[];
const original=fs.readFileSync(path.join(root,'.build/wd-native/tuning/fused-memory/fused8-768.bin'));
for(const job of cases){
 const deadline=Date.now()+600000;let before,thermal;
 while(true){before=JSON.parse(cp.execFileSync('macmon',['pipe','-s','1'],{encoding:'utf8'}));thermal=Number(cp.execFileSync(native,['--thermal'],{encoding:'utf8'}));if(thermal===0&&before.temp.gpu_temp_avg<=maxGPU&&before.cpu_usage_pct<=0.4)break;if(Date.now()>deadline)throw Error('Idle/cooling gate expired');console.log(`Cooling for ${job.name}: ${before.temp.gpu_temp_avg.toFixed(1)} C`);await new Promise(r=>setTimeout(r,15000));}
 console.log('Running '+job.name);
 const report=path.join(out,job.name+'.json'),bin=path.join(out,job.name+'.bin'),log=fs.openSync(path.join(out,job.name+'.log'),'w');
 const args=['--weights',path.join(root,'.build/wd-native/model.safetensors'),'--inputs',path.join(root,job.rgba?'.build/wd-native/inputs-rgba.bin':'.build/wd-optimization/inputs.bin'),'--frames','64','--batch','8','--repeats','3','--memory-mb',String(job.memory),'--cache-mb',String(job.cache),'--report',report,'--probabilities',bin,...(job.rgba?['--rgba-input']:[]),...(job.stream?['--stream-input']:[])];
 const installed=path.join(root,'models/published/WD-SwinV2-Tagger-v3-CoreML');
 const command=job.coreml?path.join(root,'.build/wd-native/memory-pass/coreml-stream'):native;
 const commandArgs=job.coreml?['--model',path.join(root,'.build/wd-native/memory-pass/coreml-model.mlmodelc'),'--labels',path.join(installed,'wd-tagger-labels.csv'),'--tensor-input',path.join(root,'.build/wd-optimization/inputs.bin'),'--stream-input','--repeats','3','--report',report,'--probability-output',bin]:args;
 const result=cp.spawnSync(command,commandArgs,{env:{...process.env,...job.env},stdio:['ignore',log,log]});fs.closeSync(log);if(result.status!==0)throw Error(job.name+' failed, inspect log');
 const bytes=fs.readFileSync(bin);const record={configuration:job,...JSON.parse(fs.readFileSync(report)),probabilitiesByteIdentical:bytes.equals(job.coreml?fs.readFileSync(path.join(root,'.build/wd-native/tuning/coreml/coreml8.bin')):original),outputSHA256:crypto.createHash('sha256').update(bytes).digest('hex'),maximumStartingGPUTemperatureC:maxGPU,systemBefore:{gpuTemperatureC:before.temp.gpu_temp_avg,cpuScaledUsage:before.cpu_usage_pct},systemAfter:JSON.parse(cp.execFileSync('macmon',['pipe','-s','1'],{encoding:'utf8'}))};
 if(job.coreml)record.predictionFramesPerSecond=64/record.stageSeconds.map(s=>s.prediction).sort((a,b)=>a-b)[1];
 if(!record.probabilitiesByteIdentical)throw Error(job.name+' changed predictions');
 records.push(record);fs.writeFileSync(path.join(out,'comparison.json'),JSON.stringify(records,null,2)+'\n');console.log(`${job.name}: ${record.predictionFramesPerSecond.toFixed(3)} fps, ${(record.peakPhysicalFootprintBytes/1e6).toFixed(1)} MB, MLX peak ${(record.peakMLXActiveBytes/1e6).toFixed(1)} MB`);
}
