#!/usr/bin/env node
// Native inference only. Node records cooling conditions and launches candidates.
import fs from 'node:fs';
import path from 'node:path';
import cp from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {coremlDirectory as installed} from './wd-tagger-benchmark-paths.mjs';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const compactRun=process.argv.includes('--compact');
const coremlRun=process.argv.includes('--coreml');
const fusedRun=process.argv.includes('--fused-speed');
const memoryRun=process.argv.includes('--fused-memory');
const out=path.join(root,'.build/wd-native/tuning',...(memoryRun?['fused-memory']:fusedRun?['fused-standalone']:coremlRun?['coreml']:compactRun?['compact']:[])),native=path.join(root,'Runtime/.build/release/waifu-wdtagger-validation');
fs.mkdirSync(out,{recursive:true});
const cases=memoryRun?[
  {name:'fused8-1024',tile:'32x64x32',simd:false,compact:false,memory:1024,batch:8,cache:128,fused:true},
  {name:'fused8-768',tile:'32x64x32',simd:false,compact:false,memory:768,batch:8,cache:128,fused:true},
]:fusedRun?[
  {name:'original8',tile:'32x64x32',simd:false,compact:false,memory:1536,batch:8,cache:128,fused:false},
  {name:'fused8',tile:'32x64x32',simd:false,compact:false,memory:1536,batch:8,cache:128,fused:true},
]:coremlRun?[
  {name:'coreml8',coreml:true},
  {name:'native8',tile:'relaxed32x64x32',simd:false,compact:false,memory:1536,batch:8,cache:128},
  {name:'native8-confirmation',tile:'relaxed32x64x32',simd:false,compact:false,memory:1536,batch:8,cache:128},
  {name:'coreml8-confirmation',coreml:true},
]:compactRun?[
  {name:'baseline2',tile:'32x64x32',simd:false,memory:384},
  {name:'compact2',tile:'32x64x32',simd:false,compact:true,memory:357},
  {name:'compact2-confirmation',tile:'32x64x32',simd:false,compact:true,memory:357},
  {name:'baseline2-confirmation',tile:'32x64x32',simd:false,memory:384},
]:[
  {name:'baseline2',tile:'32x64x32',simd:false},
  {name:'tile64-2',tile:'64x64x32',simd:false},
  {name:'simd2',tile:'32x64x32',simd:true},
  {name:'simd2-confirmation',tile:'32x64x32',simd:true},
  {name:'baseline2-confirmation',tile:'32x64x32',simd:false},
];
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const records=[];
for(const job of cases){
  const deadline=Date.now()+600000;let before,thermal;
  while(true){
    before=JSON.parse(cp.execFileSync('macmon',['pipe','-s','1'],{encoding:'utf8'}));
    thermal=Number(cp.execFileSync(native,['--thermal'],{encoding:'utf8'}));
    if(thermal===0&&before.temp.gpu_temp_avg<=55&&before.cpu_usage_pct<=0.4)break;
    if(Date.now()>deadline)throw Error('Cooling/CPU-idle gate timed out; no timed result for this candidate');
    console.log(`Waiting for ${job.name}: GPU ${before.temp.gpu_temp_avg.toFixed(1)} C, CPU ${(before.cpu_usage_pct*100).toFixed(0)}%, thermal ${thermal}`);
    await sleep(15000);
  }
  console.log(`Running ${job.name} at ${before.temp.gpu_temp_avg.toFixed(1)} C`);
  const report=path.join(out,job.name+'.json'),log=fs.openSync(path.join(out,job.name+'.log'),'w');
  const coreml=path.join(root,'.build/wd-optimization/shape-benchmark-v4');
  const args=job.coreml?[
    '--model',path.join(installed,'model.mlmodelc'),'--labels',path.join(installed,'wd-tagger-labels.csv'),
    '--tensor-input',path.join(root,'.build/wd-optimization/inputs.bin'),'--repeats','3','--report',report,
    '--probability-output',path.join(out,job.name+'.bin'),
  ]:['--weights',path.join(root,'.build/wd-native/model.safetensors'),'--inputs',path.join(root,'.build/wd-optimization/inputs.bin'),
    '--frames','64','--batch',String(job.batch??2),'--repeats','3','--memory-mb',String(job.memory??384),'--cache-mb',String(job.cache??64),'--gemm-tile',job.tile,
    '--report',report,'--probabilities',path.join(out,job.name+'.bin'),...(job.simd?['--simd-norm']:[]),...(job.compact?['--compact-bias']:[]),...(job.fused?['--fused-qkv']:['--unfused-qkv'])];
  const result=cp.spawnSync(job.coreml?coreml:native,args,{stdio:['ignore',log,log]});fs.closeSync(log);
  if(result.status!==0)throw Error(`${job.name} failed: ${result.status}`);
  const record={name:job.name,...JSON.parse(fs.readFileSync(report)),systemBefore:before};
  if(job.coreml){const times=record.stageSeconds.map(s=>s.prediction).sort((a,b)=>a-b);record.predictionFramesPerSecond=64/times[1];}
  record.systemAfter=JSON.parse(cp.execFileSync('macmon',['pipe','-s','1'],{encoding:'utf8'}));
  fs.writeFileSync(report,JSON.stringify(record,null,2)+'\n');records.push(record);
  fs.writeFileSync(path.join(out,'comparison.json'),JSON.stringify(records,null,2)+'\n');
  console.log(`${job.name}: ${record.predictionFramesPerSecond.toFixed(2)} fps, ${(record.peakPhysicalFootprintBytes/1e6).toFixed(1)} MB, thermal ${record.thermalBefore}->${record.thermalAfter}`);
}
