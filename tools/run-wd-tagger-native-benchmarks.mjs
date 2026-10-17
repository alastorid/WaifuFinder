#!/usr/bin/env node
// One cooled comparison after the frozen native implementation's profile.
// Both inference executables are native; Node only launches them and records data.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import cp from 'node:child_process';
import {coremlDirectory as installed} from './wd-tagger-benchmark-paths.mjs';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),out=path.join(root,'.build/wd-native');
const native=path.join(root,'Runtime/.build/release/waifu-wdtagger-validation'),coreml=path.join(root,'.build/wd-optimization/shape-benchmark-v4');
const inputs=path.join(root,'.build/wd-optimization/inputs.bin');
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const jobs=[
  {name:'coreml8',exe:coreml,args:['--model',path.join(installed,'model.mlmodelc'),'--labels',path.join(installed,'wd-tagger-labels.csv'),'--tensor-input',inputs]},
  {name:'native8',exe:native,args:['--weights',path.join(out,'model.safetensors'),'--inputs',inputs,'--frames','64','--batch','8','--memory-mb','1536','--cache-mb','128','--unfused-qkv']},
  {name:'native2-memory',exe:native,args:['--weights',path.join(out,'model.safetensors'),'--inputs',inputs,'--frames','64','--batch','2','--memory-mb','384','--cache-mb','64','--unfused-qkv']},
  {name:'coreml2',exe:coreml,args:['--model',path.join(root,'.build/wd-optimization/batch2/model.mlmodelc'),'--labels',path.join(installed,'wd-tagger-labels.csv'),'--tensor-input',inputs]},
];
const records=[];
for(const job of jobs){
  let temperature,thermal;const deadline=Date.now()+600000;
  while(true){
    temperature=JSON.parse(cp.execFileSync('macmon',['pipe','-s','1'],{encoding:'utf8'})).temp.gpu_temp_avg;
    thermal=Number(cp.execFileSync(native,['--thermal'],{encoding:'utf8'}));
    if(thermal===0&&temperature<=55)break;
    if(Date.now()>deadline)throw Error('Cooling timed out');
    console.log(`Cooling for ${job.name}: ${temperature.toFixed(1)} C, thermal ${thermal}`);await sleep(15000);
  }
  console.log(`Running ${job.name} at ${temperature.toFixed(1)} C`);
  const report=path.join(out,`${job.name}-benchmark.json`),probabilities=path.join(out,`${job.name}-probabilities.bin`),log=fs.openSync(path.join(out,`${job.name}-benchmark.log`),'w');
  const probabilityFlag=job.exe===native?'--probabilities':'--probability-output';
  const result=cp.spawnSync(job.exe,[...job.args,'--report',report,'--repeats','3',probabilityFlag,probabilities],{stdio:['ignore',log,log]});fs.closeSync(log);
  if(result.status!==0)throw Error(`${job.name} failed: ${result.status}`);
  const record=JSON.parse(fs.readFileSync(report,'utf8'));record.startGpuTemperatureC=temperature;record.coolingThresholdC=55;
  if(job.exe===coreml){const prediction=record.stageSeconds.map(s=>s.prediction).sort((a,b)=>a-b);record.predictionFramesPerSecond=64/prediction[1];}
  fs.writeFileSync(report,JSON.stringify(record,null,2)+'\n');records.push({name:job.name,...record});
  fs.writeFileSync(path.join(out,'comparison.json'),JSON.stringify(records,null,2)+'\n');
  console.log(`${job.name}: ${record.predictionFramesPerSecond.toFixed(2)} prediction fps, ${(record.peakPhysicalFootprintBytes/1e6).toFixed(1)} MB, thermal ${record.thermalBefore}->${record.thermalAfter}`);
}
