#!/usr/bin/env node
// Adjacent predictions in one native process; ANE defaults to single-image calls.
import fs from 'node:fs';import path from 'node:path';import cp from 'node:child_process';import {fileURLToPath} from 'node:url';
import {coremlDirectory} from './wd-tagger-benchmark-paths.mjs';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),exe=path.join(root,'Runtime/.build/release/waifu-wdtagger-validation');
const baseline=process.argv.includes('--baseline'),compact=process.argv.includes('--compact');
const tileOption=process.argv.indexOf('--tile'),tile=tileOption>=0?process.argv[tileOption+1]:baseline?'32x64x32':'relaxed32x64x32';
if(!tile||!/^[a-z0-9-]+$/.test(tile))throw Error('Invalid --tile');
const includeOriginal=process.argv.includes('--compare-original');
const fused=process.argv.includes('--fused-qkv');
const coremlUnitsIndex=process.argv.indexOf('--coreml-units'),coremlUnits=coremlUnitsIndex>=0?process.argv[coremlUnitsIndex+1]:'cpuAndGPU';
if(!['cpuAndGPU','cpuAndNeuralEngine','all'].includes(coremlUnits))throw Error('Invalid --coreml-units');
const batchIndex=process.argv.indexOf('--batch-size'),batchSize=batchIndex>=0?Number(process.argv[batchIndex+1]):coremlUnits==='cpuAndNeuralEngine'?1:8;
if(![1,2,4,8,16,32].includes(batchSize))throw Error('Invalid --batch-size');
const inputsIndex=process.argv.indexOf('--inputs'),inputs=inputsIndex>=0?path.resolve(process.argv[inputsIndex+1]):path.join(root,'.build/wd-optimization/inputs.bin');
const modelIndex=process.argv.indexOf('--coreml-model');
const defaultModelDirectory=process.env.WAIFU_WD_COREML_DIRECTORY?coremlDirectory:batchSize===1?path.join(root,'.build/wd-ane-baseline/coreml-batch1'):coremlDirectory;
const model=modelIndex>=0?path.resolve(process.argv[modelIndex+1]):path.join(defaultModelDirectory,'model.mlmodelc');
const labelIndex=process.argv.indexOf('--label'),label=labelIndex>=0?process.argv[labelIndex+1]:'';
if(label&&!/^[a-z0-9-]+$/.test(label))throw Error('Invalid --label');
const unitsLabel=coremlUnits==='cpuAndGPU'?'gpu':coremlUnits==='cpuAndNeuralEngine'?'ane':'all';
const unitsSuffix=coremlUnits==='cpuAndGPU'?'':`-coreml-${unitsLabel}`;
const out=path.join(root,'.build/wd-native/tuning',`paired-${tileOption>=0?tile:baseline?'baseline':'relaxed'}-batch${batchSize}${compact?'-compact':''}${fused?'-fused':''}${includeOriginal?'-triad':''}${unitsSuffix}${label?'-'+label:''}`);fs.mkdirSync(out,{recursive:true});
const sleep=ms=>new Promise(r=>setTimeout(r,ms));let before;const deadline=Date.now()+600000;
while(true){before=JSON.parse(cp.execFileSync('macmon',['pipe','-s','1'],{encoding:'utf8'}));const thermal=Number(cp.execFileSync(exe,['--thermal'],{encoding:'utf8'}));
  if(thermal===0&&before.temp.gpu_temp_avg<=55&&before.cpu_usage_pct<=0.4)break;
  if(Date.now()>deadline)throw Error('Cooling/CPU-idle gate timed out');
  console.log(`Waiting: GPU ${before.temp.gpu_temp_avg.toFixed(1)} C, CPU ${(before.cpu_usage_pct*100).toFixed(0)}%, thermal ${thermal}`);await sleep(15000);
}
console.log(`Starting paired ${tile}, batch=${batchSize}, Core ML units=${coremlUnits}, compact=${compact} at ${before.temp.gpu_temp_avg.toFixed(1)} C`);
const log=fs.openSync(path.join(out,'prediction.log'),'w'),report=path.join(out,'comparison.json');
const result=cp.spawnSync(exe,['--weights',path.join(root,'.build/wd-native/model.safetensors'),'--inputs',inputs,
  '--compare-coreml',model,'--batch-size',String(batchSize),
  '--coreml-units',coremlUnits,'--gemm-tile',tile,'--memory-mb','1536','--cache-mb','128','--repeats','3','--report',report,'--probabilities',path.join(out,'probabilities'),...(compact?['--compact-bias']:[]),...(includeOriginal?['--compare-original']:[]),...(fused?['--fused-qkv']:['--unfused-qkv'])],{stdio:['ignore',log,log]});
fs.closeSync(log);if(result.status!==0)throw Error(`Paired inference failed: ${result.status}`);
const record=JSON.parse(fs.readFileSync(report));record.systemBefore=before;record.systemAfter=JSON.parse(cp.execFileSync('macmon',['pipe','-s','1'],{encoding:'utf8'}));
fs.writeFileSync(report,JSON.stringify(record,null,2)+'\n');console.log(`Native ${record.nativePredictionFPS.toFixed(2)}, Core ML ${record.coremlPredictionFPS.toFixed(2)} fps; ratio ${record.medianSweepSpeedRatio.toFixed(3)}`);
if(includeOriginal)console.log(`Original native ${record.originalPredictionFPS.toFixed(2)} fps; candidate/original ${record.nativeOverOriginalSpeedRatio.toFixed(3)}; bitwise equal ${record.nativeOriginalProbabilitiesBitwiseEqual}`);
