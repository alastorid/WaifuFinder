#!/usr/bin/env node
// Offline numerical comparison only. Inference is the native Swift executable.
import fs from 'node:fs';
import crypto from 'node:crypto';

const args = process.argv.slice(2);
function option(name) { const i=args.indexOf(name); if(i<0 || i+1>=args.length)throw Error(`Missing ${name}`);return args[i+1]; }
function floats(file) {
  const bytes=fs.readFileSync(file);
  if(bytes.length%4)throw Error(`Unaligned FP32 file: ${file}`);
  return new Float32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4);
}
function csv(text) {
  const rows=[];let row=[],field='',quoted=false;
  for(let i=0;i<text.length;i++){
    const c=text[i];
    if(c==='"'){
      if(quoted && text[i+1]==='"'){field+='"';i++;}else quoted=!quoted;
    }else if(c===','&&!quoted){row.push(field);field='';}
    else if(c==='\n'&&!quoted){row.push(field.replace(/\r$/,''));rows.push(row);row=[];field='';}
    else field+=c;
  }
  if(field||row.length){row.push(field.replace(/\r$/,''));rows.push(row);}
  return rows;
}
const actualPath=option('--actual'), referencePath=option('--reference'), baselinePath=option('--baseline');
const actual=floats(actualPath),reference=floats(referencePath),baseline=floats(baselinePath),heads=10861;
if(actual.length!==reference.length||actual.length!==baseline.length||actual.length%heads)throw Error('Corpus shape mismatch');
if(actual.some(x=>!Number.isFinite(x)||x<0||x>1))throw Error('Invalid probabilities');
const labels=csv(fs.readFileSync(option('--labels'),'utf8')),category=labels[0].indexOf('category');
if(category<0||labels.length-1!==heads)throw Error('Label table mismatch');
const general=labels.slice(1).flatMap((row,i)=>row[category]==='0'?[i]:[]),all=Array.from({length:heads},(_,i)=>i);
const frames=actual.length/heads;
const naturalOption=args.indexOf('--natural-frames');
const natural=naturalOption<0?0:Number(args[naturalOption+1]);
if(!Number.isInteger(natural)||natural<0||natural>frames)throw Error('Invalid --natural-frames');
function top(values,offset,indices) {return [...indices].sort((i,j)=>values[offset+j]-values[offset+i]).slice(0,15);}
function overlap(a,b) {const s=new Set(a);return b.reduce((n,i)=>n+Number(s.has(i)),0);}
function metrics(expected,begin,end){
  if(begin===end)return null;
  let maximum=0,total=0,thresholds=0,generalThresholds=0,ratings=0,top15=15,top15General=15;
  const perImage=[];
  for(let frame=begin;frame<end;frame++){
    const offset=frame*heads;let imageMax=0;
    for(let i=0;i<heads;i++){
      const error=Math.abs(actual[offset+i]-expected[offset+i]);maximum=Math.max(maximum,error);imageMax=Math.max(imageMax,error);total+=error;
      if((actual[offset+i]>.35)!==(expected[offset+i]>.35))thresholds++;
    }
    for(const i of general)if((actual[offset+i]>.35)!==(expected[offset+i]>.35))generalThresholds++;
    const rating=x=>[0,1,2,3].reduce((best,i)=>x[offset+i]>x[offset+best]?i:best,0);
    ratings+=Number(rating(actual)!==rating(expected));
    top15=Math.min(top15,overlap(top(actual,offset,all),top(expected,offset,all)));
    top15General=Math.min(top15General,overlap(top(actual,offset,general),top(expected,offset,general)));
    perImage.push(imageMax);
  }
  return {maximumAbsoluteError:maximum,meanAbsoluteError:total/((end-begin)*heads),thresholdDifferencesAllHeads:thresholds,
    thresholdDifferencesGeneralHeads:generalThresholds,ratingDifferences:ratings,minimumTop15Overlap:top15,minimumTop15GeneralOverlap:top15General,perImageMaximumAbsoluteError:perImage};
}
const report={frames,naturalFrames:natural,syntheticFrames:frames-natural,
  actualSHA256:crypto.createHash('sha256').update(fs.readFileSync(actualPath)).digest('hex'),
  referenceSHA256:crypto.createHash('sha256').update(fs.readFileSync(referencePath)).digest('hex'),
  baselineSHA256:crypto.createHash('sha256').update(fs.readFileSync(baselinePath)).digest('hex'),
  againstReference:metrics(reference,0,frames),naturalAgainstReference:metrics(reference,0,natural),
  syntheticAgainstReference:metrics(reference,natural,frames),againstBaseline:metrics(baseline,0,frames),
  note:'Numerical fidelity against the supplied FP32 reference and baseline; not labeled tagging accuracy.'};
fs.writeFileSync(option('--report'),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({natural:report.naturalAgainstReference,baseline:report.againstBaseline},null,2));
