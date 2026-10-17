#!/usr/bin/env node
// Read an existing Instruments export. Does not record or replay GPU work.
import fs from 'node:fs';
const [input,output]=process.argv.slice(2);
if(!input||!output)throw Error('Usage: inspect-wd-tagger-profile.mjs EXPORT_XML REPORT_JSON');
const xml=fs.readFileSync(input,'utf8'),root={name:'root',attrs:{},children:[],text:''},stack=[root],ids=new Map();
const decode=s=>s.replace(/&(?:amp|lt|gt|quot|apos|#\d+|#x[\da-f]+);/gi,v=>({'&amp;':'&','&lt;':'<','&gt;':'>','&quot;':'"','&apos;':"'"}[v]??String.fromCodePoint(v[2]==='x'?parseInt(v.slice(3,-1),16):Number(v.slice(2,-1)))));
for(const token of xml.matchAll(/<\/?([\w-]+)\b([^>]*?)\/?\s*>|([^<]+)/g)){
  if(token[3]!==undefined){stack.at(-1).text+=decode(token[3]);continue;}
  if(token[0].startsWith('</')){stack.pop();continue;}
  const attrs=Object.fromEntries([...token[2].matchAll(/([\w-]+)=(?:"([^"]*)"|'([^']*)')/g)].map(m=>[m[1],decode(m[2]??m[3])]));
  const node={name:token[1],attrs,children:[],text:''};stack.at(-1).children.push(node);
  if(attrs.id)ids.set(attrs.id,node);
  if(!token[0].endsWith('/>'))stack.push(node);
}
const resolve=n=>n.attrs.ref?ids.get(n.attrs.ref)??n:n;
const value=n=>{n=resolve(n);return n.attrs.fmt??n.text.trim();};
function descendants(n,name,result=[]){if(n.name===name)result.push(n);for(const c of n.children)descendants(c,name,result);return result;}
const tables=[],aggregated=new Map();let targetCount=0,targetDuration=0,first=Infinity,last=0;
for(const node of descendants(root,'node')){
  const rows=node.children.filter(c=>c.name==='row'),schema=node.children.find(c=>c.name==='schema');
  const firstTags=rows[0]?.children.map(c=>c.name)??[];
  const name=schema?.attrs.name??(firstTags.includes('gpu-channel-name')?'metal-gpu-intervals':'metal-application-intervals');
  tables.push({name,rows:rows.length,xpath:node.attrs.xpath});
  if(name!=='metal-gpu-intervals')continue;
  for(const row of rows){
    const children=row.children,process=children.filter(c=>c.name==='process').at(-1);
    if(!process||!value(process).includes('waifu-wdtagger-validation'))continue;
    const duration=resolve(children.find(c=>c.name==='duration')),start=resolve(children.find(c=>c.name==='start-time'));
    const label=children.find(c=>c.name==='formatted-label');
    const depth=children.find(c=>c.name==='metal-nesting-level');
    if(depth&&Number(resolve(depth).text)!==0)continue;
    const milliseconds=Number(duration.text)/1e6,begin=Number(start.text)/1e6;
    targetCount++;targetDuration+=milliseconds;first=Math.min(first,begin);last=Math.max(last,begin+milliseconds);
    const key=(label?value(label):'Unlabeled').replace(/0x[\da-f]+/gi,'ID').replace(/Frame\s+\d+/g,'Frame N');
    const item=aggregated.get(key)??{label:key,intervals:0,totalDurationMilliseconds:0};item.intervals++;item.totalDurationMilliseconds+=milliseconds;aggregated.set(key,item);
  }
}
const report={tables,targetDepthZeroGPUIntervals:targetCount,targetSummedEncoderDurationMilliseconds:targetDuration,targetGPUActivitySpanMilliseconds:last-first,
  labels:[...aggregated.values()].sort((a,b)=>b.totalDurationMilliseconds-a.totalDurationMilliseconds),
  limitations:['The capture includes startup/warmup and one warmed prediction batch.','Encoder durations are not individual shader costs; summed durations may overlap across GPU channels.','No per-shader hotspot is established when the shader interval table is empty.']};
fs.writeFileSync(output,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({...report,labels:report.labels.slice(0,5)},null,2));
