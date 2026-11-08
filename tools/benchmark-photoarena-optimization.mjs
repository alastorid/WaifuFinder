// One final comparative session, never an implementation-time microbenchmark.
// All subprocesses run serially. Inputs remain in anonymous pipes; only scalar
// memory traces and reproducible judgments are written to the retained report.
import { spawn, spawnSync } from 'node:child_process';
import { readFileSync, writeFileSync, readdirSync, mkdirSync, cpSync, rmSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { resolve, join } from 'node:path';
import { createHash } from 'node:crypto';
import os from 'node:os';
const root=resolve(import.meta.dirname,'..');
const reference=JSON.parse(readFileSync(join(root,'.build/reports/photoarena-optimization-reference.json')));
const temporary=reference.directory;
const model=resolve(root,'models/photoarena');
const hash=p=>createHash('sha256').update(readFileSync(p)).digest('hex');
if(hash(join(temporary,'original-worker'))!==reference.workerSHA256)throw Error('Reference binary changed');
if(hash(join(root,'Sources/Waifu/UI/FrameGridView.swift'))!==reference.unrelatedUIHash)throw Error('Unrelated user changes were modified');
const traceRoot=join(root,'.build/reports/photoarena-final-memory');
mkdirSync(traceRoot,{recursive:true});
const median=values=>{const v=[...values].sort((a,b)=>a-b);return v.length%2?v[v.length>>1]:(v[v.length/2-1]+v[v.length/2])/2;};
function environment(variant,directory,replies) {
    return {...process.env,DYLD_INSERT_LIBRARIES:join(temporary,'memory-observer.dylib'),
        ARENA_MEMORY_DIRECTORY:directory,ARENA_REPLIES:replies,ARENA_MODEL:model,ARENA_WORKER:join(temporary,variant+'-worker')};
}
function memory(directory,parentPID=null) {
    const processes=readdirSync(directory).filter(x=>x.endsWith('.csv')).map(name=>{
        const samples=readFileSync(join(directory,name),'utf8').trim().split('\n').slice(1).map(line=>line.split(',').map(Number)).filter(x=>x.length===6);
        const max=index=>Math.max(0,...samples.map(row=>row[index]));
        return {pid:Number(name.split('.')[0]),sampleCount:samples.length,
            peakPhysicalBytes:Math.max(max(1),max(2)),sampledPeakCompressedBytes:max(3),sampledPeakMetalAllocatedBytes:max(4),
            ledgerSwapinsRaw:max(5),finalPhysicalBytes:samples.at(-1)?.[1]??0};
    });
    const parent=processes.find(x=>x.pid===parentPID);
    const workers=processes.filter(x=>x.pid!==parentPID);
    if(!workers.length || workers.some(x=>x.sampleCount===0 || x.sampledPeakMetalAllocatedBytes===0))throw Error('Memory/Metal observer did not attach');
    const workerMax=field=>Math.max(0,...workers.map(p=>p[field]));
    return {processes,clientPeakPhysicalBytes:parent?.peakPhysicalBytes??null,
        workerPeakPhysicalBytes:workerMax('peakPhysicalBytes'),
        clientPlusWorkerPhysicalUpperBoundBytes:(parent?.peakPhysicalBytes??0)+workerMax('peakPhysicalBytes'),
        sampledWorkerPeakMetalAllocatedBytes:workerMax('sampledPeakMetalAllocatedBytes'),
        sampledWorkerPeakCompressedBytes:workerMax('sampledPeakCompressedBytes'),
        maximumWorkerLedgerSwapinsRaw:workerMax('ledgerSwapinsRaw'),
        notes:'Client peak plus maximum worker peak is a conservative bound, not a simultaneous GUI-app sample. Baseline workers are sequential. Metal allocations are included in physical footprint; do not add them again.'};
}
async function cold(variant,workload,repeat) {
    const directory=join(traceRoot,`${workload.id}-${repeat}-${variant}`); mkdirSync(directory,{recursive:true});
    const replies=join(directory,'replies.jsonl');
    const epoch=Date.now()/1000,start=performance.now();
    const child=spawn(join(temporary,variant+'-client'),workload.files,{env:environment(variant,directory,replies),stdio:['ignore','pipe','pipe']});
    let output='',error='';child.stdout.on('data',b=>output+=b);child.stderr.on('data',b=>{if(error.length<16384)error+=b;});
    const timer=setTimeout(()=>child.kill('SIGTERM'),660000);
    const code=await new Promise((done,reject)=>{child.on('error',reject);child.on('exit',done);});clearTimeout(timer);
    if(code!==0)throw Error(`Cold ${variant} failed (${code}): ${error}`);
    const reply=JSON.parse(output.trim());
    const passes=readFileSync(replies,'utf8').trim().split('\n').map(x=>JSON.parse(x));
    if(passes.length!==2)throw Error('Expected two native order judgments');
    return {variant,repeat,wallIncludingClientStartupSeconds:(performance.now()-start)/1000,
        firstJudgmentFromProcessStartSeconds:passes[0].recordedAt-epoch,...reply,
        rawPasses:passes.map(p=>p.reply),memory:memory(directory,child.pid)};
}
async function warm(variant,workload) {
    const directory=join(traceRoot,`${workload.id}-warm-${variant}`);mkdirSync(directory,{recursive:true});
    const child=spawn(join(temporary,variant+'-worker'),[join(model,'model.gguf'),join(model,'mmproj.gguf')],
        {env:environment(variant,directory,join(directory,'unused')),stdio:['pipe','pipe','pipe']});
    let observedGatedRMSPipeline=false;
    child.stderr.on('data',bytes=>{if(bytes.toString().includes('kernel_rms_norm_gate'))observedGatedRMSPipeline=true;});
    const input=workload.files.map(p=>readFileSync(p).toString('base64'));
    const lines=createInterface({input:child.stdout});let waiting=[],queued=[];
    lines.on('line',line=>{try {const object=JSON.parse(line);if(waiting.length)waiting.shift().resolve(object);else queued.push(object);}catch(e){if(waiting.length)waiting.shift().reject(e);}});
    let ended=false;const exit=new Promise(done=>child.on('exit',code=>{ended=true;for(const w of waiting)w.reject(Error(`worker ended ${code}`));waiting=[];done(code);}));
    async function next(){if(queued.length)return queued.shift();if(ended)throw Error('worker ended');return new Promise((resolve,reject)=>waiting.push({resolve,reject}));}
    const timer=setTimeout(()=>child.kill('SIGTERM'),660000);
    const ready=await next();if(!ready.ready)throw Error('Worker failed initialization');
    const rounds=[];
    for(let round=0;round<2;round++) {
        const start=performance.now();const passes=[];
        if(ready.pairedComparison) {
            child.stdin.write(JSON.stringify({images:input,compareBoth:true})+'\n');passes.push(await next(),await next());
        }else{
            for(const images of [input,input.toReversed()]) {child.stdin.write(JSON.stringify({images})+'\n');passes.push(await next());}
        }
        if(passes.some(p=>p.error || !p.judgment))throw Error(`Warm ${variant} failed: ${JSON.stringify(passes)}`);
        rounds.push({round,seconds:(performance.now()-start)/1000,passes});
    }
    child.stdin.end();const code=await exit;clearTimeout(timer);lines.close();if(code!==0)throw Error('Warm worker failed');
    return {variant,rounds,observedGatedRMSPipeline,memory:memory(directory)};
}
const crop=spawnSync('sips',['--cropToHeightWidth','336','448',join(temporary,'watermarked.png'),'--out',join(temporary,'asymmetric.png')],{encoding:'utf8'});
if(crop.status!==0)throw Error('Apple crop generation failed');
const workloads=[
    {id:'native448',files:[join(temporary,'clean.png'),join(temporary,'watermarked.png')]},
    {id:'overview',files:[join(root,'Tests/Fixtures/RealESRGAN/nasa-collins-896x1134.jpg'),join(temporary,'overview/watermarked.png')]},
    {id:'unequal-native-crops',files:[join(temporary,'clean.png'),join(temporary,'asymmetric.png')]}
];
const report={session:'one-final-comparative-evaluation',date:new Date().toISOString(),reference,
    optimizedSourceSHA256:hash(join(root,'Runtime/PhotoArena/main.mm')),patchSHA256:hash(join(root,'Runtime/PhotoArena/patches/0001-gated-rms-metal.patch')),
    originalWorkerSHA256:hash(join(temporary,'original-worker')),optimizedWorkerSHA256:hash(join(temporary,'optimized-worker')),
    hardware:os.cpus()[0]?.model,operatingSystem:os.release(),precision:'Unchanged official F16 weights; unchanged FP32 recurrent state and gated RMS math',
    protocol:'Unchanged prompt, 512 output tokens, batch/ubatch512, threads4, Flash Attention; optimized context fits same full prompt and output budget',
    instrumentation:'Identical DYLD observer in both clients and workers. TASK_VM_INFO lifetime physical peak, 10ms sampled compression and Metal currentAllocatedSize; scalar traces only. No purge, CPU affinity, GPU eviction, paging controls or precision overrides.',
    limitations:['Production ArenaJudge.swift clients with only path lookup redirected; pre-encoded app inputs. Full SwiftUI GUI is not launched or included. PNG encoding is outside timed judging.',
        'Process lifetime physical peak is queried during inference; Metal/compression are sampled, so brief allocation spikes may be missed.',
        'Peak client plus peak sequential worker is a conservative upper bound, not a simultaneous measured GUI process tree peak.',
        'OS/file/Metal compiler caches are left intact. Cold means a fresh client/model process, not a physically cold disk/cache.',
        'Warm comparisons reuse one native worker on both sides; shipped application releases its worker after each complete action.',
        'Exact outputs on these controlled pairs are not broad calibrated model-quality validation or a universal memory bound.',
        'ledgerSwapinsRaw retains the kernel counter without assuming its units; ANE placement is not requested or measured.'],workloads:[]};
let failure;
try {
    for(const workload of workloads) {
        console.log(`Final session: ${workload.id}`);
        const item={id:workload.id,inputSHA256:workload.files.map(hash),cold:[],warm:[]};report.workloads.push(item);
        for(let repeat=0;repeat<2;repeat++)for(const variant of repeat%2?['optimized','original']:['original','optimized'])item.cold.push(await cold(variant,workload,repeat));
        for(const variant of ['original','optimized'])item.warm.push(await warm(variant,workload));
        const before=item.cold.filter(x=>x.variant==='original'),after=item.cold.filter(x=>x.variant==='optimized');
        const originalWarm=item.warm.find(x=>x.variant==='original').rounds.at(-1),optimizedWarm=item.warm.find(x=>x.variant==='optimized').rounds.at(-1);
        // Compare objects by sorted JSON keys, since Foundation serialization order is unspecified.
        const stable=x=>JSON.stringify(x,Object.keys(x).sort());
        const exactJudgment=(a,b)=>stable(a)===stable(b);
        const stableWarmState=item.warm.every(w=>w.rounds[0].passes.every((p,i)=>exactJudgment(p.judgment,w.rounds[1].passes[i].judgment)));
        const exactPasses=before.every(b=>b.rawPasses.every((p,i)=>exactJudgment(p.judgment,after.find(a=>a.repeat===b.repeat).rawPasses[i].judgment))) &&
            originalWarm.passes.every((p,i)=>exactJudgment(p.judgment,optimizedWarm.passes[i].judgment));
        const bounds=variant=>Math.max(...item.cold.filter(x=>x.variant===variant).map(x=>x.memory.clientPlusWorkerPhysicalUpperBoundBytes));
        const workerPeak=variant=>Math.max(item.warm.find(x=>x.variant===variant).memory.workerPeakPhysicalBytes,...item.cold.filter(x=>x.variant===variant).map(x=>x.memory.workerPeakPhysicalBytes));
        const promptTokens=after[0].rawPasses.reduce((s,p)=>s+p.promptTokens,0);
        item.summary={originalMedianPairSeconds:median(before.map(x=>x.wallSeconds)),optimizedMedianPairSeconds:median(after.map(x=>x.wallSeconds)),
            originalMedianColdFirstJudgmentSeconds:median(before.map(x=>x.firstJudgmentFromProcessStartSeconds)),optimizedMedianColdFirstJudgmentSeconds:median(after.map(x=>x.firstJudgmentFromProcessStartSeconds)),
            originalWarmPairSeconds:originalWarm.seconds,optimizedWarmPairSeconds:optimizedWarm.seconds,
            originalPrefillTokensPerSecond:promptTokens/median(before.map(x=>x.result.prefillSeconds)),optimizedPrefillTokensPerSecond:promptTokens/median(after.map(x=>x.result.prefillSeconds)),
            prefillScope:'Same unchanged prompt-token count; prefill includes preprocessing/vision and multimodal decode, excludes optimized decoder context creation.',
            originalDecodeTokensPerSecond:median(before.map(x=>x.result.tokens/x.result.decodeSeconds)),optimizedDecodeTokensPerSecond:median(after.map(x=>x.result.tokens/x.result.decodeSeconds)),
            originalClientWorkerPhysicalBoundBytes:bounds('original'),optimizedClientWorkerPhysicalBoundBytes:bounds('optimized'),
            originalWorkerPeakPhysicalBytes:workerPeak('original'),optimizedWorkerPeakPhysicalBytes:workerPeak('optimized'),
            originalSteadyStateWorkerPhysicalBytes:median(originalWarm.passes.map(p=>p.physicalFootprintBytes)),optimizedSteadyStateWorkerPhysicalBytes:median(optimizedWarm.passes.map(p=>p.physicalFootprintBytes)),
            originalSampledPeakMetalBytes:Math.max(...item.cold.filter(x=>x.variant==='original').map(x=>x.memory.sampledWorkerPeakMetalAllocatedBytes),item.warm.find(x=>x.variant==='original').memory.sampledWorkerPeakMetalAllocatedBytes),
            optimizedSampledPeakMetalBytes:Math.max(...item.cold.filter(x=>x.variant==='optimized').map(x=>x.memory.sampledWorkerPeakMetalAllocatedBytes),item.warm.find(x=>x.variant==='optimized').memory.sampledWorkerPeakMetalAllocatedBytes),
            productionGatedRMSDispatchObserved:item.warm.find(x=>x.variant==='optimized').observedGatedRMSPipeline,
            exactRawJudgmentAgreement:exactPasses,stableWarmState,exactColdReconciledAgreement:before.every(b=>exactJudgment(b.result.judgment,after.find(a=>a.repeat===b.repeat).result.judgment))};
        item.accepted=item.summary.optimizedMedianPairSeconds<item.summary.originalMedianPairSeconds && bounds('optimized')<=bounds('original') && workerPeak('optimized')<=workerPeak('original') && item.summary.exactRawJudgmentAgreement && item.summary.exactColdReconciledAgreement && stableWarmState && item.summary.productionGatedRMSDispatchObserved;
        console.log(JSON.stringify({workload:item.id,...item.summary,accepted:item.accepted}));
    }
    report.accepted=report.workloads.every(x=>x.accepted);
    report.acceptanceScope='Shipped complete-action latency, worker physical peak, conservative headless client+worker bound, exact judgment/state parity and actual fused dispatch. Not uniform improvement of every warm latency metric or a GUI process-tree measurement.';
    report.allWarmLatencyCasesImproved=report.workloads.every(w=>w.summary.optimizedWarmPairSeconds<=w.summary.originalWarmPairSeconds);
    report.warmLatencyRegressions=report.workloads.filter(w=>w.summary.optimizedWarmPairSeconds>w.summary.originalWarmPairSeconds).map(w=>({workload:w.id,originalSeconds:w.summary.originalWarmPairSeconds,optimizedSeconds:w.summary.optimizedWarmPairSeconds,note:'Cause not established; no extra performance session selected a replacement result.'}));
    if(!report.accepted)failure=Error('At least one final acceptance gate failed; do not ship as successful');
}catch(error){failure=error;report.accepted=false;report.error=String(error);}
finally {
    writeFileSync(join(root,'docs/photoarena-optimization-final.json'),JSON.stringify(report,null,2)+'\n');
    // Keep the reference only on a failed gate so a responsible fix/reversion can
    // be validated without reconstructing or estimating the original baseline.
    if(report.accepted)rmSync(temporary,{recursive:true,force:true});
}
if(failure)throw failure;
