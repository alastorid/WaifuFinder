// Download an immutable release into a throwaway directory and execute its real model.
import { createHash } from 'node:crypto';
import { createWriteStream, createReadStream, readFileSync, writeFileSync } from 'node:fs';
import { mkdtemp, rm, open } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { resolve, join } from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
const root=resolve(import.meta.dirname,'..');
const release=JSON.parse(readFileSync(resolve(root,'tools/photoarena-release.json')));
const base=`https://huggingface.co/${release.repository}/resolve/${release.revision}/${release.pathPrefix}/`;
const directory=await mkdtemp(join(tmpdir(),'photoarena-release-'));
async function download(name,sha) {
    const file=join(directory,name);
    const url=base+name;
    const header=await fetch(url,{method:'HEAD'});
    if(!header.ok)throw Error(`Header failed for ${name}: ${header.status}`);
    const size=Number(header.headers.get('content-length'));
    if(size>64*1024*1024) {
        const handle=await open(file,'w');
        try {
            await handle.truncate(size);
            const chunkSize=32*1024*1024;
            let next=0,completed=0;
            async function lane() {
                while(next<size) {
                    const start=next;next+=chunkSize;
                    const end=Math.min(size,start+chunkSize)-1;
                    const response=await fetch(url,{headers:{Range:`bytes=${start}-${end}`}});
                    if(response.status!==206)throw Error('Published download does not support checked ranges');
                    let offset=start;
                    for await(const chunk of Readable.fromWeb(response.body)) {
                        if(offset+chunk.length>end+1)throw Error('Range exceeds declared length');
                        let written=0;
                        while(written<chunk.length) {
                            const result=await handle.write(chunk,written,chunk.length-written,offset+written);
                            if(result.bytesWritten<=0)throw Error('Range write failed');
                            written+=result.bytesWritten;
                        }
                        offset+=chunk.length;
                    }
                    if(offset!==end+1)throw Error('Truncated range');
                    completed+=end-start+1;
                    if(completed===size || Math.floor(completed/chunkSize)%8===0)console.log(name,Math.round(completed/size*100)+'%');
                }
            }
            const lanes=await Promise.allSettled(Array.from({length:4},lane));
            const failed=lanes.find(x=>x.status==='rejected');if(failed)throw failed.reason;
        }finally{await handle.close();}
    }else{
        const response=await fetch(url);
        if (!response.ok) throw Error(`Download failed for ${name}: ${response.status}`);
        await pipeline(Readable.fromWeb(response.body),createWriteStream(file));
    }
    const hash=createHash('sha256');
    for await(const chunk of createReadStream(file))hash.update(chunk);
    if(hash.digest('hex')!==sha)throw Error(`Checksum mismatch for ${name}`);
    return file;
}
try {
    const manifestFile=await download('manifest.json',release.manifestSHA256);
    const manifest=JSON.parse(readFileSync(manifestFile));
    for(const [name,sha] of Object.entries(manifest.fileSHA256)) {
        if(name.includes('/') || name.includes('..'))throw Error('Unexpected manifest path');
        await download(name,sha);
    }
    const response=await new Promise((resolveResult,reject)=>{
        const worker=spawn(resolve(root,'.native-runtime/waifu-photoarena'),[join(directory,'model.gguf'),join(directory,'mmproj.gguf')],{stdio:['pipe','pipe','ignore']});
        const timer=setTimeout(()=>{worker.kill();reject(Error('Native inference timed out'));},120000);
        let result;
        createInterface({input:worker.stdout}).on('line',line=>{
            try{
                const reply=JSON.parse(line);
                if(reply.ready) {
                    const images=[resolve(root,'Tests/Fixtures/RealESRGAN/nasa-collins-896x1134.jpg'),resolve(root,'.native-runtime/photoarena-fixtures/watermarked.png')].map(p=>readFileSync(p).toString('base64'));
                    worker.stdin.end(JSON.stringify({images})+'\n');
                } else if(reply.error) { worker.kill(); reject(Error(reply.error)); }
                else result=reply;
            }catch(e){worker.kill();reject(e);}
        });
        worker.on('error',reject);
        worker.on('exit',code=>{clearTimeout(timer);if(code!==0 || !result?.judgment)reject(Error('Published model inference failed'));else resolveResult(result);});
    });
    writeFileSync(resolve(root,'docs/photoarena-published-verification.json'),JSON.stringify({status:'passed',repository:release.repository,revision:release.revision,manifestSHA256:release.manifestSHA256,verifiedFiles:Object.keys(manifest.fileSHA256),nativeInference:response,qualityValidated:false},null,2)+'\n');
    console.log('Published checkpoint downloaded, checksummed and inferred successfully');
}finally{await rm(directory,{recursive:true,force:true});}
