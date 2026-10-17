import fs from 'node:fs';
import {spawn} from 'node:child_process';
import assert from 'node:assert/strict';
const root=process.cwd(),out=root+'/.build/wd-native/memory-pass',size=8*448*448*4;
const input=out+'/ipc-input.bin',fd=fs.openSync(input,'w+');fs.ftruncateSync(fd,size);
const worker=root+'/dist/WaifuFinder-native-wdtagger-v3.app/Contents/Resources/Runtime/waifu-wdtagger';
const log=fs.openSync(out+'/ipc-worker.log','w');
const child=spawn(worker,['--weights',root+'/.build/wd-native/model.safetensors','--input',input],{stdio:['pipe','pipe',log]});
const iterator=child.stdout[Symbol.asyncIterator]();let pending=Buffer.alloc(0);
const timeout=setTimeout(()=>{child.kill('SIGKILL');throw Error('Worker timeout');},120000);
async function read(n){while(pending.length<n){const x=await iterator.next();if(x.done)throw Error('Unexpected worker EOF');pending=Buffer.concat([pending,x.value]);}const result=pending.subarray(0,n);pending=pending.subarray(n);return result;}
try{
 assert.equal((await read(4)).toString(),'WDT3');
 const corpus=fs.readFileSync(root+'/.build/wd-native/inputs-rgba.bin');
 const results=[];
 for(let batch=0;batch<8;batch++){fs.writeSync(fd,corpus,batch*size,size,0);child.stdin.write(Buffer.from([8]));results.push(await read(8*10861*4));}
 fs.writeFileSync(out+'/ipc-probabilities.bin',Buffer.concat(results));
 assert.ok(Buffer.concat(results).equals(fs.readFileSync(root+'/.build/wd-native/tuning/fused-memory/fused8-768.bin')),'all 64 outputs match the validated prior fused path exactly');
 // Duplicate frame 63 into every slot and request a one-frame tail.
 const frame=corpus.subarray(63*size/8,64*size/8);for(let row=0;row<8;row++)fs.writeSync(fd,frame,0,frame.length,row*frame.length);
 child.stdin.write(Buffer.from([1]));const tail=await read(10861*4);assert.ok(tail.equals(results[7].subarray(7*10861*4)),'tail preserves row identity');
 child.stdin.end();const code=await new Promise(resolve=>child.on('exit',resolve));assert.equal(code,0);
 console.log(JSON.stringify({frames:64,allOutputsByteIdentical:true,outputBytes:Buffer.concat(results).length,tailPaddingExact:true,cleanEOF:true,worker},null,2));
}finally{clearTimeout(timeout);child.kill();fs.closeSync(fd);fs.closeSync(log);}
