#!/usr/bin/env node
// Stage the optimized native WD Tagger bundle for Hugging Face, next to the
// restoration bundles. The release descriptor is not written here: the Hub
// revision only exists after the first push, so tools/upload-hf-bundles.py
// reads it back and writes tools/wd-tagger-release.json.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
const root=path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const args=process.argv.slice(2);
const option=name=>{const i=args.indexOf(name);return i<0?undefined:args[i+1];};
const weights=option('--weights'),labels=option('--labels'),license=option('--license');
const output=option('--output')??path.join(root,'models/published/WD-SwinV2-Tagger-v3-Swift-MLX');
if(!weights||!labels||!license)throw Error('Use --weights prepared.safetensors --labels labels.csv --license upstream-LICENSE [--output staging-directory]');
fs.mkdirSync(output,{recursive:true});
for(const [source,dest] of [[weights,'model.safetensors'],[labels,'wd-tagger-labels.csv'],[license,'LICENSE'],
 [path.join(root,'Runtime/WDTagger-NOTICE.md'),'NOTICE'],[path.join(root,'docs/WDTagger-native-model-card.md'),'README.md'],
 [path.join(root,'docs/wd-tagger-native-tuning.json'),'validation.json']])fs.copyFileSync(source,path.join(output,dest));
const hash=name=>crypto.createHash('sha256').update(fs.readFileSync(path.join(output,name))).digest('hex');
const manifest={algorithm:'wd-swinv2-tagger-v3',format:'swinv2-native-metal32-v1',implementation:'swinv2-native-metal32-fused-qkv-v2',batchSize:8,inputShape:[8,3,448,448],outputShape:[8,10861],sourceWeightsSHA256:'39fe04000d5907a0a7616df54af744d43fbd70b5fba7abc3135d3675734c03ef',fileSHA256:Object.fromEntries(['model.safetensors','wd-tagger-labels.csv'].map(n=>[n,hash(n)]))};
fs.writeFileSync(path.join(output,'manifest.json'),JSON.stringify(manifest,null,2)+'\n');
console.log(`Staged ${output}. Verify native inference, then publish with:\n  tools/upload-hf-bundles.py --only WD-SwinV2-Tagger-v3-Swift-MLX`);
