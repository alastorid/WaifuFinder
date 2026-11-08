// Native inference only: Node orchestrates pipes; no Python model execution.
import { spawn } from 'node:child_process';
import { readFileSync, writeFileSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { resolve } from 'node:path';
import { createHash } from 'node:crypto';
import os from 'node:os';
const root = resolve(import.meta.dirname, '..');
const worker = spawn(resolve(root, '.native-runtime/waifu-photoarena'),
    [resolve(root, 'models/photoarena/model.gguf'), resolve(root, 'models/photoarena/mmproj.gguf')],
    { stdio: ['pipe', 'pipe', 'inherit'] });
const watermark = process.argv.includes('--watermark');
const images = [resolve(root, 'Tests/Fixtures/RealESRGAN/nasa-collins-896x1134.jpg'),
                watermark ? resolve(root, '.native-runtime/photoarena-fixtures/watermarked.png') : resolve(root, 'Tests/Fixtures/RealESRGAN/nasa-collins-896x1134.jpg')]
    .map(path => readFileSync(path).toString('base64'));
const started = performance.now();
let requested = 0;
const results = [];
const timer = setTimeout(() => worker.kill('SIGTERM'), 300000);
createInterface({ input: worker.stdout }).on('line', line => {
    const reply = JSON.parse(line);
    if (reply.ready) {
        results.push({ loadSeconds: (performance.now() - started) / 1000, ...reply });
        worker.stdin.write(JSON.stringify({ images }) + '\n');
        requested++;
    } else {
        results.push(reply);
        console.log(JSON.stringify(reply));
        if (requested < 2 && !reply.error) {
            worker.stdin.write(JSON.stringify({ images: images.toReversed() }) + '\n');
            requested++;
        } else worker.stdin.end();
    }
});
worker.on('exit', code => {
    clearTimeout(timer);
    writeFileSync(resolve(root, watermark ? 'docs/photoarena-watermark-benchmark.json' : 'docs/photoarena-native-benchmark.json'), JSON.stringify({
        sourceSHA256: createHash('sha256').update(readFileSync(resolve(root, 'Runtime/PhotoArena/main.mm'))).digest('hex'),
        hardware: os.cpus()[0]?.model, operatingSystem: os.release(),
        qualityValidated: false,
        checkpoint: '345adbfa02c4a4671e61b7c75502dc9bf24fa93d',
        runtime: '7071ba378afa96dcd69c41e1ec2b4912364dbee4',
        fixture: watermark ? 'NASA Collins vs Apple-native synthetic watermark, then A/B reversed' : 'NASA Collins, identical-image pair repeated',
        status: code === 0 && results.length === 3 && results.every(r => !r.error) ? 'passed' : 'failed',
        anePlacement: 'not measured; Metal/CPU backend', results,
    }, null, 2) + '\n');
    process.exitCode = code === 0 && results.length === 3 && results.every(r => !r.error) ? 0 : 1;
});
