// New hypothesis: native-resolution 448px ROI prevents small-overlay dilution.
// Node coordinates native Swift fixture generation and native model pipes only.
import { spawn, spawnSync } from 'node:child_process';
import { readFileSync, writeFileSync, mkdtempSync, rmSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { resolve, join } from 'node:path';
import { createHash } from 'node:crypto';
import os from 'node:os';
const root = resolve(import.meta.dirname, '..');
const temporary = mkdtempSync(join(os.tmpdir(), 'photoarena-roi-'));
const hash = data => createHash('sha256').update(data).digest('hex');
let worker;
try {
    const sdk = spawnSync('xcrun', ['--sdk', 'macosx', '--show-sdk-path'], { encoding: 'utf8' });
    if (sdk.status !== 0) throw Error('Cannot locate Apple SDK');
    const fixture = spawnSync('xcrun', ['swift', '-sdk', sdk.stdout.trim(), resolve(root, 'tools/fixtures-photoarena-roi.swift'),
        resolve(root, 'Tests/Fixtures/RealESRGAN/nasa-collins-896x1134.jpg'), temporary], { encoding: 'utf8' });
    if (fixture.status !== 0) throw Error(fixture.stderr || 'Native fixture generation failed');
    const images = ['clean.png', 'watermarked.png'].map(name => readFileSync(join(temporary, name)));
    const started = performance.now();
    const results = [];
    let loadSeconds = null, requestStart = null, parseError = null;
    worker = spawn(resolve(root, '.native-runtime/waifu-photoarena'),
        [resolve(root, 'models/photoarena/model.gguf'), resolve(root, 'models/photoarena/mmproj.gguf')], { stdio: ['pipe', 'pipe', 'pipe'] });
    // Drain verbose compiler logs without writing transient inference artifacts.
    worker.stderr.on('data', () => {});
    const timer = setTimeout(() => worker.kill('SIGTERM'), 300000);
    const lines = createInterface({ input: worker.stdout });
    lines.on('line', line => {
        try {
            const reply = JSON.parse(line);
            if (reply.ready) {
                loadSeconds = (performance.now() - started) / 1000;
                requestStart = performance.now();
                worker.stdin.write(JSON.stringify({ images: images.map(data => data.toString('base64')) }) + '\n');
            } else {
                const reversed = results.length === 1;
                results.push({ order: reversed ? ['watermarked', 'clean'] : ['clean', 'watermarked'],
                    expectedCleaner: reversed ? 'B' : 'A', wallSeconds: (performance.now()-requestStart)/1000,
                    processSwapIns: null, sampleScope: 'whole worker process end-of-request, not peak and not whole app', ...reply });
                if (results.length === 1 && !reply.error) {
                    requestStart = performance.now();
                    worker.stdin.write(JSON.stringify({ images: images.toReversed().map(data => data.toString('base64')) }) + '\n');
                } else worker.stdin.end();
            }
        } catch (error) { parseError = String(error); worker.kill('SIGTERM'); }
    });
    const exit = await new Promise(done => { worker.on('error', error => done({ code: null, error: String(error) })); worker.on('exit', (code, signal) => done({ code, signal })); });
    clearTimeout(timer); lines.close();
    const valid = exit.code === 0 && results.length === 2 && results.every(result => !result.error) && !parseError;
    const observed = results.map(result => ({ order: result.order, expected: result.expectedCleaner,
        watermarkWinner: result.judgment?.watermark ?? null, overallWinner: result.judgment?.recommended_source ?? null,
        watermarkCorrect: result.judgment?.watermark === result.expectedCleaner }));
    const reverse = outcome => outcome === 'A' ? 'B' : outcome === 'B' ? 'A' : outcome;
    const reconciled = {};
    for (const category of ['watermark', 'authenticity', 'detail', 'quality', 'consistency', 'recommended_source']) {
        const a = results[0]?.judgment?.[category], b = reverse(results[1]?.judgment?.[category]);
        reconciled[category] = a && a === b ? a : 'unknown';
    }
    reconciled.uncertain = results.some(result => result.judgment?.uncertain) || Object.values(reconciled).includes('unknown');
    const report = {
        experiment: 'native448-watermark-visibility', date: new Date().toISOString(),
        hypothesis: 'Native-resolution 448px corresponding crops reduce overlay dilution relative to whole-image overviews; this experiment tests recognition, not authenticity.',
        fixture: JSON.parse(readFileSync(join(temporary, 'fixture.json'), 'utf8')),
        inputSHA256: images.map((data, index) => ({ source: index === 0 ? 'clean' : 'watermarked', sha256: hash(data) })),
        nativeSourceSHA256: hash(readFileSync(resolve(root, 'Runtime/PhotoArena/main.mm'))),
        fixtureGeneratorSHA256: hash(readFileSync(resolve(root, 'tools/fixtures-photoarena-roi.swift'))),
        checkpoint: '345adbfa02c4a4671e61b7c75502dc9bf24fa93d', runtime: '7071ba378afa96dcd69c41e1ec2b4912364dbee4',
        hardware: os.cpus()[0]?.model, operatingSystem: os.release(), execution: 'native llama.cpp Metal/CPU',
        loadSeconds, wallSeconds: (performance.now()-started)/1000,
        workerReuse: 'one loaded worker for both orders; app currently reloads each order',
        status: valid ? 'execution-passed' : 'execution-failed', exit, parseError, observed, reconciled, results,
        conclusion: !valid ? 'Execution did not complete; no quality conclusion is supported.' :
            observed.every(result => result.watermarkCorrect) ? 'The clean source won the watermark category in both orders for this single controlled sample; no general accuracy is established.' :
            'Native crop did not resolve watermark category reversal bias in this controlled sample. Overall and category outcomes must be inspected separately; disagreeing categories abstain after reversal.',
        limitations: ['Single controlled real-image crop pair; not a calibrated accuracy dataset.',
            'Overlay style and size differ from the earlier whole-image test; no controlled causal speed/accuracy comparison is claimed.',
            'Footprint and compression are whole-worker end samples, not peak whole-app measurements.',
            'Process swap-ins are unavailable from this worker protocol.', 'No Core ML export, FP16 tensor parity or ANE operation placement is claimed.',
            'Reversal agreement is not proof of original detail or authenticity.'],
        anePlacement: 'not measured; backend is Metal/CPU', fp16ReferenceParity: 'not measured',
        qualityValidated: false
    };
    writeFileSync(resolve(root, 'docs/photoarena-native-roi-validation.json'), JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify({ status: report.status, observed, loadSeconds, results: results.map(({ wallSeconds, physicalFootprintBytes, compressedBytes }) => ({ wallSeconds, physicalFootprintBytes, compressedBytes })) }));
    if (!valid) process.exitCode = 1;
} finally {
    if (worker && worker.exitCode === null) worker.kill('SIGTERM');
    rmSync(temporary, { recursive: true, force: true });
}
