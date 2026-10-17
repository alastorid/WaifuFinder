#!/usr/bin/env node
// Run on the SSH target, inside an isolated folder populated with bin/ and data/.
import fs from "node:fs";
import path from "node:path";
import cp from "node:child_process";
import crypto from "node:crypto";
const root = path.resolve(process.argv[2] ?? "."),
  label = process.argv[3] ?? "screen";
if (!/^[a-z0-9-]+$/.test(label)) throw Error("Invalid label");
const out = path.join(root, "results", label);
if (fs.existsSync(out)) throw Error("Preserve existing run; use another label");
fs.mkdirSync(out, { recursive: true });
const native = path.join(root, "bin/waifu-wdtagger-validation");
const jobs = [
  { name: "coreml-start", coreml: true },
  { name: "current-start" },
  { name: "compact448", compact: true },
  { name: "compact416", compact: true, memory: 416 },
  { name: "cache32-416", memory: 416, cache: 32 },
  { name: "submit16", env: { MLX_MAX_MB_PER_BUFFER: "16" } },
  { name: "tile64x64", tile: "64x64x32" },
  { name: "tile32x128", tile: "32x128x32" },
  { name: "tile64x128", tile: "64x128x32" },
  { name: "relaxed", tile: "relaxed32x64x32" },
  { name: "current-end" },
  { name: "coreml-end", coreml: true },
];
const records = [];
let reference, coremlReference;
for (const job of jobs) {
  let before;
  const deadline = Date.now() + 600000;
  while (true) {
    before = JSON.parse(
      cp.execFileSync("/opt/homebrew/bin/macmon", ["pipe", "-s", "1"], {
        encoding: "utf8",
      }),
    );
    const thermal = Number(
      cp.execFileSync(native, ["--thermal"], { encoding: "utf8" }),
    );
    const cpu =
      before.cpu_usage_pct ??
      Math.max(before.pcpu_usage[1], before.ecpu_usage[1]);
    if (thermal === 0 && before.temp.gpu_temp_avg <= 55 && cpu <= 0.4) break;
    if (Date.now() > deadline) throw Error("Cooling gate expired");
    console.log(
      `Cooling ${job.name}: ${before.temp.gpu_temp_avg.toFixed(1)} C, CPU ${(cpu * 100).toFixed(0)}%`,
    );
    await new Promise((r) => setTimeout(r, 10000));
  }
  console.log("Running " + job.name);
  const report = path.join(out, job.name + ".json"),
    bin = path.join(out, job.name + ".bin"),
    log = fs.openSync(path.join(out, job.name + ".log"), "w");
  const args = job.coreml
    ? [
        "--model",
        path.join(root, "data/coreml-model.mlmodelc"),
        "--labels",
        path.join(root, "data/wd-tagger-labels.csv"),
        "--tensor-input",
        path.join(root, "data/inputs.bin"),
        "--stream-input",
        "--repeats",
        "3",
        "--report",
        report,
        "--probability-output",
        bin,
      ]
    : [
        "--weights",
        path.join(root, "data/model.safetensors"),
        "--inputs",
        path.join(root, "data/inputs-rgba.bin"),
        "--rgba-input",
        "--stream-input",
        "--frames",
        "64",
        "--batch",
        "8",
        "--repeats",
        "3",
        "--memory-mb",
        String(job.memory ?? 448),
        "--cache-mb",
        String(job.cache ?? 64),
        "--gemm-tile",
        job.tile ?? "32x64x32",
        "--report",
        report,
        "--probabilities",
        bin,
        ...(job.compact ? ["--compact-bias"] : []),
      ];
  const r = cp.spawnSync(
    job.coreml ? path.join(root, "bin/coreml-stream") : native,
    args,
    { env: { ...process.env, ...job.env }, stdio: ["ignore", log, log] },
  );
  fs.closeSync(log);
  if (r.status !== 0) throw Error(job.name + " failed");
  const x = JSON.parse(fs.readFileSync(report)),
    bytes = fs.readFileSync(bin);
  if (job.coreml && coremlReference === undefined) coremlReference = bytes;
  if (!job.coreml && reference === undefined) reference = bytes;
  const record = {
    configuration: job,
    ...x,
    probabilitiesByteIdenticalToDeviceBaseline: bytes.equals(
      job.coreml ? coremlReference : reference,
    ),
    outputSHA256: crypto.createHash("sha256").update(bytes).digest("hex"),
    systemBefore: before,
    systemAfter: JSON.parse(
      cp.execFileSync("/opt/homebrew/bin/macmon", ["pipe", "-s", "1"], {
        encoding: "utf8",
      }),
    ),
  };
  if (job.coreml)
    record.predictionFramesPerSecond =
      64 / x.stageSeconds.map((s) => s.prediction).sort((a, b) => a - b)[1];
  records.push(record);
  fs.writeFileSync(
    path.join(out, "comparison.json"),
    JSON.stringify(records, null, 2) + "\n",
  );
  console.log(
    `${job.name}: ${record.predictionFramesPerSecond.toFixed(3)} fps, ${(x.peakPhysicalFootprintBytes / 1e6).toFixed(1)} MB, exact=${record.probabilitiesByteIdenticalToDeviceBaseline}`,
  );
}
