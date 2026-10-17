#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import cp from "node:child_process";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), ".."),
  args = process.argv.slice(2);
const option = (name, fallback) => {
  const i = args.indexOf(name);
  return i < 0 ? fallback : args[i + 1];
};
const label = option("--label", "compact"),
  tile = option("--tile", "32x64x32"),
  compact = args.includes("--compact");
if (!/^[a-z0-9-]+$/.test(label) || !/^[a-z0-9-]+$/.test(tile))
  throw Error("Invalid label/tile");
const out = path.join(root, ".build/wd-native/next-pass", label);
if (fs.existsSync(out)) throw Error("Use fresh label");
fs.mkdirSync(out, { recursive: true });
const exe = path.join(root, "Runtime/.build/release/waifu-wdtagger-validation");
const deadline = Date.now() + 600000;
let before;
while (true) {
  before = JSON.parse(
    cp.execFileSync("macmon", ["pipe", "-s", "1"], { encoding: "utf8" }),
  );
  const thermal = Number(
    cp.execFileSync(exe, ["--thermal"], { encoding: "utf8" }),
  );
  if (
    thermal === 0 &&
    before.temp.gpu_temp_avg <= 60 &&
    before.cpu_usage_pct <= 0.4
  )
    break;
  if (Date.now() > deadline) throw Error("Cooling gate expired");
  console.log(`Cooling: ${before.temp.gpu_temp_avg.toFixed(1)} C`);
  await new Promise((r) => setTimeout(r, 15000));
}
const report = path.join(out, "comparison.json"),
  prefix = path.join(out, "probabilities"),
  log = fs.openSync(path.join(out, "prediction.log"), "w");
console.log(`Comparing current, ${tile}, compact=${compact}, Core ML`);
const r = cp.spawnSync(
  exe,
  [
    "--weights",
    path.join(root, ".build/wd-native/model.safetensors"),
    "--inputs",
    path.join(root, ".build/wd-optimization/inputs.bin"),
    "--compare-coreml",
    path.join(root, ".build/wd-native/memory-pass/coreml-model.mlmodelc"),
    "--compare-current",
    "--gemm-tile",
    tile,
    "--memory-mb",
    "448",
    "--cache-mb",
    "64",
    "--repeats",
    option("--rounds", "3"),
    "--report",
    report,
    "--probabilities",
    prefix,
    ...(compact ? ["--compact-bias"] : []),
  ],
  { stdio: ["ignore", log, log] },
);
fs.closeSync(log);
if (r.status !== 0) throw Error("Comparison failed; inspect prediction.log");
const x = JSON.parse(fs.readFileSync(report));
const reference = fs.readFileSync(
    path.join(root, ".build/wd-native/tuning/fused-memory/fused8-768.bin"),
  ),
  coreml = fs.readFileSync(
    path.join(root, ".build/wd-native/tuning/coreml/coreml8.bin"),
  );
x.configuration = { tile, compact, memoryMiB: 448, cacheMiB: 64 };
x.systemBefore = {
  timestamp: before.timestamp,
  gpuTemperatureC: before.temp.gpu_temp_avg,
  cpuScaledUsage: before.cpu_usage_pct,
};
x.systemAfter = JSON.parse(
  cp.execFileSync("macmon", ["pipe", "-s", "1"], { encoding: "utf8" }),
);
x.exactParity = {
  current: fs.readFileSync(prefix + "-original.bin").equals(reference),
  candidate: fs.readFileSync(prefix + "-native.bin").equals(reference),
  coreml: fs.readFileSync(prefix + "-coreml.bin").equals(coreml),
};
x.batchesWonAgainstCurrent = x.pairs.filter(
  (p) => p.nativeSeconds < p.originalSeconds,
).length;
x.batchesWonAgainstCoreML = x.pairs.filter(
  (p) => p.nativeSeconds < p.coremlSeconds,
).length;
x.binarySHA256 = crypto
  .createHash("sha256")
  .update(fs.readFileSync(exe))
  .digest("hex");
fs.writeFileSync(report, JSON.stringify(x, null, 2) + "\n");
console.log(
  JSON.stringify(
    {
      current: x.originalPredictionFPS,
      candidate: x.nativePredictionFPS,
      coreml: x.coremlPredictionFPS,
      gainOverCurrent: x.nativeOverOriginalSpeedRatio,
      winsAgainstCurrent: x.batchesWonAgainstCurrent,
      batches: x.pairs.length,
      exact: x.exactParity,
    },
    null,
    2,
  ),
);
