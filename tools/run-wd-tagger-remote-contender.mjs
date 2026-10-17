#!/usr/bin/env node
// Run in the isolated remote benchmark directory after remote-matrix has completed.
import fs from "node:fs";
import path from "node:path";
import cp from "node:child_process";
import crypto from "node:crypto";

const args = process.argv.slice(2);
const option = (name, fallback) => {
  const i = args.indexOf(name);
  return i < 0 ? fallback : args[i + 1];
};
const root = path.resolve(option("--root", "."));
const label = option("--label", "compact-lut");
const bundle = option("--bundle", "bin-lut");
const tile = option("--tile", "32x64x32");
for (const value of [label, bundle, tile]) {
  if (!/^[a-z0-9-]+$/.test(value))
    throw Error("Invalid label, bundle, or tile");
}
const out = path.join(root, "results", label);
if (fs.existsSync(out)) throw Error("Use a fresh result label");
fs.mkdirSync(out, { recursive: true });
const exe = path.join(root, bundle, "waifu-wdtagger-validation");
const telemetry = () =>
  JSON.parse(
    cp.execFileSync("/opt/homebrew/bin/macmon", ["pipe", "-s", "1"], {
      encoding: "utf8",
    }),
  );
const deadline = Date.now() + 600000;
let before;
while (true) {
  before = telemetry();
  const thermal = Number(
    cp.execFileSync(exe, ["--thermal"], { encoding: "utf8" }),
  );
  const cpu =
    before.cpu_usage_pct ??
    Math.max(before.pcpu_usage[1], before.ecpu_usage[1]);
  if (thermal === 0 && before.temp.gpu_temp_avg <= 55 && cpu <= 0.4) break;
  if (Date.now() > deadline) throw Error("Cooling gate expired");
  console.log(`Cooling: ${before.temp.gpu_temp_avg.toFixed(1)} C`);
  await new Promise((resolve) => setTimeout(resolve, 10000));
}
const report = path.join(out, "comparison.json");
const prefix = path.join(out, "probabilities");
const log = fs.openSync(path.join(out, "prediction.log"), "w");
const compact = !args.includes("--dense");
const result = cp.spawnSync(
  exe,
  [
    "--weights",
    path.join(root, "data/model.safetensors"),
    "--inputs",
    path.join(root, "data/inputs.bin"),
    "--compare-coreml",
    path.join(root, "data/coreml-model.mlmodelc"),
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
if (result.status !== 0)
  throw Error("Comparison failed; inspect prediction.log");
const x = JSON.parse(fs.readFileSync(report));
const baseline = fs.readFileSync(
  path.join(root, "results/screen/current-start.bin"),
);
const coreml = fs.readFileSync(
  path.join(root, "results/screen/coreml-start.bin"),
);
x.configuration = { bundle, tile, compact, memoryMiB: 448, cacheMiB: 64 };
x.systemBefore = before;
x.systemAfter = telemetry();
x.exactParity = {
  current: fs.readFileSync(prefix + "-original.bin").equals(baseline),
  candidate: fs.readFileSync(prefix + "-native.bin").equals(baseline),
  coreml: fs.readFileSync(prefix + "-coreml.bin").equals(coreml),
};
x.batchesWonAgainstCurrent = x.pairs.filter(
  (p) => p.nativeSeconds < p.originalSeconds,
).length;
x.binarySHA256 = crypto
  .createHash("sha256")
  .update(fs.readFileSync(exe))
  .digest("hex");
x.flashSourceSHA256 = crypto
  .createHash("sha256")
  .update(
    fs.readFileSync(
      path.join(
        root,
        bundle,
        "WaifuRuntime_WDTaggerMLX.bundle/Metal/Flash32.metal",
      ),
    ),
  )
  .digest("hex");
fs.writeFileSync(report, JSON.stringify(x, null, 2) + "\n");
console.log(
  JSON.stringify(
    {
      current: x.originalPredictionFPS,
      candidate: x.nativePredictionFPS,
      coreml: x.coremlPredictionFPS,
      gainOverCurrent: x.nativeOverOriginalSpeedRatio,
      wins: x.batchesWonAgainstCurrent,
      batches: x.pairs.length,
      exact: x.exactParity,
    },
    null,
    2,
  ),
);
