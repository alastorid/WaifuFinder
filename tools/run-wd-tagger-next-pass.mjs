#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import cp from "node:child_process";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const label = process.argv[2] ?? "screen";
if (!/^[a-z0-9-]+$/.test(label)) throw Error("Invalid label");
const out = path.join(root, ".build/wd-native/next-pass", label);
if (fs.existsSync(out)) throw Error("Choose new label");
fs.mkdirSync(out, { recursive: true });
const exe = path.join(root, "Runtime/.build/release/waifu-wdtagger-validation");
const golden = fs.readFileSync(
  path.join(root, ".build/wd-native/tuning/fused-memory/fused8-768.bin"),
);
const requested = (process.env.WAIFU_NEXT_JOBS ?? "")
  .split(",")
  .filter(Boolean);
const jobs = [
  { name: "current-start" },
  { name: "compact448", compact: true },
  { name: "compact416", compact: true, memory: 416 },
  { name: "submit16", env: { MLX_MAX_MB_PER_BUFFER: "16" } },
  { name: "cache32-416", memory: 416, cache: 32 },
  { name: "tile64x64", tile: "64x64x32" },
  { name: "tile32x128", tile: "32x128x32" },
  { name: "tile64x128", tile: "64x128x32" },
  { name: "relaxed", tile: "relaxed32x64x32" },
  { name: "current-end" },
].filter((job) => requested.length === 0 || requested.includes(job.name));
const records = [];
for (const job of jobs) {
  const deadline = Date.now() + 600000;
  let system;
  while (true) {
    system = JSON.parse(
      cp.execFileSync("macmon", ["pipe", "-s", "1"], { encoding: "utf8" }),
    );
    const thermal = Number(
      cp.execFileSync(exe, ["--thermal"], { encoding: "utf8" }),
    );
    if (
      thermal === 0 &&
      system.temp.gpu_temp_avg <= 60 &&
      system.cpu_usage_pct <= 0.4
    )
      break;
    if (Date.now() > deadline) throw Error("Cooling gate expired");
    console.log(
      `Cooling for ${job.name}: ${system.temp.gpu_temp_avg.toFixed(1)} C`,
    );
    await new Promise((r) => setTimeout(r, 15000));
  }
  const report = path.join(out, job.name + ".json"),
    bin = path.join(out, job.name + ".bin"),
    log = fs.openSync(path.join(out, job.name + ".log"), "w");
  console.log("Running " + job.name);
  const args = [
    "--weights",
    path.join(root, ".build/wd-native/model.safetensors"),
    "--inputs",
    path.join(root, ".build/wd-native/inputs-rgba.bin"),
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
  const r = cp.spawnSync(exe, args, {
    env: { ...process.env, ...job.env },
    stdio: ["ignore", log, log],
  });
  fs.closeSync(log);
  if (r.status !== 0) throw Error(job.name + " failed");
  const bytes = fs.readFileSync(bin),
    x = JSON.parse(fs.readFileSync(report));
  const record = {
    configuration: job,
    ...x,
    probabilitiesByteIdentical: bytes.equals(golden),
    probabilitiesSHA256: crypto
      .createHash("sha256")
      .update(bytes)
      .digest("hex"),
    systemBefore: {
      timestamp: system.timestamp,
      gpuTemperatureC: system.temp.gpu_temp_avg,
      cpuScaledUsage: system.cpu_usage_pct,
    },
    systemAfter: JSON.parse(
      cp.execFileSync("macmon", ["pipe", "-s", "1"], { encoding: "utf8" }),
    ),
  };
  records.push(record);
  fs.writeFileSync(
    path.join(out, "comparison.json"),
    JSON.stringify(records, null, 2) + "\n",
  );
  console.log(
    `${job.name}: ${x.predictionFramesPerSecond.toFixed(3)} fps, ${(x.peakPhysicalFootprintBytes / 1e6).toFixed(1)} MB, exact=${record.probabilitiesByteIdentical}`,
  );
}
