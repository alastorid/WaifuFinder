#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import cp from "node:child_process";
import crypto from "node:crypto";

const root = path.resolve(process.argv[2] ?? ".");
const label = process.argv[3] ?? "app-prefetch";
if (!/^[a-z0-9-]+$/.test(label)) throw Error("Invalid label");
const out = path.join(root, "results", label);
if (fs.existsSync(out)) throw Error("Use a fresh label");
fs.mkdirSync(out, { recursive: true });
const telemetry = () =>
  JSON.parse(
    cp.execFileSync("/opt/homebrew/bin/macmon", ["pipe", "-s", "1"], {
      encoding: "utf8",
    }),
  );
const sha256 = (file) =>
  crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const probe = path.join(root, "bin-shared/app-benchmark");
const selected = process.argv[4] === "selected";
const jobs = selected
  ? [
      { name: "prefetch-copy-b", copy: true, prefetch: true },
      {
        name: "cached-prefetch-copy",
        copy: true,
        prefetch: true,
        cached: true,
      },
    ]
  : [
      { name: "serial-copy-a", copy: true },
      { name: "serial-shared-a" },
      { name: "prefetch-shared-a", prefetch: true },
      { name: "prefetch-copy-a", copy: true, prefetch: true },
      { name: "serial-copy-b", copy: true },
      { name: "prefetch-shared-b", prefetch: true },
      { name: "cached-serial-copy", copy: true, cached: true },
      { name: "cached-prefetch-shared", prefetch: true, cached: true },
    ];
if (process.argv[4] === "cached") {
  jobs.splice(0, jobs.length,
    { name: "cached-prefetch-copy-repeat", copy: true, prefetch: true, cached: true },
    { name: "cached-serial-copy-repeat", copy: true, cached: true },
  );
}
const records = [],
  references = new Map();
if (selected) {
  const prior = process.argv[5];
  if (!prior || !/^[a-z0-9-]+$/.test(prior))
    throw Error("Supply the prior matrix label");
  for (const record of JSON.parse(
    fs.readFileSync(path.join(root, "results", prior, "comparison.json")),
  )) {
    if (record.configuration.copy && !record.configuration.prefetch) {
      references.set(
        record.configuration.cached ? "cached" : "video",
        record.outputSHA256,
      );
    }
  }
  if (references.size !== 2)
    throw Error("Prior matrix must contain cached and video controls");
}
for (const job of jobs) {
  const deadline = Date.now() + 600000;
  let before;
  while (true) {
    before = telemetry();
    const thermal = Number(
      cp.execFileSync(
        path.join(root, "bin/waifu-wdtagger-validation"),
        ["--thermal"],
        { encoding: "utf8" },
      ),
    );
    const cpu =
      before.cpu_usage_pct ??
      Math.max(before.pcpu_usage[1], before.ecpu_usage[1]);
    if (thermal === 0 && before.temp.gpu_temp_avg <= 55 && cpu <= 0.4) break;
    if (Date.now() > deadline) throw Error("Cooling gate expired");
    console.log(
      `Cooling ${job.name}: ${before.temp.gpu_temp_avg.toFixed(1)} C`,
    );
    await new Promise((resolve) => setTimeout(resolve, 10000));
  }
  const worker = path.join(
    root,
    job.copy ? "bin-copy/waifu-wdtagger" : "bin-shared/waifu-wdtagger",
  );
  const report = path.join(out, job.name + ".json");
  const log = fs.openSync(path.join(out, job.name + ".log"), "w");
  console.log("Running " + job.name);
  const result = cp.spawnSync(
    probe,
    [
      "--model",
      path.join(root, "data"),
      "--worker",
      worker,
      "--report",
      report,
      ...(job.cached
        ? ["--rgba-input", path.join(root, "data/inputs-rgba.bin")]
        : [
            "--video",
            path.join(root, "data/tagging-source.mp4"),
            "--width",
            "512",
          ]),
      ...(job.prefetch ? ["--prefetch"] : []),
    ],
    {
      env: { ...process.env, WAIFU_WD_SHARED_INPUT: job.copy ? "0" : "1" },
      stdio: ["ignore", log, log],
    },
  );
  fs.closeSync(log);
  if (result.status !== 0) throw Error(job.name + " failed; inspect its log");
  const x = JSON.parse(fs.readFileSync(report));
  const key = job.cached ? "cached" : "video";
  if (!references.has(key)) references.set(key, x.outputSHA256);
  const record = {
    configuration: job,
    ...x,
    systemBefore: before,
    systemAfter: telemetry(),
    outputMatchesControl: references.get(key) === x.outputSHA256,
    probeSHA256: sha256(probe),
    workerSHA256: sha256(worker),
    sourceSHA256: sha256(
      path.join(
        root,
        job.cached ? "data/inputs-rgba.bin" : "data/tagging-source.mp4",
      ),
    ),
  };
  records.push(record);
  fs.writeFileSync(
    path.join(out, "comparison.json"),
    JSON.stringify(records, null, 2) + "\n",
  );
  if (!record.outputMatchesControl)
    throw Error("Output changed against control");
  console.log(
    `${job.name}: ${x.appPipelineFPS.toFixed(3)} fps, ${(x.memory.peakSimultaneousCombinedBytes / 1e6).toFixed(1)} MB combined, exact`,
  );
}
