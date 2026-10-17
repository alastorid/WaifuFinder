#!/usr/bin/env node
// Run on the isolated M4 test host; bin-shared contains worker-benchmark,
// waifu-wdtagger, mlx.metallib and the WDTaggerMLX resource bundle.
import fs from "node:fs";
import path from "node:path";
import cp from "node:child_process";
import crypto from "node:crypto";

const root = path.resolve(process.argv[2] ?? ".");
const label = process.argv[3] ?? "shared-input";
if (!/^[a-z0-9-]+$/.test(label)) throw Error("Invalid label");
const out = path.join(root, "results", label);
if (fs.existsSync(out)) throw Error("Use a fresh label");
fs.mkdirSync(out, { recursive: true });
const worker = path.join(root, "bin-shared/waifu-wdtagger");
const telemetry = () =>
  JSON.parse(
    cp.execFileSync("/opt/homebrew/bin/macmon", ["pipe", "-s", "1"], {
      encoding: "utf8",
    }),
  );
const sha256 = (file) =>
  crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const records = [];
for (const name of [
  "copy-a",
  "shared-a",
  "shared-b",
  "copy-b",
  "copy-c",
  "shared-c",
]) {
  const copy = name.startsWith("copy");
  let before;
  const deadline = Date.now() + 600000;
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
    console.log(`Cooling ${name}: ${before.temp.gpu_temp_avg.toFixed(1)} C`);
    await new Promise((resolve) => setTimeout(resolve, 10000));
  }
  console.log("Running " + name);
  const report = path.join(out, name + ".json");
  const log = fs.openSync(path.join(out, name + ".log"), "w");
  const result = cp.spawnSync(
    path.join(root, "bin-shared/worker-benchmark"),
    [
      "--worker",
      worker,
      "--weights",
      path.join(root, "data/model.safetensors"),
      "--inputs",
      path.join(root, "data/inputs-rgba.bin"),
      "--reference",
      path.join(root, "results/screen/current-start.bin"),
      "--report",
      report,
      ...(copy ? ["--copy-input"] : ["--shared-input"]),
    ],
    { stdio: ["ignore", log, log] },
  );
  fs.closeSync(log);
  if (result.status !== 0) throw Error(name + " failed; inspect its log");
  const workerLog = fs.readFileSync(
    path.join(out, name + ".worker.log"),
    "utf8",
  );
  if (!workerLog.includes("WDTagger input: " + (copy ? "copy" : "shared")))
    throw Error("Unexpected input mode");
  const record = {
    name,
    ...JSON.parse(fs.readFileSync(report)),
    systemBefore: before,
    systemAfter: telemetry(),
    workerSHA256: sha256(worker),
    probeSHA256: sha256(path.join(root, "bin-shared/worker-benchmark")),
  };
  records.push(record);
  fs.writeFileSync(
    path.join(out, "comparison.json"),
    JSON.stringify(records, null, 2) + "\n",
  );
  console.log(
    `${name}: ${record.predictionAndIPCFPS.toFixed(3)} fps, ${(record.peakWorkerPhysicalFootprintBytes / 1e6).toFixed(1)} MB, exact`,
  );
}
