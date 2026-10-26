#!/usr/bin/env node
// Capture full-precision outputs from the selected WD Tagger worker for a fixed
// RGBA corpus. The result can be passed to wd-tagger-worker-benchmark.swift.
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";

const options = new Map();
for (let i = 2; i < process.argv.length; i += 2) {
  if (!process.argv[i]?.startsWith("--") || !process.argv[i + 1]) {
    throw new Error("Use --worker PATH --weights PATH --inputs PATH --output PATH");
  }
  options.set(process.argv[i], process.argv[i + 1]);
}
for (const key of ["--worker", "--weights", "--inputs", "--output"]) {
  if (!options.has(key)) throw new Error(`Missing ${key}`);
}

const frameBytes = 448 * 448 * 4;
const rowBytes = 10861 * 4;
const batchSize = 8;
const batchBytes = frameBytes * batchSize;
const input = fs.readFileSync(options.get("--inputs"));
if (input.length === 0 || input.length % batchBytes !== 0) {
  throw new Error(`Input corpus must contain whole batches of ${batchSize} RGBA frames`);
}

const scratch = fs.mkdtempSync(path.join(os.tmpdir(), "wd-tagger-reference-"));
const inputPath = path.join(scratch, "input.bin");
const fd = fs.openSync(inputPath, "w+");
fs.ftruncateSync(fd, batchBytes);
const child = spawn(options.get("--worker"), [
  "--weights", options.get("--weights"), "--input", inputPath, "--copy-input",
], { stdio: ["pipe", "pipe", "inherit"] });

let buffer = Buffer.alloc(0);
const waiters = [];
child.stdout.on("data", (chunk) => {
  buffer = Buffer.concat([buffer, chunk]);
  while (waiters.length && buffer.length >= waiters[0].size) {
    const waiter = waiters.shift();
    const value = buffer.subarray(0, waiter.size);
    buffer = buffer.subarray(waiter.size);
    waiter.resolve(value);
  }
});
child.on("error", (error) => {
  while (waiters.length) waiters.shift().reject(error);
});
child.on("exit", (code) => {
  if (code !== 0) {
    const error = new Error(`WD Tagger worker exited with ${code}`);
    while (waiters.length) waiters.shift().reject(error);
  }
});
const read = (size) => new Promise((resolve, reject) => {
  waiters.push({ size, resolve, reject });
  while (waiters.length && buffer.length >= waiters[0].size) {
    const waiter = waiters.shift();
    const value = buffer.subarray(0, waiter.size);
    buffer = buffer.subarray(waiter.size);
    waiter.resolve(value);
  }
});

try {
  const magic = await read(4);
  if (magic.toString("ascii") !== "WDT4") throw new Error("Unexpected worker protocol header");
  const outputs = [];
  const batches = input.length / batchBytes;
  for (let batch = 0; batch < batches; batch += 1) {
    const bytes = input.subarray(batch * batchBytes, (batch + 1) * batchBytes);
    fs.writeSync(fd, bytes, 0, bytes.length, 0);
    child.stdin.write(Buffer.from([batchSize]));
    outputs.push(await read(batchSize * rowBytes));
  }
  child.stdin.end();
  const status = await new Promise((resolve) => child.once("close", resolve));
  if (status !== 0) throw new Error(`WD Tagger worker exited with ${status}`);
  const outputPath = path.resolve(options.get("--output"));
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, Buffer.concat(outputs));
  console.log(`Captured ${input.length / frameBytes} WD Tagger reference rows to ${outputPath}`);
} finally {
  fs.closeSync(fd);
  fs.rmSync(scratch, { recursive: true, force: true });
}
