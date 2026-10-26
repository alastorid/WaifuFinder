#!/usr/bin/env python3
"""Paired, end-to-end benchmark of the native TruFor worker (stdlib only).

Case file format: {"name": ["/absolute/image.png", ...]}. Each worker is warm
across sequential requests; image decoding, inference, response and serialization
are timed together. No interactive UI is used.
"""
import argparse, hashlib, json, os, platform, statistics, subprocess, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values, fraction):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    low, high = int(position), min(len(ordered) - 1, int(position) + 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def image_geometry(path):
    with open(path, "rb") as stream:
        header = stream.read(24)
    if len(header) == 24 and header[:8] == b"\x89PNG\r\n\x1a\n":
        import struct
        return {"width": struct.unpack_from(">I", header, 16)[0],
                "height": struct.unpack_from(">I", header, 20)[0]}
    return None


def request(proc, paths, output, mode):
    payload = json.dumps({"inputs": paths, "output": str(output), "mode": mode}, separators=(",", ":"))
    started = time.perf_counter()
    proc.stdin.write(payload + "\n")
    proc.stdin.flush()
    reply = proc.stdout.readline().strip()
    elapsed = time.perf_counter() - started
    if reply != "OK":
        raise RuntimeError("Worker request failed: " + reply)
    if mode == "scores":
        result = json.loads(output.read_text())
        scores = result.get("scores")
        if not isinstance(scores, list) or len(scores) != len(paths):
            raise RuntimeError("Worker returned an invalid score response")
    else:
        header = output.read_bytes()[:20]
        if len(header) != 20 or header[:4] != b"TFM1":
            raise RuntimeError("Worker returned invalid TFM1 evidence")
        width, height, maps = __import__("struct").unpack_from("<III", header, 4)
        if maps != 3 or output.stat().st_size != 20 + width * height * maps * 4:
            raise RuntimeError("Worker returned incomplete TFM1 evidence")
        scores = [__import__("struct").unpack_from("<f", header, 16)[0]]
    return elapsed, scores


def run_variant(name, row_positions, score_stream, cases, samples, evidence_samples, model, worker, root):
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("WAIFU_TRUFOR_"):
            del env[key]
    env.update({
        "WAIFU_TRUFOR_PRECISION": "fp32",
        "WAIFU_TRUFOR_ROW_POSITIONS": str(row_positions),
        "WAIFU_TRUFOR_SCORE_ONLY_STREAM": "1" if score_stream else "0",
        "WAIFU_TRUFOR_MEMORY_REPORT": "1",
    })
    log_path = root / (name + ".log")
    results, memories, model_memory = [], [], None
    with log_path.open("w") as log:
        proc = subprocess.Popen([str(worker), "--model", str(model), "--serve"], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=log, text=True, env=env)
        started = time.perf_counter()
        banner = proc.stdout.readline().strip()
        initialization = time.perf_counter() - started
        if banner != "TRF1":
            proc.kill()
            raise RuntimeError(f"Worker {name} did not start: {banner!r}")
        try:
            for case, paths in cases.items():
                if len(paths) > 4 or not paths:
                    raise ValueError(f"Case {case} must contain 1-4 image paths")
                score_times, score_values = [], []
                for sample in range(samples):
                    output = root / f"{name}-{case}-score-{sample}.json"
                    elapsed, values = request(proc, paths, output, "scores")
                    score_times.append(elapsed); score_values.append(values); output.unlink(missing_ok=True)
                row = {
                    "case": case, "mode": "score-only", "imagesPerRequest": len(paths),
                    "inputGeometry": [image_geometry(path) for path in paths],
                    "sampleSeconds": score_times, "coldFirstPredictionSeconds": score_times[0],
                    "warmP50Seconds": statistics.median(score_times[1:] or score_times),
                    "warmP95Seconds": percentile(score_times[1:] or score_times, .95),
                    "sustainedImagesPerSecond": len(paths) * len(score_times) / sum(score_times),
                    "scores": score_values[-1],
                }
                results.append(row)
                if len(paths) == 1 and evidence_samples:
                    evidence_times, evidence_scores = [], []
                    for sample in range(evidence_samples):
                        output = root / f"{name}-{case}-evidence-{sample}.bin"
                        elapsed, values = request(proc, paths, output, "evidence")
                        evidence_times.append(elapsed); evidence_scores.append(values[0]); output.unlink(missing_ok=True)
                    results.append({
                        "case": case, "mode": "evidence", "imagesPerRequest": 1,
                        "inputGeometry": [image_geometry(paths[0])],
                        "sampleSeconds": evidence_times, "coldFirstPredictionSeconds": evidence_times[0],
                        "warmP50Seconds": statistics.median(evidence_times[1:] or evidence_times),
                        "warmP95Seconds": percentile(evidence_times[1:] or evidence_times, .95),
                        "sustainedImagesPerSecond": len(evidence_times) / sum(evidence_times),
                        "scores": evidence_scores,
                    })
            proc.stdin.close()
            if proc.wait(timeout=300) != 0:
                raise RuntimeError(f"Worker {name} exited with status {proc.returncode}")
        finally:
            if proc.poll() is None:
                proc.kill(); proc.wait()
    for line in log_path.read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("phase") == "request-memory": memories.append(event)
        elif event.get("phase") == "model-memory": model_memory = event
    for row, memory in zip(results, memories):
        row["memory"] = memory
    return {
        "name": name, "backend": "mlx-metal", "precision": "fp32",
        "rowPositions": row_positions, "scoreOnlyStream": score_stream,
        "workerInitializationSeconds": initialization, "modelMemory": model_memory,
        "peakPhysicalBytes": max((x.get("peakPhysicalBytes", 0) for x in memories), default=0),
        "requests": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--worker", type=Path, default=ROOT / ".native-runtime/waifu-trufor")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/trufor-worker-benchmark.json")
    parser.add_argument("--variants", default="16384-stream,16384-legacy,32768-stream,65536-stream")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--evidence-samples", type=int, default=3)
    args = parser.parse_args()
    if args.samples < 1 or args.evidence_samples < 0:
        parser.error("sample counts must be positive (evidence may be zero)")
    cases = json.loads(args.cases.read_text())
    if not cases or not all(isinstance(v, list) for v in cases.values()):
        parser.error("cases must be a JSON object of image-path arrays")
    with tempfile.TemporaryDirectory(prefix="trufor-benchmark-") as directory:
        root = Path(directory)
        variants = []
        for item in args.variants.split(","):
            positions, mode = item.split("-", 1)
            if int(positions) not in (8_192, 16_384, 32_768, 65_536) or mode not in ("stream", "legacy"):
                parser.error("variants use 8192|16384|32768|65536-stream|legacy")
            variants.append(run_variant(item, int(positions), mode == "stream", cases,
                                        args.samples, args.evidence_samples, args.model.resolve(),
                                        args.worker.resolve(), root))
    record = {
        "measuredDate": time.strftime("%Y-%m-%d"), "machine": platform.platform(),
        "macOS": subprocess.check_output(["sw_vers", "-productVersion"], text=True).strip(),
        "chip": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        "workerSHA256": sha256(args.worker), "modelSHA256": sha256(args.model / "model.safetensors"),
        "precision": "fp32", "requestTiming": "warm TRF1 worker; decode through response and serialization",
        "variants": variants,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "chip": record["chip"],
                      "variants": [{"name": v["name"], "initializationSeconds": round(v["workerInitializationSeconds"], 3),
                                   "peakPhysicalBytes": v["peakPhysicalBytes"],
                                   "requests": [{"case": r["case"], "mode": r["mode"], "warmP50Seconds": round(r["warmP50Seconds"], 3),
                                                "warmP95Seconds": round(r["warmP95Seconds"], 3)} for r in v["requests"]]}
                                  for v in variants]}, indent=2))


if __name__ == "__main__":
    main()
