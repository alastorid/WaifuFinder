#!/usr/bin/env python3
"""Run native WDTagger probes with cooling between processes on a fanless Mac.

Requires macmon for temperature sampling. Each process uses precompiled models;
conversion, PyTorch reference work and other benchmarks must finish first.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--installed", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=["batch8", "batch1", "batch2", "batch4", "batch16", "batch32",
                                                               "rsqrt", "fold-input", "int8", "int4", "fast", "low-precision"])
    parser.add_argument("--round", type=int, default=2)
    parser.add_argument("--max-start-gpu-temperature", type=float, default=55)
    args = parser.parse_args()
    executable = str(args.executable.resolve())
    for index, variant in enumerate(args.variants):
        deadline = time.monotonic() + 600
        while True:
            sample = json.loads(subprocess.check_output(["macmon", "pipe", "-s", "1"], text=True))
            thermal = int(subprocess.check_output([executable, "--thermal"], text=True))
            temperature = sample["temp"]["gpu_temp_avg"]
            if thermal == 0 and temperature <= args.max_start_gpu_temperature:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("The machine did not cool to the requested starting condition in ten minutes")
            print(f"Cooling before {variant}: GPU {temperature:.1f} C, thermal state {thermal}", flush=True)
            time.sleep(15)
        baseline = variant in ["batch8", "fast", "low-precision", "all"]
        model = args.installed / "model.mlmodelc" if baseline else args.directory / variant / "model.mlmodelc"
        report = args.directory / f"{variant}-run{args.round}.json"
        command = [executable, "--model", str(model), "--labels", str(args.installed / "wd-tagger-labels.csv"),
                   "--images", str(args.directory / "images.json"), "--report", str(report), "--repeats", "3",
                   "--probability-output", str(args.directory / f"{variant}-probabilities.bin")]
        if variant == "fast": command.append("--fast-prediction")
        if variant == "low-precision": command.append("--low-precision-accumulation")
        if variant == "all": command.extend(["--units", "all"])
        print(f"Running {variant} at GPU {temperature:.1f} C", flush=True)
        with (args.directory / f"{variant}-run{args.round}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        record = json.loads(report.read_text())
        record["startGpuTemperatureC"] = temperature
        record["coolingMaxStartGpuTemperatureC"] = args.max_start_gpu_temperature
        report.write_text(json.dumps(record, indent=2) + "\n")
        print(f"{variant}: {record['framesPerSecond']:.2f} fps, {record['peakPhysicalFootprintBytes']/1e6:.1f} MB, "
              f"thermal {record['thermalBefore']} -> {record['thermalAfter']}", flush=True)
        if index + 1 < len(args.variants):
            time.sleep(15)


if __name__ == "__main__":
    main()
