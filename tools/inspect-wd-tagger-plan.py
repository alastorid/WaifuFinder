#!/usr/bin/env python3
"""Inspect Core ML compiler placement/cost estimates, not an execution trace."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan


def inspect(compiled):
    plan = MLComputePlan.load_from_path(str(compiled), compute_units=ct.ComputeUnit.CPU_AND_GPU)
    counts, devices = Counter(), Counter()
    costs = defaultdict(float)
    detailed = []
    missing_costs = 0
    for op in plan.model_structure.program.functions["main"].block.operations:
        counts[op.operator_name] += 1
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        device = type(usage.preferred_compute_device).__name__ if usage else "unassigned"
        devices[device] += 1
        cost = plan.get_estimated_cost_for_mlprogram_operation(op)
        if cost is None:
            missing_costs += 1
        else:
            costs[op.operator_name] += cost.weight
        if op.operator_name != "const":
            detailed.append({"operator": op.operator_name, "outputs": [v.name for v in op.outputs],
                             "preferredDevice": device, "estimatedCostWeight": cost.weight if cost else None})
    return {"operationCounts": counts, "preferredDeviceCounts": devices,
            "estimatedCostWeightByOperator": dict(sorted(costs.items(), key=lambda p: -p[1])),
            "operationsWithoutEstimatedCost": missing_costs,
            "highestEstimatedCosts": sorted(detailed, key=lambda p: p["estimatedCostWeight"] or 0, reverse=True)[:40]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", nargs=2, action="append", metavar=("NAME", "COMPILED"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = {"note": "Compiler estimates and preferred placement; not measured timings, GPU kernel counts or an allocation trace.",
              "models": {name: inspect(Path(path)) for name, path in args.model}}
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    for name, model in report["models"].items():
        print(name, model["estimatedCostWeightByOperator"])


if __name__ == "__main__":
    main()
