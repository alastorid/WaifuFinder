#!/usr/bin/env python3
"""Offline CPU-only calibration and W8A16/W8A8 export for either RRDB model.

Exported models are experiments: neither conversion nor file-size reduction
constitutes runtime/quality acceptance. Compare native ANE results to FP16.
"""
import argparse
import inspect
import json
from pathlib import Path
import coremltools as ct
import coremltools.optimize.coreml as cto
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--precision', choices=['w8a16','w8a8'], required=True)
    p.add_argument('--fixtures', type=Path, help='Representative planar FP32 calibration binaries and cases.json')
    p.add_argument('--size', type=int)
    p.add_argument('--group-size', type=int, default=128)
    a = p.parse_args()
    model = ct.models.MLModel(str(a.model),compute_units=ct.ComputeUnit.CPU_ONLY,skip_model_load=True)
    samples = []
    if a.precision == 'w8a8':
        if not a.fixtures: p.error('W8A8 requires representative calibration --fixtures')
        # Core ML Tools 9's debugger defaults to CPU_ONLY. Fail closed if a later
        # toolchain changes this; calibration must never choose a GPU implicitly.
        from coremltools.optimize.coreml.experimental._model_debugger import ModelDebugger
        for method in [ModelDebugger.predict_intermediate_outputs, ModelDebugger.step]:
            assert inspect.signature(method).parameters['compute_units'].default == ct.ComputeUnit.CPU_ONLY
        cases = json.loads((a.fixtures/'cases.json').read_text())
        for item in cases:
            size = item['size']
            if a.size and size != a.size: continue
            data = np.fromfile(a.fixtures/(item['name']+'-input.bin'),dtype=np.float32)
            assert data.size == 3*size*size and np.isfinite(data).all()
            samples.append({'input':data.reshape(1,3,size,size).astype(np.float16)})
        assert len(samples) >= 3, 'At least three calibration examples are required'
        config = cto.OptimizationConfig(global_config=cto.experimental.OpActivationLinearQuantizerConfig(mode='linear_symmetric'))
        model = cto.experimental.linear_quantize_activations(model, config, samples, calibration_op_group_size=a.group_size)
    config = cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(mode='linear_symmetric',dtype=np.int8,granularity='per_channel',weight_threshold=512))
    model = cto.linear_quantize_weights(model,config)
    model.user_defined_metadata.update({'waifu.validated':'false','waifu.precision':a.precision,'waifu.computeUnits':'ane'})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    model.save(str(a.output))
    a.output.with_suffix('.json').write_text(json.dumps(dict(precision=a.precision,calibrationSamples=len(samples),calibrationComputeUnits='cpuOnly',calibrationGroupSize=a.group_size,coremltools=ct.__version__,validated=False),indent=2)+'\n')

if __name__ == '__main__': main()
