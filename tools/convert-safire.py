#!/usr/bin/env python3
"""Extract the official SAFIRE checkpoint without importing PyTorch or NumPy.

The restricted unpickler permits ONLY the three constructors used by the pinned
upstream checkpoint. It never executes arbitrary checkpoint globals. Float32
storage bytes are preserved exactly; Swift prepares convolution layouts once.
This creates a candidate package, never a validated release or an install pin.
"""
import argparse
import collections
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import struct
import zipfile

UPSTREAM = 'ec2af6af8c789fb6359493a3e352f89c6fd8107f'

class FloatStorage:
    pass

class Storage:
    def __init__(self, key, count):
        self.key, self.count = key, count

class Tensor:
    def __init__(self, storage, offset, shape, strides, requires_grad, hooks, *metadata):
        self.storage, self.offset = storage, offset
        self.shape, self.strides = tuple(shape), tuple(strides)
        expected = 1
        for size, stride in reversed(list(zip(shape, strides))):
            if size > 1 and stride != expected:
                raise ValueError('noncontiguous checkpoint tensor: explicit conversion needed')
            expected *= size
        self.count = expected
        if offset < 0 or self.count + offset > storage.count:
            raise ValueError('invalid tensor storage view')

class CheckpointReader(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {('collections','OrderedDict'): collections.OrderedDict,
                   ('torch','FloatStorage'): FloatStorage,
                   ('torch._utils','_rebuild_tensor_v2'): Tensor}
        if (module, name) not in allowed:
            raise ValueError(f'checkpoint constructor refused: {module}.{name}')
        return allowed[module, name]

    def persistent_load(self, pid):
        if (not isinstance(pid,tuple) or len(pid) != 5 or pid[0] != 'storage'
                or pid[1] is not FloatStorage or not str(pid[2]).isdigit()
                or not isinstance(pid[4],int) or not 0 < pid[4] <= 200_000_000):
            raise ValueError('unsupported checkpoint storage')
        return Storage(str(pid[2]),pid[4])

def digest(path):
    with path.open('rb') as handle:
        value = hashlib.sha256()
        for chunk in iter(lambda: handle.read(8*1024*1024),b''):
            value.update(chunk)
        return value.hexdigest()

def convert(source, output):
    if output.exists():
        raise ValueError('destination exists; use a new versioned package directory')
    if digest(source) != '65bdf6880380162d57f77d71b3fd367c3f16aebca316f942a821da16e9225b12':
        raise ValueError('source checkpoint does not match the verified official SAFIRE release')
    with zipfile.ZipFile(source) as archive:
        name = next(n for n in archive.namelist() if n.endswith('/data.pkl'))
        prefix = name[:-len('data.pkl')]
        if archive.read(prefix+'byteorder') != b'little':
            raise ValueError('checkpoint must be little endian')
        state = CheckpointReader(io.BytesIO(archive.read(name))).load()
        tensors = {key.removeprefix('module.'): value for key,value in state['model'].items()}
        if not tensors or not all(isinstance(t,Tensor) for t in tensors.values()):
            raise ValueError('incomplete model state')
        header, position = {}, 0
        for key,tensor in sorted(tensors.items()):
            length = tensor.count * 4
            header[key] = dict(dtype='F32',shape=list(tensor.shape),data_offsets=[position,position+length])
            position += length
        encoded = json.dumps(header,separators=(',',':')).encode()
        encoded += b' ' * ((8-len(encoded)%8)%8)
        output.mkdir(parents=True)
        with (output/'model.safetensors').open('wb') as handle:
            handle.write(struct.pack('<Q',len(encoded))); handle.write(encoded)
            for key,tensor in sorted(tensors.items()):
                raw = archive.read(prefix+'data/'+tensor.storage.key)
                if len(raw) != tensor.storage.count*4:
                    raise ValueError('checkpoint storage length mismatch')
                handle.write(raw[tensor.offset*4:(tensor.offset+tensor.count)*4])
    metadata = dict(architecture='safire-vit-b-adaptor',upstreamRevision=UPSTREAM,
        sourceSHA256=digest(source),conversionVersion='safire-extract-f32-v1',
        runtimeRevision='safire-native-v1',preprocessingVersion='safire-opencv-linear-rgb-square-v1',
        inputShape=[1,1024,1024,3],embeddingShape=[1,64,64,256],pointsPerSide=16,
        weightLayout='pytorch-f32',weightsLicense='CC-BY-4.0',codeLicense='Apache-2.0',
        validationStatus='pending',tensorCount=len(tensors))
    (output/'architecture.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (output/'NOTICE').write_text('SAFIRE: Segment Any Forged Image Region (AAAI 2025)\n'
        'Myung-Joon Kwon, Wonjun Lee, Seung-Hun Nam, Minji Son, Changick Kim.\n'
        'https://github.com/mjkwon2021/SAFIRE\n'
        'Weights: CC BY 4.0. Upstream code: Apache 2.0 (upstream update 2026-08-05).\n'
        'Native Swift adaptation by WaifuFinder (MIT); weight bytes retained as FP32.\n'
        'SAM components: Copyright Meta Platforms, Inc. and affiliates; Apache 2.0.\n'
        'ViTDet/MViT relative-position architecture: Meta Platforms, Inc.; Apache 2.0.\n')
    (output/'README.md').write_text('# SAFIRE Swift MLX candidate\n\n'
        'This package is NOT yet a validated release. No parity, accuracy, or performance claims are made.\n'
        'Source checkpoint SHA-256 and conversion provenance are in architecture.json.\n'
        'Binary localization and multi-source partitions do not establish chronology or editing software.\n'
        'Retain attribution and the CC BY 4.0 weight terms when redistributing converted weights.\n')
    hashes = {p.name:digest(p) for p in output.iterdir() if p.is_file()}
    (output/'manifest.json').write_text(json.dumps(dict(fileSHA256=hashes),indent=2)+'\n')
    print(json.dumps(metadata,indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path); parser.add_argument('output',type=Path)
    args = parser.parse_args()
    convert(args.checkpoint,args.output)
