#!/usr/bin/env python3
"""Create a LOCAL native TruFor candidate from the checksum-verified official checkpoint.

No Torch/NumPy imports or arbitrary pickle globals. No upload or redistribution.
Upstream code/weights are informational/nonprofit-only; retain all notices.
"""
import argparse,collections,hashlib,importlib.util,io,json,pickle,struct,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('safire_conversion',ROOT/'tools/convert-safire.py')
safe=importlib.util.module_from_spec(spec);spec.loader.exec_module(safe)
SOURCE_SHA='ac1d90e329a72e0d66e8665e123a19e94bfae3209c3ef8a4f9ca3b91578c7844'
REVISION='ae54475df6f41a491d7615100feb19263dec13f7'
class LongStorage:pass
class DType:
    def __init__(self,code,*unused):
        if code not in ['f8','f4','i8','i4']:raise ValueError('Unsupported numeric metadata dtype')
        self.code=code;self.endian='<'
    def __setstate__(self,state):
        if state[1] not in ['<','=','|']:raise ValueError('Unsupported metadata byte order')
def scalar(dtype,value):
    if not isinstance(dtype,DType) or len(value)!=int(dtype.code[1:]):raise ValueError('Invalid numeric metadata')
    return struct.unpack('<'+{'f8':'d','f4':'f','i8':'q','i4':'i'}[dtype.code],value)[0]
def encode(value,encoding):
    if encoding not in ['latin1','latin-1']:raise ValueError('Unsupported checkpoint byte string')
    return value.encode(encoding)
class Reader(safe.CheckpointReader):
    def find_class(self,module,name):
        additional={('torch','LongStorage'):LongStorage,('numpy','dtype'):DType,('numpy.core.multiarray','scalar'):scalar,('_codecs','encode'):encode}
        return additional[(module,name)] if (module,name) in additional else super().find_class(module,name)
    def persistent_load(self,pid):
        if not isinstance(pid,tuple) or len(pid)!=5 or pid[0]!='storage' or pid[1] not in [safe.FloatStorage,LongStorage] or not str(pid[2]).isdigit() or not isinstance(pid[4],int) or not 0<pid[4]<=200_000_000:raise ValueError('Invalid storage')
        value=safe.Storage(str(pid[2]),pid[4]);value.dtype='F32' if pid[1] is safe.FloatStorage else 'I64';value.width=4 if value.dtype=='F32' else 8
        return value

def convert(checkpoint,output,licenses):
    if output.exists():raise ValueError('Use a new output directory')
    if safe.digest(checkpoint)!=SOURCE_SHA:raise ValueError('Checkpoint does not match the checksum-verified official archive')
    with zipfile.ZipFile(checkpoint) as archive:
        name=next(n for n in archive.namelist() if n.endswith('/data.pkl'));prefix=name[:-8]
        state=Reader(io.BytesIO(archive.read(name))).load()
        tensors={k.removeprefix('module.'):v for k,v in state['state_dict'].items() if not k.endswith('num_batches_tracked')}
        if not tensors or not all(isinstance(v,safe.Tensor) and v.storage.dtype=='F32' for v in tensors.values()):raise ValueError('Unsupported inference tensors')
        header={};position=0
        for key,t in sorted(tensors.items()):
            length=t.count*4;header[key]=dict(dtype='F32',shape=list(t.shape),data_offsets=[position,position+length]);position+=length
        encoded=json.dumps(header,separators=(',',':')).encode();encoded+=b' '*((-len(encoded))%8)
        output.mkdir(parents=True)
        with (output/'model.safetensors').open('wb') as target:
            target.write(struct.pack('<Q',len(encoded)));target.write(encoded)
            for key,t in sorted(tensors.items()):
                raw=archive.read(prefix+'data/'+t.storage.key)
                if len(raw)!=t.storage.count*4:raise ValueError('Storage length mismatch')
                target.write(raw[t.offset*4:(t.offset+t.count)*4])
    metadata=dict(architecture='trufor-cmx-mit-b2',sourceSHA256=SOURCE_SHA,upstreamRevision=REVISION,conversionVersion='trufor-extract-f32-v1',runtimeRevision='trufor-native-v1',preprocessingVersion='trufor-native-rgb-div256-v1',tensorCount=len(tensors),inputPolicy='native oriented RGB; divide uint8 by 256; no resize/padding',detection='official confidence-weighted statistics plus trained 8-128-1 MLP and sigmoid',license='GRIP-UNINA informational/nonprofit only',distribution='LOCAL ONLY; no verified permission for public weight redistribution',validationStatus='pending')
    (output/'architecture.json').write_text(json.dumps(metadata,indent=2)+'\n')
    for name in ['LICENSE.txt','LICENSE_CMX.txt']:(output/name).write_bytes((licenses/name).read_bytes())
    (output/'NOTICE').write_text('TruFor: Leveraging All-Round Clues for Trustworthy Image Forgery Detection and Localization (CVPR 2023).\nFabrizio Guillaro, Davide Cozzolino, Avneesh Sud, Nicholas Dufour, Luisa Verdoliva.\nCopyright (c) 2023 GRIP-UNINA. Informational and nonprofit use only; preserve all notices.\nhttps://github.com/grip-unina/TruFor\nCMX components Copyright (c) 2022 Huayao Liu, MIT.\nNative adaptation retains upstream restrictions; independently authored WaifuFinder integration is MIT.\nWeights remain local, not published or relicensed as MIT.\n')
    (output/'manifest.json').write_text(json.dumps(dict(fileSHA256={p.name:safe.digest(p) for p in output.iterdir() if p.is_file()}),indent=2)+'\n')
    print(json.dumps(metadata,indent=2));print('Converted weights SHA-256:',safe.digest(output/'model.safetensors'))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('checkpoint',type=Path);p.add_argument('output',type=Path);p.add_argument('--licenses',type=Path,required=True);a=p.parse_args();convert(a.checkpoint,a.output,a.licenses)
