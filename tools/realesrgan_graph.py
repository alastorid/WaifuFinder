"""Shared offline, CPU-only algebraic RRDB export transformations.

No inference in the application depends on this module. Weight changes are
performed in FP32 before the converter rounds once to FP16.
"""
import copy
import torch
from torch import nn


class FusedDenseBlock(nn.Module):
    def __init__(self, source, output_scale=1.0, identity=True):
        super().__init__()
        self.conv1, self.conv2 = source.conv1, source.conv2
        self.conv3, self.conv4 = source.conv3, source.conv4
        self.conv5 = copy.deepcopy(source.conv5)
        self.lrelu = source.lrelu
        self.identity = identity
        with torch.no_grad():
            self.conv5.weight.mul_(0.2)
            self.conv5.bias.mul_(0.2)
            if identity:
                # cat starts with x; identity is a diagonal at the kernel centre.
                channels = self.conv5.out_channels
                indices = torch.arange(channels)
                self.conv5.weight[indices, indices, 1, 1] += 1.0
                self.conv5.weight.mul_(output_scale)
                self.conv5.bias.mul_(output_scale)
        self.output_scale = output_scale

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        out = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return out if self.identity else (out + x)


class FusedRRDB(nn.Module):
    def __init__(self, source, identity):
        super().__init__()
        self.rdb1 = FusedDenseBlock(source.rdb1, identity=identity)
        self.rdb2 = FusedDenseBlock(source.rdb2, identity=identity)
        self.rdb3 = FusedDenseBlock(source.rdb3, output_scale=0.2 if identity else 1, identity=identity)
        self.identity = identity

    def forward(self, x):
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return out + x if self.identity else out * 0.2 + x


def optimize_rrdb(model, mode="identity"):
    """Keep all convolutions/dense inputs; absorb scales and optionally skips."""
    if mode not in ("none", "scale", "identity"):
        raise ValueError(mode)
    result = copy.deepcopy(model)
    if mode != "none":
        assert len(result.body) == 23
        result.body = nn.Sequential(*(FusedRRDB(block, mode == "identity") for block in result.body))
    return result.eval().requires_grad_(False)


def fuse_x2_input(model):
    """Preserve the validated pixel-unshuffle + conv stride-2 reparameterization."""
    assert model.scale == 2 and tuple(model.conv_first.weight.shape) == (64, 12, 3, 3)
    old = model.conv_first
    new = nn.Conv2d(3, 64, 6, stride=2, padding=2)
    with torch.no_grad():
        for c in range(3):
            for y in range(2):
                for x in range(2):
                    new.weight[:, c, y::2, x::2].copy_(old.weight[:, 4*c+2*y+x])
        new.bias.copy_(old.bias)
    model.conv_first = new
    model.scale = 4  # Skip unshuffle in original forward; stride still gives 2x.
    return model


def load_official(source, variant):
    """Strict checkpoint/architecture loader; avoid importing BasicSR's tool stack."""
    import hashlib
    import importlib.util
    import sys
    import types
    expected = {
        'x2plus': '49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb',
        'x4plus': '4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1',
    }[variant]
    def digest(path):
        with path.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
    architecture = source/'rrdbnet_arch.py'
    checkpoint = source/f'RealESRGAN_{variant}.pth'
    assert digest(architecture) == '6aa8eb3f77e918b90eeff4ebfaf8745abe0c40eb8fcc03abbc1928693d70ab58', 'Unexpected architecture'
    assert digest(checkpoint) == expected, 'Unexpected checkpoint'
    class Registry:
        def register(self): return lambda cls: cls
    for name in ['basicsr','basicsr.archs','basicsr.utils','basicsr.utils.registry','basicsr.archs.arch_util']:
        sys.modules[name] = types.ModuleType(name)
    sys.modules['basicsr.utils.registry'].ARCH_REGISTRY = Registry()
    utils = sys.modules['basicsr.archs.arch_util']
    utils.default_init_weights = lambda *args, **kwargs: None
    utils.make_layer = lambda block, n, **kw: nn.Sequential(*(block(**kw) for _ in range(n)))
    utils.pixel_unshuffle = lambda x, scale: torch.nn.functional.pixel_unshuffle(x, scale)
    spec = importlib.util.spec_from_file_location('basicsr.archs.rrdbnet_arch', architecture)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.RRDBNet(3,3,scale=2 if variant == 'x2plus' else 4,num_feat=64,num_block=23,num_grow_ch=32).eval().requires_grad_(False)
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['params_ema'],strict=True)
    return model, expected


class FusedUpsamplingNet(nn.Module):
    """Nearest-2x then 3x3 conv equals a stride-2, 4x4 transpose conv.

    Zero padding outside the nearest-neighbour image maps exactly to padding=1.
    The kernel includes the 2x2 replication box, preserving borders and bias.
    """
    def __init__(self, source):
        super().__init__()
        self.scale = source.scale
        self.conv_first, self.body, self.conv_body = source.conv_first, source.body, source.conv_body
        self.conv_hr, self.conv_last, self.lrelu = source.conv_hr, source.conv_last, source.lrelu
        self.up1, self.up2 = self.reparameterize(source.conv_up1), self.reparameterize(source.conv_up2)

    @staticmethod
    def reparameterize(conv):
        assert conv.kernel_size == (3,3) and conv.padding == (1,1) and conv.groups == 1
        result = nn.ConvTranspose2d(conv.in_channels,conv.out_channels,4,stride=2,padding=1)
        with torch.no_grad():
            result.weight.zero_()
            weights = conv.weight.permute(1,0,2,3).flip((2,3))
            for y in range(3):
                for x in range(3):
                    result.weight[:,:,y:y+2,x:x+2] += weights[:,:,y:y+1,x:x+1]
            result.bias.copy_(conv.bias)
        return result

    def forward(self, x):
        if self.scale == 2: x = torch.nn.functional.pixel_unshuffle(x,2)
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.up1(feat))
        feat = self.lrelu(self.up2(feat))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))
