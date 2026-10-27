#!/usr/bin/env python3
"""Small CPU-only regression tests for offline algebraic export passes."""
import unittest
import torch
from torch import nn
from realesrgan_graph import FusedDenseBlock, FusedRRDB, FusedUpsamplingNet


class Dense(nn.Module):
    def __init__(self):
        super().__init__()
        for index in range(1,6):
            setattr(self,f'conv{index}',nn.Conv2d(4+2*(index-1),4 if index == 5 else 2,3,padding=1))
        self.lrelu=nn.LeakyReLU(0.2)
    def forward(self,x):
        features=[x]
        for index in range(1,5): features.append(self.lrelu(getattr(self,f'conv{index}')(torch.cat(features,1))))
        return x+0.2*self.conv5(torch.cat(features,1))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.rdb1,self.rdb2,self.rdb3=Dense(),Dense(),Dense()
    def forward(self,x): return x+0.2*self.rdb3(self.rdb2(self.rdb1(x)))


class FusionTests(unittest.TestCase):
    def test_residual_reparameterization_preserves_signed_border_values(self):
        torch.manual_seed(7)
        for h,w in [(1,1),(2,3),(7,9)]:
            x=torch.randn(1,4,h,w)
            block=Block().eval()
            with torch.inference_mode():
                for identity in [False,True]:
                    torch.testing.assert_close(FusedRRDB(block,identity)(x),block(x),rtol=2e-6,atol=2e-6)
                torch.testing.assert_close(FusedDenseBlock(block.rdb1)(x),block.rdb1(x),rtol=2e-6,atol=2e-6)

    def test_nearest_conv_transpose_preserves_shapes_bias_and_all_borders(self):
        torch.manual_seed(8)
        conv=nn.Conv2d(3,5,3,padding=1).eval()
        fused=FusedUpsamplingNet.reparameterize(conv).eval()
        for h,w in [(1,1),(1,7),(2,3),(7,9)]:
            for x in [torch.zeros(1,3,h,w),torch.randn(1,3,h,w)]:
                with torch.inference_mode():
                    reference=conv(torch.nn.functional.interpolate(x,scale_factor=2,mode='nearest'))
                    result=fused(x)
                    self.assertEqual(tuple(result.shape),(1,5,h*2,w*2))
                    torch.testing.assert_close(result,reference,rtol=2e-6,atol=2e-6)

if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
