"""Experimental inference layouts for fixed-resolution SwinV2.

All permutations preserve each window's token order and original attention mask.
They change storage order only; no tokens, masks or network layers are removed.
"""
import types

import torch


def window_order(height, width, window, shift):
    """Grid indices in the exact order produced by roll + window_partition."""
    wh, ww = window
    if height % wh or width % ww:
        raise ValueError("This experiment requires complete windows without padding")
    grid = torch.arange(height * width).reshape(1, height, width, 1)
    grid = torch.roll(grid, shifts=(-shift[0], -shift[1]), dims=(1, 2))
    return grid.reshape(1, height // wh, wh, width // ww, ww, 1).permute(0, 1, 3, 2, 4, 5).reshape(-1)


def gathered_attention(self, x):
    batch, height, width, channels = x.shape
    windows = x.reshape(batch, height * width, channels).index_select(1, self._layout_into)
    windows = windows.reshape(-1, self.window_area, channels)
    attended = self.attn(windows, mask=self.attn_mask)
    restored = attended.reshape(batch, height * width, channels).index_select(1, self._layout_out)
    return restored.reshape(batch, height, width, channels)


def persistent_stage(self, x):
    x = self.downsample(x)
    batch, height, width, channels = x.shape
    x = x.reshape(batch, height * width, channels)
    for index, block in enumerate(self.blocks):
        if not self._layout_identity[index]:
            x = x.index_select(1, getattr(self, f"_layout_transition_{index}"))
        windows = x.reshape(-1, block.window_area, channels)
        attended = block.attn(windows, mask=block.attn_mask).reshape(batch, height * width, channels)
        x = x + block.drop_path1(block.norm1(attended))
        x = x + block.drop_path2(block.norm2(block.mlp(x)))
    if not self._layout_output_identity:
        x = x.index_select(1, self._layout_output)
    return x.reshape(batch, height, width, channels)


@torch.inference_mode(False)
def install_layout(model, persistent):
    """Install an isolated gather layout, returning permutation invariants."""
    summary = {"persistent": persistent, "stages": [], "runtimeGathers": 0}
    for stage in model.layers:
        height, width = stage.blocks[0].input_resolution
        identity = torch.arange(height * width)
        previous = identity
        noops = []
        for index, block in enumerate(stage.blocks):
            if block.dynamic_mask:
                raise ValueError("Dynamic masks are outside the fixed-shape experiment")
            order = window_order(height, width, block.window_size, block.shift_size)
            if not torch.equal(torch.sort(order).values, identity):
                raise RuntimeError("Window order is not a full permutation")
            if persistent:
                transition = torch.argsort(previous)[order]
                is_identity = torch.equal(transition, identity)
                stage.register_buffer(f"_layout_transition_{index}", transition)
                noops.append(is_identity)
                previous = order
                summary["runtimeGathers"] += int(not is_identity)
            else:
                block.register_buffer("_layout_into", order)
                block.register_buffer("_layout_out", torch.argsort(order))
                block._attn = types.MethodType(gathered_attention, block)
                summary["runtimeGathers"] += 2
        if persistent:
            inverse = torch.argsort(previous)
            stage.register_buffer("_layout_output", inverse)
            stage._layout_identity = tuple(noops)
            stage._layout_output_identity = torch.equal(inverse, identity)
            stage.forward = types.MethodType(persistent_stage, stage)
            summary["runtimeGathers"] += int(not stage._layout_output_identity)
        summary["stages"].append({"resolution": [height, width], "blocks": len(stage.blocks)})
    return summary
