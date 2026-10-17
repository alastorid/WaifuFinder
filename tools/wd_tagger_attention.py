"""Inference-only SwinV2 attention preserving cosine math with rank-four SDPA.

The window axis is folded into heads rather than introducing a rank-five query
and broadcast mask. No mask is omitted; no batch rows are duplicated here.
"""
import math
import types
import torch
import torch.nn.functional as F

def sdpa_forward(self, x, mask=None):
    windows, tokens, channels = x.shape
    qkv = F.linear(x, self.qkv.weight, self._sdpa_qkv_bias)
    qkv = qkv.reshape(windows, tokens, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q = F.normalize(q, dim=-1) * self._sdpa_query_scale
    k = F.normalize(k, dim=-1)
    bias = self._sdpa_bias
    if mask is not None:
        count = mask.shape[0]
        q = q.reshape(-1, count * self.num_heads, tokens, channels // self.num_heads)
        k = k.reshape_as(q)
        v = v.reshape_as(q)
        bias = (bias.unsqueeze(1) + mask.unsqueeze(0).unsqueeze(2)).reshape(1, count * self.num_heads, tokens, tokens)
    # A non-default explicit scale would make coremltools decompose this op.
    result = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.0)
    result = result.reshape(windows, self.num_heads, tokens, -1).transpose(1, 2).reshape(windows, tokens, channels)
    return self.proj(result)


def install_sdpa(model):
    from timm.models.swin_transformer_v2 import WindowAttention
    count = 0
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, WindowAttention):
                continue
            bias_table = module.cpb_mlp(module.relative_coords_table).view(-1, module.num_heads)
            tokens = module.window_size[0] * module.window_size[1]
            bias = bias_table[module.relative_position_index.view(-1)].view(tokens, tokens, -1)
            bias = 16 * torch.sigmoid(bias.permute(2, 0, 1).contiguous())
            module.register_buffer('_sdpa_bias', bias.unsqueeze(0))
            scale = module.logit_scale.clamp(max=math.log(100)).exp()
            module.register_buffer('_sdpa_query_scale', scale * math.sqrt(module.qkv.in_features // module.num_heads))
            module.register_buffer('_sdpa_qkv_bias', torch.cat((module.q_bias, module.k_bias, module.v_bias)))
            module.forward = types.MethodType(sdpa_forward, module)
            count += 1
    if count != 24:
        raise RuntimeError(f'Expected 24 SwinV2 attention blocks, found {count}')
    return count



def check_sdpa_spec(spec):
    """Qualify the exported tensor ranks, not just the number of SDPA operations."""
    function = spec.mlProgram.functions['main']
    block = list(function.block_specializations.values())[0]
    variables = {value.name: value.type for value in function.inputs}
    count = 0
    for operation in block.operations:
        if operation.type == 'scaled_dot_product_attention':
            for key in ['query', 'key', 'value', 'attn_mask']:
                name = operation.inputs[key].arguments[0].name
                rank = len(variables[name].tensorType.dimensions)
                if rank != 4:
                    raise RuntimeError(f'{operation.outputs[0].name}: {key} rank {rank}, expected rank4')
            count += 1
        variables.update({value.name: value.type for value in operation.outputs})
    if count != 24:
        raise RuntimeError(f'Expected 24 native rank4 attention blocks; found {count}')
    return count
