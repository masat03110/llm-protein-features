"""Weight-only int8 quantisation for inference (no extra packages).

Every nn.Linear is replaced by Int8Linear: the weight is stored as int8 with one
scale per output channel (symmetric, absmax), and de-quantised on the fly to the
input dtype for each matmul.  Weight memory drops to ~1/2 of fp16/bf16 (~1/4 of
fp32); activations are unchanged.  Only the weights of one layer exist in
floating point at a time.

    from qwenfold.quant import quantize_linears_int8
    n = quantize_linears_int8(model.esm)      # returns the number of replaced layers
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Int8Linear(nn.Module):
    def __init__(self, lin: nn.Linear):
        super().__init__()
        w = lin.weight.detach().float()
        scale = w.abs().amax(dim=1).clamp(min=1e-8) / 127.0
        self.register_buffer("qweight", torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8))
        self.register_buffer("scale", scale.to(torch.float16))
        self.register_buffer("bias", None if lin.bias is None else lin.bias.detach().clone())
        self.in_features, self.out_features = lin.in_features, lin.out_features

    @property
    def weight(self):  # for code that only inspects .weight.dtype / .device / .shape
        return self.qweight.to(self.scale.dtype) * self.scale[:, None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.qweight.to(x.dtype) * self.scale.to(x.dtype)[:, None]
        b = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, w, b)

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, int8"


def quantize_linears_int8(module: nn.Module, min_params: int = 4096) -> int:
    """Replace nn.Linear children (recursively) in place; tiny layers are kept."""
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and child.weight.numel() >= min_params:
            setattr(module, name, Int8Linear(child))
            n += 1
        else:
            n += quantize_linears_int8(child, min_params)
    return n


def param_bytes(module: nn.Module) -> int:
    """Bytes of parameters + buffers (what sits in GPU memory for the weights)."""
    return sum(t.numel() * t.element_size() for t in list(module.parameters()) + list(module.buffers()))
