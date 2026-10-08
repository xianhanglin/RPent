"""Pointer head from Kev (kev/model.py PointerHead).

The decide token is a query and each option's closing token is a key. A scaled dot product
produces one logit per option. Temperature is applied only in eval mode, so training always
sees raw logits and the winning option does not change when a temperature is fitted later.
"""
import math

import torch
import torch.nn as nn


class PointerHead(nn.Module):
    def __init__(self, d: int, dp: int = 256):
        super().__init__()
        self.q = nn.Linear(d, dp)
        self.k = nn.Linear(d, dp)
        self.scale = 1 / math.sqrt(dp)
        self.temperature = 1.0

    def forward(self, h_decide: torch.Tensor, h_opts: torch.Tensor) -> torch.Tensor:
        logits = (self.k(h_opts) @ self.q(h_decide)) * self.scale
        if self.training or self.temperature == 1.0:
            return logits
        return logits / self.temperature
