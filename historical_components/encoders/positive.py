import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def inverse_softplus(value):
    return float(np.log(np.expm1(value)))

class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features, init_weight):
        super().__init__()
        self.raw_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.raw_weight, mean=inverse_softplus(init_weight), std=.15)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return F.linear(x, F.softplus(self.raw_weight) + 1e-12, self.bias)

class PositiveComponentEncoder(nn.Module):
    pass
    def __init__(self, dim, masks=(), hidden=32):
        super().__init__()
        self.masks = list(masks)
        if len(set(self.masks)) != len(self.masks) or any(i < 0 or i >= dim for i in self.masks):
            raise ValueError("Component mask indices must be unique and within the input width")
        self.scores = [j for j in range(dim) if j not in self.masks]
        if not self.scores:
            raise ValueError("At least one predictive score component is required")
        self.first = PositiveLinear(len(self.scores), hidden, .1)
        self.second = PositiveLinear(hidden, hidden, .035)
        self.context = nn.Linear(len(self.masks), hidden, bias=False) if self.masks else None
        if self.context is not None:
            nn.init.normal_(self.context.weight, std=.02)

    def forward(self, x):
        hidden = self.first(x[:, self.scores])
        if self.context is not None:
            hidden = hidden + self.context(x[:, self.masks])
        return F.softplus(self.second(F.softplus(hidden)))

class JointHead(nn.Module):
    pass
    def __init__(self, hidden=32):
        super().__init__()
        self.h = PositiveLinear(hidden, hidden, .035)
        self.s = nn.Linear(hidden, hidden, bias=False)
        nn.init.zeros_(self.s.weight)
        self.mix = PositiveLinear(hidden, hidden, .035)
        self.out = PositiveLinear(hidden, 1, .04)

    def forward(self, h, s):
        hidden = F.softplus(self.h(h) + self.s(s))
        return self.out(F.softplus(self.mix(hidden))).flatten()
