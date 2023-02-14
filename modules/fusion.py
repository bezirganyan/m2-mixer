import torch
from torch import nn


class BiModalGatedUnit(nn.Module):
    def __init__(self, mod1_in, mod2_in, out_size, **kwargs):
        super(BiModalGatedUnit, self).__init__()
        self.mod1_hidden = nn.Linear(mod1_in, out_size)
        self.mod2_hidden = nn.Linear(mod2_in, out_size)

        self.z_hidden = nn.Linear(mod1_in + mod2_in, out_size)

    def forward(self, mod1, mod2):
        mod1_hidden = torch.tanh(self.mod1_hidden(mod1))
        mod2_hidden = torch.tanh(self.mod2_hidden(mod2))

        z_hidden = self.z_hidden(torch.cat([mod1, mod2], dim=-1))
        z = torch.sigmoid(z_hidden)

        return z * mod1_hidden + (1 - z) * mod2_hidden


class ConcatFusion:
    def __init__(self, dim=1, **kwargs):
        self.dim = dim

    def __call__(self, *args):
        return torch.cat(args, dim=self.dim)


class MaxFusion:
    def __init__(self, **kwargs):
        pass

    def __call__(self, *args):
        return torch.maximum(*args)


class SumFusion:
    def __init__(self, **kwargs):
        pass

    def __call__(self, *args):
        return torch.add(*args)


class MeanFusion:
    def __init__(self, **kwargs):
        pass

    def __call__(self, *args):
        return torch.mean(torch.stack(args), 0)
