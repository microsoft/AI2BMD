import math

import torch
from torch import nn

act_class_mapping = {"silu": nn.SiLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}


class CosineCutoff(nn.Module):
    def __init__(self, cutoff):
        super(CosineCutoff, self).__init__()

        self.cutoff = cutoff

    def forward(self, distances):
        cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff).float()
        return cutoffs


class ExpNormalSmearing(nn.Module):
    def __init__(self, cutoff=5.0, num_rbf=50, trainable=True):
        super(ExpNormalSmearing, self).__init__()

        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(cutoff)
        self.alpha = 5.0 / cutoff

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        start_value = torch.exp(torch.scalar_tensor(-self.cutoff))
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf
        )
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(
            -self.betas * (torch.exp(self.alpha * (-dist)) - self.means) ** 2
        )


class VecLayerNorm(nn.Module):
    def __init__(self, hidden_channels, trainable, norm_type="max_min"):
        super(VecLayerNorm, self).__init__()

        self.hidden_channels = hidden_channels
        self.eps = 1e-6

        weight = torch.ones(self.hidden_channels)
        if trainable:
            self.register_parameter("weight", nn.Parameter(weight))
        else:
            self.register_buffer("weight", weight)

        if norm_type == "max_min":
            self.norm = self.max_min_norm
        else:
            self.norm = self.none_norm

        self.reset_parameters()

    def reset_parameters(self):
        weight = torch.ones(self.hidden_channels)
        self.weight.data.copy_(weight)

    def none_norm(self, vec):
        return vec

    def max_min_norm(self, vec):
        # vec: (B, N, 3 or 5, hidden_channels)
        dist = torch.norm(vec, dim=-2, keepdim=True)

        if (dist == 0).all():
            return torch.zeros_like(vec)

        dist = dist.clamp(min=self.eps)
        direct = vec / dist

        max_val, _ = torch.max(dist, dim=-1)
        min_val, _ = torch.min(dist, dim=-1)
        # delta: (B, N, 1)
        delta = max_val - min_val
        delta = torch.where(delta == 0, torch.ones_like(delta), delta)
        dist = (dist - min_val.unsqueeze(-1)) / delta.unsqueeze(-1)

        return dist * direct

    def forward(self, vec):
        # vec: (num_atoms, 3 or 8, hidden_channels)
        if vec.shape[-2] == 3:
            vec = self.norm(vec)
            return vec * self.weight.view(1, 1, 1, -1)
        elif vec.shape[-2] == 8:
            vec1, vec2 = torch.split(vec, [3, 5], dim=-2)
            vec1 = self.norm(vec1)
            vec2 = self.norm(vec2)
            vec = torch.cat([vec1, vec2], dim=-2)
            return vec * self.weight.view(1, 1, 1, -1)
        else:
            raise ValueError("VecLayerNorm only support 3 or 8 channels")
