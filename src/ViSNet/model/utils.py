import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius_graph
from torch_geometric.nn import MessagePassing
from torch_sparse import SparseTensor

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


class GaussianSmearing(nn.Module):
    def __init__(self, cutoff=5.0, num_rbf=50, trainable=True):
        super(GaussianSmearing, self).__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.trainable = trainable

        offset, coeff = self._initial_params()
        if trainable:
            self.register_parameter("coeff", nn.Parameter(coeff))
            self.register_parameter("offset", nn.Parameter(offset))
        else:
            self.register_buffer("coeff", coeff)
            self.register_buffer("offset", offset)

    def _initial_params(self):
        offset = torch.linspace(0, self.cutoff, self.num_rbf)
        coeff = -0.5 / (offset[1] - offset[0]) ** 2
        return offset, coeff

    def reset_parameters(self):
        offset, coeff = self._initial_params()
        self.offset.data.copy_(offset)
        self.coeff.data.copy_(coeff)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * torch.pow(dist, 2))


rbf_class_mapping = {"gauss": GaussianSmearing, "expnorm": ExpNormalSmearing}


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


act_class_mapping = {
    "ssp": ShiftedSoftplus,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "swish": Swish,
}


class Sphere(nn.Module):
    def __init__(self, lmax=2):
        super(Sphere, self).__init__()
        self.l = lmax

    def forward(self, edge_vec):
        edge_sh = self._spherical_harmonics(
            self.l, edge_vec[..., 0], edge_vec[..., 1], edge_vec[..., 2]
        )
        return edge_sh

    @staticmethod
    def _spherical_harmonics(
        lmax: int, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        sh_1_0, sh_1_1, sh_1_2 = x, y, z

        if lmax == 1:
            return torch.stack([sh_1_0, sh_1_1, sh_1_2], dim=-1)

        sh_2_0 = math.sqrt(3.0) * x * z
        sh_2_1 = math.sqrt(3.0) * x * y
        y2 = y.pow(2)
        x2z2 = x.pow(2) + z.pow(2)
        sh_2_2 = y2 - 0.5 * x2z2
        sh_2_3 = math.sqrt(3.0) * y * z
        sh_2_4 = math.sqrt(3.0) / 2.0 * (z.pow(2) - x.pow(2))

        if lmax == 2:
            return torch.stack(
                [
                    sh_1_0,
                    sh_1_1,
                    sh_1_2,
                    sh_2_0,
                    sh_2_1,
                    sh_2_2,
                    sh_2_3,
                    sh_2_4,
                ],
                dim=-1,
            )
        else:
            raise ValueError(f"lmax={lmax}")


class VecLayerNorm(nn.Module):
    def __init__(self, hidden_channels, trainable, norm_type="max_min"):
        super(VecLayerNorm, self).__init__()

        self.hidden_channels = hidden_channels
        self.eps = 1e-12

        weight = torch.ones(self.hidden_channels)
        if trainable:
            self.register_parameter("weight", nn.Parameter(weight))
        else:
            self.register_buffer("weight", weight)

        self.norm_type = norm_type

        self.reset_parameters()

    def reset_parameters(self):
        weight = torch.ones(self.hidden_channels)
        self.weight.data.copy_(weight)

    def none_norm(self, vec):
        return vec

    def rms_norm(self, vec):
        # vec: (num_atoms, 3 or 5, hidden_channels)
        dist = torch.norm(vec, dim=1)

        if (dist == 0).all():
            return torch.zeros_like(vec)

        dist = dist.clamp(min=self.eps)
        dist = torch.sqrt(torch.mean(dist**2, dim=-1))
        return vec / F.relu(dist).unsqueeze(-1).unsqueeze(-1)

    def max_min_norm(self, vec):
        # vec: (num_atoms, 3 or 5, hidden_channels)
        dist = torch.norm(vec, dim=1, keepdim=True)

        if (dist == 0).all():
            return torch.zeros_like(vec)

        dist = dist.clamp(min=self.eps)
        direct = vec / dist

        max_val, _ = torch.max(dist, dim=-1)
        min_val, _ = torch.min(dist, dim=-1)
        delta = (max_val - min_val).view(-1)
        delta = torch.where(delta == 0, torch.ones_like(delta), delta)
        dist = (dist - min_val.view(-1, 1, 1)) / delta.view(-1, 1, 1)

        return F.relu(dist) * direct

    def forward(self, vec):
        # vec: (num_atoms, 3 or 8, hidden_channels)
        if vec.shape[1] == 3:
            if self.norm_type == "rms":
                vec = self.rms_norm(vec)
            elif self.norm_type == "max_min":
                vec = self.max_min_norm(vec)
            else:
                vec = self.none_norm(vec)

            return vec * self.weight.unsqueeze(0).unsqueeze(0)
        elif vec.shape[1] == 8:
            vec1, vec2 = torch.split(vec, [3, 5], dim=1)

            if self.norm_type == "rms":
                vec1 = self.rms_norm(vec1)
            elif self.norm_type == "max_min":
                vec1 = self.max_min_norm(vec1)
            else:
                vec1 = self.none_norm(vec1)

            if self.norm_type == "rms":
                vec2 = self.rms_norm(vec2)
            elif self.norm_type == "max_min":
                vec2 = self.max_min_norm(vec2)
            else:
                vec2 = self.none_norm(vec2)

            vec = torch.cat([vec1, vec2], dim=1)
            return vec * self.weight.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError("VecLayerNorm only support 3 or 8 channels")


class Distance(nn.Module):
    def __init__(self, cutoff, max_num_neighbors=32, loop=True):
        super(Distance, self).__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.loop = loop

    def forward(self, pos, batch):
        edge_index = radius_graph(
            pos,
            r=self.cutoff,
            batch=batch,
            loop=self.loop,
            max_num_neighbors=self.max_num_neighbors,
        )
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]

        if self.loop:
            mask = edge_index[0] != edge_index[1]
            edge_weight = torch.zeros(edge_vec.size(0), device=edge_vec.device)
            edge_weight[mask] = torch.norm(edge_vec[mask], dim=-1)
        else:
            edge_weight = torch.norm(edge_vec, dim=-1)

        return edge_index, edge_weight, edge_vec


class NeighborEmbedding(MessagePassing):
    def __init__(self, hidden_channels, num_rbf, cutoff, max_z):
        super(NeighborEmbedding, self).__init__(aggr="add")
        self.embedding = nn.Embedding(max_z, hidden_channels)
        self.distance_proj = nn.Linear(num_rbf, hidden_channels)
        self.combine = nn.Linear(hidden_channels * 2, hidden_channels)
        self.cutoff = CosineCutoff(cutoff)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        nn.init.xavier_uniform_(self.distance_proj.weight)
        nn.init.xavier_uniform_(self.combine.weight)
        self.distance_proj.bias.data.fill_(0)
        self.combine.bias.data.fill_(0)

    def forward(self, z, x, edge_index, edge_weight, edge_attr):
        # remove self loops
        mask = edge_index[0] != edge_index[1]
        if not mask.all():
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask]
            edge_attr = edge_attr[mask]

        C = self.cutoff(edge_weight)
        W = self.distance_proj(edge_attr) * C.view(-1, 1)

        x_neighbors = self.embedding(z)

        # propagate_type: (x: Tensor, W: Tensor)
        x_neighbors = self.propagate(
            edge_index, x=x_neighbors, W=W, size=None
        )
        x_neighbors = self.combine(torch.cat([x, x_neighbors], dim=1))
        return x_neighbors

    def message(self, x_j, W):
        return x_j * W


class EdgeEmbedding(MessagePassing):
    def __init__(self, num_rbf, hidden_channels):
        super(EdgeEmbedding, self).__init__(aggr=None)
        self.edge_proj = nn.Linear(num_rbf, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.edge_proj.weight)
        self.edge_proj.bias.data.fill_(0)

    def forward(self, edge_index: torch.Tensor, edge_attr, x):
        # propagate_type: (x: Tensor, edge_attr: Tensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        return out

    def message(self, x_i, x_j, edge_attr):
        return (x_i + x_j) * self.edge_proj(edge_attr)

    def aggregate(self, features, index):
        # no aggregate
        return features
