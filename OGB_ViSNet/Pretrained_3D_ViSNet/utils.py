import math
import torch
from torch import nn
from torch_geometric.nn import MessagePassing
import math

EPS = 1e-8

class VecLayerNorm(nn.Module):
    def __init__(self, hidden_channels, trainable, norm_type="max_min"):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden_channels), requires_grad=trainable)
        
        if norm_type == "rms":
            self.norm = self.rms_norm
        elif norm_type == "max_min":
            self.norm = self.max_min_norm
        else:
            self.norm = self.none_norm
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
    
    def none_norm(self, vec):
        return vec
        
    def rms_norm(self, vec):
        dist = torch.norm(vec, dim=1)
        
        if (dist == 0).all():
            return torch.zeros_like(vec)
        
        dist = dist.clamp(min=EPS)
        dist = torch.sqrt(torch.mean(dist ** 2, dim=-1) + EPS)
        return vec / dist.unsqueeze(-1).unsqueeze(-1)
    
    def max_min_norm(self, vec):
        dist = torch.norm(vec, dim=1, keepdim=True)
        
        if (dist == 0).all():
            return torch.zeros_like(vec)
        
        dist =  dist.clamp(min=EPS)
        direct = vec / dist
        
        max_val, _ = torch.max(dist, dim=-1)
        min_val, _ = torch.min(dist, dim=-1)
        delta = (max_val - min_val).view(-1)
        delta = torch.where(delta == 0, torch.ones_like(delta), delta)
        dist = (dist - min_val.view(-1, 1, 1)) / delta.view(-1, 1, 1)
        
        return dist * direct

    def forward(self, vec):
        
        if vec.shape[1] == 3:
            vec = self.norm(vec)
            return vec * self.weight.unsqueeze(0).unsqueeze(0)
        elif vec.shape[1] == 8:
            vec1, vec2 = torch.split(vec, [3, 5], dim=1)
            vec1 = self.norm(vec1)
            vec2 = self.norm(vec2)
            vec = torch.cat([vec1, vec2], dim=1)
            return vec * self.weight.unsqueeze(0).unsqueeze(0)
        else:
            NotImplementedError()

class NeighborEmbedding(MessagePassing):
    def __init__(self, hidden_channels, cutoff_lower, cutoff_upper, atom_feature):
        super(NeighborEmbedding, self).__init__(aggr="add")

        self.embedding = IntEmbedding(atom_feature, hidden_channels, usage="atom")
        self.combine = nn.Linear(hidden_channels * 2, hidden_channels)
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.combine.weight)
        self.combine.bias.data.fill_(0)

    def forward(self, data, x, edge_weight, edge_attr, use_pos_kind):
        # remove self loops
        edge_index = data[f"{use_pos_kind}_edge_index"]
        mask = edge_index[0] != edge_index[1]
        if not mask.all():
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask]
            edge_attr = edge_attr[mask]

        C = self.cutoff(edge_weight)
        W = edge_attr * C.view(-1, 1)

        x_neighbors = self.embedding(data)
        # propagate_type: (x: Tensor, W: Tensor)
        x_neighbors = self.propagate(edge_index, x=x_neighbors, W=W, size=None)
        x_neighbors = self.combine(torch.cat([x, x_neighbors], dim=1))
        return x_neighbors

    def message(self, x_j, W):
        return x_j * W
    
class EdgeEmbedding(MessagePassing):
    
    def __init__(self):
        super(EdgeEmbedding, self).__init__(aggr=None)
        
    def forward(self, edge_index, edge_attr, x):
        # propagate_type: (x: Tensor, edge_attr: Tensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out
    
    def message(self, x_i, x_j, edge_attr):
        return (x_i + x_j) * edge_attr
    
    def aggregate(self, features, index):
        # no aggregate
        return features

class ExpNormalSmearing(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50, trainable=True):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(0, cutoff_upper)
        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = torch.exp(
            torch.scalar_tensor(-self.cutoff_upper + self.cutoff_lower)
        )
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
            -self.betas
            * (torch.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2
        )

class CosineCutoff(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0):
        super(CosineCutoff, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

    def forward(self, distances):
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (
                torch.cos(
                    math.pi
                    * (
                        2
                        * (distances - self.cutoff_lower)
                        / (self.cutoff_upper - self.cutoff_lower)
                        + 1.0
                    )
                )
                + 1.0
            )
            # remove contributions below the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            cutoffs = cutoffs * (distances > self.cutoff_lower).float()
            return cutoffs
        else:
            cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff_upper) + 1.0)
            # remove contributions beyond the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            return cutoffs


class Distance(nn.Module):
    def __init__(
        self,
        cutoff_lower,
        cutoff_upper,
        return_vecs=False,
    ):
        super(Distance, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.return_vecs = return_vecs

    def forward(self, data, use_pos_kind):
        
        edge_index, pos = data[f"{use_pos_kind}_edge_index"], data[f"{use_pos_kind}_atom_pos"]
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]

        mask = edge_index[0] != edge_index[1]
        edge_weight = torch.zeros(edge_vec.size(0), device=edge_vec.device)
        edge_weight[mask] = torch.norm(edge_vec[mask], dim=-1)

        lower_mask = edge_weight >= self.cutoff_lower
        edge_index = edge_index[:, lower_mask]
        edge_weight = edge_weight[lower_mask]

        if self.return_vecs:
            edge_vec = edge_vec[lower_mask]
            return edge_index, edge_weight.clamp(1e-8), edge_vec

        return edge_index, edge_weight.clamp(1e-8), None
        
class Sphere(nn.Module):
    
    def __init__(self, l=2):
        
        super(Sphere, self).__init__()
        
        self.l = l
        
    def forward(self, edge_vec):
        
        # edge_vec = F.normalize(edge_vec, p=2, dim=-1)
        edge_sh = _spherical_harmonics(self.l, edge_vec[..., 0], edge_vec[..., 1], edge_vec[..., 2])
        
        return edge_sh
        
# @torch.jit.script
def _spherical_harmonics(lmax: int, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:

    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    if lmax == 1:
        return torch.stack([
            sh_1_0, sh_1_1, sh_1_2
        ], dim=-1)

    sh_2_0 = math.sqrt(3.0) * x * z
    sh_2_1 = math.sqrt(3.0) * x * y
    y2 = y.pow(2)
    x2z2 = x.pow(2) + z.pow(2)
    sh_2_2 = y2 - 0.5 * x2z2
    sh_2_3 = math.sqrt(3.0) * y * z
    sh_2_4 = math.sqrt(3.0) / 2.0 * (z.pow(2) - x.pow(2))

    if lmax == 2:
        return torch.stack([
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4
        ], dim=-1)
        
class GatedEquivariantBlock(nn.Module):
    """Gated Equivariant Block as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra
    """

    def __init__(
        self,
        hidden_channels,
        out_channels,
        intermediate_channels=None,
        activation="silu",
        scalar_activation=False,
    ):
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = out_channels

        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)

        act_class = act_class_mapping[activation]
        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, intermediate_channels),
            act_class(),
            nn.Linear(intermediate_channels, out_channels * 2),
        )

        self.act = act_class() if scalar_activation else None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def forward(self, x, v):
        vec1 = torch.norm(self.vec1_proj(v), dim=-2)
        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(1) * vec2

        if self.act is not None:
            x = self.act(x)
        return x, v
    
rbf_class_mapping = {"expnorm": ExpNormalSmearing}

act_class_mapping = {
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}

class IntEmbedding(nn.Module):
    """
    Atom Encoder
    """
    def __init__(self, names, embed_dim, usage='atom'):
        super(IntEmbedding, self).__init__()
        self.names = names
        self.usage = usage
        self.embed_dict = nn.ModuleDict()
        self.atom_feature_size = {'atomic_num': 124, 'formal_charge': 22, 'degree': 17, 'chiral_tag': 14, 'total_numHs': 15, 'is_aromatic': 7, 'hybridization': 14}
        self.bond_feature_size = {'bond_dir': 12, 'bond_type': 27, 'is_in_ring': 7}
            
        embed_params = self._get_embed_params()
        for name in self.names:
            embed = nn.Embedding(embed_params[name]['vocab_size'], embed_dim)
            self.embed_dict[name] = embed

    def _get_embed_params(self):
        
        embed_params = {}
        for name in self.names:
            if self.usage == 'atom':
                embed_params[name] = {
                    'vocab_size': self.atom_feature_size[name]}
            elif self.usage == 'bond':
                embed_params[name] = {
                    'vocab_size': self.bond_feature_size[name]}
            else:
                NotImplementedError('usage should be atom or bond')

        return embed_params

    def reset_parameters(self):
        for name in self.names:
            self.embed_dict[name].reset_parameters()

    def forward(self, input):
        """
        Args: 
            input(dict of tensor): node features.
        """
        out_embed = 0
        for name in self.names:
            out_embed += self.embed_dict[name](input[name])
        return out_embed


