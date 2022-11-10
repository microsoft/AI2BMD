from typing import Optional, Tuple
import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
from OGB_ViSNet.Pretrained_3D_ViSNet.utils import (
    NeighborEmbedding,
    EdgeEmbedding,
    CosineCutoff,
    Distance,
    Sphere,
    VecLayerNorm,
    IntEmbedding,
    rbf_class_mapping,
    act_class_mapping,
)

EPS = 1e-12

class ViSNetBlock(nn.Module):

    def __init__(
        self,
        hidden_channels=128,
        num_layers=6,
        num_rbf=50,
        rbf_type="expnorm",
        trainable_rbf=True,
        activation="silu",
        attn_activation="silu",
        neighbor_embedding=True,
        num_heads=8,
        distance_influence="both",
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        lmax=1,
        vecnorm_type="max_min",
        vecnorm_trainable=True,
        atom_feature=['atomic_num'],
        bond_feature=[],
        dropout=0.0,
    ):
        super(ViSNetBlock, self).__init__()

        assert distance_influence in ["keys", "values", "both", "none"]
        assert rbf_type in rbf_class_mapping, (
            f'Unknown RBF type "{rbf_type}". '
            f'Choose from {", ".join(rbf_class_mapping.keys())}.'
        )
        assert activation in act_class_mapping, (
            f'Unknown activation function "{activation}". '
            f'Choose from {", ".join(act_class_mapping.keys())}.'
        )
        assert attn_activation in act_class_mapping, (
            f'Unknown attention activation function "{attn_activation}". '
            f'Choose from {", ".join(act_class_mapping.keys())}.'
        )

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.rbf_type = rbf_type
        self.trainable_rbf = trainable_rbf
        self.activation = activation
        self.attn_activation = attn_activation
        self.neighbor_embedding = neighbor_embedding
        self.num_heads = num_heads
        self.distance_influence = distance_influence
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.lmax = lmax
        self.vecnorm_type = vecnorm_type
        self.vecnorm_trainable = vecnorm_trainable
        self.atom_feature = atom_feature
        self.bond_feature = bond_feature

        act_class = act_class_mapping[activation]

        self.distance = Distance(
            cutoff_lower,
            cutoff_upper,
            return_vecs=True,
        )

        self.sphere = Sphere(l=self.lmax)
        self.distance_expansion = rbf_class_mapping[rbf_type](
            cutoff_lower, cutoff_upper, num_rbf, trainable_rbf
        )
        self.rbf_proj = nn.Linear(num_rbf, hidden_channels)

        if len(self.atom_feature) > 0:
            self.atom_embedding = IntEmbedding(self.atom_feature, hidden_channels, usage='atom')
        else:
            raise ValueError('atom_feature must be specified')

        if len(self.bond_feature) > 0:
            self.bond_embedding = IntEmbedding(self.bond_feature, hidden_channels, usage='bond')
        else:
            self.bond_embedding = None
        
        self.neighbor_embedding = NeighborEmbedding(hidden_channels, cutoff_lower, cutoff_upper, self.atom_feature).jittable() if neighbor_embedding else None

        self.edge_embedding = EdgeEmbedding().jittable()

        self.attention_layers = nn.ModuleList()
        block_params = dict(hidden_channels=hidden_channels, distance_influence=distance_influence,
                            num_heads=num_heads, activation=act_class, attn_activation=attn_activation,
                            cutoff_lower=cutoff_lower, cutoff_upper=cutoff_upper,
                            vecnorm_trainable=vecnorm_trainable, vecnorm_type=vecnorm_type,
                            dropout=dropout)
        
        for _ in range(num_layers - 1):
            layer = EquivariantMultiHeadAttention(**block_params,last_layer=False).jittable()
            self.attention_layers.append(layer)
        self.attention_layers.append(EquivariantMultiHeadAttention(**block_params,last_layer=True).jittable())

        self.x_out_norm = nn.LayerNorm(hidden_channels)
        self.v_out_norm = VecLayerNorm(hidden_channels, vecnorm_trainable, vecnorm_type)

        self.reset_parameters()

    def reset_parameters(self):
        self.atom_embedding.reset_parameters()
        if self.bond_embedding is not None:
            self.bond_embedding.reset_parameters()
        self.distance_expansion.reset_parameters()
        nn.init.xavier_uniform_(self.rbf_proj.weight)
        self.rbf_proj.bias.data.fill_(0)
        if self.neighbor_embedding is not None:
            self.neighbor_embedding.reset_parameters()
        for attn in self.attention_layers:
            attn.reset_parameters()
        self.x_out_norm.reset_parameters()
        self.v_out_norm.reset_parameters()

    def forward(self,
                data: Data,
                use_pos_kind,
                **kwargs
                ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

        x = self.atom_embedding(data)

        edge_index, edge_weight, edge_vec = self.distance(data, use_pos_kind)
        assert (
            edge_vec is not None
        ), "Distance module did not return directional information"

        edge_attr = self.rbf_proj(self.distance_expansion(edge_weight))

        if self.bond_embedding is not None:
            edge_attr += self.bond_embedding(data)

        edge_vec = edge_vec / torch.norm(edge_vec, dim=1).unsqueeze(1).clamp(min=1e-8)
        edge_vec = self.sphere(edge_vec)

        if self.neighbor_embedding is not None:
            x = self.neighbor_embedding(data, x, edge_weight, edge_attr, use_pos_kind)

        vec = torch.zeros(x.size(0), ((self.lmax + 1) ** 2) - 1, x.size(1), device=x.device)
        edge_attr = self.edge_embedding(edge_index, edge_attr, x)
        
        for attn in self.attention_layers[:-1]:
            dx, dvec, dedge_attr = attn(x, vec, edge_index, edge_weight, edge_attr, edge_vec, use_pos_kind)
            x = x + dx
            vec = vec + dvec
            edge_attr = edge_attr + dedge_attr

        dx, dvec, _ = self.attention_layers[-1](x, vec, edge_index, edge_weight, edge_attr, edge_vec, use_pos_kind)
        x = x + dx
        vec = vec + dvec
        
        x = self.x_out_norm(x)
        vec = self.v_out_norm(vec)
        
        return x, vec, data.atomic_num, data.pos, data.batch

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels}, "
            f"num_layers={self.num_layers}, "
            f"num_rbf={self.num_rbf}, "
            f"rbf_type={self.rbf_type}, "
            f"trainable_rbf={self.trainable_rbf}, "
            f"activation={self.activation}, "
            f"attn_activation={self.attn_activation}, "
            f"neighbor_embedding={self.neighbor_embedding}, "
            f"num_heads={self.num_heads}, "
            f"distance_influence={self.distance_influence}, "
            f"cutoff_lower={self.cutoff_lower}, "
            f"cutoff_upper={self.cutoff_upper})"
        )


class EquivariantMultiHeadAttention(MessagePassing):
    def __init__(
        self,
        hidden_channels,
        distance_influence,
        num_heads,
        activation,
        attn_activation,
        cutoff_lower,
        cutoff_upper,
        vecnorm_type,
        vecnorm_trainable,
        last_layer=False,
        dropout=0.0,
    ):
        super(EquivariantMultiHeadAttention, self).__init__(aggr="add", node_dim=0)
        assert hidden_channels % num_heads == 0, (
            f"The number of hidden channels ({hidden_channels}) "
            f"must be evenly divisible by the number of "
            f"attention heads ({num_heads})"
        )

        self.distance_influence = distance_influence
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads
        self.last_layer = last_layer
        self.vecnorm_type = vecnorm_type
        self.vecnorm_trainable = vecnorm_trainable

        self.x_layernorm = nn.LayerNorm(hidden_channels)
        self.f_layernorm = nn.LayerNorm(hidden_channels)
        self.v_layernorm = VecLayerNorm(hidden_channels, vecnorm_trainable, vecnorm_type)
        self.act = activation()
        self.attn_activation = act_class_mapping[attn_activation]()
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)

        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels)
        self.s_proj = nn.Linear(hidden_channels, hidden_channels * 2)

        self.v_dot_proj = nn.Linear(hidden_channels, hidden_channels)
        
        if not self.last_layer:
            self.f_proj = nn.Linear(hidden_channels, hidden_channels * 2)
            self.src_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
            self.trg_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
            self.w_dot_proj = nn.Linear(hidden_channels, hidden_channels)

        self.o_proj = nn.Linear(hidden_channels, hidden_channels * 3)
        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 3, bias=False)

        self.dk_proj = None
        if distance_influence in ["keys", "both"]:
            self.dk_proj = nn.Linear(hidden_channels, hidden_channels)

        self.dv_proj = None
        if distance_influence in ["values", "both"]:
            self.dv_proj = nn.Linear(hidden_channels, hidden_channels)
            
        self.scalar_dropout = nn.Dropout(dropout)
        self.vector_dropout = nn.Dropout2d(dropout)

        self.reset_parameters()
        
    def vector_rejection(self, vec, d_ij):
        
        vec_proj = (vec * d_ij.unsqueeze(2)).sum(dim=1, keepdim=True)
        return vec - vec_proj * d_ij.unsqueeze(2)

    def reset_parameters(self):
        self.x_layernorm.reset_parameters()
        self.f_layernorm.reset_parameters()
        self.v_layernorm.reset_parameters()

        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.s_proj.weight)
        self.s_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_dot_proj.weight)
        self.v_dot_proj.bias.data.fill_(0)
        
        if not self.last_layer:
            nn.init.xavier_uniform_(self.f_proj.weight)
            self.f_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.src_proj.weight)
            nn.init.xavier_uniform_(self.trg_proj.weight)
            nn.init.xavier_uniform_(self.w_dot_proj.weight)
            self.w_dot_proj.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.vec_proj.weight)
        if self.dk_proj:
            nn.init.xavier_uniform_(self.dk_proj.weight)
            self.dk_proj.bias.data.fill_(0)
        if self.dv_proj:
            nn.init.xavier_uniform_(self.dv_proj.weight)
            self.dv_proj.bias.data.fill_(0)

    def forward(self, x, vec, edge_index, r_ij, f_ij, d_ij, use_pos_kind):
        x = self.x_layernorm(x)
        f_ij = self.f_layernorm(f_ij)
        vec = self.v_layernorm(vec)
        
        q = self.q_proj(x).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads, self.head_dim)

        vec1, vec2, vec3 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)
        vec_dot = (vec1 * vec2).sum(dim=1)
        vec_dot = self.act(self.v_dot_proj(vec_dot))
        if use_pos_kind == "rdkit":
            vec_dot = self.scalar_dropout(vec_dot)

        dk = (
            self.act(self.dk_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
            if self.dk_proj is not None
            else None
        )
        dv = (
            self.act(self.dv_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
            if self.dv_proj is not None
            else None
        )

        # propagate_type: (q: Tensor, k: Tensor, v: Tensor, vec: Tensor, dk: Tensor, dv: Tensor, r_ij: Tensor, d_ij: Tensor)
        x, vec_out = self.propagate(
            edge_index,
            q=q,
            k=k,
            v=v,
            vec=vec,
            dk=dk,
            dv=dv,
            r_ij=r_ij,
            d_ij=d_ij,
            size=None,
        )
        # edge_updater_type: (vec: Tensor, d_ij: Tensor, f_ij: Tensor, use_pos_kind: str)
        o1, o2, o3 = torch.split(self.o_proj(x), self.hidden_channels, dim=1)
        dx = vec_dot * o2 + o3
        if use_pos_kind == "rdkit":
            dx = self.scalar_dropout(dx)
        dvec = vec3 * o1.unsqueeze(1) + vec_out
        if use_pos_kind == "rdkit":
            dvec = self.vector_dropout(dvec)
        if not self.last_layer:
            df_ij = self.edge_updater(edge_index, vec=vec, d_ij=d_ij, f_ij=f_ij, use_pos_kind=use_pos_kind)
            if use_pos_kind == "rdkit":
                df_ij = self.scalar_dropout(df_ij)
            return dx, dvec, df_ij
        else:
            return dx, dvec, None

    def message(self, q_i, k_j, v_j, vec_j, dk, dv, r_ij, d_ij):
        # attention mechanism
        if dk is None:
            attn = (q_i * k_j).sum(dim=-1)
        else:
            attn = (q_i * k_j * dk).sum(dim=-1)

        # attention activation function
        attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)

        # value pathway
        if dv is not None:
            v_j = v_j * dv

        # update scalar features
        v_j = (v_j * attn.unsqueeze(2)).view(-1, self.hidden_channels)

        s1, s2 = torch.split(self.act(self.s_proj(v_j)), self.hidden_channels, dim=1)
        
        # update vector features
        vec = vec_j * s1.unsqueeze(1) + s2.unsqueeze(1) * d_ij.unsqueeze(2)
        
        return v_j, vec
    
    def edge_update(self, vec_i, vec_j, d_ij, f_ij, use_pos_kind):

        w1 = self.vector_rejection(self.trg_proj(vec_i), d_ij)
        w2 = self.vector_rejection(self.src_proj(vec_j), -d_ij)
        w_dot = (w1 * w2).sum(dim=1)
        w_dot = self.act(self.w_dot_proj(w_dot))
        if use_pos_kind == "rdkit":
            w_dot = self.scalar_dropout(w_dot)

        f1, f2 = torch.split(
            self.act(self.f_proj(f_ij)),
            self.hidden_channels,
            dim=1
        )
        
        return f1 * w_dot + f2


    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec

    def update(
        self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs
