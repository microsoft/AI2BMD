import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Callable


class AtomFeature(nn.Module):

    def __init__(self, num_atoms, num_in_degree, num_out_degree, hidden_dim, no_2d=False):
        super(AtomFeature, self).__init__()
        self.num_atoms = num_atoms
        self.hidden_dim = hidden_dim
        self.no_2d = no_2d

        self.atom_encoder = nn.ModuleList([nn.Embedding(num_atoms + 1, hidden_dim, padding_idx=0) for _ in range(9)])
        self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(num_out_degree, hidden_dim, padding_idx=0)
        self.graph_token = nn.Embedding(1, hidden_dim)
    
    def forward(self, batched_data, mask_2d=None):
        x, in_degree, out_degree = batched_data['x'],batched_data['in_degree'], batched_data['out_degree']
        n_graph, n_node, n_feature = x.size()

        results_x = torch.zeros(n_graph, n_node, n_feature, self.hidden_dim, device=x.device)
        
        for i in range(9):
            results_x[:, :, i] = self.atom_encoder[i](x[:, :, i])
        node_feature = results_x.sum(dim=-2)

        if not self.no_2d:
            degree_feature = self.in_degree_encoder(in_degree) + self.out_degree_encoder(out_degree)
            if mask_2d is not None:
                degree_feature = degree_feature * mask_2d[:, None, None]
            node_feature = node_feature + degree_feature

        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)

        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)

        return graph_node_feature


class MoleculeAttnBias(nn.Module):
    
    def __init__(self, num_heads, num_edges, num_spatial, num_edge_dis, multi_hop_max_dist, no_2d=False):
        super(MoleculeAttnBias, self).__init__()
        
        self.num_heads = num_heads
        self.multi_hop_max_dist = multi_hop_max_dist
        self.no_2d = no_2d

        self.edge_encoder = nn.ModuleList([nn.Embedding(num_edges + 1, num_heads, padding_idx=0) for _ in range(3)])
        
        self.edge_dis_encoder = nn.Embedding(num_edge_dis * num_heads * num_heads, 1)
        self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

    def forward(self, batched_data, mask_2d=None):
    
        graph_attn_bias, spatial_pos = batched_data['attn_bias'], batched_data['spatial_pos']
        edge_input = batched_data['edge_input']

        n_graph, n_node, _, _, n_feature = edge_input.size()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        if not self.no_2d:
            spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
            if mask_2d is not None:
                spatial_pos_bias = spatial_pos_bias * mask_2d[:, None, None, None]
            graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias

        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        if not self.no_2d:

            spatial_pos_ = spatial_pos.clone()
            spatial_pos_[spatial_pos_ == 0] = 1
            spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
            if self.multi_hop_max_dist > 0:
                spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                edge_input = edge_input[:, :, :, :self.multi_hop_max_dist, :]
            
            results_edge = torch.zeros(n_graph, n_node, n_node, self.multi_hop_max_dist, n_feature, self.num_heads, device=edge_input.device, dtype=graph_attn_bias.dtype)
            for i in range(3):
                results_edge[:, :, :, :, i] = self.edge_encoder[i](edge_input[:, :, :, :, i])
            edge_input = results_edge.mean(-2)
        
            max_dist = edge_input.size(-2)
            edge_input_flat = edge_input.permute(3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
            
            edge_input_flat = torch.bmm(edge_input_flat, self.edge_dis_encoder.weight.reshape(-1, self.num_heads, self.num_heads)[:max_dist, :, :])
            edge_input = edge_input_flat.reshape(max_dist, n_graph, n_node, n_node, self.num_heads).permute(1, 2, 3, 0, 4)
            edge_input = (edge_input.sum(-2) / (spatial_pos_.float().unsqueeze(-1))).permute(0, 3, 1, 2)

            if mask_2d is not None:
                edge_input = edge_input * mask_2d[:, None, None, None]
            graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:,:, 1:, 1:] + edge_input

        return graph_attn_bias


class ViSRGCFeatureV1(nn.Module):

    def __init__(self, num_atoms, num_heads, num_edges, n_layers, embed_dim, num_kernel, no_share_rpe=False):
        super(ViSRGCFeatureV1, self).__init__()
        
        self.num_atoms = num_atoms
        self.num_heads = num_heads
        self.num_edges = num_edges
        self.n_layers = n_layers
        self.no_share_rpe = no_share_rpe
        self.num_kernel = num_kernel
        self.embed_dim = embed_dim

        self.atom_encoder = nn.ModuleList([nn.Embedding(num_atoms + 1, embed_dim, padding_idx=0) for _ in range(9)])
        
        self.left_angle_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.right_angle_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.left_dihedral_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.right_dihedral_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        rpe_heads = self.num_heads * self.n_layers if self.no_share_rpe else self.num_heads
        self.gbf = GaussianLayer(self.num_kernel, num_edges)
        self.gbf_proj = NonLinear(self.num_kernel, rpe_heads)
        self.dihedral_proj = NonLinear(self.embed_dim, rpe_heads)
        self.angle_proj = NonLinear(self.embed_dim, self.embed_dim)

    def forward(self, batched_data):

        pos, x, node_type_edge = batched_data['pos'], batched_data['x'], batched_data['node_type_edge']

        no_pos_mask =  ~(pos.eq(0).all(dim=-1).all(dim=-1))
        
        n_graph, n_node, n_feature = x.size()
        
        results_x = torch.zeros(n_graph, n_node, n_feature, self.embed_dim, device=x.device)
        for i in range(9):
            results_x[:, :, i] = self.atom_encoder[i](x[:, :, i])
        node_feature = results_x.sum(dim=-2)
        
        padding_mask = x.eq(0).all(dim=-1)

        delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = delta_pos.norm(dim=-1).view(-1, n_node, n_node)
        delta_pos /= dist.unsqueeze(-1) + 1e-5
        
        edge_vec_feature = node_feature.unsqueeze(2).unsqueeze(3) * delta_pos.unsqueeze(-1)
        node_vec_feature = edge_vec_feature.masked_fill(
            padding_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).to(torch.bool), 0.0).sum(dim=-3)
        node_angle_feature = (self.left_angle_proj(node_vec_feature) * self.right_angle_proj(node_vec_feature)).sum(dim=-2)
        node_angle_feature = self.angle_proj(node_angle_feature)

        node_angle_feature = node_angle_feature * no_pos_mask.reshape(-1, 1, 1)

        edge_feature = self.gbf(dist, torch.zeros_like(dist).long() if node_type_edge is None else node_type_edge.long())
        gbf_result = self.gbf_proj(edge_feature)
        
        w_left_vec = self.vector_rejection(self.left_dihedral_proj(node_vec_feature).unsqueeze(1).repeat(1, n_node, 1, 1, 1), delta_pos)
        w_right_vec = self.vector_rejection(self.right_dihedral_proj(node_vec_feature).unsqueeze(2).repeat(1, 1, n_node, 1, 1), -delta_pos)
        
        edge_dihedral_feature = (w_left_vec * w_right_vec).sum(dim=-2)
        edge_dihedral_feature = self.dihedral_proj(edge_dihedral_feature)
        
        graph_attn_bias = gbf_result + edge_dihedral_feature

        graph_attn_bias = graph_attn_bias * no_pos_mask.reshape(-1, 1, 1, 1)

        graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
        graph_attn_bias.masked_fill_(
            padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
        )

        return graph_attn_bias, node_angle_feature, delta_pos

    
    def vector_rejection(self, vec, d_ij):
        # vec: [n_graph, n_nodes, n_nodes, 3, n_hidden] 
        # d_ij: [n_graphs, n_nodes, n_nodes, 3]
        vec_proj = (vec * d_ij.unsqueeze(-1)).sum(dim=-2, keepdim=True)
        return vec - vec_proj * d_ij.unsqueeze(-1)


class ViSRGCFeatureV2(nn.Module):

    def __init__(self, num_atoms, num_heads, num_edges, n_layers, embed_dim, num_kernel, no_share_rpe=False):
        super(ViSRGCFeatureV2, self).__init__()

        self.num_atoms = num_atoms
        self.num_heads = num_heads
        self.num_edges = num_edges
        self.n_layers = n_layers
        self.no_share_rpe = no_share_rpe
        self.num_kernel = num_kernel
        self.embed_dim = embed_dim

        self.atom_encoder = nn.ModuleList([nn.Embedding(num_atoms + 1, embed_dim, padding_idx=0) for _ in range(9)])
        self.edge_feature_proj = NonLinear(2 * embed_dim, embed_dim)

        self.left_angle_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.right_angle_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.left_dihedral_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.right_dihedral_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        rpe_heads = self.num_heads * self.n_layers if self.no_share_rpe else self.num_heads
        self.gbf = GaussianLayer(self.num_kernel, num_edges)
        self.gbf_proj = NonLinear(self.num_kernel, rpe_heads)
        self.dihedral_proj = NonLinear(self.embed_dim, rpe_heads)
        self.angle_proj = NonLinear(self.embed_dim, self.embed_dim)

    def forward(self, batched_data):

        pos, x, node_type_edge = batched_data['pos'], batched_data['x'], batched_data['node_type_edge']

        no_pos_mask = ~(pos.eq(0).all(dim=-1).all(dim=-1))

        n_graph, n_node, n_feature = x.size()
        
        results_x = torch.zeros(n_graph, n_node, n_feature, self.embed_dim, device=x.device)
        for i in range(9):
            results_x[:, :, i] = self.atom_encoder[i](x[:, :, i])
        node_feature = results_x.sum(dim=-2)
        
        padding_mask = x.eq(0).all(dim=-1)
        
        delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = delta_pos.norm(dim=-1).view(-1, n_node, n_node)
        delta_pos /= dist.unsqueeze(-1) + 1e-5

        edge_feature = self.edge_feature_proj(torch.cat([node_feature.unsqueeze(1).repeat(1, n_node, 1, 1), node_feature.unsqueeze(2).repeat(1, 1, n_node, 1)], dim=-1))
        edge_vec_feature = edge_feature.unsqueeze(-2) * delta_pos.unsqueeze(-1)
        node_vec_feature = edge_vec_feature.masked_fill(
            padding_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).to(torch.bool), 0.0).sum(dim=-3)
        node_angle_feature = (self.left_angle_proj(
            node_vec_feature) * self.right_angle_proj(node_vec_feature)).sum(dim=-2)
        node_angle_feature = self.angle_proj(node_angle_feature)

        node_angle_feature = node_angle_feature * no_pos_mask.reshape(-1, 1, 1)

        edge_feature = self.gbf(dist, torch.zeros_like(
            dist).long() if node_type_edge is None else node_type_edge.long())
        gbf_result = self.gbf_proj(edge_feature)

        w_left_vec = self.vector_rejection(self.left_dihedral_proj(
            node_vec_feature).unsqueeze(1).repeat(1, n_node, 1, 1, 1), delta_pos)
        w_right_vec = self.vector_rejection(self.right_dihedral_proj(
            node_vec_feature).unsqueeze(2).repeat(1, 1, n_node, 1, 1), -delta_pos)

        edge_dihedral_feature = (w_left_vec * w_right_vec).sum(dim=-2)
        edge_dihedral_feature = self.dihedral_proj(edge_dihedral_feature)

        graph_attn_bias = gbf_result + edge_dihedral_feature

        graph_attn_bias = graph_attn_bias * no_pos_mask.reshape(-1, 1, 1, 1)

        graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
        graph_attn_bias.masked_fill_(
            padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
        )

        return graph_attn_bias, node_angle_feature, delta_pos

    def vector_rejection(self, vec, d_ij):
        # vec: [n_graph, n_nodes, n_nodes, 3, n_hidden]
        # d_ij: [n_graphs, n_nodes, n_nodes, 3]
        vec_proj = (vec * d_ij.unsqueeze(-1)).sum(dim=-2, keepdim=True)
        return vec - vec_proj * d_ij.unsqueeze(-1)


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=512):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.src_mul = nn.Embedding(edge_types, 1, padding_idx=0)
        self.tgt_mul = nn.Embedding(edge_types, 1, padding_idx=0)
        self.src_bias = nn.Embedding(edge_types, 1, padding_idx=0)
        self.tgt_bias = nn.Embedding(edge_types, 1, padding_idx=0)

    def forward(self, x, edge_types):   
        src_mul = self.src_mul(edge_types[:, :, :, 0])
        src_bias = self.src_bias(edge_types[:, :, :, 0])
        tgt_mul = self.tgt_mul(edge_types[:, :, :, 1])
        tgt_bias = self.tgt_bias(edge_types[:, :, :, 1])
        mul = src_mul + tgt_mul
        bias = src_bias + tgt_bias
        
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-2
        return gaussian(x.float(), mean, std).type_as(self.means.weight)

class NonLinear(nn.Module):
    def __init__(self, input, output_size, hidden=None):
        super(NonLinear, self).__init__()
        
        if hidden is None:
            hidden = input
            
        self.layer1 = nn.Linear(input, hidden)
        self.layer2 = nn.Linear(hidden, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = F.gelu(x)
        x = self.layer2(x)
        return x


class AtomTaskHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
        self.k_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
        self.v_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.scaling = (embed_dim // num_heads) ** -0.5
        self.force_proj1: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)
        self.force_proj2: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)
        self.force_proj3: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)

        self.dropout_module = nn.Dropout(0.1)

    def forward(
        self,
        query: Tensor,
        attn_bias: Tensor,
        delta_pos: Tensor,
    ) -> Tensor:
        query = query.contiguous().transpose(0, 1)
        bsz, n_node, _ = query.size()
        q = (self.q_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)* self.scaling)
        k = self.k_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
        v = self.v_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
        attn = q @ k.transpose(-1, -2)
        attn_probs_float = F.softmax(attn.view(-1, n_node, n_node) + attn_bias.contiguous().view(-1, n_node, n_node), dim=-1)
        attn_probs = attn_probs_float.type_as(attn)
        attn_probs = self.dropout_module(attn_probs).view(bsz, self.num_heads, n_node, n_node)
        rot_attn_probs = attn_probs.unsqueeze(-1) * delta_pos.unsqueeze(1).type_as(attn_probs)
        rot_attn_probs = rot_attn_probs.permute(0, 1, 4, 2, 3)
        x = rot_attn_probs @ v.unsqueeze(2)
        x = x.permute(0, 3, 2, 1, 4).contiguous().view(bsz, n_node, 3, -1)
        f1 = self.force_proj1(x[:, :, 0, :]).view(bsz, n_node, 1)
        f2 = self.force_proj2(x[:, :, 1, :]).view(bsz, n_node, 1)
        f3 = self.force_proj3(x[:, :, 2, :]).view(bsz, n_node, 1)
        cur_force = torch.cat([f1, f2, f3], dim=-1).float()
        return cur_force