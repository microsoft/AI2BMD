from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from .multihead_attention import MultiheadAttention
from .layers import AtomFeature, MoleculeAttnBias, AtomTaskHead, ViSRGCFeatureV1, ViSRGCFeatureV2
from .encoder_layer import EncoderLayer

def init_params(module):

    def normal_(data):
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


class Encoder(nn.Module):

    def __init__(
        self,
        num_atoms: int,
        num_in_degree: int,
        num_out_degree: int,
        num_edges: int,
        num_spatial: int,
        num_edge_dis: int,
        multi_hop_max_dist: int,
        num_encoder_layers: int = 6,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        sandwich_ln: bool = False,
        droppath_prob: float = 0.0,
        add_3d: bool = False,
        num_3d_bias_kernel: int = 128,
        no_2d: bool = False,
        mode_prob: str = "0.2,0.2,0.6",
        version: str = "v2",
    ) -> None:

        super().__init__()
        
        self.dropout_module = nn.Dropout(dropout)
        self.embedding_dim = embedding_dim
        self.mode_prob = mode_prob

        try:
            mode_prob = [float(item) for item in mode_prob.split(',')]
            assert len(mode_prob) == 3
            assert sum(mode_prob) == 1.0
        except:
            mode_prob = [0.2, 0.2, 0.6]
            
        self.mode_prob = mode_prob

        self.atom_feature = AtomFeature(
            num_atoms=num_atoms,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            hidden_dim=embedding_dim,
            no_2d=no_2d,
        )

        self.molecule_attn_bias = MoleculeAttnBias(
            num_heads=num_attention_heads,
            num_edges=num_edges,
            num_spatial=num_spatial,
            num_edge_dis=num_edge_dis,
            multi_hop_max_dist=multi_hop_max_dist,
            no_2d=no_2d,
        )

        if version == "v1":
            self.molecule_3d_bias = ViSRGCFeatureV1(
                num_atoms=num_atoms,
                num_heads=num_attention_heads,
                num_edges=num_edges,
                n_layers=num_encoder_layers,
                embed_dim=embedding_dim,
                num_kernel=num_3d_bias_kernel,
                no_share_rpe=False,
            ) if add_3d else None
        elif version == "v2":
            self.molecule_3d_bias = ViSRGCFeatureV2(
                num_atoms=num_atoms,
                num_heads=num_attention_heads,
                num_edges=num_edges,
                n_layers=num_encoder_layers,
                embed_dim=embedding_dim,
                num_kernel=num_3d_bias_kernel,
                no_share_rpe=False,
            ) if add_3d else None
        else:
            self.molecule_3d_bias = None
                
        self.atom_proc = AtomTaskHead(embedding_dim, num_attention_heads)

        self.emb_layer_norm = nn.LayerNorm(self.embedding_dim)
            
        self.layers = nn.ModuleList([])

        droppath_probs = [x.item() for x in torch.linspace(0, droppath_prob, num_encoder_layers)]

        self.layers.extend(
            [
                EncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout_module.p,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    sandwich_ln=sandwich_ln,
                    droppath_prob=droppath_probs[i],
                )
                for i in range(num_encoder_layers)
            ]
        )


    def forward(self, batched_data) -> Tuple[torch.Tensor, torch.Tensor]:
        
        data_x = batched_data["x"]
        n_mol, n_atom = data_x.size()[:2]
        padding_mask = (data_x[:,:,0]).eq(0)
        padding_mask_cls = torch.zeros(n_mol, 1, device=padding_mask.device, dtype=padding_mask.dtype)
        padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)
        mask_dict = {0: [1, 1], 1: [1, 0], 2: [0, 1]}
        mask_2d = mask_3d = None
        if self.training:
            mask_choice = np.random.choice(np.arange(3), n_mol, p=self.mode_prob)
            mask = torch.tensor([mask_dict[i] for i in mask_choice]).to(batched_data['pos'])
            mask_2d = mask[:, 0]
            mask_3d = mask[:, 1]

        x = self.atom_feature(batched_data, mask_2d=mask_2d)

        attn_bias = self.molecule_attn_bias(batched_data, mask_2d=mask_2d)

        delta_pos = None
        if self.molecule_3d_bias is not None and not (batched_data["pos"] == 0).all():
            attn_bias_3d, merged_edge_features, delta_pos = self.molecule_3d_bias(batched_data)
            if mask_3d is not None:
                merged_edge_features, delta_pos = merged_edge_features * mask_3d[:, None, None], delta_pos * mask_3d[:, None, None, None]
                attn_bias_3d = attn_bias_3d.masked_fill_(((attn_bias_3d != float('-inf')) * (1 - mask_3d[:, None, None, None])).bool(), 0.0)
            attn_bias[:, :, 1:, 1:] = attn_bias[:, :, 1:, 1:] + attn_bias_3d
            x[:, 1:, :] = x[:, 1:, :] + merged_edge_features * 0.01

        x = self.emb_layer_norm(x)
        x = self.dropout_module(x)

        x = x.transpose(0, 1)

        for layer in self.layers:
            x = layer(x, self_attn_padding_mask=padding_mask, self_attn_bias=attn_bias)

        atom_output = None
        if delta_pos is not None:
            atom_output = self.atom_proc(x[1:, :, :], attn_bias[:, :, 1:, 1:], delta_pos)
            if mask_3d is not None:
                mask_3d_only = (mask == torch.tensor([0.0, 1.0]).to(mask)[None, :]).all(dim=-1)
                atom_output = atom_output * mask_3d_only[:, None, None]

        return x, atom_output