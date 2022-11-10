import warnings
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import init_params, Encoder

def create_model(args, mean=None, std=None):
    
    molecule_encoder = Encoder(
        num_atoms=args['num_atoms'],
        num_in_degree=args['num_in_degree'],
        num_out_degree=args['num_out_degree'],
        num_edges=args['num_edges'],
        num_spatial=args['num_spatial'],
        num_edge_dis=args['num_edge_dis'],
        multi_hop_max_dist=args['multi_hop_max_dist'],
        num_encoder_layers=args['encoder_layers'],
        embedding_dim=args['encoder_embed_dim'],
        ffn_embedding_dim=args['encoder_ffn_embed_dim'],
        num_attention_heads=args['encoder_attention_heads'],
        dropout=args['dropout'],
        attention_dropout=args['attention_dropout'],
        activation_dropout=args['act_dropout'],
        sandwich_ln=args['sandwich_ln'],
        droppath_prob=args['droppath_prob'],
        add_3d=args['add_3d'],
        num_3d_bias_kernel=args['num_3d_bias_kernel'],
        no_2d=args['no_2d'],
        mode_prob=args['mode_prob'],
        version=args['version'],
    )
    
    model = TransformerMViSNet(molecule_encoder, args['encoder_embed_dim'], mean, std)
    
    return model


def load_model(filepath, args=None, device="cpu", **kwargs):
    ckpt = torch.load(filepath, map_location="cpu")
    if args is None:
        args = ckpt["hyper_parameters"]

    for key, value in kwargs.items():
        if not key in args:
            warnings.warn(f"Unknown hyperparameter: {key}={value}")
        args[key] = value

    model = create_model(args)

    state_dict = {re.sub(r"^model\.", "", k): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state_dict)
    return model.to(device)


class TransformerMViSNet(nn.Module):

    def __init__(self, molecule_encoder, encoder_embed_dim, mean, std):
        
        super(TransformerMViSNet, self).__init__()
        
        self.molecule_encoder = molecule_encoder
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
        self.lm_head_transform_weight = nn.Linear(encoder_embed_dim, encoder_embed_dim)
        self.activation_fn = F.gelu
        self.layer_norm = nn.LayerNorm(encoder_embed_dim)
        self.embed_out = nn.Linear(encoder_embed_dim, 1, bias=False)
        
        mean = torch.scalar_tensor(0) if mean is None else mean
        self.register_buffer("mean", mean)
        std = torch.scalar_tensor(1) if std is None else std
        self.register_buffer("std", std)
        
        self.apply(init_params)

    def forward(self, batched_data):

        x, atom_output = self.molecule_encoder(batched_data)

        x = x.transpose(0, 1)

        x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))
        x = self.embed_out(x)
        x = x + self.lm_output_learned_bias
        
        if self.std is not None:
            x = x * self.std
            
        if self.mean is not None:
            x = x + self.mean
        
        return x, atom_output


