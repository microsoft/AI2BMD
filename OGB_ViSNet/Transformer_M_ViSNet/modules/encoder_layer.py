from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .multihead_attention import MultiheadAttention
from .droppath import DropPath

class EncoderLayer(nn.Module):

    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        droppath_prob: float = 0.1,
        dropout: float = 0.0,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        sandwich_ln: bool = False,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout


        if droppath_prob > 0.0:
            self.dropout_module = DropPath(droppath_prob)
        else:
            self.dropout_module = nn.Dropout(dropout)

        self.activation_dropout_module = nn.Dropout(activation_dropout)

        self.self_attn = MultiheadAttention(self.embedding_dim, num_attention_heads, attention_dropout)

        self.sandwich_ln = sandwich_ln

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)

        # layer norm associated with the self attention layer, sandwich
        self.self_attn_sandwich_layer_norm = nn.LayerNorm(self.embedding_dim) if self.sandwich_ln else None

        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim) 
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.final_sandwich_layer_norm = nn.LayerNorm(self.embedding_dim) if self.sandwich_ln else None

    def forward(
        self,
        x: torch.Tensor,
        self_attn_bias: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
    ):
        residual = x
        if self.sandwich_ln:
            x = self.self_attn_layer_norm(x)
            
        x = self.self_attn(
            query=x,
            attn_bias=self_attn_bias,
            key_padding_mask=self_attn_padding_mask
        )
        
        x = self.dropout_module(x)
        
        if self.sandwich_ln:
            x = self.self_attn_sandwich_layer_norm(x)
            
        x = residual + x
        
        if not self.sandwich_ln:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.sandwich_ln:
            x = self.final_layer_norm(x)
        x = F.gelu(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        if self.sandwich_ln:
            x = self.final_sandwich_layer_norm(x)
        x = residual + x
        if not self.sandwich_ln:
            x = self.final_layer_norm(x)
        return x