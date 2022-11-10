from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

""" Geoformer model configuration"""

logger = logging.get_logger(__name__)


class GeoformerConfig(PretrainedConfig):
    model_type = "geoformer"

    def __init__(
        self,
        max_z: int = 100,
        embedding_dim: int = 512,
        ffn_embedding_dim: int = 2048,
        num_layers: int = 9,
        num_attention_heads: int = 8,
        cutoff: int = 5.0,
        num_rbf: int = 64,
        rbf_trainable: bool = True,
        norm_type: str = "max_min",
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation_dropout: float = 0.0,
        activation_function: str = "silu",
        decoder_type: str = "scalar",
        aggr="sum",
        dataset_root=None,
        dataset_arg=None,
        mean=None,
        std=None,
        prior_model=None,
        num_classes: int = 1,
        pad_token_id: int = 0,
        **kwargs
    ):
        self.max_z = max_z
        self.embedding_dim = embedding_dim
        self.ffn_embedding_dim = ffn_embedding_dim
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.rbf_trainable = rbf_trainable
        self.norm_type = norm_type
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.decoder_type = decoder_type
        self.aggr = aggr
        self.dataset_root = dataset_root
        self.dataset_arg = dataset_arg
        self.mean = mean
        self.std = std
        self.prior_model = prior_model
        self.num_classes = num_classes

        super(GeoformerConfig, self).__init__(
            pad_token_id=pad_token_id, **kwargs
        )
