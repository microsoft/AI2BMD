from typing import Any, Dict, List

import torch


class GeoformerDataCollator:
    def __init__(self, max_nodes=None) -> None:
        self.max_nodes = max_nodes

    @staticmethod
    def _pad_attn_bias(attn_bias: torch.Tensor, max_node: int) -> torch.Tensor:
        N = attn_bias.shape[0]
        if N <= max_node:
            attn_bias_padded = torch.zeros(
                [max_node, max_node], dtype=torch.float
            ).fill_(float("-inf"))
            attn_bias_padded[:N, :N] = attn_bias
            attn_bias_padded[N:, :N] = 0
        else:
            print(
                f"Warning: max_node {max_node} is too small to hold all nodes {N} in a batch"
            )
            print("Play truncation...")

        return attn_bias_padded

    @staticmethod
    def _pad_feats(feats: torch.Tensor, max_node: int) -> torch.Tensor:
        N, *_ = feats.shape
        if N <= max_node:
            feats_padded = torch.zeros([max_node, *_], dtype=feats.dtype)
            feats_padded[:N] = feats
        else:
            print(
                f"Warning: max_node {max_node} is too small to hold all nodes {N} in a batch"
            )
            print("Play truncation...")

        return feats_padded

    def _check_attn_bias(self, feat: dict):
        num_node = len(feat["z"])
        if "attn_bias" not in feat:
            return torch.zeros([num_node, num_node]).float()
        else:
            return torch.tensor(feat["attn_bias"]).float()

    def __call__(self, features: List[dict]) -> Dict[str, Any]:
        batch = dict()

        max_node = (
            max(feat["z"].shape[0] for feat in features)
            if self.max_nodes is None
            else self.max_nodes
        )

        batch["z"] = torch.stack(
            [self._pad_feats(feat["z"], max_node) for feat in features]
        )
        batch["pos"] = torch.stack(
            [self._pad_feats(feat["pos"], max_node) for feat in features]
        )

        batch["labels"] = torch.concatenate([feat["y"] for feat in features])

        return batch
