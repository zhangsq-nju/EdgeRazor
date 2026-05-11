# ruff: noqa N812
import logging

import torch
from torch.nn import functional as F


def resolve_layer_indices(
    layer_indices: list[int | str],
    num_layers: int,
    loss_key: str,
    logger: logging.Logger,
) -> list[int]:
    # Resolve string layer names to actual indices
    resolved_indices = []
    for idx in layer_indices:
        if isinstance(idx, str):
            # Map predefined string choices to actual layer indices
            if idx == "low":
                actual_idx = 1 if num_layers > 1 else 0
            elif idx == "mid":
                actual_idx = num_layers // 2
            elif idx == "high":
                actual_idx = num_layers - 1
            else:
                logger.warning(
                    f'{loss_key}: unknown layer_index string "{idx}", skipping'
                )
                continue
            resolved_indices.append(actual_idx)
            logger.debug(
                f'{loss_key}: resolved "{idx}" to layer {actual_idx}'
            )
        else:
            # Handle negative indexing for integer indices
            actual_idx = idx if idx >= 0 else num_layers + idx
            resolved_indices.append(actual_idx)
    return resolved_indices


def resolve_layer_indices_adaptive(
    hidden_states: tuple[torch.Tensor, ...],
    metric: str,
    topk: int,
) -> list[int]:
    # hidden_states: tuple(len=L+1) of [1, seq, hidden]; idx0 is embedding, idx1..L are transformer layers
    embedding = hidden_states[0]  # layer 0: embedding
    layers = hidden_states[1:]    # layer 1..L: transformer layers
    L = len(layers)
    metric_list = []
    selected_list = []
    
    for i in range(L):
        h = layers[i].squeeze(0)  # [seq, hidden]
        
        if metric == "l2":
            metric_list.append(float(h.norm(dim=-1).mean().detach()))

        elif metric == "variance":
            metric_list.append(float(h.var(dim=0).mean().detach()))
        
        elif metric == "cosine_similarity":
            # Compute cosine similarity with previous layer
            # For i=0 (layer 1), compare with embedding (layer 0)
            # For i>0 (layer 2+), compare with previous transformer layer
            if i == 0:
                prev = embedding.squeeze(0)
            else:
                prev = layers[i-1].squeeze(0)
            cos = F.cosine_similarity(h, prev, dim=-1).mean()
            metric_list.append(float(cos.detach()))

    # Select top-k layers based on the metric => list[int=layer_index] -> layer_index_scope [1, L]
    # For "l2" and "variance", higher is better; for "cosine_similarity", lower is better
    if metric in ["l2", "variance"]:
        # Sort indices by metric values in descending order (higher is better)
        sorted_indices = sorted(range(L), key=lambda i: metric_list[i], reverse=True)
        # Take top-k and add 1 to convert from 0-indexed to layer indices [1, L]
        selected_list = [idx + 1 for idx in sorted_indices[:topk]]
    elif metric == "cosine_similarity":
        # Sort indices by metric values in ascending order (lower is better)
        sorted_indices = sorted(range(L), key=lambda i: metric_list[i])
        # Take top-k and add 1 to convert from 0-indexed to layer indices [1, L]
        selected_list = [idx + 1 for idx in sorted_indices[:topk]]
    else:
        raise ValueError(f"Unknown adaptive metric: {metric}")

    return selected_list
