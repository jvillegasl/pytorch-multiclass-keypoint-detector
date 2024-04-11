import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F


def get_padded_queries(indexes: Tensor, query_embeddings: nn.ParameterList):
    """
    Args:
        - indexes (Tensor): a list of indexes of the query_embeddings.
        - query_embeddings (Tensor): the embeddings to pad.

    Returns:
        - (padded_queries, padding_masks)
            - padded_queries (Tensor): the padded queries.
            - padding_masks (Tensor): the boolean masks of the queries.

    Shape:
        - indexes: `(N)`.
        - query_embeddings: list of parameters `(T_i, E)` where T_i ≤ T_max.

        - padded_queries: `(N, T_max, E)`.
        - padding_masks: `(N, T_max)`
    """
    queries: list[nn.Parameter] = [
        query_embeddings[t] for t in indexes
    ]

    queries_lens = [t.size(0) for t in queries]
    max_query_lens = max(queries_lens)

    queries_pads = [max_query_lens - t for t in queries_lens]

    padded_queries = [
        F.pad(t, (0, pad), "constant", 0).unsqueeze(0)
        for t, pad in zip(queries, queries_pads)
    ]

    padded_queries = []
    padding_masks = []
    for query, pad in zip(queries, queries_pads):
        padded_queries.append(F.pad(query, (0, 0, 0, pad), "constant", 0).unsqueeze(0))

        mask = torch.zeros(max_query_lens, dtype=torch.bool)
        print("pad", pad)
        if pad > 0:
            mask[-pad:] = True
        print("mask", mask)
        padding_masks.append(mask.unsqueeze(0))
    
    padded_queries = torch.cat(padded_queries)
    padding_masks = torch.cat(padding_masks)

    return padded_queries, padding_masks
