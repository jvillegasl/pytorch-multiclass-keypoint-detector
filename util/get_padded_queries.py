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
        - query_embeddings: list of parameters `(E_i, T)` where E_i ≤ E_max.

        - padded_queries: `(N, E_max, T)`.
        - padding_masks: `(N, E_max, T)`
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
    for t, pad in zip(queries, queries_pads):
        padded_queries.append(F.pad(t, (0, 0, 0, pad), "constant", 0).unsqueeze(0))
        # padded_queries.append(t)
        
        mask = torch.full_like(t, False, dtype=torch.bool)
        mask = F.pad(mask, (0, 0, 0, pad), "constant", True).unsqueeze(0)
        padding_masks.append(mask)
    
    padded_queries = torch.cat(padded_queries)
    padding_masks = torch.cat(padding_masks)

    return padded_queries, padding_masks
