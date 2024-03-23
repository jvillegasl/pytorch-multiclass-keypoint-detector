import torch
from torch import nn
from random import randint

from utils.get_padded_queries import get_padded_queries


def test_get_padded_queries():
    NUM_QUERIES = randint(1, 100)
    NUM_EMBEDDINGS = randint(1, 100)
    MAX_EMBEDDING_LEN = randint(1, 100)
    D_MODEL = randint(1, 100)

    indexes = torch.randint(0, NUM_EMBEDDINGS, (NUM_QUERIES,))
    set_indexes = [t.item() for t in torch.unique(indexes)]
    set_expected_queries = [torch.rand(
        randint(1, MAX_EMBEDDING_LEN), D_MODEL) for _ in set_indexes]
    set_expected_queries_by_index = dict(
        zip(set_indexes, set_expected_queries))

    embeddings = []
    for idx in range(NUM_EMBEDDINGS):
        if idx in set_expected_queries_by_index:
            embeddings.append(set_expected_queries_by_index[idx])
            continue

        embedding = torch.rand(
            randint(1, MAX_EMBEDDING_LEN), D_MODEL)
        embeddings.append(embedding)

    embeddings = nn.ParameterList(embeddings)

    padded_queries, padding_masks = get_padded_queries(indexes, embeddings)

    expected_queries = [set_expected_queries_by_index[idx.item()]
                        for idx in indexes]

    assert padded_queries.size(0) == NUM_QUERIES

    actual_queries = [torch.narrow(a, 0, 0, b.size(0))
                      for a, b in zip(padded_queries, expected_queries)]

    for a, b in zip(actual_queries, expected_queries):
        assert torch.all(torch.eq(a, b))
