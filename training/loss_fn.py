import torch
from torch import Tensor
from torch import nn
from torch.nn.functional import mse_loss

from models.model import MKDPred


class MKDCriterion(nn.Module):
    pass


def loss_fn(pred: MKDPred, tgt: list[Tensor]) -> Tensor:
    """
    Shape:
        - tgt: List of N items tgt_i `[num_kpts_i, 2]`
    """

    flat_kpts = pred.flat_unmasked_kpts
    flat_tgt = torch.cat([t.unsqueeze(0) for t in tgt])

    loss = mse_loss(flat_kpts, flat_tgt)

    return loss
