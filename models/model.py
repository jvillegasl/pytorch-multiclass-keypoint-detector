from torch import Tensor, nn
import torch.nn.functional as F
import torch

from models.backbone import Backbone
from models.positional_encoding import PositionalEncoding
from util.get_padded_queries import get_padded_queries


class MulticlassKptsDetector(nn.Module):
    backbone: Backbone
    pos_encoder: PositionalEncoding
    query_embeddings: nn.ParameterList
    transformer: nn.Transformer
    mlp: nn.Module

    list_num_kpts: list[int]
    d_model: int

    def __init__(
            self,
            list_num_kpts: list[int],
            d_model: int,
            n_head: int,
            num_encoder_layers: int,
            num_decoder_layers: int,
            dim_feedforward: int
    ):
        assert d_model % n_head == 0, "d_model must be divisible by n_head"

        super(MulticlassKptsDetector, self).__init__()

        self.list_num_kpts = list_num_kpts
        self.d_model = d_model
        self.query_embeddings = nn.ParameterList(
            [torch.rand(num_kpts, self.d_model)
             for num_kpts in self.list_num_kpts]
        )

        self.backbone = Backbone(
            name="resnet18",
            d_model=self.d_model,
            dilation=False
        )

        self.pos_encoder = PositionalEncoding(d_model=self.d_model)

        self.transformer = nn.Transformer(
            nhead=n_head,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
        )

    def forward(self, x: Tensor, classes: Tensor):
        """
        Shape:
            - x: `(N, C, H, W)`
            - classes: `(N)`, where: `∀ t ∈ classes, 0 ≤ t < num_classes ∩ t ∈ Z`
        """
        padded_queries, padding_masks = get_padded_queries(
            classes,
            self.query_embeddings
        )

        src = self.backbone(x).permute(2, 0, 1)
        src = self.pos_encoder(src)

        # implement positional encoding

        y = self.transformer(
            src=src,
            tgt=padded_queries,
            tgt_key_padding_mask=padding_masks
        )

        return y

    @property
    def num_classes(self) -> int:
        return len(self.list_num_kpts)
