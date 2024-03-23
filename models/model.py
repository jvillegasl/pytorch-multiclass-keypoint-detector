from torch import Tensor, nn
import torch

from models.backbone import Backbone
from models.positional_encoding import PositionalEncoding
from utils.get_padded_queries import get_padded_queries


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
            n_head: int = 4,
            num_encoder_layers: int = 4,
            num_decoder_layers: int = 4,
            dim_feedforward: int = 256,
    ):
        super(MulticlassKptsDetector, self).__init__()

        assert d_model % n_head == 0, "d_model must be divisible by n_head"

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
            d_model=self.d_model,
            nhead=n_head,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            batch_first=False
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

        tgt = padded_queries.permute(1, 0, 2)
        print(src.shape)
        print(padded_queries.shape)
        y = self.transformer(
            src=src,
            tgt=tgt,
            tgt_key_padding_mask=padding_masks
        )
        y = y.permute(1, 0, 2)

        return y, padding_masks

    @property
    def num_classes(self) -> int:
        return len(self.list_num_kpts)
