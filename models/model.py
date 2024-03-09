from torch import Tensor, nn
import torch.nn.functional as F
import torch

from models.backbone import Backbone


class MulticlassKptsDetector(nn.Module):
    backbone: Backbone
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

        self.transformer = nn.Transformer(
            nhead=n_head,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward
        )

    def forward(self, x: Tensor, classes: Tensor):
        """
        Shape:
            - x: `(N, C, H, W)`
            - classes: `(N)`, where: `∀ t ∈ classes, 0 ≤ t < num_classes ∩ t ∈ Z`
        """
        pass

    def get_targets(self, classes: Tensor) -> Tensor:
        list_embeddings: list[nn.Parameter] = [
            self.query_embeddings[t] for t in classes]

        list_num_kpts = [t.size(0) for t in list_embeddings]

        max_num_kpts = max(list_num_kpts)

        list_pad_len = [max_num_kpts - t for t in list_num_kpts]

        tgt = [
            F.pad(t, (0, 0, 0, pad), "constant", 0).unsqueeze(0)
            for t, pad in zip(list_embeddings, list_pad_len)
        ]
        tgt = torch.cat(tgt)

        

        return Tensor()

    @property
    def num_classes(self) -> int:
        return len(self.list_num_kpts)
