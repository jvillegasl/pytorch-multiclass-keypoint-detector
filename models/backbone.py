import torchvision
from typing import Literal, Union
import typing
from torch import Tensor, nn

from models.frozenBatchNorm2d import FrozenBatchNorm2d

SmallBackbone = Literal["resnet18", "resnet34"]
LargeBackbone = Literal["resnet50", "resnet101", "resnet152"]
BackboneName = Union[SmallBackbone, LargeBackbone]

smallBackbones = typing.get_args(SmallBackbone)


class Backbone(nn.Module):
    name: BackboneName

    backbone: nn.Module
    input_proj: nn.Conv2d
    flatten: nn.Flatten

    def __init__(self, name: BackboneName, d_model: int, dilation: bool):
        super(Backbone, self).__init__()

        self.name = name
        self.backbone = getattr(torchvision.models, self.name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=True,
            norm_layer=FrozenBatchNorm2d,
        )
        self.input_proj = nn.Conv2d(self.num_channels, d_model, kernel_size=1)
        self.flatten = nn.Flatten(2)

    def forward(self, x: Tensor) -> Tensor:
        """
        Shape:
            - Input: `(N, C, H, W)`
            - Output: `(N, d_model, *)`
        """

        y = x
        for name, module in self.backbone.named_children():
            y = module(y)
            if name == "layer4":
                break

        y = self.input_proj(y)
        y = self.flatten(y)

        return y

    @property
    def num_channels(self) -> int:
        return 512 if self.name in smallBackbones else 2048
