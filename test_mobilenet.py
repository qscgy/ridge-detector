import torch
from rloss.pytorch.deeplabv3plus.modeling.backbone.mobilenet import MobileNetV2
import torch.nn as nn

if __name__ == "__main__":
    input = torch.rand(1, 3, 512, 512)
    model = MobileNetV2(output_stride=16, BatchNorm=nn.BatchNorm2d, in_channels=3)
    output, low_level_feat = model(input)
    print(output.size())
    print(low_level_feat.size())