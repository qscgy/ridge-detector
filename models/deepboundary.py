import torch
import torch.nn as nn
import torch.nn.functional as F
from rloss.pytorch.deeplabv3plus.modeling.aspp import build_aspp
from rloss.pytorch.deeplabv3plus.modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from rloss.pytorch.deeplabv3plus.modeling.decoder import build_decoder
from rloss.pytorch.deeplabv3plus.modeling.backbone import build_backbone
from .boundary import BoundaryBranch

# Adapted from Meng Tang et al. (2018)
class DeepBoundary(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(DeepBoundary, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)
        self.boundary_branch = BoundaryBranch(256, order=3, stride=1, norm=BatchNorm)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x0, low_level_feat = self.backbone(input)
        x1 = self.aspp(x0)
        x = self.decoder(x1, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        mask = self.boundary_branch(x1).permute(2,0,1).unsqueeze(1)
        mask = F.interpolate(mask, size=input.size()[2:], mode='bilinear', align_corners=True)
        mask = torch.round(mask)    # interpolate may create fractional entries which make casting to bool unpredictable

        return x, mask

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == "__main__":
    model = DeepBoundary(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())

