from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from rloss.pytorch.deeplabv3plus.modeling.aspp import build_aspp
from rloss.pytorch.deeplabv3plus.modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from rloss.pytorch.deeplabv3plus.modeling.decoder import build_decoder
from rloss.pytorch.deeplabv3plus.modeling.backbone import build_backbone
from .boundary import BoundaryBranch

class ScalarMult(nn.Module):
    def __init__(self):
        super().__init__()
        self.k = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        return x*self.k

# Adapted from https://github.com/panzhiyi/URSS/blob/main/model_RW.py
# Position Attention Module, I believe from https://arxiv.org/pdf/1809.02983.pdf
class PAM(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim

        self.query = nn.Conv2d(in_dim, in_dim//8, kernel_size=1)
        self.key = nn.Conv2d(in_dim, in_dim//8, kernel_size=1)
        self.value = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.alpha = ScalarMult()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batchsize, chan, height, width = x.size()
        Q = self.query(x).reshape(batchsize, -1, height*width).permute(0, 2, 1)
        K = self.key(x).reshape(batchsize, -1, height*width)
        QK = torch.bmm(Q, K)
        attention = self.softmax(QK)
        V = self.value(x).reshape(batchsize, -1, height*width)
        QKV = torch.bmm(V, attention.permute(0,2,1)).reshape(batchsize, chan, height, width)
        out = self.alpha(QKV) + x

        return out, QK

# Adapted from Meng Tang et al. (2018)
class DeepBoundary(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False, boundary=False, in_channels=3, rw=False):
        super(DeepBoundary, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d
        
        self.boundary = boundary
        self.rw = rw    # Random walks

        self.backbone = build_backbone(backbone, output_stride, BatchNorm, in_channels=in_channels)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        if self.boundary:
            # Decoupled ASPP and decoder for instance prediction, inspired by Panoptic-DeepLab
            self.shape_aspp = build_aspp(backbone, output_stride, BatchNorm)
            self.shape_decoder = build_decoder(num_classes, backbone, BatchNorm)
            self.boundary_branch = BoundaryBranch(256, order=3, stride=1, norm=BatchNorm)
        
        if self.rw:
            self.pam = PAM(320)     # mobilenet out dim

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x0, low_level_feat = self.backbone(input)
        if self.rw:
            x0, energy = self.pam(x0)
        x1 = self.aspp(x0)
        x = self.decoder(x1, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        if self.boundary:
            s = self.shape_aspp(x0)
            # s = self.shape_decoder(s, low_level_feat)
            mask, params = self.boundary_branch(s)
            mask = mask.permute(2,0,1).unsqueeze(1)
            mask = F.interpolate(mask, size=input.size()[2:], mode='bilinear', align_corners=True)
            mask = torch.round(mask)    # interpolate may create fractional entries which make casting to bool unpredictable

            return x, mask, params
        
        return x

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
        if self.rw:
            modules.extend([self.pam])
        if self.boundary:
            modules.extend([self.shape_aspp, self.boundary_branch])
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d) or 'alpha' in m[0]:
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

if __name__ == "__main__":
    model = DeepBoundary(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())


