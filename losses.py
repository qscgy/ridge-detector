from numpy import dtype
import torch.nn as nn
import torch
from rloss.pytorch.deeplabv3plus.NormalizedCutLoss import NormalizedCutLoss
from rloss.pytorch.deeplabv3plus.DenseCRFLoss import DenseCRFLoss
from rloss.pytorch.deeplabv3plus.dataloaders.custom_transforms import denormalizeimage

if __name__=='__main__':
    criterion = NormalizedCutLoss(1, 15., 40., 1.)
    N, K, H, W = 10, 2, 50, 50
    images = torch.randn((N, 3, H, W), dtype=torch.float32, device='cuda:0')
    images = denormalizeimage(images, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    mask = torch.rand((N, 1, H, W), dtype=torch.float32, requires_grad=True, device='cuda:0')
    segmentations = torch.cat((mask, 1-mask), dim=1)
    ROIs = torch.ones((N, H, W), device='cuda:0').float()
    loss = criterion(images, segmentations, ROIs)
    loss.retain_grad()
    print(loss.data)
    print(loss.grad)
    loss.backward()
    print(loss.grad)