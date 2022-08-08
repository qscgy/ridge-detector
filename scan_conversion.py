import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt

class VectorSobel2D(nn.Module):
    def __init__(self):
        super().__init__()
        Gx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32)
        Gy = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32)
        G = torch.stack((Gx, Gy), 0).unsqueeze(1)
        self.filter = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.filter.weight = nn.Parameter(G, requires_grad=False)
    
    def forward(self, x):
        y = self.filter(x)
        return y

class ScanConversion2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel = VectorSobel2D()
        self.thresh = nn.Threshold(1.5, 0)
        self.nms = nn.MaxPool2d(3, 1, padding=1)

        Sx = torch.tensor([[[[1,-1]]]], dtype=torch.float32)
        Sy = Sx.clone().permute(0,1,3,2)
        self.sb_x = nn.Conv2d(1, 1, kernel_size=(1,2), bias=False)
        self.sb_y = nn.Conv2d(1, 1, kernel_size=(2,1), bias=False)
        self.sb_x.weight = nn.Parameter(Sx, requires_grad=False)
        self.sb_y.weight = nn.Parameter(Sy, requires_grad=False)
    
    def forward(self, x):
        y = self.sobel(x)
        y0 = self.sb_x(torch.sign(y[:,0]).unsqueeze(1))[...,:-1,:]
        # y0 = self.thresh(y0)
        y1 = self.sb_y(torch.sign(y[:,1]).unsqueeze(1))[...,:-1]
        # y1 = self.thresh(y1)
        return y, y0, y1

class ProposalGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.scan_conv = ScanConversion2D()

if __name__=='__main__':
    scan_conv = ScanConversion2D()
    img = Image.open('022_frame021781.jpg')
    img = img.filter(ImageFilter.MedianFilter(size=3))
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.GaussianBlur(5, sigma=2),
        transforms.ToTensor()
    ])
    x = transform(img)
    x.unsqueeze_(0)
    y, bx, by= scan_conv(x)

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2)
    ax0.imshow(img)
    ax1.imshow(torch.sqrt((y**2).sum(1)).permute(1,2,0).numpy(), cmap='gray')
    ax2.imshow(bx[0].permute(1,2,0).numpy(), cmap='gray')
    ax3.imshow((torch.abs(bx)+torch.abs(by))[0].permute(1,2,0).numpy(), cmap='gray')

    plt.show()