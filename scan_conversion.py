import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
import numpy as np
import scipy.signal as ss
from scipy.special import ive
import matplotlib.pyplot as plt

class VectorSobel2D(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

        # Has some interesting effects (I believe this is the cross Laplacian)
        # array([[  1,   4,   5,   0,  -5,  -4,  -1],
        #        [  4,  16,  20,   0, -20, -16,  -4],
        #        [  5,  20,  25,   0, -25, -20,  -5],
        #        [  0,   0,   0,   0,   0,   0,   0],
        #        [ -5, -20, -25,   0,  25,  20,   5],
        #        [ -4, -16, -20,   0,  20,  16,   4],
        #        [ -1,  -4,  -5,   0,   5,   4,   1]])

        # L = np.array([[1,4,6,4,1]]).T@np.array([[1,4,6,4,1]]).astype(np.float32)
        L = ive(np.arange(-7,8), self.scale).reshape(1,-1)
        L = L.T@L
        print(L)
        Gx = torch.from_numpy(ss.convolve2d(L, np.array([[1,0,-1]]), 'same')).float()
        Gy = Gx.clone().T
        G = torch.stack((Gx, Gy), 0).unsqueeze(1)
        self.filter = nn.Conv2d(1, 2, kernel_size=Gx.shape, stride=1, padding=1, bias=False)
        self.filter.weight = nn.Parameter(G, requires_grad=False)
    
    def forward(self, x):
        y = self.filter(x)*self.scale
        return y

class ScanConversion2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel = VectorSobel2D(16)
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
    img = Image.open('seg.png')
    img = img.filter(ImageFilter.MedianFilter(size=3))
    transform = transforms.Compose([
        transforms.Grayscale(),
        # transforms.GaussianBlur(5, sigma=4),
        transforms.ToTensor()
    ])
    x = transform(img)
    x.unsqueeze_(0)
    y, bx, by= scan_conv(x)

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2)
    ax0.imshow(img)
    ax1.imshow(torch.sqrt((y**2).sum(1)).permute(1,2,0).numpy(), cmap='gray')
    yc = torch.cat((y[0], torch.zeros(1,*y.shape[2:])), dim=0)
    yc /= yc.max()
    ax2.imshow(yc.permute(1,2,0).numpy())
    ax3.imshow((bx+by)[0].permute(1,2,0).numpy(), cmap='gray')

    plt.show()