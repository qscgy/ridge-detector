import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
import numpy as np
import scipy.signal as ss
from scipy.special import ive
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage.filters as filters

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
        # L = ive(np.arange(-7,8), self.scale).reshape(1,-1)
        # L = L.T@L
        # print(L)
        # Gx = torch.from_numpy(ss.convolve2d(L, np.array([[1,0,-1]]), 'same')).float()
        # Gy = Gx.clone().T

        Gx = torch.FloatTensor([[1,0,-1],[2,0,-2],[1,0,-1]])
        Gy = Gx.clone().T

        G = torch.stack((Gx, Gy), 0).unsqueeze(1)
        self.filter =nn.Conv2d(1, 2, kernel_size=Gx.shape, stride=1, padding=1, bias=False)
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
    img_ = Image.open('labels.png')
    wi, he = img_.size
    padding = 200

    img = Image.new(img_.mode, (wi+2*padding, he+2*padding), (0,0,0))
    img.paste(img_, (padding, padding))

    img = img.filter(ImageFilter.MedianFilter(size=3))
    transform = transforms.Compose([
        transforms.Grayscale(),
        # transforms.GaussianBlur(5, sigma=4),
        transforms.ToTensor()
    ])
    x = transform(img)
    x.unsqueeze_(0)
    y, bx, by= scan_conv(x)
    img_cv = np.array(img)

    scan = (bx+by)[0].permute(1,2,0).numpy()
    # scan_arr = (scan+np.min(scan))/(np.max(scan)-np.min(scan))*255
    scan_arr = (np.sign(scan)+1)*255/2
    scan_arr = scan_arr.astype(np.uint8).squeeze()

    y_flat = (torch.sqrt((y**2).sum(1))).numpy()
    y_flat -= y_flat.min()
    y_flat /= y_flat.max()
    y_flat *= 255
    y_flat = y_flat.astype(np.uint8)[0]

    circles = cv2.HoughCircles(y_flat,cv2.HOUGH_GRADIENT,1.5,5,
                            param1=120,param2=50,minRadius=100)

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2)
    ax0.imshow(img)
    spacing = 1
    # ix = np.arange((img_cv.shape[0]-y.shape[2])//2, img_cv.shape[0]-(img_cv.shape[0]-y.shape[2])//2, spacing)
    # iy = np.arange((img_cv.shape[1]-y.shape[3])//2, img_cv.shape[1]-(img_cv.shape[1]-y.shape[3])//2, spacing)
    ix, iy = np.arange(y.shape[3]), np.arange(y.shape[2])
    xs, ys = np.meshgrid(ix, iy, indexing='xy')
    # ax0.quiver(xs, ys, y[0,0,::spacing,::spacing], y[0,1,::spacing,::spacing], color='green')

    ax1.imshow(torch.sqrt((y**2).sum(1)).permute(1,2,0).numpy(), cmap='gray')
    yc = torch.cat((y[0], torch.zeros(1,*y.shape[2:])), dim=0)
    yc -= yc.min()
    yc /= yc.max()
    yn = y.numpy()
    yn[np.abs(yn)<0.1]=0
    # ax2.imshow(yc.permute(1,2,0).numpy())
    # ax2.quiver(xs, ys, yn[0,0], yn[0,1], color='blue')
    # ax2.imshow(scan_arr, cmap='gray')

    circles_i = np.uint16(np.around(circles))

    for i in circles_i[0,:]:
        cv2.circle(img_cv,(i[0],i[1]),i[2],(0,255,0),2)
        cv2.circle(img_cv,(i[0],i[1]),2,(0,0,255),3)
    ax2.imshow(img_cv)

    center = np.average(circles[0,:,:2], axis=0)
    print(center)
    img_polar = cv2.linearPolar(np.array(img), center, 400, cv2.INTER_CUBIC)
    # ax3.imshow(img_polar)

    polar_edges = cv2.Canny(img_polar, 0, 100)
    contours, _ = cv2.findContours(polar_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.minAreaRect(contours_poly[i])
        box = np.int0(cv2.boxPoints(boundRect[i]))
        cv2.drawContours(img_polar, [box], 0, (0,255,0), 1)    
    ax3.imshow(img_polar)
    plt.show()