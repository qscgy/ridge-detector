from importlib.metadata import requires
import re
import cv2
import numpy as np
import os
import glob
import pickle
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

def calc_normals(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    files = list(data.keys())
    X = np.zeros_like(cv2.imread(files[0]), dtype=np.float64)
    X2 = np.zeros_like(X)
    for f in files:
        img = cv2.imread(f)/255.0
        X += img
        X2 += img**2
    print(X)
    mean = np.mean(X, axis=(0,1))/len(files)
    std = np.sqrt(np.mean(X2, axis=(0,1))/len(files)-mean**2)
    print(mean, std)

def plot_contour(a0, c0, a, b, c, d, outsize):
    n = torch.arange(1,a.shape[-1]+1)
    t = torch.arange(0,1,0.0001).reshape(-1,1)
    sins = torch.sin(2*np.pi*n*t).T
    coss = torch.cos(2*np.pi*n*t).T
    X = a0 + torch.sum(a[...,None]*sins + b[...,None]*coss, axis=-2)
    Y = c0 + torch.sum(c[...,None]*sins + d[...,None]*coss, axis=-2)

    X = X.long()
    Y = Y.long()

    im = torch.zeros((*a.shape[:-1], *outsize), dtype=torch.float32, requires_grad=True)
    im2 = im.clone()
    for i in range(a.shape[-2]):
        im2[i, X[i], Y[i]] = 1
    mask = in_out(im2, a.shape[:-1]) * torch.flip(in_out(torch.flip(im2, (-2,)), a.shape[:-1]), (-2,))
    mask = mask.sum(-3)>0

    return mask

def in_out(im, par_size):
    # left = torch.tensor([[[[1,0,0],[1,0,0],[1,0,0]]]], dtype=torch.int, requires_grad=False)
    # right = torch.tensor([[[[0,0,1],[0,0,1],[0,0,1]]]], dtype=torch.int, requires_grad=False)
    # lefts = F.conv2d(im.unsqueeze(0).int(), left, padding=1)
    # rights = F.conv2d(im.unsqueeze(0).int(), right, padding=1)
    mask = torch.clone(im)
    # plt.imshow(im[0].detach().numpy())
    # plt.show()
    crossings = torch.zeros(*par_size, mask.shape[-1])
    for i in range(1, mask.shape[-2]-1):
        crossings += ((im[...,i,:]-im[...,i-1,:])>0)
        mask[...,i,:] = (crossings%2==1)
    return mask

if __name__=='__main__':
    # calc_normals('/playpen/Datasets/scribble-samples/annotations/annotations.pkl')
    a0 = 200
    c0 = 200
    a = torch.tensor([[0, 0, 0],[0,0,0]])*100
    b = torch.tensor([[1, 0.5, 0],[1,0.4,0]])*100
    c = torch.tensor([[1, 0.5, 0],[1,0.4,0]])*100
    d = torch.tensor([[0, 0, 0],[0,0,0]])*100
    mask = plot_contour(a0, c0, a, b, c, d, (400,400))
    img = mask.detach().numpy()
    plt.imshow(img)
    plt.show()