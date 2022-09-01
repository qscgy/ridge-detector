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
from models.boundary import plot_contour

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

if __name__=='__main__':
    # calc_normals('/playpen/Datasets/scribble-samples/annotations/annotations.pkl')
    a0 = torch.tensor([[100],[200]]).expand(4,2,2,1)
    c0 = torch.tensor([200]).expand(4,2,2,1)
    a = torch.tensor([[[0,0.1,0.1]]]).expand(4,2,2,3)*100
    b = torch.tensor([[[1, 0.5, 0],[1,0.4,0]]]).expand(4,2,2,3)*100
    c = torch.tensor([[[1, 0.5, 0],[1,0.4,0]]]).expand(4,2,2,3)*100
    d = torch.tensor([[[0, 0, 0],[0,0,0]]]).expand(4,2,2,3)*100
    mask = plot_contour(a0, c0, a, b, c, d, (400,400))
    img = mask.detach().numpy()
    plt.imshow(img[0])
    plt.show()