import cv2
import numpy as np
import os
import glob
import pickle

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

calc_normals('/playpen/Datasets/scribble-samples/annotations/annotations.pkl')