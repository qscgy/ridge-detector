from cv2 import EVENT_MOUSEMOVE
import numpy as np
import os
from natsort import natsorted
import cv2
import copy
import pickle
import random
import shutil

def get_all_images(base_dir):
    im_dirs = os.listdir(base_dir)
    im_dirs = natsorted(im_dirs)
    images = []
    for d in im_dirs:
        path = os.path.join(base_dir, d, 'img_corr')
        if os.path.isdir(path):
            images.extend(natsorted([os.path.join(path, i) for i in os.listdir(path)]))
    return images

def dump(annotations, fname):
    with open(fname, 'wb') as f:
        pickle.dump(annotations, f)

def update_save(annotations, fname, img_name, points=[]):
    if len(points) > 0:
        annotations[img_name] = copy.deepcopy(points)
        dump(annotations, fname)

def copy_random_sample(files, dst, n=200):
    files = random.sample(files, n)
    for im in files:
        fname = im.split('/')[-1]
        im_dir = im.split('/')[-3]
        shutil.copyfile(im, os.path.join(dst, im_dir+'_'+fname))

base_dir = '/playpen/Datasets/geodepth2'
all_images = get_all_images(base_dir)[::5]
print(len(all_images))
# all_images = natsorted([os.path.join(base_dir, im) for im in os.listdir(base_dir)])

# copy_random_sample(all_images, 200)

drawing = False
ix, iy = -1,-1
points = []
bg_points = []

def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        points.append([[ix, iy]])   # start drawing
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # cv2.circle(img, (x,y), 5, (0,0,255), -1)
            cv2.line(img, (ix, iy), (x, y), (0,255,0), 2)
            points[-1].append([x, y])
            ix, iy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

def restore_lines(lines):
    for l in lines:
        cv2.polylines(img, np.array([l], dtype=np.int32), False, (0,255,0), 2)

img = cv2.resize(cv2.imread(all_images[0]), (540, 432))
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)
idx = 0
annotations = {}
dump_file = 'folds.pkl'
if os.path.isfile(dump_file):
    with open(dump_file, 'rb') as f:
        annotations = pickle.load(f)
    if all_images[idx] in annotations:
        points = annotations[all_images[idx]]
    print(len(annotations.keys()))

while(1):
    cv2.imshow('image', img)
    if all_images[idx] in annotations:
        restore_lines(points)
    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        update_save(annotations, dump_file, all_images[idx], points)
        break
    elif k==ord('d'):
        update_save(annotations, dump_file, all_images[idx], points)
        idx += 1
        img = cv2.resize(cv2.imread(all_images[idx]), (540, 432))
        cv2.imshow('image', img)
        if all_images[idx] in annotations:
            points = copy.deepcopy(annotations[all_images[idx]])
            restore_lines(points)
        else:
            points = []
    elif k==ord('a'):
        update_save(annotations, dump_file, all_images[idx], points)
        idx -= 1
        if all_images[idx] in annotations:
            points = copy.deepcopy(annotations[all_images[idx]])
        else:
            points = []
        img = cv2.resize(cv2.imread(all_images[idx]), (540, 432))
        cv2.imshow('image', img)
        restore_lines(points)