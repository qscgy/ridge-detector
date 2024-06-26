from __future__ import annotations
from cv2 import EVENT_MOUSEMOVE
import numpy as np
import os
from natsort import natsorted
import cv2
import copy
import pickle
import random
import shutil

sequence_dir = '/playpen/Datasets/geodepth2'
dump_file = 'annotations.pkl'

def get_all_images(base_dir):
    im_dirs = os.listdir(base_dir)
    im_dirs = natsorted(im_dirs)
    images = []
    for d in im_dirs:
        path = os.path.join(base_dir, d, 'img_corr')
        if os.path.isdir(path):
            images.extend(natsorted([os.path.join(path, i) for i in os.listdir(path)]))
    print(images[0])
    return images

def copy_random_sample(files, dst, n=200):
    files = random.sample(files, n)
    for im in files:
        fname = im.split('/')[-1]
        im_dir = im.split('/')[-3]
        shutil.copyfile(im, os.path.join(dst, im_dir+'_'+fname))

class ScribbleAnnotator:
    def __init__(self, start=0):
        self.all_images = get_all_images(sequence_dir)[::5]
        print(f'Number of image files: {len(self.all_images)}')
        self.points = ([],[])
        self.drawing = False
        self.label = 0
        self.ix, self.iy = -1,-1
        self.idx = start

        self.annotations = {}
        self.dump_file = dump_file
        if os.path.isfile(self.dump_file):
            with open(self.dump_file, 'rb') as f:
                self.annotations = pickle.load(f)
                
        self.load_img()

        self.load_points()
       
    def update_save(self):
        if len(self.points[0]) > 0 or len(self.points[1]) > 0:
            self.annotations[self.all_images[self.idx]] = copy.deepcopy(self.points)
            with open(self.dump_file, 'wb') as f:
                pickle.dump(self.annotations, f)

    def draw_lines(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            self.points[self.label].append([[self.ix, self.iy]])   # start drawing
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                cv2.line(self.img, (self.ix, self.iy), (x, y), (255,0,0), 2)
                self.points[self.label][-1].append([x, y])
                self.ix, self.iy = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False

    def load_points(self):
        if self.all_images[self.idx] in self.annotations:
            self.points = copy.deepcopy(self.annotations[self.all_images[self.idx]])
        else:
            self.points = ([],[])

    def restore_lines(self):
        for l in self.points[0]:
            cv2.polylines(self.img, np.array([l], dtype=np.int32), False, (0,255,0), 2)
        for l in self.points[1]:
            cv2.polylines(self.img, np.array([l], dtype=np.int32), False, (255,0,0), 2)

    def load_img(self):
        print(self.all_images[self.idx])
        self.img = cv2.resize(cv2.imread(self.all_images[self.idx]), (540, 432))
        # self.img = cv2.putText(self.img, self.all_images[self.idx].split('/')[-1],
        #                     (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))

    def mainloop(self):
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.draw_lines)

        while(1):
            cv2.imshow('image', self.img)
            self.restore_lines()
            
            k = cv2.waitKey(1) & 0xFF

            if k==27:
                self.update_save()
                break
            elif k==ord('d'):
                self.update_save()
                self.idx += 1
                self.load_img()
                cv2.imshow('image', self.img)
                self.load_points()
                self.restore_lines()
            elif k==ord('a'):
                self.update_save()
                self.idx -= 1
                self.load_points()
                self.load_img()
                cv2.imshow('image', self.img)
                self.restore_lines()
            elif k==ord('r'):
                if len(self.points[self.label]) > 0:
                    self.load_img()
                    if self.label==0:
                        self.points = (self.points[0][:-1], self.points[1])
                    else:
                        self.points = (self.points[0], self.points[1][:-1])
                    self.restore_lines()
                    self.update_save()
            elif k==ord('m'):
                self.label = 1-self.label
        print(f'Total images with annotations: {len(self.annotations.keys())}')

if __name__=='__main__':
    scr = ScribbleAnnotator(0)
    scr.mainloop()