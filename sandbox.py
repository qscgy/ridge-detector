import pickle
import shutil
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
from data import FoldSegmentation
import numpy as np
from natsort import natsorted
import os, random
import cv2

def renormalize():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 216
    args.crop_size = 216
    args.base_dir = '/playpen/Datasets/scribble-samples/'

    voc_train = FoldSegmentation(args, split='train')

    dataloader = DataLoader(voc_train, batch_size=1, shuffle=True, num_workers=0)
    print(len(dataloader))
    for ii, sample in enumerate(dataloader):
        img = sample['image'].numpy()
        gt = sample['label'].numpy()
        for jj in range(sample["image"].size()[0]):
            tmp = np.array(gt[jj]).astype(np.uint8)
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            img_tmp[tmp==0] = np.array([255,0,0])
            img_tmp[tmp==1] = np.array([0,255,0])       

            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(tmp)

        if ii == 1:
            break

    plt.show(block=True)

def make_test_set(base_dir):
    im_dirs = os.listdir(base_dir)
    im_dirs = natsorted(im_dirs)
    images = []
    for d in im_dirs:
        path = os.path.join(base_dir, d, 'img_corr')
        if os.path.isdir(path):
            images.extend(natsorted([os.path.join(path, i) for i in os.listdir(path)]))
    
    unseen = [x for i,x in enumerate(images) if i%5!=0]
    test_set = random.sample(unseen, 64)
    with open('test_set.pkl', 'wb') as f:
        pickle.dump(test_set, f)

def depth_derivative():
    im = np.load('/playpen/Datasets/geodepth2/007/colon_norm_preall_abs_nosm/frame014287_disp.npy')
    im = im/np.max(im)
    scharry = cv2.Scharr(im, -1, 0, 1)
    scharrx = cv2.Scharr(im, -1, 1, 0)
    cv2.imshow('original', im)
    cv2.imshow('derivative', np.sqrt(scharrx**2 + scharry**2))
    cv2.waitKey(0)

def show():
    im = cv2.imread('/playpen/ridge-dtec/run/pascal/deeplab-mobilenet4/ex_0_louis/results/000_frame031221.jpg')
    cv2.imshow('image', im)
    cv2.waitKey(0)

if __name__ == '__main__':
    # make_test_set('/playpen/Datasets/geodepth2')
    # with open('test_set.pkl', 'rb') as f:
    #     images = pickle.load(f)
    # for i in images:
    #     path = i.split('/')
    #     fname = path[-1]
    #     # print(path)
    #     dst = os.path.join('/playpen/Datasets/test-set', f'{path[-3]}_{path[-1]}')
    #     shutil.copy(i, dst)
    # depth_derivative()
    show()