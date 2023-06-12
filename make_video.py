import numpy as np
import cv2
import skimage
from os.path import join
import os
import glob
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw

def generate_video(files):
    video_name = 'vid_020.avi'
    frame = cv2.imread(files[0])
    h, w, c = frame.shape
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (w, h))
    for f in files:
        video.write(cv2.imread(f))
    video.release()

border = 10

base_dir = "/playpen/Datasets/geodepth2/020"
preds = np.load(join(base_dir, "results-mine/preds.npy"))
foldit_pred_ims = sorted(glob.glob("/playpen/CEP/results/foldit_public_020/test_latest/images/*_fake_B.png"))
image_files = sorted(glob.glob(join(base_dir, "image/*.jpg")))
out_dir = join(base_dir, 'video')
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 40)
out_files = []

for i,f in enumerate(image_files):
    im = skimage.io.imread(f)
    im = skimage.transform.resize(im, (preds.shape[1], preds.shape[2]))
    im_seg = np.copy(im)
    seg_map = preds[i]>0.3
    im_seg[seg_map,0] = 0
    im_foldit = np.copy(im)
    foldit_seg = skimage.io.imread(foldit_pred_ims[i])
    foldit_seg = skimage.transform.resize(foldit_seg, im.shape)
    foldit_seg = (foldit_seg[...,1]==0) & (foldit_seg[...,2]==0)
    im_foldit[foldit_seg,0] = 0

    h, w, c = im.shape
    out_im = np.ones((h+100, w*3+4*border, c), dtype=im.dtype)
    results = [im, im_seg, im_foldit]
    captions = ['Original', 'Ours', 'FoldIt']

    for i in range(len(results)):
        out_im[border:border+h,border*(i+1)+i*w:(border+w)*(i+1)] = results[i]

    out_im = (out_im*255).astype(np.uint8)
    out_im = Image.fromarray(out_im)
    drawn = ImageDraw.Draw(out_im)
    for i in range(len(captions)):
        lth = drawn.textlength(captions[i], font=font)
        tx = border*(i+1) + w*i + w//2 - int(lth/2)
        ty = border+h+20
        drawn.text((tx,ty), captions[i], (0,0,0), font=font)
    fname = join(out_dir, f"{f.split('/')[-1][:-4]}_s.jpg")
    out_im.save(fname)
    out_files.append(fname)

generate_video(out_files)