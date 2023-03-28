import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
from glob import glob
import re
from skimage.transform import resize
from natsort import natsorted
import matplotlib.pyplot as plt
import cv2
import plotly.express as px
import time

from pytorch3d.structures import Pointclouds
from pytorch3d.ops import iterative_closest_point, utils as oputil
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.loss import chamfer_distance

from typing import NamedTuple

from cubic_spline import interp, Interpolator

torch.autograd.set_detect_anomaly(True)
device = 'cuda:0'

class MyICPSoln(NamedTuple):
    Xt: torch.Tensor
    R: torch.Tensor
    T: torch.Tensor
    s: torch.Tensor
    interp: torch.Tensor

class Aligner(nn.Module):
    def __init__(self, n_pts=9) -> None:
        super().__init__()
        self.R = nn.Parameter(torch.eye(3).unsqueeze(0))
        self.T = nn.Parameter(torch.zeros((1, 1, 3)))
        self.s = nn.Parameter(torch.ones(1, 1))
        # self.s = torch.ones((1,1)).cuda()

        self.interp = Interpolator(n_pts, max=5)
    
    def forward(self, X):
        Xt = torch.clone(X)
        Xt[0,:,2] = self.interp(Xt[0,:,2])
        Xt = self.s[:, None, None] * torch.bmm(Xt, self.R) + self.T[:, None, :]
        return Xt[0]

def icp_with_warp(X, Y, iters=500):
    aligner = Aligner()
    aligner = aligner.to(device)
    optimizer = torch.optim.Adam(aligner.parameters(), lr=3e-4)

    X, num_points_X = oputil.convert_pointclouds_to_tensor(X)
    Y, num_points_Y = oputil.convert_pointclouds_to_tensor(Y)

    best_loss = 1
    for i in range(iters):
        loss = 0
        Xt = aligner(X)
        loss, _ = chamfer_distance(Xt, Y)
        # print(loss.item())
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_sol = MyICPSoln(Xt.clone(), aligner.R.clone(), aligner.T.clone(), aligner.s.clone(), \
                                 torch.vstack((aligner.interp.x_ctl, aligner.interp.y_ctl)))
        loss.backward()
        optimizer.step()
    print(loss.item())
    
    return best_sol

def frame_num_to_index(fpath, frame):
    img_dir = os.path.dirname(fpath)
    frames = natsorted(os.listdir(img_dir))
    frames = [int(re.search(r'[0-9]{6}', f).group(0)) for f in frames]
    return frames.index(frame)

# From Blanton et al. (WACV 2022)
def align_pc(A: torch.FloatTensor, B: torch.FloatTensor, W: torch.FloatTensor):
    A = A.reshape(-1, 3)
    B = B.reshape(-1, 3)
    # W = W.reshape(1, -1).repeat(A.shape[0], 1)
    # SW = W.sum()
    # uA = (W*A).sum((-2, -1))/SW
    # uB = (W*B).sum((-2, -1))/SW
    uA = A.mean(0).unsqueeze(1)
    uB = B.mean(0).unsqueeze(1)
    print(uA.shape)

    A = A-uA.repeat(1, A.shape[0]).mT
    B = B-uB.repeat(1, B.shape[0]).mT

    print(A.shape, B.shape)
    M = A.mT @ B
    U, S, Vh = torch.linalg.svd(M)
    V = Vh.mT
    d = torch.linalg.det(V@(U.mT))
    R = V @ torch.diag_embed(torch.tensor([1, 1, d])) @ U.mT
    T = -R@uA + uB
    return R, T

def get_depth(fname, base_path='/playpen/Datasets/geodepth2/'):
    fname = fname.replace('Datasets','019')
    path = fname.split('/')
    frame = re.search(r'[0-9]{6}', path[-1]).group(0)
    num = '_'.join(path[-1].split('_')[:-1])
    orig_path = os.path.join(base_path, num, 'image', frame+'.jpg')
    index = frame_num_to_index(orig_path, int(frame))
    spec = 'depth'
    outs = os.path.join(*path[:-2], 'NFPS', 'images', f'{num}_{index:03d}', f'{num}_{index:03d}_nr_{spec}.npy')
    # outs = os.path.join(*path[:-2], 'colon_geo_light', 'nr4', f'frame{frame}_disp.npy')
    if fname[0]=='/':
        outs = '/'+outs
    return outs, num

def make_xy(depth):
    w, h = depth.shape
    xr = torch.linspace(0, 1, w)
    yr = torch.linspace(0, 1, h)
    return torch.meshgrid(xr, yr, indexing='ij')

def flatten_each_fold(depth, seg_mask):
    seg_mask *= 255
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(seg_mask.astype(np.uint8))
    for i in range(1, n):
        mask_i = (labels==i)
        mean_depth_i = depth[mask_i].mean()
        std_i = depth[mask_i].std()
        depth[mask_i] = torch.clamp(depth[mask_i], mean_depth_i-std_i, mean_depth_i+std_i)
    return depth

# seg_dir = '/playpen/ridge-dtec/run/pascal/mobilenet-normal/ex_1_erwin/results-mine-sequence'
seg_dir = '/playpen/ridge-dtec/run/pascal/mobilenet3-96/ex_0_cyrus/results-mine-jhu'

def get_point_clouds():
    seg_image_files = natsorted(os.listdir(seg_dir))
    seg_images = []
    point_clouds = []
    shape = (216,216)
    crop_border = 0
    n_samples = 1000
    for im in [seg_image_files[0], seg_image_files[-1]]:
        if len(im)>4 and im[-3:]=='png':
            seg_img = Image.open(os.path.join(seg_dir, im))
            seg_img = seg_img.crop((crop_border, crop_border, seg_img.width-crop_border, seg_img.height-crop_border))
            seg_img = np.asarray(seg_img.resize(shape))
            seg_mask = seg_img[:,:,0]==0
            seg_images.append(torch.from_numpy(seg_mask.astype(float)).unsqueeze(0))

            outs, num = get_depth(im, base_path='/bigpen/Datasets/jhu-older')
            depth_path = os.path.join('/bigpen/Datasets/jhu-older', num, outs)
            depth = np.load(depth_path)
            print(depth_path)
            h, w = depth.shape
            depth = depth[crop_border:h-crop_border, crop_border:w-crop_border]
            depth = resize(depth, shape)
            depth = torch.from_numpy(depth)
            # depth = flatten_each_fold(depth, seg_mask.astype('uint8'))
            Y, X = make_xy(depth)
            
            points = torch.stack((X, Y, depth), 0)
            fold_pts = torch.stack(torch.where(seg_images[-1]>0), 0)
            sample_inds = torch.randperm(fold_pts.shape[-1])[:n_samples]
            points = points[:,fold_pts[1, sample_inds], fold_pts[2, sample_inds]]
            points = Pointclouds(points=points.mT.unsqueeze(0))
            point_clouds.append(points)
    return point_clouds

rmses = []
point_clouds = get_point_clouds()

initial = '1.000000000000000000e+00,6.391316248943144096e-01,7.684132125700027238e-01,2.876749660781416709e-02,3.295279956880653316e+01,-1.742007229456648404e-02,1.085591701692181499e-01,-2.455592562815625546e-02,4.908259586491502091e+02,-1.785351629422295833e-02,-2.298346165463256707e-02,9.997243557531489966e-01,-1.070875008342320172e+02,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00'
final = '5.880000000000000000e+02,6.618543599570043678e-01,7.486081202417533831e-01,3.519842143738682871e-02,2.296029051361081486e+01,-2.520100448540786575e-02,1.489336108320585550e-01,-3.100679380868567869e-02,5.008819283619806697e+02,5.830388956390004235e-03,-5.265773321181702049e-02,9.988057517384720807e-01,-8.887025569089860255e+01,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00'
final_arr = [float(i) for i in final.split(',')][1:]
final_arr = np.array(final_arr).reshape(4,4)
initial_arr = [float(i) for i in initial.split(',')][1:]
initial_arr = np.array(initial_arr).reshape(4,4)
print(final_arr[:3,3]-initial_arr[:3,3])

# pose_26 = '0.874276 -0.291865 -0.387886 0.215497 0.200608 0.944868 -0.258807 0.0370422 0.442038 0.148456 0.884626 0.0734533'.split(' ')
# pose_26 = [float(p) for p in pose_26]
# pose_26 = np.array(pose_26).reshape(3,4)
# pose_33 = '0.654413 -0.283621 -0.70093 0.416737 0.330514 0.941036 -0.0721974 0.000804453 0.680077 -0.18442 0.709566 0.0763768'.split(' ')
# pose_33 = [float(p) for p in pose_33]
# pose_33 = np.array(pose_33).reshape(3, 4)
# print((pose_33-pose_26)[:,3])

icp_it = iterative_closest_point(point_clouds[0], point_clouds[-1], estimate_scale=True)
print(chamfer_distance(icp_it.Xt, point_clouds[-1]))

tic = time.perf_counter()
icp = icp_with_warp(point_clouds[0].to(device), point_clouds[-1].to(device))
toc = time.perf_counter()
print('Elapsed time: ', toc-tic)
print(icp)
# print('RMSE: ', icp.rmse[0].item())
# print('number of iterations: ', len(icp.t_history))

# rmses = np.array(rmses)
# print('Mean: ', rmses.mean())
# print('StDev: ', rmses.std())
# print(rmses)
# fig = px.histogram(rmses)

fig = plot_scene({
    "Pointcloud": {
        "pc1": point_clouds[0],
        "pc2:": point_clouds[-1],
        "aligned": Pointclouds(icp.Xt)
    }
})
fig.show()

print(icp_it.RTs.R.detach().cpu().numpy())

## Plot warp
# xy = icp.interp.detach().cpu().numpy()
# print(xy.shape)
# fig2 = px.line(x=xy[0], y=xy[1])
# fig2.show()