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
from pytorch3d.ops.knn import knn_points
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.loss import chamfer_distance
from geomloss import SamplesLoss

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

        self.interp_z = Interpolator(n_pts, max=0.8)
    
    def forward(self, X):
        Xt = torch.clone(X)
        Xt[0,:,2] = self.interp_z(X[0,:,2])
        Xt = self.s[:, None, None] * torch.bmm(Xt, self.R) + self.T[:, None, :]
        return Xt[0]

def icp_with_warp(X, Y, iters=100):
    aligner = Aligner()
    aligner = aligner.to(device)
    optimizer = torch.optim.Adam(aligner.parameters(), lr=3e-4)

    X, num_points_X = oputil.convert_pointclouds_to_tensor(X)
    Y, num_points_Y = oputil.convert_pointclouds_to_tensor(Y)

    best_loss = 1
    criterion = SamplesLoss(loss="sinkhorn", scaling=0.3)
    for i in range(iters):
        loss = 0
        Xt = aligner(X)
        loss = criterion(Xt, Y)
        # print(loss.item())
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_sol = MyICPSoln(Xt.clone(), aligner.R.clone(), aligner.T.clone(), aligner.s.clone(), \
                                 torch.vstack((aligner.interp_z.x_ctl, aligner.interp_z.y_ctl)))
        loss.backward()
        optimizer.step()
    print(loss.item())
    
    return best_sol

def frame_num_to_index(fpath, frame):
    img_dir = os.path.dirname(fpath)
    frames = natsorted(os.listdir(img_dir))
    frames = [int(re.search(r'[0-9]{6}', f).group(0)) for f in frames if f[-3:] != 'npy']
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

def get_depth(fname, base_path):
    path = fname.split('/')
    frame = re.search(r'[0-9]{6}', path[-1]).group(0)
    num = base_path.split('/')[-1]
    orig_path = os.path.join(base_path, 'image', 'frame'+frame+'.jpg')
    index = frame_num_to_index(orig_path, int(frame))
    spec = 'depth'
    outs = os.path.join(base_path, 'NFPS', 'images', f'{num}_{index:03d}', f'{num}_{index:03d}_nr_{spec}.npy')
    print(path, outs)
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


base_path = '/playpen/Datasets/geodepth2/019'
seg_dir = base_path+'/results-mine'

num = base_path.split('/')[-1]
use_nr = False

p1 = 26
p2 = 32

def get_point_clouds():
    preds = np.load(os.path.join(seg_dir, 'preds.npy'))
    point_clouds = []
    shape = (preds.shape[-2], preds.shape[-1])
    crop_border = 20
    n_samples = 500
    for i in [p1, p2]:
        im = preds[i]
        seg_mask = torch.from_numpy(im>0.3)

        if use_nr:
            depth = np.load(os.path.join(
                base_path, 
                'NFPS/images', 
                f'{num}_{i:03d}', 
                f'{num}_{i:03d}_nr_depth.npy'
            ))
        else:
            depth_list = natsorted(glob(os.path.join(
                base_path, 
                'colon_norm_preall_abs_nosm/*_disp.npy'
            )))
            depth = 1/np.load(depth_list[i])
        h, w = depth.shape
        depth = depth[crop_border:h-crop_border, crop_border:w-crop_border]
        depth = resize(depth, shape)
        depth = torch.from_numpy(depth)
        # depth = flatten_each_fold(depth, seg_mask.astype('uint8'))
        Y, X = make_xy(depth)
        
        points = torch.stack((X, Y, depth), 0)
        fold_pts = torch.stack(torch.where(seg_mask.logical_and((X-0.5)**2+(Y-0.5)**2<=0.5**2)), 0)
        sample_inds = torch.randperm(fold_pts.shape[-1])[:n_samples]
        points = points[:,fold_pts[0, sample_inds], fold_pts[1, sample_inds]]
        points = Pointclouds(points=points.mT.unsqueeze(0))
        point_clouds.append(points)
    return point_clouds

rmses = []
point_clouds = get_point_clouds()

# this appears to produce better alignments than Pytorch3D ICP
aligner = Aligner()
icp = icp_with_warp(point_clouds[0].to(device), point_clouds[1].to(device), iters=100)

fig = plot_scene({
    "Pointcloud": {
        "pc1": point_clouds[0],
        "pc2:": point_clouds[1],
        "aligned": Pointclouds(icp.Xt)
    },
}, pointcloud_marker_size=2)
fig.write_html('pointclouds/3.html')

print(icp.interp)
print(icp.R)
print(icp.T)
print(icp.s)

# plt.plot(icp.interp[0].cpu().detach().numpy(), icp.interp[1].cpu().detach().numpy())
# plt.show()
