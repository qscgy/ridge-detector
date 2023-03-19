from numpy import dtype
import torch.nn as nn
import torch
from rloss.pytorch.deeplabv3plus.NormalizedCutLoss import NormalizedCutLoss
from rloss.pytorch.deeplabv3plus.DenseCRFLoss import DenseCRFLoss
from rloss.pytorch.deeplabv3plus.dataloaders.custom_transforms import denormalizeimage
from PIL import Image
import numpy as np
from natsort import natsorted
from glob import glob

def pixel_matching_loss(mask, target):
    loss = (mask != target) & (target < 254)
    return loss.sum((1,2)).float().mean()/torch.count_nonzero(target<254)

def improved_arc_loss(output, locs, params):
    pass

class SwirlLoss(nn.Module):
    def __init__(self, side) -> None:
        super().__init__()
        self.side = side
        self.ks = (2*side+1, 2*side+1)
        self.unfold = nn.Unfold(kernel_size=self.ks)
        self.sobelx = torch.tensor([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]
        ]).reshape(1,1,3,3).float()
        self.sobely = self.sobelx.swapaxes(2, 3)

    def find_edges(self, mask):
        '''
        Finds the edges of a mask using a dilation.

        Arguments:
            mask (Tensor) : binary tensor of dimension N x H x W
        Returns:
            IntTensor : tensor of same dimensions as mask, where entries of 1 are edges and all others are 0
        '''
        
        im = mask.int().unsqueeze(1)
        kernel = torch.tensor([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ]).reshape(1,1,3,3).int()

        edges = torch.clamp(torch.nn.functional.conv2d(im, kernel, padding=(1, 1)), 0, 1)-im
        return edges

    def forward(self, mask, normals):
        '''
        Computes the alignment of the segmentation mask to the normals.

        Arguments:
            mask (Tensor): BxHxW tensor of binary values (0 or 1) where 1 means a pixel is fold
            normals (Tensor): BxHxWx2 tensor of normal vectors projected into the imaging plane
        Returns:
            loss (float): the loss
        '''
        # These should be precomputed; they are constant w.r.t. network params
        dFxdy = torch.gradient(normals[:,0])[-2]
        dFydx = torch.gradient(normals[:,1])[-1]
        # dFxdy = torch.nn.functional.conv2d(normals[:,0].unsqueeze(1), self.sobelx, padding=1)
        # dFydx = torch.nn.functional.conv2d(normals[:,1].unsqueeze(1), self.sobely, padding=1)
        curl = dFxdy-dFydx

        loss = torch.sum(curl*mask)**2/mask.sum()
        return loss


    def forward_iterative(self, seg, field):
        field = field[...,:,2:-2, 2:-2]

        kernel = torch.tensor([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ]).reshape(1,1,3,3).int()
        edges = torch.clamp(torch.nn.functional.conv2d(seg, kernel, padding=(1, 1)), 0, 1) - seg

        locs = torch.where(edges[...,2:-2,2:-2]>0)
        tangents = torch.zeros((edges.shape[-1]-4, edges.shape[-2]-4, 2))
        for x, y in zip(locs[0], locs[1]):
            window = edges[x:x+5, y:y+5]
            if torch.count_nonzero(window)<2:
                continue
            ic, jc = torch.where(window==1)
            coords = torch.stack((jc, ic), 0)
            mean = torch.mean(coords, axis=1)
            print(mean)
            slope = torch.nan_to_num((torch.sum((coords[0]-mean[0])*(coords[1]-mean[1])))/torch.sum((coords[0]-mean[0])**2), nan=torch.inf)
            if x==98 and y==186:
                print(slope, mean)
            # slope = local_tangent(window)
            # print(slope)
            if slope is None:
                slope=torch.inf
            theta = torch.arctan(slope)
            tangents[x, y, 0] = torch.cos(theta)
            tangents[x, y, 1] = torch.sin(theta)
        
        loss = torch.abs(tangents[:,:,0]*field[:,:,0]+tangents[:,:,1]*field[:,:,1])/torch.count_nonzero(torch.sum(tangents**2, axis=2))
        return loss.sum()

if __name__=='__main__':
    criterion = SwirlLoss(side=2)

    name = '000'
    number = '027'
    normals = np.load(f'/playpen/Datasets/geodepth2/{name}/NFPS/images/{name}_{number}/{name}_{number}_nr_pred.npy')
    frame = Image.open(natsorted(glob(f'/playpen/Datasets/geodepth2/{name}/image/*.jpg'))[int(number)])

    normals_f = normals.copy()
    normals_f = np.transpose(normals_f, (2, 0, 1))
    normals_f = torch.from_numpy(normals_f)

    field = torch.stack((normals_f[1], -normals_f[0]), 2)
    norms = torch.linalg.norm(field, dim=2)
    field[:,:,0] /= norms
    field[:,:,1] /= norms
    field = field[None].permute(0, 3, 1, 2)
    field = field.repeat(2,1,1,1)
    print(field.shape)

    seg_img_file = f'/playpen/ridge-dtec/run/pascal/mobilenet4-96/ex_4_lucille/results-mine/{name}_frame031221.png'
    seg_img = Image.open(seg_img_file)
    seg_img = torch.from_numpy(np.asarray(seg_img.resize((field.shape[-1], field.shape[-2]))))
    seg_img = seg_img.permute(2, 0, 1)[None]
    seg_img = seg_img.repeat(2,1,1,1).float()
    print(seg_img.shape)
    seg_img.requires_grad_()

    mask = seg_img[:,0]==0      # red channel is 0
    mask.unsqueeze_(1)
    print(mask.shape)
    mask = mask.float()
    mask.requires_grad_()
    mask.retain_grad()

    loss = criterion(mask, field)
    print(loss)
    loss.backward()
    print(loss.grad)